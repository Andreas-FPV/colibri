// npu_poc.cpp -- standalone OpenVINO feasibility probe for an Intel NPU lane in colibri.
//
// Answers, with measurements on the machine it runs on, the questions the backend design
// depends on:
//   1. is the NPU visible through OpenVINO, and what does the plugin report about it;
//   2. can a MatMul be compiled and executed there, and how far is it from a float64 reference;
//   3. which weight precisions work (f16, int8 per channel, int4 per group);
//   4. can the weights be REPLACED without recompiling (weights as graph inputs);
//   5. does an Inkling-2-Mini expert (2304 -> 2x1152 -> 2304, softcapped SwiGLU) run there;
//   6. compile / transfer / execute latencies, CPU vs iGPU vs NPU.
//
// Nothing here is colibri code: it is a probe. Every number it prints is measured here.
//
// Build (MSVC, OpenVINO from the pip wheel or an archive): see build_poc.cmd.

#include <openvino/openvino.hpp>
#include <openvino/op/ops.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace ov;
using clk = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// ---------------------------------------------------------------- host-side quantizers

enum WMode { W_F16, W_I8, W_I4 };
static const char *wmode_name(WMode m) { return m == W_F16 ? "f16" : m == W_I8 ? "i8/ch" : "i4/g128"; }

struct QWeight {             // a [N, K] weight in one of the three forms, plus its dequantized copy
    WMode mode;
    int N, K, G;             // G: group size for i4 (K % G == 0)
    std::vector<ov::float16> f16;     // W_F16: N*K
    std::vector<int8_t> i8;           // W_I8:  N*K
    std::vector<uint8_t> i4;          // W_I4:  N*K/2, packed low nibble first, signed
    std::vector<ov::float16> scale;   // W_I8: N ; W_I4: N*(K/G)
    std::vector<float> deq;           // N*K, exactly what the device should multiply by
};

static QWeight quantize(const std::vector<float> &w, int N, int K, WMode mode, int G = 128) {
    QWeight q; q.mode = mode; q.N = N; q.K = K; q.G = G; q.deq.resize((size_t)N * K);
    if (mode == W_F16) {
        q.f16.resize((size_t)N * K);
        for (size_t i = 0; i < q.f16.size(); i++) { q.f16[i] = ov::float16(w[i]); q.deq[i] = (float)q.f16[i]; }
    } else if (mode == W_I8) {
        q.i8.resize((size_t)N * K); q.scale.resize(N);
        for (int n = 0; n < N; n++) {
            float amax = 1e-12f;
            for (int k = 0; k < K; k++) amax = std::max(amax, std::fabs(w[(size_t)n * K + k]));
            ov::float16 s(amax / 127.f); q.scale[n] = s; float sf = (float)s;
            for (int k = 0; k < K; k++) {
                int v = (int)std::lround(w[(size_t)n * K + k] / sf); v = std::clamp(v, -127, 127);
                q.i8[(size_t)n * K + k] = (int8_t)v; q.deq[(size_t)n * K + k] = v * sf;
            }
        }
    } else {
        int ng = K / G; q.i4.assign((size_t)N * K / 2, 0); q.scale.resize((size_t)N * ng);
        for (int n = 0; n < N; n++) for (int g = 0; g < ng; g++) {
            float amax = 1e-12f;
            for (int k = 0; k < G; k++) amax = std::max(amax, std::fabs(w[(size_t)n * K + g * G + k]));
            ov::float16 s(amax / 7.f); q.scale[(size_t)n * ng + g] = s; float sf = (float)s;
            for (int k = 0; k < G; k++) {
                size_t idx = (size_t)n * K + g * G + k;
                int v = (int)std::lround(w[idx] / sf); v = std::clamp(v, -8, 7);
                q.deq[idx] = v * sf;
                uint8_t nib = (uint8_t)(v & 0xF);
                if (idx & 1) q.i4[idx / 2] |= (uint8_t)(nib << 4); else q.i4[idx / 2] |= nib;
            }
        }
    }
    return q;
}

// ---------------------------------------------------------------- graph builders

// A weight sub-graph producing an f32 [N, K] (or [E, N, K]) tensor.
// as_input: the weight (and its scale) are Parameters, so new weights need no recompilation.
struct WeightNode { Output<Node> out; ParameterVector params; };

static WeightNode make_weight(const QWeight &q, bool as_input, int E = 0) {
    WeightNode r;
    Shape lead = E ? Shape{(size_t)E} : Shape{};
    auto with_lead = [&](Shape s) { Shape o = lead; o.insert(o.end(), s.begin(), s.end()); return o; };
    if (q.mode == W_F16) {
        std::shared_ptr<Node> w;
        if (as_input) { auto p = std::make_shared<op::v0::Parameter>(element::f16, with_lead({(size_t)q.N, (size_t)q.K})); r.params.push_back(p); w = p; }
        else w = std::make_shared<op::v0::Constant>(element::f16, Shape{(size_t)q.N, (size_t)q.K}, q.f16.data());
        r.out = std::make_shared<op::v0::Convert>(w, element::f32);
    } else if (q.mode == W_I8) {
        std::shared_ptr<Node> w, s;
        if (as_input) {
            auto pw = std::make_shared<op::v0::Parameter>(element::i8, with_lead({(size_t)q.N, (size_t)q.K}));
            auto ps = std::make_shared<op::v0::Parameter>(element::f16, with_lead({(size_t)q.N, 1}));
            r.params.push_back(pw); r.params.push_back(ps); w = pw; s = ps;
        } else {
            w = std::make_shared<op::v0::Constant>(element::i8, Shape{(size_t)q.N, (size_t)q.K}, q.i8.data());
            s = std::make_shared<op::v0::Constant>(element::f16, Shape{(size_t)q.N, 1}, q.scale.data());
        }
        auto wf = std::make_shared<op::v0::Convert>(w, element::f16);
        auto m = std::make_shared<op::v1::Multiply>(wf, s);
        r.out = std::make_shared<op::v0::Convert>(m, element::f32);
    } else {
        size_t ng = q.K / q.G;
        std::shared_ptr<Node> w, s;
        if (as_input) {
            auto pw = std::make_shared<op::v0::Parameter>(element::i4, with_lead({(size_t)q.N, ng, (size_t)q.G}));
            auto ps = std::make_shared<op::v0::Parameter>(element::f16, with_lead({(size_t)q.N, ng, 1}));
            r.params.push_back(pw); r.params.push_back(ps); w = pw; s = ps;
        } else {
            w = std::make_shared<op::v0::Constant>(element::i4, Shape{(size_t)q.N, ng, (size_t)q.G}, q.i4.data());
            s = std::make_shared<op::v0::Constant>(element::f16, Shape{(size_t)q.N, ng, 1}, q.scale.data());
        }
        auto wf = std::make_shared<op::v0::Convert>(w, element::f16);
        auto m = std::make_shared<op::v1::Multiply>(wf, s);
        std::vector<int64_t> shp = E ? std::vector<int64_t>{E, q.N, q.K} : std::vector<int64_t>{q.N, q.K};
        auto rs = std::make_shared<op::v1::Reshape>(m, op::v0::Constant::create(element::i64, Shape{shp.size()}, shp), false);
        r.out = std::make_shared<op::v0::Convert>(rs, element::f32);
    }
    return r;
}

// y[M,N] = x[M,K] * W[N,K]^T
static std::shared_ptr<Model> build_matmul(int M, const QWeight &q, bool w_as_input, ParameterVector &wparams) {
    auto x = std::make_shared<op::v0::Parameter>(element::f32, Shape{(size_t)M, (size_t)q.K});
    x->set_friendly_name("x");
    auto w = make_weight(q, w_as_input);
    auto y = std::make_shared<op::v0::MatMul>(x, w.out, false, true);
    ParameterVector ps{x}; ps.insert(ps.end(), w.params.begin(), w.params.end());
    wparams = w.params;
    return std::make_shared<Model>(OutputVector{y}, ps, "matmul");
}

static Output<Node> softcap(Output<Node> v, float c) {
    auto cc = op::v0::Constant::create(element::f32, Shape{}, {c});
    auto ic = op::v0::Constant::create(element::f32, Shape{}, {1.0f / c});
    return std::make_shared<op::v1::Multiply>(std::make_shared<op::v0::Tanh>(std::make_shared<op::v1::Multiply>(v, ic)), cc);
}

// Inkling-2 routed expert(s): h = softcap(g,4)*sigmoid(g)*softcap(u,25); y = h * Wd^T
// E == 0: one expert, x [M, D]. E > 0: E experts in one graph, x [E, M, D] (decode: each expert its row).
static std::shared_ptr<Model> build_expert(int M, const QWeight &wg, const QWeight &wu, const QWeight &wd,
                                           bool w_as_input, int E) {
    Shape xs = E ? Shape{(size_t)E, (size_t)M, (size_t)wg.K} : Shape{(size_t)M, (size_t)wg.K};
    auto x = std::make_shared<op::v0::Parameter>(element::f32, xs);
    auto g = make_weight(wg, w_as_input, E), u = make_weight(wu, w_as_input, E), d = make_weight(wd, w_as_input, E);
    auto gm = std::make_shared<op::v0::MatMul>(x, g.out, false, true);
    auto um = std::make_shared<op::v0::MatMul>(x, u.out, false, true);
    auto h = std::make_shared<op::v1::Multiply>(
        std::make_shared<op::v1::Multiply>(softcap(gm, 4.f), std::make_shared<op::v0::Sigmoid>(gm)), softcap(um, 25.f));
    auto y = std::make_shared<op::v0::MatMul>(h, d.out, false, true);
    ParameterVector ps{x};
    for (auto *w : {&g, &u, &d}) ps.insert(ps.end(), w->params.begin(), w->params.end());
    return std::make_shared<Model>(OutputVector{y}, ps, "expert");
}

// ---------------------------------------------------------------- reference + stats

static void ref_matmul(const float *x, const float *w, double *y, int M, int K, int N) {
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double s = 0; const float *a = x + (size_t)m * K, *b = w + (size_t)n * K;
        for (int k = 0; k < K; k++) s += (double)a[k] * b[k];
        y[(size_t)m * N + n] = s;
    }
}
static double sc(double v, double c) { return c * std::tanh(v / c); }
static void ref_expert(const float *x, const QWeight &g, const QWeight &u, const QWeight &d, double *y, int M) {
    int D = g.K, F = g.N;
    std::vector<double> gg((size_t)M * F), uu((size_t)M * F);
    ref_matmul(x, g.deq.data(), gg.data(), M, D, F);
    ref_matmul(x, u.deq.data(), uu.data(), M, D, F);
    std::vector<float> h((size_t)M * F);
    for (size_t i = 0; i < h.size(); i++) h[i] = (float)(sc(gg[i], 4) * (1 / (1 + std::exp(-gg[i]))) * sc(uu[i], 25));
    ref_matmul(h.data(), d.deq.data(), y, M, F, D);
}

struct Err { double max_abs = 0, rel_l2 = 0; };
static Err compare(const float *got, const double *ref, size_t n) {
    Err e; double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        double d = got[i] - ref[i]; e.max_abs = std::max(e.max_abs, std::fabs(d));
        num += d * d; den += ref[i] * ref[i];
    }
    e.rel_l2 = std::sqrt(num / std::max(den, 1e-30));
    return e;
}

// copy a QWeight into the Parameter tensors of a request (the "transfer" of a new expert)
static void bind_weight(InferRequest &rq, const ParameterVector &ps, size_t first, const QWeight &q, int e = -1) {
    auto put = [&](size_t idx, const void *src, size_t bytes) {
        Tensor t = rq.get_tensor(ps[idx]->output(0));
        size_t per = t.get_byte_size() / (e >= 0 ? t.get_shape()[0] : 1);
        std::memcpy((char *)t.data() + (e >= 0 ? per * e : 0), src, std::min(bytes, per));
    };
    if (q.mode == W_F16) put(first, q.f16.data(), q.f16.size() * 2);
    else if (q.mode == W_I8) { put(first, q.i8.data(), q.i8.size()); put(first + 1, q.scale.data(), q.scale.size() * 2); }
    else { put(first, q.i4.data(), q.i4.size()); put(first + 1, q.scale.data(), q.scale.size() * 2); }
}

static std::vector<float> randv(std::mt19937 &rng, size_t n, float sd) {
    std::normal_distribution<float> nd(0.f, sd); std::vector<float> v(n);
    for (auto &x : v) x = nd(rng);
    return v;
}

// ---------------------------------------------------------------- tests

static Core *g_core;
static int g_iters = 50;

static void print_devices() {
    Core &c = *g_core;
    printf("OpenVINO %s\n", ov::get_openvino_version().buildNumber);
    for (auto &d : c.get_available_devices()) {
        printf("device %-4s %s\n", d.c_str(), c.get_property(d, ov::device::full_name).c_str());
    }
    auto sup = c.get_property("NPU", ov::supported_properties);
    for (const char *k : {"DEVICE_ARCHITECTURE", "NPU_DRIVER_VERSION", "NPU_COMPILER_VERSION", "NPU_DEVICE_TOTAL_MEM_SIZE",
                          "NPU_DEVICE_ALLOC_MEM_SIZE", "OPTIMIZATION_CAPABILITIES", "NPU_MAX_TILES", "DEVICE_GOPS",
                          "NPU_COMPILER_TYPE", "RANGE_FOR_ASYNC_INFER_REQUESTS"}) {
        if (std::find(sup.begin(), sup.end(), k) == sup.end()) continue;
        try { printf("  NPU %-30s = %s\n", k, c.get_property("NPU", k).as<std::string>().c_str()); }
        catch (const std::exception &ex) { printf("  NPU %-30s = <%s>\n", k, ex.what()); }
    }
}

struct Timing { double compile = 0, first = 0, infer = 0, bind = 0; Err err; bool ok = false; std::string why; };

// matmul: weights constant (compiled in) or as inputs (swapped each iteration, verified each time)
static Timing run_matmul(const std::string &dev, int M, int K, int N, WMode wm, bool w_as_input, int nswap = 4) {
    Timing T; std::mt19937 rng(1234 + M + K + N + wm);
    std::vector<QWeight> ws;
    for (int i = 0; i < (w_as_input ? nswap : 1); i++) ws.push_back(quantize(randv(rng, (size_t)N * K, 0.02f), N, K, wm));
    auto x = randv(rng, (size_t)M * K, 1.f);
    try {
        ParameterVector wp;
        auto model = build_matmul(M, ws[0], w_as_input, wp);
        auto t0 = clk::now();
        CompiledModel cm = g_core->compile_model(model, dev, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        T.compile = ms_since(t0);
        InferRequest rq = cm.create_infer_request();
        std::memcpy(rq.get_tensor(model->get_parameters()[0]->output(0)).data(), x.data(), x.size() * 4);
        if (w_as_input) bind_weight(rq, model->get_parameters(), 1, ws[0]);
        t0 = clk::now(); rq.infer(); T.first = ms_since(t0);
        // steady state: for weight inputs, each iteration installs another weight set (copy + infer)
        double tb = 0, ti = 0; Err worst;
        std::vector<double> ref((size_t)M * N);
        for (int it = 0; it < g_iters; it++) {
            const QWeight &q = ws[it % ws.size()];
            auto tq = clk::now();
            if (w_as_input) bind_weight(rq, model->get_parameters(), 1, q);
            tb += ms_since(tq);
            tq = clk::now(); rq.infer(); ti += ms_since(tq);
            if (it < (int)ws.size()) {
                ref_matmul(x.data(), q.deq.data(), ref.data(), M, K, N);
                Err e = compare(rq.get_output_tensor(0).data<float>(), ref.data(), ref.size());
                worst.max_abs = std::max(worst.max_abs, e.max_abs); worst.rel_l2 = std::max(worst.rel_l2, e.rel_l2);
            }
        }
        T.bind = tb / g_iters; T.infer = ti / g_iters; T.err = worst; T.ok = true;
    } catch (const std::exception &ex) { T.why = ex.what(); }
    return T;
}

static Timing run_expert(const std::string &dev, int M, WMode wm, bool w_as_input, int E) {
    const int D = 2304, F = 1152;
    Timing T; std::mt19937 rng(77 + M + wm + E);
    int ne = std::max(E, 1);
    std::vector<QWeight> G, U, Dn;
    for (int e = 0; e < ne; e++) {
        G.push_back(quantize(randv(rng, (size_t)F * D, 0.03f), F, D, wm));
        U.push_back(quantize(randv(rng, (size_t)F * D, 0.03f), F, D, wm));
        Dn.push_back(quantize(randv(rng, (size_t)D * F, 0.03f), D, F, wm));
    }
    auto x = randv(rng, (size_t)ne * M * D, 1.f);
    try {
        if (E > 0 && !w_as_input) throw std::runtime_error("batched experts need weight inputs");
        auto model = build_expert(M, G[0], U[0], Dn[0], w_as_input, E);
        auto t0 = clk::now();
        CompiledModel cm = g_core->compile_model(model, dev, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        T.compile = ms_since(t0);
        InferRequest rq = cm.create_infer_request();
        auto ps = model->get_parameters();
        std::memcpy(rq.get_tensor(ps[0]->output(0)).data(), x.data(), x.size() * 4);
        size_t per = (wm == W_F16) ? 1 : 2;
        auto bind_all = [&]() {
            for (int e = 0; e < ne; e++) {
                bind_weight(rq, ps, 1, G[e], E ? e : -1);
                bind_weight(rq, ps, 1 + per, U[e], E ? e : -1);
                bind_weight(rq, ps, 1 + 2 * per, Dn[e], E ? e : -1);
            }
        };
        if (w_as_input) bind_all();
        t0 = clk::now(); rq.infer(); T.first = ms_since(t0);
        std::vector<double> ref((size_t)M * D);
        Err worst;
        for (int e = 0; e < ne; e++) {
            ref_expert(x.data() + (size_t)e * M * D, G[e], U[e], Dn[e], ref.data(), M);
            Err er = compare(rq.get_output_tensor(0).data<float>() + (size_t)e * M * D, ref.data(), ref.size());
            worst.max_abs = std::max(worst.max_abs, er.max_abs); worst.rel_l2 = std::max(worst.rel_l2, er.rel_l2);
        }
        double tb = 0, ti = 0;
        for (int it = 0; it < g_iters; it++) {
            auto tq = clk::now(); if (w_as_input) bind_all(); tb += ms_since(tq);
            tq = clk::now(); rq.infer(); ti += ms_since(tq);
        }
        T.bind = tb / g_iters; T.infer = ti / g_iters; T.err = worst; T.ok = true;
    } catch (const std::exception &ex) { T.why = ex.what(); }
    return T;
}

// A whole layer-pass of a prompt chunk: n_exp experts, each with `rows` rows, f16 weights as
// inputs, `nreq` infer requests in flight (fill one while the others run). Returns ms for the
// layer-pass and checks every expert's output against the float64 reference.
struct LayerRun { double compile = 0, total = 0, fill = 0; Err err; bool ok = false; std::string why; };
static LayerRun run_layer(const std::string &dev, int rows, int n_exp, int nreq, int E_per_call) {
    const int D = 2304, F = 1152; LayerRun R;
    std::mt19937 rng(4242 + rows);
    const int NW = 6;   // distinct weight sets, cycled (keeps host RAM small); each call is still a real swap
    std::vector<QWeight> G, U, Dn;
    for (int i = 0; i < NW; i++) {
        G.push_back(quantize(randv(rng, (size_t)F * D, 0.03f), F, D, W_F16));
        U.push_back(quantize(randv(rng, (size_t)F * D, 0.03f), F, D, W_F16));
        Dn.push_back(quantize(randv(rng, (size_t)D * F, 0.03f), D, F, W_F16));
    }
    const int EB = std::max(E_per_call, 1);
    auto x = randv(rng, (size_t)EB * rows * D, 1.f);
    try {
        auto model = build_expert(rows, G[0], U[0], Dn[0], true, E_per_call);
        auto t0 = clk::now();
        CompiledModel cm = g_core->compile_model(model, dev, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        R.compile = ms_since(t0);
        auto ps = model->get_parameters();
        std::vector<InferRequest> rq; std::vector<int> busy_first(nreq, -1);
        for (int i = 0; i < nreq; i++) {
            rq.push_back(cm.create_infer_request());
            std::memcpy(rq[i].get_tensor(ps[0]->output(0)).data(), x.data(), x.size() * 4);
        }
        auto fill = [&](InferRequest &r, int first) {
            for (int e = 0; e < EB; e++) {
                int w = (first + e) % NW;
                bind_weight(r, ps, 1, G[w], E_per_call ? e : -1);
                bind_weight(r, ps, 2, U[w], E_per_call ? e : -1);
                bind_weight(r, ps, 3, Dn[w], E_per_call ? e : -1);
            }
        };
        std::vector<double> ref((size_t)rows * D);
        auto check = [&](InferRequest &r, int first) {
            for (int e = 0; e < EB; e++) {
                int w = (first + e) % NW;
                ref_expert(x.data() + (size_t)e * rows * D, G[w], U[w], Dn[w], ref.data(), rows);
                Err er = compare(r.get_output_tensor(0).data<float>() + (size_t)e * rows * D, ref.data(), ref.size());
                R.err.max_abs = std::max(R.err.max_abs, er.max_abs); R.err.rel_l2 = std::max(R.err.rel_l2, er.rel_l2);
            }
        };
        for (int rep = 0; rep < 2; rep++) {           // rep 0 warms up, rep 1 is timed
            double tf = 0; int checked = 0;
            t0 = clk::now();
            int next = 0, slot = 0;
            for (; next < n_exp; next += EB) {
                InferRequest &r = rq[slot];
                if (busy_first[slot] >= 0) {
                    r.wait();
                    if (rep == 0 && checked < 3) { check(r, busy_first[slot]); checked++; }
                }
                auto tq = clk::now(); fill(r, next); tf += ms_since(tq);
                r.start_async(); busy_first[slot] = next;
                slot = (slot + 1) % nreq;
            }
            for (int i = 0; i < nreq; i++) if (busy_first[i] >= 0) { rq[i].wait(); busy_first[i] = -1; }
            if (rep == 1) { R.total = ms_since(t0); R.fill = tf; }
        }
        R.ok = true;
    } catch (const std::exception &ex) { R.why = ex.what(); }
    return R;
}

static void print_row(const char *what, const std::string &dev, const Timing &T) {
    if (!T.ok) {
        std::string w = T.why.substr(0, 160); std::replace(w.begin(), w.end(), '\n', ' ');
        printf("%-38s %-4s FAILED: %s\n", what, dev.c_str(), w.c_str()); return;
    }
    printf("%-38s %-4s compile %8.1f ms  first %7.2f ms  bind %7.3f ms  infer %7.3f ms  max|err| %.2e  relL2 %.2e\n",
           what, dev.c_str(), T.compile, T.first, T.bind, T.infer, T.err.max_abs, T.err.rel_l2);
    fflush(stdout);
}

int main(int argc, char **argv) {
    std::vector<std::string> devs = {"CPU", "GPU", "NPU"};
    std::string only;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) g_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dev") && i + 1 < argc) devs = {argv[++i]};
        else only = argv[i];
    }
    Core core; g_core = &core;
    print_devices();
    char buf[160];

    if (only.empty() || only == "matmul") {
        printf("\n== MatMul y[M,N] = x[M,K] W[N,K]^T, weights compiled in (const) or as inputs (swapped every call) ==\n");
        struct S { int M, K, N; } shapes[] = {{1, 2304, 2304}, {1, 2304, 1152}, {1, 1152, 2304}, {32, 2304, 2304}, {128, 2304, 2304}};
        for (auto s : shapes) for (WMode wm : {W_F16, W_I8, W_I4}) for (bool in : {false, true}) for (auto &d : devs) {
            snprintf(buf, sizeof buf, "mm %dx%dx%d %s %s", s.M, s.K, s.N, wmode_name(wm), in ? "W-in" : "W-const");
            print_row(buf, d, run_matmul(d, s.M, s.K, s.N, wm, in));
        }
    }
    if (only.empty() || only == "expert") {
        printf("\n== Inkling-2-Mini routed expert (2304 -> 2x1152 -> 2304, softcapped SwiGLU) ==\n");
        for (int M : {1, 16, 128}) for (WMode wm : {W_F16, W_I8, W_I4}) for (bool in : {false, true}) for (auto &d : devs) {
            snprintf(buf, sizeof buf, "expert M=%d %s %s", M, wmode_name(wm), in ? "W-in" : "W-const");
            print_row(buf, d, run_expert(d, M, wm, in, 0));
        }
        printf("\n== 8 experts in one graph (decode: one row each), weights as inputs ==\n");
        for (WMode wm : {W_F16, W_I8, W_I4}) for (auto &d : devs) {
            snprintf(buf, sizeof buf, "8 experts M=1 %s W-in", wmode_name(wm));
            print_row(buf, d, run_expert(d, 1, wm, true, 8));
        }
    }
    if (only.empty() || only == "layer") {
        printf("\n== One layer-pass of a prompt chunk: 128 experts, f16 weights as inputs, swapped per call ==\n");
        for (int rows : {8, 32, 64, 128}) for (int E : {0, 4}) for (int nreq : {1, 2, 4}) for (auto &d : devs) {
            if (d == "CPU" && nreq > 1) continue;
            LayerRun R = run_layer(d, rows, 128, nreq, E);
            if (!R.ok) { printf("layer rows=%-3d E/call=%d req=%d %-4s FAILED: %.150s\n", rows, E, nreq, d.c_str(), R.why.c_str()); continue; }
            double pairs = 128.0 * rows;
            printf("layer rows=%-3d E/call=%d req=%d %-4s compile %7.1f ms  layer %8.2f ms (fill %7.2f)  %8.0f pairs/s  max|err| %.2e relL2 %.2e\n",
                   rows, E, nreq, d.c_str(), R.compile, R.total, R.fill, pairs / (R.total / 1000.0), R.err.max_abs, R.err.rel_l2);
            fflush(stdout);
        }
    }
    return 0;
}
