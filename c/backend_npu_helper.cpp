// backend_npu_helper.cpp -- coli_npu.dll: colibri's optional Intel NPU lane, OpenVINO side.
//
// Built with MSVC against an OpenVINO runtime (c/tools/build_npu_helper.cmd). Exports the
// C entry points declared in backend_npu.h; nothing else of colibri is linked in.
//
// One compiled graph per row bucket computes ONE routed expert over `rows` rows with the
// expert's weights as graph INPUTS (f16), so any expert runs on the same compiled graph
// without recompilation (c/tools/npu_poc measured 0.3-3.3 s to compile a graph, which rules
// out compiling per expert). Each bucket owns `nreq` infer requests; the host decodes the
// next expert into a free request while the device runs the others.
//
// After compiling, the helper reads EXECUTION_DEVICES from every compiled graph and refuses
// to run unless it is exactly the requested device: OpenVINO must never move the work to the
// CPU behind colibri's back.

#include "backend_npu.h"

#include <openvino/openvino.hpp>
#include <openvino/op/ops.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#define NPU_EXPORT extern "C" __declspec(dllexport)
#else
#define NPU_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

thread_local std::string g_err;

void set_err(const std::string &s) { g_err = s; }

enum ReqState { FREE = 0, HELD, RUNNING, DONE };

struct Req {
    ov::InferRequest rq;
    ReqState st = FREE;
    float *x = nullptr; uint16_t *w13 = nullptr; uint16_t *w2 = nullptr;
};

struct Bucket {
    int rows = 0;
    ov::CompiledModel cm;
    std::vector<Req> reqs;
};

struct Lane {
    ColiNpuConfig cfg{};
    std::string device, cache_dir;
    ov::Core core;
    std::vector<Bucket> buckets;
    ColiNpuInfo info{};
    std::mutex mu;
};

ov::Output<ov::Node> softcap(const ov::Output<ov::Node> &v, float c) {
    using namespace ov::op;
    auto ic = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f / c});
    auto cc = v0::Constant::create(ov::element::f32, ov::Shape{}, {c});
    return std::make_shared<v1::Multiply>(std::make_shared<v0::Tanh>(std::make_shared<v1::Multiply>(v, ic)), cc);
}

std::shared_ptr<ov::Model> expert_graph(const ColiNpuConfig &c, int rows) {
    using namespace ov::op;
    const size_t D = (size_t)c.D, I = (size_t)c.I;
    auto x = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{(size_t)rows, D});
    auto w13 = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{2 * I, D});
    auto w2 = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{D, I});
    x->set_friendly_name("x"); w13->set_friendly_name("w13"); w2->set_friendly_name("w2");
    // infer requests look ports up by TENSOR name, which a friendly name does not set
    x->output(0).set_names({"x"}); w13->output(0).set_names({"w13"}); w2->output(0).set_names({"w2"});
    auto w13f = std::make_shared<v0::Convert>(w13, ov::element::f32);
    auto w2f = std::make_shared<v0::Convert>(w2, ov::element::f32);
    auto gu = std::make_shared<v0::MatMul>(x, w13f, false, true);              // [rows, 2I]
    auto axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto parts = std::make_shared<v1::Split>(gu, axis, 2);
    ov::Output<ov::Node> g = parts->output(0), u = parts->output(1), h;
    if (c.act == 1) {
        h = std::make_shared<v1::Multiply>(
            std::make_shared<v1::Multiply>(softcap(g, c.act_a0), std::make_shared<v0::Sigmoid>(g)),
            softcap(u, c.act_a1));
    } else {
        h = std::make_shared<v1::Multiply>(std::make_shared<v4::Swish>(g), u);
    }
    auto y = std::make_shared<v0::MatMul>(h, w2f, false, true);                // [rows, D]
    y->set_friendly_name("y");
    y->output(0).set_names({"y"});
    return std::make_shared<ov::Model>(ov::OutputVector{y}, ov::ParameterVector{x, w13, w2}, "coli_npu_expert");
}

int encode(int b, int r) { return b * 64 + r; }

}  // namespace

NPU_EXPORT uint32_t coli_npu_helper_abi(void) { return COLI_NPU_ABI_VERSION; }

NPU_EXPORT const char *coli_npu_helper_error(void) { return g_err.c_str(); }

NPU_EXPORT void *coli_npu_helper_open(const ColiNpuConfig *cfg) {
    if (!cfg || cfg->abi != COLI_NPU_ABI_VERSION) { set_err("config ABI mismatch"); return nullptr; }
    if (cfg->D <= 0 || cfg->I <= 0 || cfg->nbuckets < 1 || cfg->nbuckets > COLI_NPU_MAX_BUCKETS ||
        cfg->nreq < 1 || cfg->nreq > 32) { set_err("invalid config"); return nullptr; }
    std::unique_ptr<Lane> L(new Lane());
    try {
        L->cfg = *cfg;
        L->device = cfg->device && *cfg->device ? cfg->device : "NPU";
        if (cfg->cache_dir && *cfg->cache_dir) {
            L->cache_dir = cfg->cache_dir;
            L->core.set_property(ov::cache_dir(L->cache_dir));
        }
        auto devs = L->core.get_available_devices();
        bool found = false;
        for (auto &d : devs) found |= (d == L->device);
        if (!found) { set_err("OpenVINO sees no " + L->device + " device"); return nullptr; }

        std::snprintf(L->info.device, sizeof L->info.device, "%s",
                      L->core.get_property(L->device, ov::device::full_name).c_str());
        std::snprintf(L->info.version, sizeof L->info.version, "%s", ov::get_openvino_version().buildNumber);
        L->info.device_mem_total = L->info.device_mem_alloc = -1;
        auto mem_prop = [&](const char *k) -> int64_t {
            try {
                auto sup = L->core.get_property(L->device, ov::supported_properties);
                for (auto &p : sup) if (p == k) return (int64_t)L->core.get_property(L->device, k).as<uint64_t>();
            } catch (...) {}
            return -1;
        };
        L->info.device_mem_total = mem_prop("NPU_DEVICE_TOTAL_MEM_SIZE");

        auto t0 = std::chrono::steady_clock::now();
        std::string exec_all;
        for (int b = 0; b < cfg->nbuckets; b++) {
            Bucket B; B.rows = cfg->buckets[b];
            if (B.rows < 1 || B.rows > 4096) { set_err("invalid bucket"); return nullptr; }
            B.cm = L->core.compile_model(expert_graph(*cfg, B.rows), L->device,
                                         ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
            // The work must run where it was asked to: no silent fallback inside OpenVINO.
            std::string exec;
            for (auto &d : B.cm.get_property(ov::execution_devices)) exec += (exec.empty() ? "" : ",") + d;
            if (exec != L->device) {
                set_err("compiled graph executes on '" + exec + "', not on " + L->device + ": refusing");
                return nullptr;
            }
            exec_all = exec;
            for (int r = 0; r < cfg->nreq; r++) {
                Req q; q.rq = B.cm.create_infer_request();
                auto tx = q.rq.get_tensor("x"), t13 = q.rq.get_tensor("w13"), t2 = q.rq.get_tensor("w2");
                q.x = tx.data<float>(); q.w13 = (uint16_t *)t13.data(); q.w2 = (uint16_t *)t2.data();
                std::memset(q.x, 0, tx.get_byte_size());   // padding rows are finite
                L->info.host_bytes += (int64_t)(tx.get_byte_size() + t13.get_byte_size() + t2.get_byte_size() +
                                                q.rq.get_tensor("y").get_byte_size());
                B.reqs.push_back(std::move(q));
            }
            L->buckets.push_back(std::move(B));
        }
        L->info.compile_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::snprintf(L->info.exec_devices, sizeof L->info.exec_devices, "%s", exec_all.c_str());
        L->info.device_mem_alloc = mem_prop("NPU_DEVICE_ALLOC_MEM_SIZE");
    } catch (const std::exception &e) {
        set_err(std::string("OpenVINO: ") + e.what());
        return nullptr;
    }
    return L.release();
}

NPU_EXPORT int coli_npu_helper_info(void *h, ColiNpuInfo *out) {
    if (!h || !out) return -1;
    *out = static_cast<Lane *>(h)->info;
    return 0;
}

NPU_EXPORT int coli_npu_helper_acquire(void *h, int rows, ColiNpuSlot *out) {
    auto *L = static_cast<Lane *>(h);
    if (!L || !out || rows < 1) { set_err("acquire: bad arguments"); return -1; }
    std::lock_guard<std::mutex> lk(L->mu);
    for (size_t b = 0; b < L->buckets.size(); b++) {
        Bucket &B = L->buckets[b];
        if (B.rows < rows) continue;
        for (size_t r = 0; r < B.reqs.size(); r++) {
            Req &q = B.reqs[r];
            if (q.st != FREE) continue;
            q.st = HELD;
            out->x = q.x; out->w13 = q.w13; out->w2 = q.w2; out->rows = B.rows;
            return encode((int)b, (int)r);
        }
        return -2;   // the right bucket is busy: the caller completes one and retries
    }
    set_err("acquire: rows above the largest bucket");
    return -1;
}

static Req *find_req(Lane *L, int id) {
    int b = id / 64, r = id % 64;
    if (!L || id < 0 || b >= (int)L->buckets.size() || r >= (int)L->buckets[b].reqs.size()) return nullptr;
    return &L->buckets[b].reqs[r];
}

NPU_EXPORT int coli_npu_helper_submit(void *h, int id) {
    auto *L = static_cast<Lane *>(h);
    Req *q = find_req(L, id);
    if (!q || q->st != HELD) { set_err("submit: request not held"); return -1; }
    try { q->rq.start_async(); q->st = RUNNING; }
    catch (const std::exception &e) { set_err(std::string("start_async: ") + e.what()); return -1; }
    return 0;
}

NPU_EXPORT int coli_npu_helper_wait(void *h, int id, const float **y) {
    auto *L = static_cast<Lane *>(h);
    Req *q = find_req(L, id);
    if (!q || (q->st != RUNNING && q->st != DONE)) { set_err("wait: request not submitted"); return -1; }
    try {
        if (q->st == RUNNING) { q->rq.wait(); q->st = DONE; }
        if (y) *y = q->rq.get_tensor("y").data<float>();
    } catch (const std::exception &e) { set_err(std::string("wait: ") + e.what()); return -1; }
    return 0;
}

NPU_EXPORT int coli_npu_helper_release(void *h, int id) {
    auto *L = static_cast<Lane *>(h);
    Req *q = find_req(L, id);
    if (!q) { set_err("release: bad id"); return -1; }
    std::lock_guard<std::mutex> lk(L->mu);
    if (q->st == RUNNING) { try { q->rq.wait(); } catch (...) {} }
    q->st = FREE;
    return 0;
}

NPU_EXPORT void coli_npu_helper_close(void *h) {
    auto *L = static_cast<Lane *>(h);
    if (!L) return;
    for (auto &B : L->buckets) for (auto &q : B.reqs) if (q.st == RUNNING) { try { q.rq.wait(); } catch (...) {} }
    delete L;
}
