/* Host side of the optional Intel NPU lane (backend_npu.c), without an NPU.
 *
 * Binds the synthetic helpers (tests/npu_fake_helper*.dll / .so) through the lane's own
 * loader and checks: the lane stays closed unless COLI_NPU asks for it; an absent helper,
 * another ABI version and a missing entry point are each refused with the normal path;
 * a helper that cannot run on the requested device is refused; the memory cap reduces
 * requests and then buckets; a request's life cycle (smallest bucket, -2 when a bucket is
 * full, submit, wait, release) computes the expert the engine asked for; a failed wait
 * turns the lane off; and the f32 -> f16 conversion rounds to nearest even, the F16C path
 * agreeing with the scalar one. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../backend_npu.h"

#ifdef _WIN32
#define DLEXT ".dll"
static void env_set(const char *k, const char *v) { _putenv_s(k, v ? v : ""); }
#else
#define DLEXT ".so"
static void env_set(const char *k, const char *v) { if (v) setenv(k, v, 1); else unsetenv(k); }
#endif

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static char g_dir[900] = ".";   /* the fake helpers sit beside this test's executable */
static void helper(const char *name) {
    char p[1024];
    snprintf(p, sizeof p, "%s/%s" DLEXT, g_dir, name);
    coli_npu_test_set_helper_path(p);
}

static uint32_t rng = 12345;
static float frand(void) { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) / 16777216.0f) * 2.f - 1.f; }

static float h2f(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexpf((float)(m | 1024), e - 25);
    return s ? -v : v;
}

static void test_f16(void) {
    static const struct { float f; uint16_t h; } k[] = {
        {0.f, 0x0000}, {-0.f, 0x8000}, {1.f, 0x3c00}, {-2.f, 0xc000}, {0.5f, 0x3800},
        {65504.f, 0x7bff}, {65520.f, 0x7c00}, {1e9f, 0x7c00}, {-1e9f, 0xfc00},
        {5.9604645e-8f, 0x0001}, {6.1035156e-5f, 0x0400}, {1e-9f, 0x0000},
        {1.00048828125f, 0x3c00},      /* 1 + 2^-11: a tie, rounds to even (down) */
        {1.00146484375f, 0x3c02},      /* 1 + 3*2^-11: a tie, rounds to even (up) */
        {1.000732421875f, 0x3c01},     /* above the tie: up */
    };
    for (size_t i = 0; i < sizeof k / sizeof k[0]; i++) {
        const uint16_t h = coli_npu_f32_to_f16_1(k[i].f);
        CHECK(h == k[i].h, "f16(%g) = 0x%04x, want 0x%04x", (double)k[i].f, h, k[i].h);
    }
    CHECK((coli_npu_f32_to_f16_1(NAN) & 0x7c00) == 0x7c00 && (coli_npu_f32_to_f16_1(NAN) & 0x3ff), "NaN stays NaN");
    /* the vector path (F16C when compiled in) against the scalar one, over many magnitudes */
    enum { N = 1 << 16 };
    float *src = malloc(N * sizeof *src); uint16_t *dst = malloc(N * sizeof *dst);
    for (int i = 0; i < N; i++) src[i] = frand() * ldexpf(1.f, (i % 48) - 30);
    coli_npu_f32_to_f16(src, dst, N);
    int bad = 0;
    for (int i = 0; i < N; i++) if (dst[i] != coli_npu_f32_to_f16_1(src[i])) bad++;
    CHECK(bad == 0, "%d of %d vector conversions differ from the scalar one", bad, N);
    free(src); free(dst);
}

static void test_refusals(void) {
    env_set("COLI_NPU", NULL);
    helper("npu_fake_helper");
    CHECK(coli_npu_lane_open("t", 64, 32, 1, 4.f, 25.f) == NULL, "lane opened without COLI_NPU");

    env_set("COLI_NPU", "0");
    CHECK(coli_npu_lane_open("t", 64, 32, 1, 4.f, 25.f) == NULL, "lane opened with COLI_NPU=0");

    env_set("COLI_NPU", "1");
    helper("no_such_npu_helper");
    CHECK(coli_npu_lane_open("t", 64, 32, 1, 4.f, 25.f) == NULL, "lane opened without a helper");
    CHECK(coli_npu_binding() == COLI_NPU_ABSENT, "absent helper: binding %d", coli_npu_binding());

    helper("npu_fake_helper_abi_old");
    CHECK(coli_npu_lane_open("t", 64, 32, 1, 4.f, 25.f) == NULL, "lane opened with another ABI");
    CHECK(coli_npu_binding() == COLI_NPU_ABI_INCOMPATIBLE, "old ABI: binding %d", coli_npu_binding());

    helper("npu_fake_helper_partial");
    CHECK(coli_npu_lane_open("t", 64, 32, 1, 4.f, 25.f) == NULL, "lane opened with a missing entry point");
    CHECK(coli_npu_binding() == COLI_NPU_SYMBOL_INCOMPLETE, "partial: binding %d", coli_npu_binding());

    helper("npu_fake_helper");
    env_set("COLI_NPU_DEVICE", "XPU");   /* the fake, like the real helper, refuses a device it would not run on */
    CHECK(coli_npu_lane_open("t", 64, 32, 1, 4.f, 25.f) == NULL, "lane opened on a device the helper refused");
    env_set("COLI_NPU_DEVICE", NULL);
}

static void test_mem_cap(void) {
    env_set("COLI_NPU", "1"); env_set("COLI_NPU_CACHE", "0");
    helper("npu_fake_helper");
    env_set("COLI_NPU_MEM_MB", "0");     /* nothing fits: one request of the smallest bucket remains */
    ColiNpuLane *L = coli_npu_lane_open("t", 64, 32, 1, 4.f, 25.f);
    CHECK(L != NULL, "lane did not open under a zero memory cap");
    CHECK(coli_npu_lane_inflight_cap(L) == 1 && coli_npu_lane_max_rows(L) == 16,
          "zero cap: %d requests, max %d rows; want 1 and 16", coli_npu_lane_inflight_cap(L), coli_npu_lane_max_rows(L));
    coli_npu_lane_close(L);
    env_set("COLI_NPU_MEM_MB", NULL);

    env_set("COLI_NPU_BUCKETS", "not,a,list");   /* unparsable: one bucket of 64 */
    L = coli_npu_lane_open("t", 64, 32, 1, 4.f, 25.f);
    CHECK(L && coli_npu_lane_max_rows(L) == 64, "bad bucket list: max %d rows", L ? coli_npu_lane_max_rows(L) : -1);
    coli_npu_lane_close(L);
    env_set("COLI_NPU_BUCKETS", NULL);
}

/* the expert the engine asks for, in double, from the same f16 weights */
static void ref_expert(const float *x, const uint16_t *w13, const uint16_t *w2, int rows, int D, int I, double *y) {
    for (int s = 0; s < rows; s++)
        for (int d = 0; d < D; d++) {
            double acc = 0;
            for (int i = 0; i < I; i++) {
                double g = 0, u = 0;
                for (int k = 0; k < D; k++) {
                    g += (double)x[s * D + k] * h2f(w13[i * D + k]);
                    u += (double)x[s * D + k] * h2f(w13[(I + i) * D + k]);
                }
                const double h = 4.0 * tanh(g / 4.0) / (1 + exp(-g)) * (25.0 * tanh(u / 25.0));
                acc += h * h2f(w2[d * I + i]);
            }
            y[s * D + d] = acc;
        }
}

static void test_lifecycle(void) {
    const int D = 64, I = 32, rows = 10;
    env_set("COLI_NPU", "1"); env_set("COLI_NPU_CACHE", "0");
    env_set("COLI_NPU_REQS", "2"); env_set("COLI_NPU_MIN_ROWS", "4");
    helper("npu_fake_helper");
    ColiNpuLane *L = coli_npu_lane_open("t", D, I, 1, 4.f, 25.f);
    CHECK(L && coli_npu_lane_ok(L), "lane did not open on the fake helper");
    if (!L) return;
    CHECK(coli_npu_lane_min_rows(L) == 4 && coli_npu_lane_max_rows(L) == 256 && coli_npu_lane_inflight_cap(L) == 2,
          "min %d max %d inflight %d", coli_npu_lane_min_rows(L), coli_npu_lane_max_rows(L), coli_npu_lane_inflight_cap(L));

    ColiNpuSlot a, b, c;
    const int ia = coli_npu_lane_acquire(L, rows, &a);
    CHECK(ia >= 0 && a.rows == 16, "acquire %d rows: id %d, bucket %d (want 16)", rows, ia, a.rows);
    const int ib = coli_npu_lane_acquire(L, rows, &b);
    CHECK(ib >= 0 && ib != ia, "second request: id %d", ib);
    const int ic = coli_npu_lane_acquire(L, rows, &c);
    CHECK(ic == -2 && coli_npu_lane_ok(L), "a full bucket must answer -2 and keep the lane: %d", ic);
    coli_npu_lane_release(L, ib);

    float *x = malloc(sizeof(float) * rows * D);
    for (int i = 0; i < rows * D; i++) x[i] = frand();
    float *w = malloc(sizeof(float) * 3 * I * D);
    for (int i = 0; i < 3 * I * D; i++) w[i] = frand() * 0.2f;
    memcpy(a.x, x, sizeof(float) * rows * D);
    coli_npu_f32_to_f16(w, a.w13, (int64_t)2 * I * D);
    coli_npu_f32_to_f16(w + 2 * I * D, a.w2, (int64_t)D * I);
    CHECK(coli_npu_lane_submit(L, ia) == 0, "submit");
    const float *y = NULL;
    CHECK(coli_npu_lane_wait(L, ia, &y) == 0 && y, "wait");
    if (y) {
        double *r = malloc(sizeof(double) * rows * D), num = 0, den = 0;
        ref_expert(x, a.w13, a.w2, rows, D, I, r);
        for (int i = 0; i < rows * D; i++) { num += (y[i] - r[i]) * (y[i] - r[i]); den += r[i] * r[i]; }
        CHECK(den > 0 && sqrt(num / den) < 1e-6, "expert output differs: rel L2 %g", sqrt(num / (den + 1e-30)));
        for (int i = rows * D; i < 16 * D; i++) if (y[i] != 0.f) { CHECK(0, "padding row %d not zero", i / D); break; }
        free(r);
    }
    coli_npu_lane_release(L, ia);

    /* a failed wait turns the lane off, and it stays off */
    const int id = coli_npu_lane_acquire(L, 3, &a);
    CHECK(id >= 0, "acquire after release");
    coli_npu_lane_submit(L, id);
    env_set("FAKE_NPU_FAIL_WAIT", "1");
    fprintf(stderr, "(the next [NPU] line is expected)\n");
    CHECK(coli_npu_lane_wait(L, id, &y) != 0, "a failed wait reported success");
    CHECK(!coli_npu_lane_ok(L), "lane still on after a failed wait");
    CHECK(coli_npu_lane_acquire(L, 3, &a) == -1, "acquire on a lane that is off");
    env_set("FAKE_NPU_FAIL_WAIT", NULL);
    coli_npu_lane_release(L, id);
    coli_npu_lane_close(L);
    free(x); free(w);
    env_set("COLI_NPU_REQS", NULL); env_set("COLI_NPU_MIN_ROWS", NULL);
}

int main(int argc, char **argv) {
    if (argc > 0 && argv[0]) {
        snprintf(g_dir, sizeof g_dir, "%s", argv[0]);
        char *s = strrchr(g_dir, '/'), *b = strrchr(g_dir, '\\');
        if (b > s) s = b;
        if (s) *s = 0; else snprintf(g_dir, sizeof g_dir, ".");
    }
    test_f16();
    test_refusals();
    test_mem_cap();
    test_lifecycle();
    coli_npu_test_set_helper_path(NULL);
    if (g_fail) { fprintf(stderr, "test_npu_lane: %d failure(s)\n", g_fail); return 1; }
    printf("test_npu_lane: ok\n");
    return 0;
}
