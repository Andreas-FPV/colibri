/* Synthetic Intel NPU helper for hardware-free tests of the host side (backend_npu.c).
 * Contains NO OpenVINO and touches no device: it implements the coli_npu_helper_* ABI in
 * plain C and computes the expert on the CPU in double from the f16 weights it was given.
 *
 * It qualifies the binding (ABI and entry-point checks), the request life cycle (acquire,
 * -2 when a bucket is full, submit, wait, release), the memory cap and the failure path.
 * The numerics of the real NPU are measured separately on hardware (COLI_NPU_CHECK);
 * nothing here substitutes for that.
 *
 * Build variants:
 *   (default)      complete, the current ABI
 *   -DFAKE_ABI_OLD reports another ABI version -- must be refused
 *   -DFAKE_PARTIAL lacks coli_npu_helper_wait -- must be refused
 * Run-time switch: FAKE_NPU_FAIL_WAIT=1 makes every wait fail. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../backend_npu.h"

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

enum { R_FREE, R_HELD, R_RUNNING };

typedef struct {
    int state;
    float *x, *y;
    uint16_t *w13, *w2;
} Req;

typedef struct {
    ColiNpuConfig cfg;
    Req *req;               /* nbuckets x nreq */
} Fake;

static char g_err[256];

static float h2f(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(m | 1024), e - 25);
    return s ? -v : v;
}

API uint32_t coli_npu_helper_abi(void) {
#ifdef FAKE_ABI_OLD
    return COLI_NPU_ABI_VERSION + 100u;
#else
    return COLI_NPU_ABI_VERSION;
#endif
}

API void *coli_npu_helper_open(const ColiNpuConfig *cfg) {
    if (!cfg || cfg->abi != COLI_NPU_ABI_VERSION || cfg->nbuckets < 1) { snprintf(g_err, sizeof g_err, "bad config"); return NULL; }
    if (cfg->device && strcmp(cfg->device, "NPU")) {
        snprintf(g_err, sizeof g_err, "graphs would execute on CPU, not %s", cfg->device);
        return NULL;
    }
    Fake *f = calloc(1, sizeof *f);
    f->cfg = *cfg;
    f->cfg.device = NULL; f->cfg.cache_dir = NULL;   /* not owned */
    f->req = calloc((size_t)cfg->nbuckets * cfg->nreq, sizeof(Req));
    for (int b = 0; b < cfg->nbuckets; b++)
        for (int r = 0; r < cfg->nreq; r++) {
            Req *q = &f->req[b * cfg->nreq + r];
            q->x = calloc((size_t)cfg->buckets[b] * cfg->D, sizeof(float));
            q->y = calloc((size_t)cfg->buckets[b] * cfg->D, sizeof(float));
            q->w13 = calloc((size_t)2 * cfg->I * cfg->D, sizeof(uint16_t));
            q->w2 = calloc((size_t)cfg->D * cfg->I, sizeof(uint16_t));
        }
    return f;
}

API int coli_npu_helper_info(void *h, ColiNpuInfo *out) {
    Fake *f = h;
    memset(out, 0, sizeof *out);
    snprintf(out->device, sizeof out->device, "fake NPU");
    snprintf(out->exec_devices, sizeof out->exec_devices, "NPU");
    snprintf(out->version, sizeof out->version, "fake");
    out->device_mem_total = out->device_mem_alloc = -1;
    for (int b = 0; b < f->cfg.nbuckets; b++)
        out->host_bytes += (int64_t)f->cfg.nreq * ((int64_t)f->cfg.buckets[b] * f->cfg.D * 8 + (int64_t)3 * f->cfg.I * f->cfg.D * 2);
    return 0;
}

API int coli_npu_helper_acquire(void *h, int rows, ColiNpuSlot *out) {
    Fake *f = h;
    for (int b = 0; b < f->cfg.nbuckets; b++) {
        if (rows > f->cfg.buckets[b]) continue;
        for (int r = 0; r < f->cfg.nreq; r++) {
            Req *q = &f->req[b * f->cfg.nreq + r];
            if (q->state != R_FREE) continue;
            q->state = R_HELD;
            memset(q->x, 0, (size_t)f->cfg.buckets[b] * f->cfg.D * sizeof(float));
            out->x = q->x; out->w13 = q->w13; out->w2 = q->w2; out->rows = f->cfg.buckets[b];
            return b * 64 + r;
        }
        return -2;
    }
    snprintf(g_err, sizeof g_err, "%d rows exceed the largest bucket", rows);
    return -1;
}

static Req *find(Fake *f, int id, int *bucket) {
    const int b = id / 64, r = id % 64;
    if (id < 0 || b >= f->cfg.nbuckets || r >= f->cfg.nreq) return NULL;
    *bucket = f->cfg.buckets[b];
    return &f->req[b * f->cfg.nreq + r];
}

API int coli_npu_helper_submit(void *h, int id) {
    Fake *f = h; int rows;
    Req *q = find(f, id, &rows);
    if (!q || q->state != R_HELD) { snprintf(g_err, sizeof g_err, "submit of a request not held"); return -1; }
    const int D = f->cfg.D, I = f->cfg.I;
    double *hid = malloc(sizeof(double) * I);
    for (int s = 0; s < rows; s++) {
        const float *x = q->x + (size_t)s * D;
        for (int i = 0; i < I; i++) {
            double g = 0, u = 0;
            for (int d = 0; d < D; d++) {
                g += (double)x[d] * h2f(q->w13[(size_t)i * D + d]);
                u += (double)x[d] * h2f(q->w13[(size_t)(I + i) * D + d]);
            }
            if (f->cfg.act == 1)
                hid[i] = f->cfg.act_a0 * tanh(g / f->cfg.act_a0) / (1 + exp(-g)) * (f->cfg.act_a1 * tanh(u / f->cfg.act_a1));
            else
                hid[i] = g / (1 + exp(-g)) * u;
        }
        for (int d = 0; d < D; d++) {
            double y = 0;
            for (int i = 0; i < I; i++) y += hid[i] * h2f(q->w2[(size_t)d * I + i]);
            q->y[(size_t)s * D + d] = (float)y;
        }
    }
    free(hid);
    q->state = R_RUNNING;
    return 0;
}

#ifndef FAKE_PARTIAL
API int coli_npu_helper_wait(void *h, int id, const float **y) {
    Fake *f = h; int rows;
    Req *q = find(f, id, &rows);
    const char *fail = getenv("FAKE_NPU_FAIL_WAIT");
    if (fail && atoi(fail)) { snprintf(g_err, sizeof g_err, "synthetic wait failure"); return -1; }
    if (!q || q->state != R_RUNNING) { snprintf(g_err, sizeof g_err, "wait on a request not submitted"); return -1; }
    *y = q->y;
    return 0;
}
#endif

API int coli_npu_helper_release(void *h, int id) {
    Fake *f = h; int rows;
    Req *q = find(f, id, &rows);
    if (!q) return -1;
    q->state = R_FREE;
    return 0;
}

API void coli_npu_helper_close(void *h) {
    Fake *f = h;
    if (!f) return;
    for (int i = 0; i < f->cfg.nbuckets * f->cfg.nreq; i++) {
        free(f->req[i].x); free(f->req[i].y); free(f->req[i].w13); free(f->req[i].w2);
    }
    free(f->req);
    free(f);
}

API const char *coli_npu_helper_error(void) { return g_err; }
