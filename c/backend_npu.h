#ifndef COLIBRI_BACKEND_NPU_H
#define COLIBRI_BACKEND_NPU_H

/* Optional Intel NPU compute lane (OpenVINO) -- host side.
 *
 * Like the XDNA lane, the NPU is an OPTIONAL lane, never an engine. colibri keeps the
 * model's semantics, routing, expert cache, scheduling and fallback; the helper only
 * multiplies an expert that colibri already chose, decoded and handed over.
 *
 * This header and backend_npu.c NEVER include or link OpenVINO. OpenVINO lives behind an
 * optional helper DLL (coli_npu.dll, built with MSVC from backend_npu_helper.cpp), resolved
 * at run time beside the executable. A build with NPU=1 has the same import table as a
 * default one and starts normally on a machine with no NPU, no OpenVINO and no helper.
 *
 * What the helper computes (measured feasible in c/tools/npu_poc): ONE routed expert of
 * width D and intermediate I over R rows,
 *     g = x W13[0:I]^T,  u = x W13[I:2I]^T,  h = act(g, u),  y = h W2^T
 * with act the SwiGLU (act 0) or Inkling-2's softcapped one (act 1:
 * softcap(g, a0) * sigmoid(g) * softcap(u, a1)). x and y are f32; W13 [2I, D] and W2 [D, I]
 * are IEEE f16, written by colibri into the helper's own input tensors (no extra copy).
 * The NPU computes in fp16: this is a REDUCED-PRECISION path, opt-in only.
 *
 * The helper keeps, per row bucket, a small pool of infer requests so colibri can decode
 * the next expert while the NPU runs the previous ones. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bump when any structure or entry point below changes: a helper reporting another value
 * is refused outright (no partial binding). */
#define COLI_NPU_ABI_VERSION 1u

#ifdef _WIN32
#define COLI_NPU_HELPER_DLL "coli_npu.dll"
#else
#define COLI_NPU_HELPER_DLL "libcoli_npu.so"
#endif

#define COLI_NPU_MAX_BUCKETS 6

typedef struct {
    uint32_t abi;                  /* COLI_NPU_ABI_VERSION */
    int32_t  D, I;                 /* expert width and intermediate size */
    int32_t  act;                  /* 0 SwiGLU, 1 Inkling-2 softcapped SwiGLU */
    float    act_a0, act_a1;       /* act 1: gate and up softcaps */
    int32_t  nbuckets;             /* row buckets, ascending (each a compiled graph) */
    int32_t  buckets[COLI_NPU_MAX_BUCKETS];
    int32_t  nreq;                 /* infer requests per bucket (in flight at once) */
    int32_t  verbose;
    const char *device;            /* OpenVINO device: "NPU" (default); "GPU" for comparison runs */
    const char *cache_dir;         /* OpenVINO compiled-blob cache, or NULL */
} ColiNpuConfig;

/* One request's buffers, valid from acquire to release. */
typedef struct {
    float    *x;                   /* [rows, D] f32 input rows (rows = the bucket) */
    uint16_t *w13;                 /* [2I, D] f16: gate rows, then up rows */
    uint16_t *w2;                  /* [D, I] f16: down rows */
    int32_t   rows;                /* the bucket's row count (>= the rows asked for) */
} ColiNpuSlot;

typedef struct {
    char     device[96];           /* OpenVINO full device name */
    char     exec_devices[64];     /* EXECUTION_DEVICES of the compiled graphs: must be the device */
    char     version[64];          /* OpenVINO build */
    int64_t  device_mem_total;     /* NPU_DEVICE_TOTAL_MEM_SIZE, or -1 */
    int64_t  device_mem_alloc;     /* NPU_DEVICE_ALLOC_MEM_SIZE after compilation, or -1 */
    double   compile_ms;           /* all buckets, cold or from cache_dir */
    int64_t  host_bytes;           /* bytes of the request buffers */
} ColiNpuInfo;

/* Helper entry points (exported by coli_npu.dll with C linkage). Return 0 / a handle on
 * success; on failure a negative value / NULL and a message for coli_npu_helper_error. */
typedef uint32_t (*coli_npu_abi_fn)(void);
typedef void *(*coli_npu_open_fn)(const ColiNpuConfig *cfg);
typedef int (*coli_npu_info_fn)(void *h, ColiNpuInfo *out);
/* A free request of the smallest bucket >= rows: its id (>= 0) and buffers. -2: every
 * request of that bucket is in flight or unreleased (wait/release one, then retry).
 * -1: rows larger than the largest bucket, or an error. */
typedef int (*coli_npu_acquire_fn)(void *h, int rows, ColiNpuSlot *out);
typedef int (*coli_npu_submit_fn)(void *h, int id);
/* Wait for a submitted request; *y -> its [rows, D] f32 output, valid until release. */
typedef int (*coli_npu_wait_fn)(void *h, int id, const float **y);
typedef int (*coli_npu_release_fn)(void *h, int id);
typedef void (*coli_npu_close_fn)(void *h);
typedef const char *(*coli_npu_error_fn)(void);

/* ---- host side (backend_npu.c) ------------------------------------------------------ */

typedef enum {
    COLI_NPU_UNPROBED = 0,
    COLI_NPU_AVAILABLE,            /* helper bound, ABI matched, every entry point resolved */
    COLI_NPU_ABSENT,               /* no helper beside the executable */
    COLI_NPU_LOAD_FAILED,          /* present but would not load (missing OpenVINO DLLs?) */
    COLI_NPU_ABI_INCOMPATIBLE,
    COLI_NPU_SYMBOL_INCOMPLETE
} ColiNpuBinding;

ColiNpuBinding coli_npu_binding(void);
const char *coli_npu_binding_text(ColiNpuBinding b);

/* The lane as an engine sees it. coli_npu_lane_open reads the environment:
 *   COLI_NPU=1                 the request (nothing happens without it)
 *   COLI_NPU_DEVICE=NPU|GPU    OpenVINO device (default NPU)
 *   COLI_NPU_BUCKETS=16,64,256 row buckets (compiled graphs)
 *   COLI_NPU_REQS=3            requests in flight per bucket
 *   COLI_NPU_MEM_MB=512        cap on the helper's request buffers (fewer requests / buckets)
 *   COLI_NPU_MIN_ROWS=8        an expert with fewer rows in a chunk stays on the CPU
 *   COLI_NPU_CACHE=DIR         compiled-graph cache (default %LOCALAPPDATA%\colibri\npu-cache)
 *   COLI_NPU_OPENVINO=DIR      directory of openvino.dll, if not beside the helper or on PATH
 *   COLI_NPU_STATS=1, COLI_NPU_VERBOSE=1
 * Returns NULL (and says why on stderr, once) when the lane is not requested or cannot run;
 * the engine then continues on its normal path. */
typedef struct ColiNpuLane ColiNpuLane;

typedef struct {
    int64_t experts, rows, pad_rows, cpu_experts, cpu_rows, chunks;
    double  t_decode, t_wait, t_cpu, t_total;   /* seconds */
} ColiNpuStats;

ColiNpuLane *coli_npu_lane_open(const char *engine, int D, int I, int act, float a0, float a1);
int  coli_npu_lane_min_rows(const ColiNpuLane *L);
int  coli_npu_lane_max_rows(const ColiNpuLane *L);
int  coli_npu_lane_acquire(ColiNpuLane *L, int rows, ColiNpuSlot *out);
int  coli_npu_lane_submit(ColiNpuLane *L, int id);
int  coli_npu_lane_wait(ColiNpuLane *L, int id, const float **y);
int  coli_npu_lane_release(ColiNpuLane *L, int id);
int  coli_npu_lane_inflight_cap(const ColiNpuLane *L);
ColiNpuStats *coli_npu_lane_stats(ColiNpuLane *L);
/* A runtime failure: say so once, close the helper; later calls see a closed lane. */
void coli_npu_lane_fail(ColiNpuLane *L, const char *what);
int  coli_npu_lane_ok(const ColiNpuLane *L);
void coli_npu_lane_report(const ColiNpuLane *L, const char *scope);
void coli_npu_lane_close(ColiNpuLane *L);

/* f32 -> IEEE f16, round to nearest even (F16C when compiled for it). */
void coli_npu_f32_to_f16(const float *src, uint16_t *dst, int64_t n);
uint16_t coli_npu_f32_to_f16_1(float f);

/* test seams */
void coli_npu_test_set_helper_path(const char *path);

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_BACKEND_NPU_H */
