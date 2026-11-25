/* backend_npu.c -- optional Intel NPU lane, host side (see backend_npu.h).
 *
 * Resolves coli_npu.dll beside the executable at run time and never links OpenVINO, so an
 * NPU=1 build starts normally where there is no NPU, no OpenVINO and no helper. Every
 * outcome other than a working lane is reported on stderr once, and the engine continues
 * on its normal path. */

#include "backend_npu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__F16C__)
#include <immintrin.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#endif

/* ---- binding --------------------------------------------------------------------- */

static ColiNpuBinding g_bind = COLI_NPU_UNPROBED;
static char g_helper_path[1024];
#ifdef _WIN32
static HMODULE g_mod;
#else
static void *g_mod;
#endif

static struct {
    coli_npu_abi_fn abi;
    coli_npu_open_fn open;
    coli_npu_info_fn info;
    coli_npu_acquire_fn acquire;
    coli_npu_submit_fn submit;
    coli_npu_wait_fn wait;
    coli_npu_release_fn release;
    coli_npu_close_fn close;
    coli_npu_error_fn error;
} F;

void coli_npu_test_set_helper_path(const char *path) {
    snprintf(g_helper_path, sizeof g_helper_path, "%s", path ? path : "");
    g_bind = COLI_NPU_UNPROBED;
}

static int exe_dir(char *out, size_t n) {
#ifdef _WIN32
    DWORD k = GetModuleFileNameA(NULL, out, (DWORD)n);
    if (k == 0 || k >= n) return -1;
    char *s = strrchr(out, '\\'); if (!s) s = strrchr(out, '/');
    if (!s) return -1;
    *s = 0;
#else
    ssize_t k = readlink("/proc/self/exe", out, n - 1);
    if (k <= 0) return -1;
    out[k] = 0;
    char *s = strrchr(out, '/'); if (!s) return -1;
    *s = 0;
#endif
    return 0;
}

static void *sym(const char *name) {
#ifdef _WIN32
    return (void *)GetProcAddress(g_mod, name);
#else
    return dlsym(g_mod, name);
#endif
}

static void unload(void) {
    memset(&F, 0, sizeof F);
    if (g_mod) {
#ifdef _WIN32
        FreeLibrary(g_mod);
#else
        dlclose(g_mod);
#endif
    }
    g_mod = NULL;
}

ColiNpuBinding coli_npu_binding(void) {
    if (g_bind != COLI_NPU_UNPROBED) return g_bind;
    char path[1200];
    if (g_helper_path[0]) snprintf(path, sizeof path, "%s", g_helper_path);
    else {
        char dir[1024];
        if (exe_dir(dir, sizeof dir)) return g_bind = COLI_NPU_ABSENT;
#ifdef _WIN32
        snprintf(path, sizeof path, "%s\\%s", dir, COLI_NPU_HELPER_DLL);
#else
        snprintf(path, sizeof path, "%s/%s", dir, COLI_NPU_HELPER_DLL);
#endif
    }
#ifdef _WIN32
    if (GetFileAttributesA(path) == INVALID_FILE_ATTRIBUTES) return g_bind = COLI_NPU_ABSENT;
    /* OpenVINO's DLLs: beside the helper, in COLI_NPU_OPENVINO, or the default search. */
    const char *ovd = getenv("COLI_NPU_OPENVINO");
    if (ovd && *ovd) {
        wchar_t w[1024];
        if (MultiByteToWideChar(CP_UTF8, 0, ovd, -1, w, 1024) > 0) AddDllDirectory(w);
    }
    g_mod = LoadLibraryExA(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                                        LOAD_LIBRARY_SEARCH_USER_DIRS);
    if (!g_mod) g_mod = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);   /* PATH too */
#else
    if (access(path, R_OK)) return g_bind = COLI_NPU_ABSENT;
    g_mod = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
    if (!g_mod) return g_bind = COLI_NPU_LOAD_FAILED;
    F.abi = (coli_npu_abi_fn)sym("coli_npu_helper_abi");
    if (!F.abi || F.abi() != COLI_NPU_ABI_VERSION) { unload(); return g_bind = COLI_NPU_ABI_INCOMPATIBLE; }
    F.open = (coli_npu_open_fn)sym("coli_npu_helper_open");
    F.info = (coli_npu_info_fn)sym("coli_npu_helper_info");
    F.acquire = (coli_npu_acquire_fn)sym("coli_npu_helper_acquire");
    F.submit = (coli_npu_submit_fn)sym("coli_npu_helper_submit");
    F.wait = (coli_npu_wait_fn)sym("coli_npu_helper_wait");
    F.release = (coli_npu_release_fn)sym("coli_npu_helper_release");
    F.close = (coli_npu_close_fn)sym("coli_npu_helper_close");
    F.error = (coli_npu_error_fn)sym("coli_npu_helper_error");
    if (!F.open || !F.info || !F.acquire || !F.submit || !F.wait || !F.release || !F.close || !F.error) {
        unload(); return g_bind = COLI_NPU_SYMBOL_INCOMPLETE;
    }
    return g_bind = COLI_NPU_AVAILABLE;
}

const char *coli_npu_binding_text(ColiNpuBinding b) {
    switch (b) {
    case COLI_NPU_UNPROBED: return "not probed";
    case COLI_NPU_AVAILABLE: return "helper available";
    case COLI_NPU_ABSENT: return "no " COLI_NPU_HELPER_DLL " beside the executable";
    case COLI_NPU_LOAD_FAILED: return COLI_NPU_HELPER_DLL " would not load (OpenVINO's DLLs not found? set COLI_NPU_OPENVINO)";
    case COLI_NPU_ABI_INCOMPATIBLE: return COLI_NPU_HELPER_DLL " has another ABI version";
    case COLI_NPU_SYMBOL_INCOMPLETE: return COLI_NPU_HELPER_DLL " lacks an entry point";
    }
    return "?";
}

/* ---- f16 --------------------------------------------------------------------------- */

uint16_t coli_npu_f32_to_f16_1(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x007fffffu;
    int32_t exp = (int32_t)((x >> 23) & 0xff);
    if (exp == 0xff) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));   /* inf / nan */
    exp = exp - 127 + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);                           /* overflow */
    if (exp <= 0) {                                                              /* subnormal */
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x00800000u;
        const int shift = 14 - exp;
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;                      /* may carry into exp: correct */
    return (uint16_t)(sign | h);
}

void coli_npu_f32_to_f16(const float *src, uint16_t *dst, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
    for (; i + 8 <= n; i += 8)
        _mm_storeu_si128((__m128i *)(dst + i), _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
#endif
    for (; i < n; i++) dst[i] = coli_npu_f32_to_f16_1(src[i]);
}

/* ---- the lane ------------------------------------------------------------------------ */

struct ColiNpuLane {
    void *h;
    int ok;
    int min_rows, max_rows, nreq, nbuckets, stats, verbose;
    ColiNpuInfo info;
    ColiNpuStats st;
    char engine[32];
};

static int env_int(const char *k, int d) {
    const char *v = getenv(k);
    return v && *v ? atoi(v) : d;
}

static int parse_buckets(const char *s, int *b) {
    int n = 0;
    while (s && *s && n < COLI_NPU_MAX_BUCKETS) {
        int v = atoi(s);
        if (v > 0 && (n == 0 || v > b[n - 1])) b[n++] = v;
        const char *c = strchr(s, ',');
        if (!c) break;
        s = c + 1;
    }
    return n;
}

ColiNpuLane *coli_npu_lane_open(const char *engine, int D, int I, int act, float a0, float a1) {
    const char *req = getenv("COLI_NPU");
    if (!req || !*req || !strcmp(req, "0")) return NULL;
    ColiNpuBinding b = coli_npu_binding();
    if (b != COLI_NPU_AVAILABLE) {
        fprintf(stderr, "[NPU] requested, but %s: continuing on the normal path\n", coli_npu_binding_text(b));
        return NULL;
    }
    ColiNpuConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.abi = COLI_NPU_ABI_VERSION; cfg.D = D; cfg.I = I; cfg.act = act; cfg.act_a0 = a0; cfg.act_a1 = a1;
    cfg.nbuckets = parse_buckets(getenv("COLI_NPU_BUCKETS") ? getenv("COLI_NPU_BUCKETS") : "16,64,256", cfg.buckets);
    if (cfg.nbuckets < 1) { cfg.nbuckets = 1; cfg.buckets[0] = 64; }
    cfg.nreq = env_int("COLI_NPU_REQS", 3);
    if (cfg.nreq < 1) cfg.nreq = 1;
    if (cfg.nreq > 16) cfg.nreq = 16;
    /* the memory cap: a request holds x and y (rows x D f32) and the f16 weights */
    const int64_t cap = (int64_t)env_int("COLI_NPU_MEM_MB", 512) << 20;
    for (;;) {
        int64_t need = 0;
        for (int i = 0; i < cfg.nbuckets; i++)
            need += (int64_t)cfg.nreq * ((int64_t)cfg.buckets[i] * D * 8 + (int64_t)3 * I * D * 2);
        if (need <= cap) break;
        if (cfg.nreq > 1) cfg.nreq--;
        else if (cfg.nbuckets > 1) cfg.nbuckets--;   /* drop the largest bucket */
        else break;
    }
    cfg.verbose = env_int("COLI_NPU_VERBOSE", 0);
    cfg.device = getenv("COLI_NPU_DEVICE") ? getenv("COLI_NPU_DEVICE") : "NPU";
    char cache[1024] = "";
    const char *cd = getenv("COLI_NPU_CACHE");
    if (cd) snprintf(cache, sizeof cache, "%s", cd);
#ifdef _WIN32
    else if (getenv("LOCALAPPDATA")) snprintf(cache, sizeof cache, "%s\\colibri\\npu-cache", getenv("LOCALAPPDATA"));
    if (cache[0] && strcmp(cache, "0")) {
        char tmp[1024]; snprintf(tmp, sizeof tmp, "%s", cache);
        for (char *p = tmp + 3; *p; p++) if (*p == '\\') { *p = 0; CreateDirectoryA(tmp, NULL); *p = '\\'; }
        CreateDirectoryA(tmp, NULL);
    }
#endif
    cfg.cache_dir = (cache[0] && strcmp(cache, "0")) ? cache : NULL;

    ColiNpuLane *L = calloc(1, sizeof *L);
    if (!L) return NULL;
    snprintf(L->engine, sizeof L->engine, "%s", engine ? engine : "");
    L->h = F.open(&cfg);
    if (!L->h) {
        fprintf(stderr, "[NPU] requested, but the helper could not open %s: %s: continuing on the normal path\n",
                cfg.device, F.error());
        free(L);
        return NULL;
    }
    F.info(L->h, &L->info);
    L->ok = 1;
    L->nreq = cfg.nreq; L->nbuckets = cfg.nbuckets;
    L->max_rows = cfg.buckets[cfg.nbuckets - 1];
    L->min_rows = env_int("COLI_NPU_MIN_ROWS", 8);
    if (L->min_rows < 1) L->min_rows = 1;
    L->stats = env_int("COLI_NPU_STATS", 0);
    L->verbose = cfg.verbose;
    fprintf(stderr,
            "[NPU] experimental: %s routed experts of prompt chunks run on %s (%s, OpenVINO %s) with a\n"
            "[NPU] reduced-precision fp16 path; output may differ from the CPU path. Decode stays on the\n"
            "[NPU] CPU. Graphs execute on: %s. Buckets:",
            L->engine, cfg.device, L->info.device, L->info.version, L->info.exec_devices);
    for (int i = 0; i < cfg.nbuckets; i++) fprintf(stderr, " %d", cfg.buckets[i]);
    fprintf(stderr, " rows x %d requests, %.0f MiB host buffers, compiled in %.0f ms; min %d rows/expert\n",
            cfg.nreq, L->info.host_bytes / 1048576.0, L->info.compile_ms, L->min_rows);
    return L;
}

int coli_npu_lane_ok(const ColiNpuLane *L) { return L && L->ok; }
int coli_npu_lane_min_rows(const ColiNpuLane *L) { return L ? L->min_rows : 0; }
int coli_npu_lane_max_rows(const ColiNpuLane *L) { return L ? L->max_rows : 0; }
int coli_npu_lane_inflight_cap(const ColiNpuLane *L) { return L ? L->nreq : 0; }
ColiNpuStats *coli_npu_lane_stats(ColiNpuLane *L) { return L ? &L->st : NULL; }

void coli_npu_lane_fail(ColiNpuLane *L, const char *what) {
    if (!L || !L->ok) return;
    fprintf(stderr, "[NPU] %s failed: %s -- the NPU lane is off, continuing on the CPU path\n",
            what, F.error ? F.error() : "?");
    L->ok = 0;
}

int coli_npu_lane_acquire(ColiNpuLane *L, int rows, ColiNpuSlot *out) {
    if (!L || !L->ok) return -1;
    int id = F.acquire(L->h, rows, out);
    if (id == -1) coli_npu_lane_fail(L, "acquire");
    return id;
}
int coli_npu_lane_submit(ColiNpuLane *L, int id) {
    if (!L || !L->ok) return -1;
    if (F.submit(L->h, id)) { coli_npu_lane_fail(L, "submit"); return -1; }
    return 0;
}
int coli_npu_lane_wait(ColiNpuLane *L, int id, const float **y) {
    if (!L || !L->ok) return -1;
    if (F.wait(L->h, id, y)) { coli_npu_lane_fail(L, "wait"); return -1; }
    return 0;
}
int coli_npu_lane_release(ColiNpuLane *L, int id) {
    if (!L || !L->h) return -1;
    return F.release(L->h, id);
}

void coli_npu_lane_report(const ColiNpuLane *L, const char *scope) {
    if (!L || (!L->stats && !L->verbose)) return;
    const ColiNpuStats *s = &L->st;
    if (!s->chunks) return;
    fprintf(stderr,
            "[NPU] %s: %lld chunks; NPU %lld experts / %lld rows (+%lld padding); CPU %lld experts / %lld rows; "
            "decode->f16 %.2f s, NPU wait %.2f s, CPU experts %.2f s, MoE total %.2f s\n",
            scope ? scope : "stats", (long long)s->chunks, (long long)s->experts, (long long)s->rows,
            (long long)s->pad_rows, (long long)s->cpu_experts, (long long)s->cpu_rows,
            s->t_decode, s->t_wait, s->t_cpu, s->t_total);
}

void coli_npu_lane_close(ColiNpuLane *L) {
    if (!L) return;
    coli_npu_lane_report(L, "final");
    if (L->h && F.close) F.close(L->h);
    free(L);
}
