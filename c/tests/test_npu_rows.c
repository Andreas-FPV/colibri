/* backend_npu_rows.h: the NPU lane's GGUF row -> f16 decode. For every type it accelerates,
 * the vector path must give the scalar path's bits exactly (gq_dequant_row, times the
 * expert's scale, round-to-nearest-even f16), on random blocks with random block scales
 * and several multipliers. A type it does not accelerate must take the scalar path and
 * agree too, and a partial block must be refused. Runs on any build: without AVX2 + F16C
 * every type is the scalar path and the comparison is trivially exact. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ggml_quants.h"
#include "../backend_npu.h"
#include "../backend_npu_rows.h"

static uint32_t rng = 0x1234567u;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

int main(void) {
    static const int types[] = { GQ_IQ2_XXS, GQ_IQ2_XS, GQ_IQ2_S, GQ_IQ3_XXS, GQ_IQ3_S,
                                 GQ_IQ4_NL, GQ_IQ4_XS, GQ_Q4_0, GQ_Q8_0, GQ_Q4_K };
    static const float muls[] = { 1.f, 0.731f, -3.25f, 1e-3f };
    const int cols = 2304;   /* Inkling-2-Mini's hidden size: 9 blocks of 256, 72 of 32 */
    int fail = 0, fast = 0;
    float *row = malloc(sizeof(float) * cols), *ref32 = malloc(sizeof(float) * cols);
    uint16_t *a = malloc(2 * cols), *b = malloc(2 * cols);
    uint8_t *src = malloc(4 * cols);
    for (size_t ti = 0; ti < sizeof types / sizeof types[0]; ti++) {
        const int t = types[ti], bl = GQ_TYPES[t].blck, by = GQ_TYPES[t].bytes;
        fast += coli_npu_rows_fast(t);
        for (int trial = 0; trial < 50; trial++) {
            const int nb = cols / bl;
            for (int i = 0; i < nb * by; i++) src[i] = (uint8_t)rnd();
            for (int i = 0; i < nb; i++) {   /* a finite, normal f16 block scale (Q4_K: d and dmin) */
                for (int k = 0; k < (t == GQ_Q4_K ? 2 : 1); k++) {
                    const uint16_t h = (uint16_t)((rnd() & 0x83FF) | ((6 + rnd() % 14) << 10));
                    memcpy(src + (int64_t)i * by + 2 * k, &h, 2);
                }
            }
            const float mul = muls[trial % 4];
            if (coli_npu_row_f16(t, src, cols, mul, a, row)) { printf("FAIL type %d: refused\n", t); fail++; break; }
            if (gq_dequant_row(t, src, ref32, cols)) { printf("FAIL type %d: scalar refused\n", t); fail++; break; }
            if (mul != 1.f) for (int i = 0; i < cols; i++) ref32[i] *= mul;
            coli_npu_f32_to_f16(ref32, b, cols);
            if (memcmp(a, b, 2 * (size_t)cols)) {
                int i = 0; while (a[i] == b[i]) i++;
                printf("FAIL type %d trial %d: element %d is 0x%04x, scalar 0x%04x\n", t, trial, i, a[i], b[i]);
                fail++; break;
            }
        }
        if (coli_npu_row_f16(t, src, bl / 2 > 0 ? bl / 2 : 1, 1.f, a, row) == 0 && bl > 1) {
            printf("FAIL type %d: a partial block was accepted\n", t); fail++;
        }
    }
    free(row); free(ref32); free(a); free(b); free(src);
    if (fail) { printf("test_npu_rows: %d failure(s)\n", fail); return 1; }
    printf("test_npu_rows: ok (%d types on the vector path, the rest scalar)\n", fast);
    return 0;
}
