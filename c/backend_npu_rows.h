/* backend_npu_rows.h -- one GGUF row to f16 for the Intel NPU lane (docs/intel-npu.md).
 *
 * The NPU lane feeds every routed expert to the device as f16 weights, decoded on the host
 * from the GGUF's blocks; on Inkling-2-Mini Mix-12G that decode was 39 of a turn's 58 s of
 * MoE while the NPU waited 0.1 s. The scalar path decodes a row to f32 (gq_dequant_row),
 * scales it and converts it. Here the types whose value is exactly "group scale x small
 * integer" go through gq_ublk, the engine's own vector unpack (the integers and scales its
 * CPU kernels use), and the integers are scaled and converted eight at a time:
 *
 *     scalar:  y = (s * (float)q) * mul -> f16        vector: the same two products -> f16
 *
 * Both round the same products once each and convert with round-to-nearest-even, so the
 * result is bit-identical (tests/test_npu_rows.c checks every listed type against the
 * scalar path). Other types, and a build without AVX2 + F16C, keep the scalar path.
 *
 * Include after ggml_quants.h and backend_npu.h. */
#ifndef COLI_BACKEND_NPU_ROWS_H
#define COLI_BACKEND_NPU_ROWS_H

#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#define COLI_NPU_ROWS_VECTOR 1
#else
#define COLI_NPU_ROWS_VECTOR 0
#endif

/* types whose gq_ublk values carry no per-group minimum and whose scalar decode is s * q */
static inline int coli_npu_rows_fast(int t) {
#if COLI_NPU_ROWS_VECTOR
    switch (t) {
    case GQ_IQ2_XXS: case GQ_IQ2_XS: case GQ_IQ2_S: case GQ_IQ3_XXS: case GQ_IQ3_S:
    case GQ_IQ4_NL: case GQ_IQ4_XS: case GQ_Q4_0:
        return 1;
    default:
        return 0;
    }
#else
    (void)t;
    return 0;
#endif
}

/* cols elements of type t at src (whole blocks) -> dst f16, each times mul. row is a scratch of
 * cols floats for the scalar path. 0, or -1 for a type gq_dequant_row cannot decode. */
static inline int coli_npu_row_f16(int t, const uint8_t *src, int cols, float mul, uint16_t *dst, float *row) {
#if COLI_NPU_ROWS_VECTOR
    if (coli_npu_rows_fast(t)) {
        const int bl = GQ_TYPES[t].blck, by = GQ_TYPES[t].bytes;
        if (cols % bl) return -1;
        int8_t w8[256]; float sw[16], mw[16];
        const __m256 vm = _mm256_set1_ps(mul);
        for (int i = 0; i < cols / bl; i++) {
            gq_ublk(t, src + (int64_t)i * by, w8, sw, mw);
            uint16_t *o = dst + (int64_t)i * bl;
            for (int g = 0; g < bl / GQ_G; g++) {
                const __m128i q = _mm_loadu_si128((const __m128i *)(w8 + GQ_G * g));
                const __m256 s = _mm256_set1_ps(sw[g]);
                const __m256 lo = _mm256_mul_ps(_mm256_mul_ps(s, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q))), vm);
                const __m256 hi = _mm256_mul_ps(_mm256_mul_ps(s, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8)))), vm);
                _mm_storeu_si128((__m128i *)(o + GQ_G * g), _mm256_cvtps_ph(lo, _MM_FROUND_TO_NEAREST_INT));
                _mm_storeu_si128((__m128i *)(o + GQ_G * g + 8), _mm256_cvtps_ph(hi, _MM_FROUND_TO_NEAREST_INT));
            }
        }
        return 0;
    }
#endif
    if (gq_dequant_row(t, src, row, cols)) return -1;
    if (mul != 1.f) for (int i = 0; i < cols; i++) row[i] *= mul;
    coli_npu_f32_to_f16(row, dst, cols);
    return 0;
}

#endif
