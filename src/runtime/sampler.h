// runtime/sampler.h
// Logit-sampling helpers for Gemma-4 E2B decode loop.
//
// Logits come out of the QNN/Vulkan backend as fp16 (__fp16 on ARM).
// All samplers below operate directly on the fp16 buffer using
// ARMv8.2-A FP16 intrinsics when available (the OnePlus 10 Pro / SM8450
// supports FEAT_FP16). On older targets we fall back to a scalar widen.

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#  define GEMMA4_HAS_ARM_FP16 1
#  include <arm_neon.h>
   using float16_t_local = __fp16;
#else
#  define GEMMA4_HAS_ARM_FP16 0
   using float16_t_local = uint16_t;  // raw bits; widened in software
#endif

namespace gemma4 {

// Greedy argmax over an fp16 logit vector of length `vocab`.
int32_t greedy_sample(const float16_t_local* logits, size_t vocab);

// Temperature sampling. `temp` <= 0 falls back to greedy.
// Uses Gumbel-max trick so we never materialize a full softmax.
int32_t temp_sample(const float16_t_local* logits, size_t vocab,
                    float temp, uint64_t* rng_state);

// Top-K sampling. Keeps the K largest logits, applies softmax over them,
// then samples. K defaults to 50.
int32_t top_k_sample(const float16_t_local* logits, size_t vocab,
                     int k, float temp, uint64_t* rng_state);

// xorshift64* — small, fast, deterministic; seed with non-zero.
inline uint64_t xorshift64(uint64_t* s) {
  uint64_t x = *s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *s = x;
  return x * 0x2545F4914F6CDD1DULL;
}
inline float uniform01(uint64_t* s) {
  return (xorshift64(s) >> 40) * (1.0f / 16777216.0f);  // 24-bit mantissa
}

}  // namespace gemma4
