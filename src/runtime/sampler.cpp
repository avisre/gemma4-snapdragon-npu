// runtime/sampler.cpp
// fp16-native sampling for Gemma-4 E2B on ARMv8.2 (OnePlus 10 Pro: SM8450).

#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace gemma4 {

// --- fp16 widening helper -------------------------------------------------
static inline float f16_to_f32(float16_t_local x) {
#if GEMMA4_HAS_ARM_FP16
  return static_cast<float>(x);   // ARMv8.2 FP16: single FCVT, no library call
#else
  // Software widen of IEEE 754 half-precision bits.
  uint16_t h = static_cast<uint16_t>(x);
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp  = (h & 0x7C00u) >> 10;
  uint32_t mant = (h & 0x03FFu);
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) { f = sign; }
    else {
      // subnormal
      while (!(mant & 0x0400u)) { mant <<= 1; exp -= 1; }
      exp += 1; mant &= 0x03FFu;
      f = sign | ((exp + 112u) << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out; std::memcpy(&out, &f, 4); return out;
#endif
}

// =====================================================================
// greedy_sample — ARMv8.2 FP16 vectorized argmax (8 lanes per iteration)
// =====================================================================
int32_t greedy_sample(const float16_t_local* logits, size_t vocab) {
#if GEMMA4_HAS_ARM_FP16
  float16x8_t best_v = vdupq_n_f16((__fp16)-65504.0f);  // -fp16 max
  uint16x8_t  idx_v  = {0,1,2,3,4,5,6,7};
  uint16x8_t  best_i = idx_v;
  const uint16x8_t step = vdupq_n_u16(8);
  size_t i = 0;
  for (; i + 8 <= vocab; i += 8) {
    float16x8_t v = vld1q_f16(reinterpret_cast<const __fp16*>(logits + i));
    uint16x8_t  m = vcgtq_f16(v, best_v);              // v > best ?
    best_v = vbslq_f16(m, v, best_v);
    best_i = vbslq_u16(m, idx_v, best_i);
    idx_v  = vaddq_u16(idx_v, step);
  }
  // Horizontal reduce 8 lanes -> single max.
  __fp16   bv[8]; uint16_t bi[8];
  vst1q_f16(bv, best_v);
  vst1q_u16(bi, best_i);
  float    best = -INFINITY;
  int32_t  best_idx = 0;
  for (int k = 0; k < 8; ++k) {
    float f = static_cast<float>(bv[k]);
    if (f > best) { best = f; best_idx = bi[k]; }
  }
  // Tail.
  for (; i < vocab; ++i) {
    float f = f16_to_f32(logits[i]);
    if (f > best) { best = f; best_idx = static_cast<int32_t>(i); }
  }
  return best_idx;
#else
  float best = -INFINITY; int32_t best_idx = 0;
  for (size_t i = 0; i < vocab; ++i) {
    float f = f16_to_f32(logits[i]);
    if (f > best) { best = f; best_idx = static_cast<int32_t>(i); }
  }
  return best_idx;
#endif
}

// =====================================================================
// temp_sample — Gumbel-max over fp16 logits; no full softmax allocated.
// =====================================================================
int32_t temp_sample(const float16_t_local* logits, size_t vocab,
                    float temp, uint64_t* rng_state) {
  if (temp <= 1e-6f) return greedy_sample(logits, vocab);
  const float inv_t = 1.0f / temp;
  float best = -INFINITY; int32_t best_idx = 0;
  for (size_t i = 0; i < vocab; ++i) {
    float l = f16_to_f32(logits[i]) * inv_t;
    float u = uniform01(rng_state);
    if (u < 1e-20f) u = 1e-20f;
    float g = -std::log(-std::log(u));   // standard Gumbel
    float s = l + g;
    if (s > best) { best = s; best_idx = static_cast<int32_t>(i); }
  }
  return best_idx;
}

// =====================================================================
// top_k_sample — partial-sort to K, softmax over K, multinomial sample.
// =====================================================================
int32_t top_k_sample(const float16_t_local* logits, size_t vocab,
                     int k, float temp, uint64_t* rng_state) {
  if (k <= 1) return greedy_sample(logits, vocab);
  if (static_cast<size_t>(k) > vocab) k = static_cast<int>(vocab);

  std::vector<std::pair<float,int32_t>> heap;
  heap.reserve(vocab);
  for (size_t i = 0; i < vocab; ++i) {
    heap.emplace_back(f16_to_f32(logits[i]), static_cast<int32_t>(i));
  }
  std::partial_sort(
      heap.begin(), heap.begin() + k, heap.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });

  const float t = (temp > 1e-6f) ? temp : 1.0f;
  float max_l = heap[0].first;
  float sum   = 0.0f;
  std::vector<float> probs(k);
  for (int i = 0; i < k; ++i) {
    probs[i] = std::exp((heap[i].first - max_l) / t);
    sum += probs[i];
  }
  float r = uniform01(rng_state) * sum;
  float acc = 0.0f;
  for (int i = 0; i < k; ++i) {
    acc += probs[i];
    if (r <= acc) return heap[i].second;
  }
  return heap[k - 1].second;
}

}  // namespace gemma4
