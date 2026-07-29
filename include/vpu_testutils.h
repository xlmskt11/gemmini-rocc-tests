// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_INCLUDE_VPU_TESTUTILS_H_
#define GEMMINI_ROCC_TESTS_INCLUDE_VPU_TESTUTILS_H_

#include <math.h>
#include <stddef.h>

static inline void vpu_ref_rmsnorm(const float *input, const float *weight,
                                   float *output, size_t n, float epsilon) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    sum += input[i] * input[i];
  }
  const float scale = 1.0f / sqrtf(sum / (float)n + epsilon);
  for (size_t i = 0; i < n; ++i) {
    output[i] = input[i] * scale * weight[i];
  }
}

static inline float vpu_ref_silu_scalar(float x) {
  return x / (1.0f + expf(-x));
}

static inline void vpu_ref_silu(const float *input, float *output, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    output[i] = vpu_ref_silu_scalar(input[i]);
  }
}

static inline void vpu_ref_swiglu(const float *gate, const float *up,
                                  float *output, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    output[i] = vpu_ref_silu_scalar(gate[i]) * up[i];
  }
}

static inline void vpu_ref_softmax(const float *input, float *output,
                                   size_t n) {
  float maximum = -INFINITY;
  for (size_t i = 0; i < n; ++i) {
    maximum = fmaxf(maximum, input[i]);
  }
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    output[i] = expf(input[i] - maximum);
    sum += output[i];
  }
  for (size_t i = 0; i < n; ++i) {
    output[i] /= sum;
  }
}

static inline int vpu_ref_close(float actual, float expected,
                                float relative_tolerance,
                                float absolute_tolerance) {
  if (isnan(expected)) {
    return isnan(actual);
  }
  if (isinf(expected)) {
    return actual == expected;
  }
  const float difference = fabsf(actual - expected);
  const float scale = fmaxf(fabsf(actual), fabsf(expected));
  return difference <= absolute_tolerance + relative_tolerance * scale;
}

#endif  // GEMMINI_ROCC_TESTS_INCLUDE_VPU_TESTUTILS_H_
