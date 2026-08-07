// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_INCLUDE_VPU_TESTUTILS_H_
#define GEMMINI_ROCC_TESTS_INCLUDE_VPU_TESTUTILS_H_

#include <math.h>
#include <stddef.h>

#include "vpu_kernels.h"

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

static inline void vpu_ref_relu(const float *input, float *output, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    output[i] = fmaxf(input[i], 0.0f);
  }
}

static inline void vpu_ref_sigmoid(const float *input, float *output,
                                   size_t n) {
  for (size_t i = 0; i < n; ++i) {
    output[i] = 1.0f / (1.0f + expf(-input[i]));
  }
}

static inline void vpu_ref_tanh(const float *input, float *output, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    /* Match the emitted identity instead of libm tanhf's implementation. */
    output[i] = 2.0f / (1.0f + expf(-2.0f * input[i])) - 1.0f;
  }
}

static inline void vpu_ref_gelu(const float *input, float *output, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    output[i] = input[i] / (1.0f + expf(-1.702f * input[i]));
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

/* Mathematical reference for vpu_rope_auto().  Pair/half values are captured
 * before either output is written so exact in-place use is also well-defined. */
static inline void vpu_ref_rope(
    const float *input, const float *cosine, const float *sine,
    float *output, size_t rows, size_t rotary_dim,
    enum vpu_rope_layout layout) {
  const size_t half = rotary_dim / 2u;
  for (size_t row = 0; row < rows; ++row) {
    const size_t base = row * rotary_dim;
    if (layout == VPU_ROPE_INTERLEAVED) {
      for (size_t pair = 0; pair < half; ++pair) {
        const size_t first = base + 2u * pair;
        const size_t second = first + 1u;
        const float x_first = input[first];
        const float x_second = input[second];
        output[first] = x_first * cosine[2u * pair] -
                        x_second * sine[2u * pair];
        output[second] = x_second * cosine[2u * pair + 1u] +
                         x_first * sine[2u * pair + 1u];
      }
    } else {
      for (size_t element = 0; element < half; ++element) {
        const size_t first = base + element;
        const size_t second = first + half;
        const float x_first = input[first];
        const float x_second = input[second];
        output[first] = x_first * cosine[element] -
                        x_second * sine[element];
        output[second] = x_second * cosine[half + element] +
                         x_first * sine[half + element];
      }
    }
  }
}

/* Reference for a group-local gather. A fixed-size snapshot makes input ==
 * output behave like the VPU, which finishes the group load before storing. */
static inline void vpu_ref_permute(
    const float *input, const vpu_index_t *indices, float *output,
    size_t groups, size_t group_elements) {
  float snapshot[VPU_VLEN];
  for (size_t group = 0; group < groups; ++group) {
    const size_t base = group * group_elements;
    for (size_t element = 0; element < group_elements; ++element) {
      snapshot[element] = input[base + element];
    }
    for (size_t element = 0; element < group_elements; ++element) {
      const size_t index = (size_t)indices[element];
      output[base + element] = index < group_elements
          ? snapshot[index] : 0.0f;
    }
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
