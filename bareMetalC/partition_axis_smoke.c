// See LICENSE for license details.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/gemmini_testutils_all.h"

/* Every dimension has a non-zero DIM tail. Mask 0x5 also exercises physical
 * members which are selected in a non-contiguous order. */
#define AXIS_SMOKE_I (2 * DIM + 3)
#define AXIS_SMOKE_J (3 * DIM - 5)
#define AXIS_SMOKE_K (2 * DIM + 1)
#define AXIS_SMOKE_MASK 0x5u

/* full_C=false moves out elem_t values in MAX_BYTES-sized groups.  Make the
 * logical row four full groups plus one partial DIM tile, while padding the
 * physical stride to the next full group so every row remains aligned. */
#define AXIS_SMOKE_PACKED_J (4 * MAX_BLOCK_LEN * DIM + 3)
#define AXIS_SMOKE_PACKED_STRIDE (5 * MAX_BLOCK_LEN * DIM)

typedef enum {
  AXIS_SMOKE_NO_BIAS = 0,
  AXIS_SMOKE_FULL_BIAS,
  AXIS_SMOKE_REPEATING_BIAS,
} axis_smoke_bias_t;

static elem_t axis_smoke_A[AXIS_SMOKE_I][AXIS_SMOKE_K]
    row_align(MAX_BLOCK_LEN);
static elem_t axis_smoke_B[AXIS_SMOKE_K][AXIS_SMOKE_J]
    row_align(MAX_BLOCK_LEN);
static acc_t axis_smoke_D[AXIS_SMOKE_I][AXIS_SMOKE_J]
    row_align_acc(MAX_BLOCK_LEN_ACC);
static acc_t axis_smoke_D_repeating[1][AXIS_SMOKE_J]
    row_align_acc(MAX_BLOCK_LEN_ACC);
static acc_t axis_smoke_C[AXIS_SMOKE_I][AXIS_SMOKE_J]
    row_align_acc(MAX_BLOCK_LEN_ACC);
static acc_t axis_smoke_gold[AXIS_SMOKE_I][AXIS_SMOKE_J]
    row_align_acc(MAX_BLOCK_LEN_ACC);
static elem_t axis_smoke_B_packed[AXIS_SMOKE_K][AXIS_SMOKE_PACKED_STRIDE]
    row_align(MAX_BLOCK_LEN);
static elem_t axis_smoke_C_packed[AXIS_SMOKE_I][AXIS_SMOKE_PACKED_STRIDE]
    row_align(MAX_BLOCK_LEN);
static acc_t axis_smoke_gold_packed[AXIS_SMOKE_I][AXIS_SMOKE_PACKED_J]
    row_align_acc(MAX_BLOCK_LEN_ACC);

static elem_t axis_smoke_elem_from_int(int value) {
#if defined(ELEM_T_IS_LOWPREC_FLOAT) && ELEM_T_EXP_BITS == 8 && \
    ELEM_T_SIG_BITS == 8
  union {
    float value;
    uint32_t bits;
  } encoded = {(float)value};
  return elem_t_bits_to_elem_t((elem_t_bits)(encoded.bits >> 16));
#else
  return (elem_t)value;
#endif
}

static acc_t axis_smoke_elem_to_acc(elem_t value) {
#if defined(ELEM_T_IS_LOWPREC_FLOAT) && ELEM_T_EXP_BITS == 8 && \
    ELEM_T_SIG_BITS == 8
  union {
    uint32_t bits;
    float value;
  } decoded = {(uint32_t)elem_t_to_elem_t_bits(value) << 16};
  return (acc_t)decoded.value;
#else
  return (acc_t)value;
#endif
}

static void axis_smoke_flush_mask(unsigned mask) {
  if ((mask & 0x1u) != 0)
    gemmini_flush(custom0, 0);
  if ((mask & 0x2u) != 0)
    gemmini_flush(custom1, 0);
  if ((mask & 0x4u) != 0)
    gemmini_flush(custom2, 0);
  if ((mask & 0x8u) != 0)
    gemmini_flush(custom3, 0);
}

static void axis_smoke_initialize(void) {
  for (size_t i = 0; i < AXIS_SMOKE_I; ++i) {
    for (size_t k = 0; k < AXIS_SMOKE_K; ++k) {
      const int value = (int)((i + 2 * k) % 5) - 2;
      axis_smoke_A[i][k] = axis_smoke_elem_from_int(value);
    }
  }

  for (size_t k = 0; k < AXIS_SMOKE_K; ++k) {
    for (size_t j = 0; j < AXIS_SMOKE_J; ++j) {
      const int value = (int)((3 * k + j) % 5) - 2;
      axis_smoke_B[k][j] = axis_smoke_elem_from_int(value);
    }
    for (size_t j = 0; j < AXIS_SMOKE_PACKED_J; ++j) {
      const int value = (int)((3 * k + j) % 5) - 2;
      axis_smoke_B_packed[k][j] = axis_smoke_elem_from_int(value);
    }
  }

  for (size_t i = 0; i < AXIS_SMOKE_I; ++i) {
    for (size_t j = 0; j < AXIS_SMOKE_J; ++j) {
      axis_smoke_D[i][j] = (acc_t)((int)((i + j) % 7) - 3);
    }
  }
  for (size_t j = 0; j < AXIS_SMOKE_J; ++j) {
    axis_smoke_D_repeating[0][j] = (acc_t)((int)(j % 5) - 2);
  }
}

static void axis_smoke_reference(axis_smoke_bias_t bias) {
  for (size_t i = 0; i < AXIS_SMOKE_I; ++i) {
    for (size_t j = 0; j < AXIS_SMOKE_J; ++j) {
      acc_t sum = 0;
      if (bias == AXIS_SMOKE_FULL_BIAS)
        sum = axis_smoke_D[i][j];
      else if (bias == AXIS_SMOKE_REPEATING_BIAS)
        sum = axis_smoke_D_repeating[0][j];

      for (size_t k = 0; k < AXIS_SMOKE_K; ++k) {
        sum += axis_smoke_elem_to_acc(axis_smoke_A[i][k]) *
               axis_smoke_elem_to_acc(axis_smoke_B[k][j]);
      }
      axis_smoke_gold[i][j] = sum;
    }
  }
}

static bool axis_smoke_equal(
    gemmini_partition_axis_t axis, acc_t got, acc_t expected) {
#ifdef ELEM_T_IS_FLOAT
  if (axis == GEMMINI_PARTITION_AXIS_K) {
    const float got_f = (float)got;
    const float expected_f = (float)expected;
    const float diff = got_f >= expected_f
                           ? got_f - expected_f
                           : expected_f - got_f;
    const float magnitude = expected_f >= 0.0f
                                ? expected_f
                                : -expected_f;
    return diff <= 1.0e-4f * (1.0f + magnitude);
  }
#else
  (void)axis;
#endif
  return acc_t_to_acc_t_bits(got) == acc_t_to_acc_t_bits(expected);
}

static bool axis_smoke_run_one(
    gemmini_partition_axis_t axis, axis_smoke_bias_t bias) {
  shared_multi_matmul_job_t job;
  memset(&job, 0, sizeof(job));
  memset(axis_smoke_C, 0xa5, sizeof(axis_smoke_C));
  axis_smoke_reference(bias);

  job.gemmini_list = AXIS_SMOKE_MASK;
  job.tile_id = 0;
  job.sp_addr_range = TOTAL_SPAD_ROWS;
  job.acc_addr_range = TOTAL_ACC_ROWS;
  job.dim_I = AXIS_SMOKE_I;
  job.dim_J = AXIS_SMOKE_J;
  job.dim_K = AXIS_SMOKE_K;
  job.A = &axis_smoke_A[0][0];
  job.B = &axis_smoke_B[0][0];
  job.D = bias == AXIS_SMOKE_NO_BIAS
              ? NULL
              : (bias == AXIS_SMOKE_FULL_BIAS
                     ? (const void *)&axis_smoke_D[0][0]
                     : (const void *)&axis_smoke_D_repeating[0][0]);
  job.C = &axis_smoke_C[0][0];
  job.stride_A = AXIS_SMOKE_K;
  job.stride_B = AXIS_SMOKE_J;
  job.stride_D = bias == AXIS_SMOKE_NO_BIAS ? 0 : AXIS_SMOKE_J;
  job.stride_C = AXIS_SMOKE_J;
  job.A_scale_factor = MVIN_SCALE_IDENTITY;
  job.B_scale_factor = MVIN_SCALE_IDENTITY;
  job.D_scale_factor = MVIN_SCALE_IDENTITY;
  job.act = NO_ACTIVATION;
  job.scale = ACC_SCALE_IDENTITY;
  job.repeating_bias = bias == AXIS_SMOKE_REPEATING_BIAS;
  job.full_C = true;
  job.low_D = false;
  job.weightA = 1;
  job.dataflow = WEIGHT_STATIONARY;
  job.partition_axis = axis;

  shared_multi_choose_tiling_factors(&job);
  shared_multi_tiled_matmul_job_init(&job);
  if (job.partition_status != SHARED_MULTI_PARTITION_OK) {
    printf("partition-axis init failed: axis=%d bias=%d status=%d\n",
           (int)axis, (int)bias, (int)job.partition_status);
    return false;
  }

  while (!job.done)
    shared_multi_tiled_matmul_job_step(&job);
  gemmini_fence();

  if (job.partition_status != SHARED_MULTI_PARTITION_OK) {
    printf("partition-axis step failed: axis=%d bias=%d status=%d\n",
           (int)axis, (int)bias, (int)job.partition_status);
    return false;
  }

  for (size_t i = 0; i < AXIS_SMOKE_I; ++i) {
    for (size_t j = 0; j < AXIS_SMOKE_J; ++j) {
      if (!axis_smoke_equal(axis,
                            axis_smoke_C[i][j], axis_smoke_gold[i][j])) {
        printf("partition-axis mismatch: axis=%d bias=%d i=%zu j=%zu "
               "got=0x%x expected=0x%x\n",
               (int)axis, (int)bias, i, j,
               (unsigned)acc_t_to_acc_t_bits(axis_smoke_C[i][j]),
               (unsigned)acc_t_to_acc_t_bits(axis_smoke_gold[i][j]));
        return false;
      }
    }
  }
  return true;
}

static bool axis_smoke_run_k_packed_store(void) {
  shared_multi_matmul_job_t job;
  memset(&job, 0, sizeof(job));
  memset(axis_smoke_C_packed, 0xa5, sizeof(axis_smoke_C_packed));

  for (size_t i = 0; i < AXIS_SMOKE_I; ++i) {
    for (size_t j = 0; j < AXIS_SMOKE_PACKED_J; ++j) {
      acc_t sum = 0;
      for (size_t k = 0; k < AXIS_SMOKE_K; ++k) {
        sum += axis_smoke_elem_to_acc(axis_smoke_A[i][k]) *
               axis_smoke_elem_to_acc(axis_smoke_B_packed[k][j]);
      }
      axis_smoke_gold_packed[i][j] = sum;
    }
  }

  job.gemmini_list = AXIS_SMOKE_MASK;
  job.tile_id = 0;
  job.sp_addr_range = TOTAL_SPAD_ROWS;
  job.acc_addr_range = TOTAL_ACC_ROWS;
  job.dim_I = AXIS_SMOKE_I;
  job.dim_J = AXIS_SMOKE_PACKED_J;
  job.dim_K = AXIS_SMOKE_K;
  job.A = &axis_smoke_A[0][0];
  job.B = &axis_smoke_B_packed[0][0];
  job.D = NULL;
  job.C = &axis_smoke_C_packed[0][0];
  job.stride_A = AXIS_SMOKE_K;
  job.stride_B = AXIS_SMOKE_PACKED_STRIDE;
  job.stride_D = 0;
  job.stride_C = AXIS_SMOKE_PACKED_STRIDE;
  job.A_scale_factor = MVIN_SCALE_IDENTITY;
  job.B_scale_factor = MVIN_SCALE_IDENTITY;
  job.D_scale_factor = MVIN_SCALE_IDENTITY;
  job.act = NO_ACTIVATION;
  job.scale = ACC_SCALE_IDENTITY;
  job.repeating_bias = false;
  job.full_C = false;
  job.low_D = false;
  job.weightA = 1;
  job.dataflow = WEIGHT_STATIONARY;
  job.partition_axis = GEMMINI_PARTITION_AXIS_K;

  shared_multi_choose_tiling_factors(&job);
  shared_multi_tiled_matmul_job_init(&job);
  if (job.partition_status != SHARED_MULTI_PARTITION_OK) {
    printf("partition-axis packed-store init failed: status=%d\n",
           (int)job.partition_status);
    return false;
  }

  while (!job.done)
    shared_multi_tiled_matmul_job_step(&job);
  gemmini_fence();

  if (job.partition_status != SHARED_MULTI_PARTITION_OK) {
    printf("partition-axis packed-store step failed: status=%d\n",
           (int)job.partition_status);
    return false;
  }

  for (size_t i = 0; i < AXIS_SMOKE_I; ++i) {
    for (size_t j = 0; j < AXIS_SMOKE_PACKED_J; ++j) {
      const elem_t expected =
          axis_smoke_elem_from_int((int)axis_smoke_gold_packed[i][j]);
      if (elem_t_to_elem_t_bits(axis_smoke_C_packed[i][j]) !=
          elem_t_to_elem_t_bits(expected)) {
        printf("partition-axis packed-store mismatch: i=%zu j=%zu "
               "got=0x%x expected=0x%x\n",
               i, j,
               (unsigned)elem_t_to_elem_t_bits(axis_smoke_C_packed[i][j]),
               (unsigned)elem_t_to_elem_t_bits(expected));
        return false;
      }
    }
  }

  printf("partition_axis_smoke: PASS "
         "(K, no-bias, full_C=false, MAX_BYTES/tail)\n");
  return true;
}

int main(void) {
#if !defined(GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION) || \
    GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION != 1
  printf("partition_axis_smoke: SKIP (generated shared-axis ABI unavailable)\n");
  return 0;
#else
  axis_smoke_initialize();
  axis_smoke_flush_mask(AXIS_SMOKE_MASK);

  for (int axis = GEMMINI_PARTITION_AXIS_M;
       axis <= GEMMINI_PARTITION_AXIS_K; ++axis) {
    for (int bias = AXIS_SMOKE_NO_BIAS;
         bias <= AXIS_SMOKE_REPEATING_BIAS; ++bias) {
      if (!axis_smoke_run_one(
              (gemmini_partition_axis_t)axis,
              (axis_smoke_bias_t)bias))
        return 1;
    }
  }

  if (!axis_smoke_run_k_packed_store())
    return 1;

  printf("partition_axis_smoke: PASS (M/N/K, mask=0x%x)\n",
         AXIS_SMOKE_MASK);
  return 0;
#endif
}
