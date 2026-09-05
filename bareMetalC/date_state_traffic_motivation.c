// See LICENSE for license details.

/*
 * DATE motivation microbenchmark for shared-state traffic.
 *
 * Each binary runs exactly one cold-boot case.  DATE_TRAFFIC_PRIVATE_LIKE=0
 * uses one four-pod M/N/K collective over the shared SPAD/ACC.  Setting it to
 * one partitions the same GEMM into four independent one-pod jobs with
 * disjoint local-memory quarters.  The latter is a command-stream emulation
 * of compulsory private-pod traffic, not a performance model of the private
 * RTL configuration.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils_all.h"

#ifndef DATE_TRAFFIC_I
#define DATE_TRAFFIC_I 32u
#endif
#ifndef DATE_TRAFFIC_J
#define DATE_TRAFFIC_J 256u
#endif
#ifndef DATE_TRAFFIC_K
#define DATE_TRAFFIC_K 256u
#endif
#ifndef DATE_TRAFFIC_AXIS
#define DATE_TRAFFIC_AXIS GEMMINI_PARTITION_AXIS_M
#endif
#ifndef DATE_TRAFFIC_PRIVATE_LIKE
#define DATE_TRAFFIC_PRIVATE_LIKE 0
#endif

#define DATE_TRAFFIC_PODS 4u
#define DATE_COUNTER_RDMA 0u
#define DATE_COUNTER_WDMA 1u

#if GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION != 1
#error "date_state_traffic_motivation requires the shared M/N/K RTL ABI"
#endif
#if DATE_TRAFFIC_PRIVATE_LIKE != 0 && DATE_TRAFFIC_PRIVATE_LIKE != 1
#error "DATE_TRAFFIC_PRIVATE_LIKE must be zero or one"
#endif
#if DATE_TRAFFIC_I == 0 || DATE_TRAFFIC_J == 0 || DATE_TRAFFIC_K == 0
#error "matrix dimensions must be nonzero"
#endif
#if DATE_TRAFFIC_I % DATE_TRAFFIC_PODS != 0 || \
    DATE_TRAFFIC_J % DATE_TRAFFIC_PODS != 0 || \
    DATE_TRAFFIC_K % DATE_TRAFFIC_PODS != 0
#error "all dimensions must be divisible by four"
#endif

static elem_t date_a[DATE_TRAFFIC_I][DATE_TRAFFIC_K]
    row_align(MAX_BLOCK_LEN);
static elem_t date_b[DATE_TRAFFIC_K][DATE_TRAFFIC_J]
    row_align(MAX_BLOCK_LEN);
static acc_t date_c[DATE_TRAFFIC_I][DATE_TRAFFIC_J]
    row_align_acc(MAX_BLOCK_LEN_ACC);
static acc_t date_partial[DATE_TRAFFIC_PODS][DATE_TRAFFIC_I][DATE_TRAFFIC_J]
    row_align_acc(MAX_BLOCK_LEN_ACC);

static inline uint64_t date_read_cycles(void) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles) : : "memory");
  return cycles;
}

static inline void date_cpu_fence(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static elem_t date_elem_one(void) {
#if defined(ELEM_T_IS_LOWPREC_FLOAT) && ELEM_T_EXP_BITS == 8 && \
    ELEM_T_SIG_BITS == 8
  union {
    float value;
    uint32_t bits;
  } encoded = {1.0f};
  return elem_t_bits_to_elem_t((elem_t_bits)(encoded.bits >> 16));
#else
  return (elem_t)1;
#endif
}

static const char *date_axis_name(gemmini_partition_axis_t axis) {
  switch (axis) {
    case GEMMINI_PARTITION_AXIS_M: return "M";
    case GEMMINI_PARTITION_AXIS_N: return "N";
    case GEMMINI_PARTITION_AXIS_K: return "K";
    default: return "?";
  }
}

static void date_initialize(void) {
  const elem_t one = date_elem_one();
  for (size_t i = 0; i < DATE_TRAFFIC_I; ++i) {
    for (size_t k = 0; k < DATE_TRAFFIC_K; ++k)
      date_a[i][k] = one;
  }
  for (size_t k = 0; k < DATE_TRAFFIC_K; ++k) {
    for (size_t j = 0; j < DATE_TRAFFIC_J; ++j)
      date_b[k][j] = one;
  }
  memset(date_c, 0, sizeof(date_c));
  memset(date_partial, 0, sizeof(date_partial));
}

static void date_flush_all(void) {
  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
  gemmini_flush(custom3, 0);
  gemmini_fence();
}

static void date_reset_counters(void) {
  for (int custom = 0; custom < (int)DATE_TRAFFIC_PODS; ++custom) {
    counter_configure(custom, DATE_COUNTER_RDMA, RDMA_BYTES_REC);
    counter_configure(custom, DATE_COUNTER_WDMA, WDMA_BYTES_SENT);
    counter_reset(custom);
  }
  gemmini_fence();
}

static bool date_configure_job(
    shared_multi_matmul_job_t *job,
    unsigned mask,
    unsigned tile_id,
    size_t sp_start,
    size_t sp_end,
    size_t acc_start,
    size_t sp_range,
    size_t acc_range,
    size_t dim_i,
    size_t dim_j,
    size_t dim_k,
    const elem_t *a,
    const elem_t *b,
    acc_t *c,
    size_t stride_a,
    size_t stride_b,
    size_t stride_c,
    gemmini_partition_axis_t axis) {
  memset(job, 0, sizeof(*job));
  job->gemmini_list = (int)mask;
  job->tile_id = (int)tile_id;
  job->sp_addr_start_stack = sp_start;
  job->sp_addr_end_stack = sp_end;
  job->acc_addr_start_stack = acc_start;
  job->sp_addr_range = sp_range;
  job->acc_addr_range = acc_range;
  job->dim_I = dim_i;
  job->dim_J = dim_j;
  job->dim_K = dim_k;
  job->A = a;
  job->B = b;
  job->D = NULL;
  job->C = c;
  job->stride_A = stride_a;
  job->stride_B = stride_b;
  job->stride_D = 0;
  job->stride_C = stride_c;
  job->A_scale_factor = MVIN_SCALE_IDENTITY;
  job->B_scale_factor = MVIN_SCALE_IDENTITY;
  job->D_scale_factor = MVIN_SCALE_IDENTITY;
  job->act = NO_ACTIVATION;
  job->scale = ACC_SCALE_IDENTITY;
  job->bert_scale = 0;
  job->repeating_bias = false;
  job->a_transpose = false;
  job->b_transpose = false;
  job->full_C = true;
  job->low_D = false;
  job->weightA = 1;
  job->dataflow = WEIGHT_STATIONARY;
  job->partition_axis = axis;

  shared_multi_choose_tiling_factors(job);
  if (job->partition_status != SHARED_MULTI_PARTITION_OK ||
      job->tile_I == 0 || job->tile_J == 0 || job->tile_K == 0)
    return false;
  shared_multi_tiled_matmul_job_init(job);
  return job->partition_status == SHARED_MULTI_PARTITION_OK && !job->done;
}

static bool date_prepare_shared(
    shared_multi_matmul_job_t jobs[DATE_TRAFFIC_PODS],
    gemmini_partition_axis_t axis) {
  return date_configure_job(
      &jobs[0], 0xfu, 0u,
      0u, 0u, 0u, TOTAL_SPAD_ROWS, TOTAL_ACC_ROWS,
      DATE_TRAFFIC_I, DATE_TRAFFIC_J, DATE_TRAFFIC_K,
      &date_a[0][0], &date_b[0][0], &date_c[0][0],
      DATE_TRAFFIC_K, DATE_TRAFFIC_J, DATE_TRAFFIC_J, axis);
}

static bool date_prepare_private_like(
    shared_multi_matmul_job_t jobs[DATE_TRAFFIC_PODS],
    gemmini_partition_axis_t requested_axis) {
  const size_t sp_rows_per_bank = BANK_ROWS / DATE_TRAFFIC_PODS;
  const size_t acc_rows_per_half = (ACC_ROWS / 2u) / DATE_TRAFFIC_PODS;
  for (size_t pod = 0; pod < DATE_TRAFFIC_PODS; ++pod) {
    size_t dim_i = DATE_TRAFFIC_I;
    size_t dim_j = DATE_TRAFFIC_J;
    size_t dim_k = DATE_TRAFFIC_K;
    const elem_t *a = &date_a[0][0];
    const elem_t *b = &date_b[0][0];
    acc_t *c = &date_c[0][0];

    if (requested_axis == GEMMINI_PARTITION_AXIS_M) {
      dim_i = DATE_TRAFFIC_I / DATE_TRAFFIC_PODS;
      a += pod * dim_i * DATE_TRAFFIC_K;
      c += pod * dim_i * DATE_TRAFFIC_J;
    } else if (requested_axis == GEMMINI_PARTITION_AXIS_N) {
      dim_j = DATE_TRAFFIC_J / DATE_TRAFFIC_PODS;
      b += pod * dim_j;
      c += pod * dim_j;
    } else {
      dim_k = DATE_TRAFFIC_K / DATE_TRAFFIC_PODS;
      a += pod * dim_k;
      b += pod * dim_k * DATE_TRAFFIC_J;
      c = &date_partial[pod][0][0];
    }

    const size_t sp_start = pod * sp_rows_per_bank;
    const size_t acc_start = pod * acc_rows_per_half;
    if (!date_configure_job(
            &jobs[pod], 1u << pod, (unsigned)pod,
            sp_start, sp_start, acc_start,
            BANK_NUM * sp_rows_per_bank, 2u * acc_rows_per_half,
            dim_i, dim_j, dim_k, a, b, c,
            DATE_TRAFFIC_K, DATE_TRAFFIC_J, DATE_TRAFFIC_J,
            GEMMINI_PARTITION_AXIS_M))
      return false;
  }
  return true;
}

static bool date_run_jobs(
    shared_multi_matmul_job_t jobs[DATE_TRAFFIC_PODS],
    size_t job_count,
    uint64_t *wall_cycles,
    size_t *scheduler_rounds) {
  gemmini_fence();
  date_reset_counters();
  date_cpu_fence();
  const uint64_t start = date_read_cycles();

  size_t live = job_count;
  size_t rounds = 0;
  while (live != 0) {
    ++rounds;
    for (size_t job = 0; job < job_count; ++job) {
      if (jobs[job].done)
        continue;
      shared_multi_tiled_matmul_job_step(&jobs[job]);
      if (jobs[job].partition_status != SHARED_MULTI_PARTITION_OK)
        return false;
      if (jobs[job].done)
        --live;
    }
  }
  gemmini_fence();
  const uint64_t end = date_read_cycles();
  date_cpu_fence();
  *wall_cycles = end - start;
  *scheduler_rounds = rounds;
  return true;
}

static bool date_check(gemmini_partition_axis_t axis, bool private_like) {
  if (private_like && axis == GEMMINI_PARTITION_AXIS_K) {
    const acc_t expected = (acc_t)(DATE_TRAFFIC_K / DATE_TRAFFIC_PODS);
    for (size_t pod = 0; pod < DATE_TRAFFIC_PODS; ++pod) {
      for (size_t i = 0; i < DATE_TRAFFIC_I; ++i) {
        for (size_t j = 0; j < DATE_TRAFFIC_J; ++j) {
          if (date_partial[pod][i][j] != expected) {
            printf("DATE_TRAFFIC mismatch partial=%u i=%u j=%u got=%f expected=%f\n",
                   (unsigned)pod, (unsigned)i, (unsigned)j,
                   (double)date_partial[pod][i][j], (double)expected);
            return false;
          }
        }
      }
    }
    return true;
  }

  const acc_t expected = (acc_t)DATE_TRAFFIC_K;
  for (size_t i = 0; i < DATE_TRAFFIC_I; ++i) {
    for (size_t j = 0; j < DATE_TRAFFIC_J; ++j) {
      if (date_c[i][j] != expected) {
        printf("DATE_TRAFFIC mismatch i=%u j=%u got=%f expected=%f\n",
               (unsigned)i, (unsigned)j,
               (double)date_c[i][j], (double)expected);
        return false;
      }
    }
  }
  return true;
}

int main(void) {
  const gemmini_partition_axis_t axis =
      (gemmini_partition_axis_t)DATE_TRAFFIC_AXIS;
  const bool private_like = DATE_TRAFFIC_PRIVATE_LIKE != 0;
  if (!gemmini_partition_axis_is_valid(axis)) {
    printf("DATE_TRAFFIC invalid axis=%d\n", (int)axis);
    return 1;
  }

  printf("DATE_TRAFFIC_START schema=1 mode=%s axis=%s dims=%ux%ux%u\n",
         private_like ? "private_like" : "shared_collective",
         date_axis_name(axis), DATE_TRAFFIC_I, DATE_TRAFFIC_J, DATE_TRAFFIC_K);

  date_initialize();
  date_flush_all();

  shared_multi_matmul_job_t jobs[DATE_TRAFFIC_PODS];
  memset(jobs, 0, sizeof(jobs));
  const size_t job_count = private_like ? DATE_TRAFFIC_PODS : 1u;
  const bool prepared = private_like
      ? date_prepare_private_like(jobs, axis)
      : date_prepare_shared(jobs, axis);
  if (!prepared) {
    printf("DATE_TRAFFIC prepare=FAIL mode=%s axis=%s\n",
           private_like ? "private_like" : "shared_collective",
           date_axis_name(axis));
    return 1;
  }

  uint64_t wall_cycles = 0;
  size_t rounds = 0;
  if (!date_run_jobs(jobs, job_count, &wall_cycles, &rounds)) {
    printf("DATE_TRAFFIC execution=FAIL mode=%s axis=%s\n",
           private_like ? "private_like" : "shared_collective",
           date_axis_name(axis));
    return 1;
  }

  uint64_t rdma_total = 0;
  uint64_t wdma_total = 0;
  for (int pod = 0; pod < (int)DATE_TRAFFIC_PODS; ++pod) {
    const uint32_t rdma = counter_read(pod, DATE_COUNTER_RDMA);
    const uint32_t wdma = counter_read(pod, DATE_COUNTER_WDMA);
    rdma_total += rdma;
    wdma_total += wdma;
    printf("DATE_TRAFFIC_POD pod=%d rdma_bytes=%u wdma_bytes=%u\n",
           pod, rdma, wdma);
  }

  const bool correct = date_check(axis, private_like);
  const uint64_t logical_a_bytes =
      (uint64_t)DATE_TRAFFIC_I * DATE_TRAFFIC_K * sizeof(elem_t);
  const uint64_t logical_b_bytes =
      (uint64_t)DATE_TRAFFIC_K * DATE_TRAFFIC_J * sizeof(elem_t);
  const uint64_t logical_c_bytes =
      (uint64_t)DATE_TRAFFIC_I * DATE_TRAFFIC_J * sizeof(acc_t);
  printf("DATE_TRAFFIC schema=1 mode=%s axis=%s dims=%ux%ux%u "
         "jobs=%u rounds=%u wall_cycles=%llu rdma_bytes=%llu "
         "wdma_bytes=%llu logical_a_bytes=%llu logical_b_bytes=%llu "
         "logical_c_bytes=%llu partial_reduction_included=0 correctness=%s\n",
         private_like ? "private_like" : "shared_collective",
         date_axis_name(axis), DATE_TRAFFIC_I, DATE_TRAFFIC_J, DATE_TRAFFIC_K,
         (unsigned)job_count, (unsigned)rounds,
         (unsigned long long)wall_cycles,
         (unsigned long long)rdma_total,
         (unsigned long long)wdma_total,
         (unsigned long long)logical_a_bytes,
         (unsigned long long)logical_b_bytes,
         (unsigned long long)logical_c_bytes,
         correct ? "PASS" : "FAIL");
  return correct ? 0 : 1;
}
