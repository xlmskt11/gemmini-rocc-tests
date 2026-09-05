// See LICENSE for license details.

/*
 * DATE motivation microbenchmark for ragged gang allocation.
 *
 * Select one policy per ELF through the volatile DATE_RAGGED_POLICY data word:
 *   0: three jobs run sequentially on all four pods
 *   1: three concurrent one-pod jobs (pod 3 is idle)
 *   2: concurrent 2+1+1 allocation, with two pods on the long job
 *
 * Policy selection is deliberately a runtime branch.  The three ELFs contain
 * identical code and buffer layouts; only the initialized selector word
 * differs.  This avoids policy-dependent cache-set and DRAM-address mappings.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils_all.h"

#ifndef DATE_RAGGED_POLICY
#define DATE_RAGGED_POLICY 0
#endif

#ifndef DATE_RAGGED_PRINT_PLAN
#define DATE_RAGGED_PRINT_PLAN 0
#endif
#if !defined(GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION) || \
    GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION != 1
#error "date_ragged_gang_motivation requires the shared M/N/K RTL ABI"
#endif

#define DATE_RAGGED_JOBS 3u
#ifndef DATE_RAGGED_I
#define DATE_RAGGED_I 16u
#endif
#ifndef DATE_RAGGED_K
#define DATE_RAGGED_K 64u
#endif
#ifndef DATE_RAGGED_LONG_N
#define DATE_RAGGED_LONG_N 128u
#endif
#ifndef DATE_RAGGED_SHORT_N
#define DATE_RAGGED_SHORT_N 32u
#endif
#define DATE_RAGGED_MAX_N DATE_RAGGED_LONG_N

#if DATE_RAGGED_I == 0 || DATE_RAGGED_K == 0 || \
    DATE_RAGGED_SHORT_N == 0 || DATE_RAGGED_LONG_N < DATE_RAGGED_SHORT_N || \
    DATE_RAGGED_I % DIM != 0 || DATE_RAGGED_K % DIM != 0 || \
    DATE_RAGGED_LONG_N % DIM != 0 || DATE_RAGGED_SHORT_N % DIM != 0
#error "ragged dimensions must be nonzero DIM multiples and long N >= short N"
#endif

static const size_t date_ragged_n[DATE_RAGGED_JOBS] = {
    DATE_RAGGED_LONG_N, DATE_RAGGED_SHORT_N, DATE_RAGGED_SHORT_N};

/* Add one so every policy selector is nonzero and occupies the same .data
 * section; policy 0 must not silently move the word into .bss. */
static volatile unsigned date_ragged_policy_code = DATE_RAGGED_POLICY + 1u;

static elem_t date_ragged_a[DATE_RAGGED_JOBS][DATE_RAGGED_I][DATE_RAGGED_K]
    __attribute__((aligned(4096)));
static elem_t date_ragged_b[DATE_RAGGED_JOBS][DATE_RAGGED_K]
                           [DATE_RAGGED_MAX_N]
    __attribute__((aligned(4096)));
static acc_t date_ragged_c[DATE_RAGGED_JOBS][DATE_RAGGED_I]
                          [DATE_RAGGED_MAX_N]
    __attribute__((aligned(4096)));

static inline uint64_t date_ragged_read_cycles(void) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles) : : "memory");
  return cycles;
}

static inline void date_ragged_cpu_fence(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static elem_t date_ragged_one(void) {
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

static const char *date_ragged_policy_name(unsigned policy) {
  switch (policy) {
    case 0u:
      return "sequential_g4";
    case 1u:
      return "static_1+1+1";
    case 2u:
      return "fixed_2+1+1";
    default:
      return "invalid";
  }
}

static void date_ragged_policy_layout(
    unsigned policy, size_t job_index,
    unsigned *mask, unsigned *q, unsigned *quarters) {
  if (policy == 0u) {
    *mask = 0xfu;
    *q = 0u;
    *quarters = 4u;
  } else if (policy == 1u) {
    *mask = 1u << job_index;
    *q = (unsigned)job_index;
    *quarters = 1u;
  } else {
    static const unsigned masks[DATE_RAGGED_JOBS] = {0x3u, 0x4u, 0x8u};
    static const unsigned starts[DATE_RAGGED_JOBS] = {0u, 2u, 3u};
    static const unsigned sizes[DATE_RAGGED_JOBS] = {2u, 1u, 1u};
    *mask = masks[job_index];
    *q = starts[job_index];
    *quarters = sizes[job_index];
  }
}

static void date_ragged_initialize(void) {
  const elem_t one = date_ragged_one();
  for (size_t job = 0; job < DATE_RAGGED_JOBS; ++job) {
    for (size_t i = 0; i < DATE_RAGGED_I; ++i) {
      for (size_t k = 0; k < DATE_RAGGED_K; ++k)
        date_ragged_a[job][i][k] = one;
    }
    for (size_t k = 0; k < DATE_RAGGED_K; ++k) {
      for (size_t j = 0; j < date_ragged_n[job]; ++j)
        date_ragged_b[job][k][j] = one;
    }
  }
  memset(date_ragged_c, 0, sizeof(date_ragged_c));
}

static void date_ragged_flush_all(void) {
  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
  gemmini_flush(custom3, 0);
  gemmini_fence();
}

static bool date_ragged_prepare_job(
    shared_multi_matmul_job_t *job, size_t job_index,
    unsigned mask, unsigned q, unsigned quarters) {
  const size_t sp_rows_per_operand_bank =
      (size_t)quarters * BANK_ROWS / 4u;
  const size_t sp_start = (size_t)q * BANK_ROWS / 4u;
  const size_t acc_rows_per_half =
      (size_t)quarters * (ACC_ROWS / 2u) / 4u;
  const size_t acc_start = (size_t)q * (ACC_ROWS / 2u) / 4u;

  if (job == NULL || mask == 0u || q + quarters > 4u ||
      quarters == 0u)
    return false;

  memset(job, 0, sizeof(*job));
  job->gemmini_list = (int)mask;
  job->tile_id = (int)job_index;
  job->sp_addr_start_stack = sp_start;
  job->sp_addr_end_stack = sp_start;
  job->acc_addr_start_stack = acc_start;
  job->sp_addr_range = 4u * sp_rows_per_operand_bank;
  job->acc_addr_range = 2u * acc_rows_per_half;
  job->dim_I = DATE_RAGGED_I;
  job->dim_J = date_ragged_n[job_index];
  job->dim_K = DATE_RAGGED_K;
  job->A = &date_ragged_a[job_index][0][0];
  job->B = &date_ragged_b[job_index][0][0];
  job->D = NULL;
  job->C = &date_ragged_c[job_index][0][0];
  job->stride_A = DATE_RAGGED_K;
  job->stride_B = DATE_RAGGED_MAX_N;
  job->stride_D = 0u;
  job->stride_C = DATE_RAGGED_MAX_N;
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
  job->weightA = 1u;
  job->dataflow = WEIGHT_STATIONARY;
  job->partition_axis = GEMMINI_PARTITION_AXIS_N;

  shared_multi_choose_tiling_factors(job);
  if (job->partition_status != SHARED_MULTI_PARTITION_OK ||
      job->tile_I == 0u || job->tile_J == 0u || job->tile_K == 0u ||
      job->sp_addr_A_stacked > sp_rows_per_operand_bank ||
      job->sp_addr_B_stacked > sp_rows_per_operand_bank ||
      job->acc_addr_stacked > acc_rows_per_half)
    return false;

  shared_multi_tiled_matmul_job_init(job);
  return job->partition_status == SHARED_MULTI_PARTITION_OK && !job->done;
}

static bool date_ragged_run(
    shared_multi_matmul_job_t jobs[DATE_RAGGED_JOBS],
    unsigned policy, uint64_t *cycles, size_t *step_calls) {
  size_t steps = 0u;
  date_ragged_cpu_fence();
  const uint64_t start = date_ragged_read_cycles();

  if (policy == 0u) {
    for (size_t job = 0; job < DATE_RAGGED_JOBS; ++job) {
      while (!jobs[job].done) {
        shared_multi_tiled_matmul_job_step(&jobs[job]);
        ++steps;
        if (jobs[job].partition_status != SHARED_MULTI_PARTITION_OK)
          return false;
      }
      gemmini_fence();
    }
  } else {
    size_t live = DATE_RAGGED_JOBS;
    size_t cursor = 0u;
    while (live != 0u) {
      shared_multi_matmul_job_t *job = &jobs[cursor];
      if (!job->done) {
        shared_multi_tiled_matmul_job_step(job);
        ++steps;
        if (job->partition_status != SHARED_MULTI_PARTITION_OK)
          return false;
        if (job->done)
          --live;
      }
      cursor = (cursor + 1u) % DATE_RAGGED_JOBS;
    }
    gemmini_fence();
  }

  const uint64_t end = date_ragged_read_cycles();
  date_ragged_cpu_fence();
  *cycles = end - start;
  *step_calls = steps;
  return true;
}

static bool date_ragged_check(void) {
  const acc_t expected = (acc_t)DATE_RAGGED_K;
  const acc_t_bits expected_bits = acc_t_to_acc_t_bits(expected);
  for (size_t job = 0; job < DATE_RAGGED_JOBS; ++job) {
    for (size_t i = 0; i < DATE_RAGGED_I; ++i) {
      for (size_t j = 0; j < date_ragged_n[job]; ++j) {
        const acc_t got = date_ragged_c[job][i][j];
        if (acc_t_to_acc_t_bits(got) != expected_bits) {
          printf("DATE_RAGGED mismatch job=%u i=%u j=%u got=0x%x "
                 "expected=0x%x\n",
                 (unsigned)job, (unsigned)i, (unsigned)j,
                 (unsigned)acc_t_to_acc_t_bits(got),
                 (unsigned)expected_bits);
          return false;
        }
      }
    }
  }
  return true;
}

int main(void) {
  shared_multi_matmul_job_t jobs[DATE_RAGGED_JOBS];
  uint64_t cycles = 0u;
  size_t step_calls = 0u;
  const unsigned policy_code = date_ragged_policy_code;
  const unsigned policy = policy_code - 1u;

  if (policy_code == 0u || policy > 2u) {
    printf("DATE_RAGGED schema=2 policy_id=%u status=INVALID_POLICY\n", policy);
    return 1;
  }

  printf("DATE_RAGGED_START schema=2 policy_id=%u policy=%s "
         "I=%u K=%u Ns=%u,%u,%u\n",
         policy, date_ragged_policy_name(policy),
         DATE_RAGGED_I, DATE_RAGGED_K, DATE_RAGGED_LONG_N,
         DATE_RAGGED_SHORT_N, DATE_RAGGED_SHORT_N);
  date_ragged_initialize();
  date_ragged_flush_all();

  for (size_t job = 0; job < DATE_RAGGED_JOBS; ++job) {
    unsigned mask, q, quarters;
    date_ragged_policy_layout(policy, job, &mask, &q, &quarters);
    if (!date_ragged_prepare_job(&jobs[job], job, mask, q, quarters)) {
      printf("DATE_RAGGED schema=2 policy_id=%u policy=%s status=INIT_FAIL "
             "job=%u\n",
             policy, date_ragged_policy_name(policy), (unsigned)job);
      return 1;
    }
    if (DATE_RAGGED_PRINT_PLAN) {
      printf("DATE_RAGGED_PLAN job=%u mask=0x%x quarters=%u "
             "tile=%ux%ux%u outer=%ux%ux%u\n",
             (unsigned)job, mask, quarters,
             (unsigned)jobs[job].tile_I, (unsigned)jobs[job].tile_J,
             (unsigned)jobs[job].tile_K, (unsigned)jobs[job].I0,
             (unsigned)jobs[job].J0, (unsigned)jobs[job].K0);
    }
  }

  /* Drain the configuration commands emitted by job initialization so the
   * timed region contains only work issue and the required completion fence. */
  gemmini_fence();
  if (!date_ragged_run(jobs, policy, &cycles, &step_calls)) {
    printf("DATE_RAGGED schema=2 policy_id=%u policy=%s status=RUN_FAIL\n",
           policy, date_ragged_policy_name(policy));
    return 1;
  }

  const bool correct = date_ragged_check();
  printf("DATE_RAGGED schema=2 policy_id=%u policy=%s axis=N "
         "I=%u K=%u Ns=%u,%u,%u cycles=%llu step_calls=%u "
         "correct=%s\n",
         policy, date_ragged_policy_name(policy), DATE_RAGGED_I,
         DATE_RAGGED_K, DATE_RAGGED_LONG_N, DATE_RAGGED_SHORT_N,
         DATE_RAGGED_SHORT_N, (unsigned long long)cycles, (unsigned)step_calls,
         correct ? "PASS" : "FAIL");
  return correct ? 0 : 1;
}
