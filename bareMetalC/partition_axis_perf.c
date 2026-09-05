// See LICENSE for license details.

/* One cold-start M, N, or K performance/correctness test. */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils_all.h"

#ifndef AXIS_PERF_I
#define AXIS_PERF_I 128u
#endif
#ifndef AXIS_PERF_J
#define AXIS_PERF_J 128u
#endif
#ifndef AXIS_PERF_K
#define AXIS_PERF_K 128u
#endif
#ifndef AXIS_PERF_REQUIRE_MULTISTEP
#define AXIS_PERF_REQUIRE_MULTISTEP 0
#endif
#ifndef AXIS_PERF_MASK
#define AXIS_PERF_MASK 0xfu
#endif
#ifndef AXIS_PERF_MEMBERS
#define AXIS_PERF_MEMBERS 4u
#endif
#if AXIS_PERF_MEMBERS < 1u || AXIS_PERF_MEMBERS > 4u
#error "AXIS_PERF_MEMBERS must be between one and four"
#endif
#if AXIS_PERF_MASK != ((1u << AXIS_PERF_MEMBERS) - 1u)
#error "AXIS_PERF_MASK must select the low AXIS_PERF_MEMBERS Gemminis"
#endif
#define AXIS_PERF_COUNTER_LOOP 0u
#define AXIS_PERF_COUNTER_EX 1u

#ifndef AXIS_PERF_PARTITION_AXIS
#define AXIS_PERF_PARTITION_AXIS GEMMINI_PARTITION_AXIS_M
#endif

static elem_t axis_perf_a[AXIS_PERF_I][AXIS_PERF_K]
    row_align(MAX_BLOCK_LEN) = {{1}};
static elem_t axis_perf_b[AXIS_PERF_K][AXIS_PERF_J]
    row_align(MAX_BLOCK_LEN) = {{1}};
static acc_t axis_perf_c[AXIS_PERF_I][AXIS_PERF_J]
    row_align_acc(MAX_BLOCK_LEN_ACC) = {{1}};

static inline void axis_perf_cpu_fence(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static inline uint64_t axis_perf_read_cycles(void) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles) : : "memory");
  return cycles;
}

static elem_t axis_perf_elem_from_int(int value) {
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

static int axis_perf_k_factor(size_t k) {
  return (int)((3u * k) % 5u) - 2;
}

static int axis_perf_j_factor(size_t j) {
  static const int factors[] = {1, -2, 3, -1, 2, -3};
  return factors[j % (sizeof(factors) / sizeof(factors[0]))];
}

static const char *axis_perf_name(gemmini_partition_axis_t axis) {
  switch (axis) {
  case GEMMINI_PARTITION_AXIS_M:
    return "M";
  case GEMMINI_PARTITION_AXIS_N:
    return "N";
  case GEMMINI_PARTITION_AXIS_K:
    return "K";
  default:
    return "?";
  }
}

static void axis_perf_initialize(void) {
  for (size_t i = 0; i < AXIS_PERF_I; ++i) {
    const elem_t a = axis_perf_elem_from_int(axis_perf_j_factor(i));
    for (size_t k = 0; k < AXIS_PERF_K; ++k)
      axis_perf_a[i][k] = a;
  }
  for (size_t k = 0; k < AXIS_PERF_K; ++k) {
    for (size_t j = 0; j < AXIS_PERF_J; ++j) {
      axis_perf_b[k][j] = axis_perf_elem_from_int(
          axis_perf_k_factor(k) * axis_perf_j_factor(j));
    }
  }
}

static bool axis_perf_check(gemmini_partition_axis_t axis) {
  acc_t k_sum = 0;
  for (size_t k = 0; k < AXIS_PERF_K; ++k)
    k_sum += (acc_t)axis_perf_k_factor(k);

  for (size_t i = 0; i < AXIS_PERF_I; ++i) {
    const acc_t row_factor = (acc_t)axis_perf_j_factor(i) * k_sum;
    for (size_t j = 0; j < AXIS_PERF_J; ++j) {
      const acc_t expected = row_factor * (acc_t)axis_perf_j_factor(j);
      if (acc_t_to_acc_t_bits(axis_perf_c[i][j]) !=
          acc_t_to_acc_t_bits(expected)) {
        printf("partition_axis_perf: axis=%s B_owner=K mismatch i=%u j=%u "
               "got=0x%x expected=0x%x\n",
               axis_perf_name(axis), (unsigned)i, (unsigned)j,
               (unsigned)acc_t_to_acc_t_bits(axis_perf_c[i][j]),
               (unsigned)acc_t_to_acc_t_bits(expected));
        return false;
      }
    }
  }
  return true;
}

static void axis_perf_flush_all(void) {
  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
  gemmini_flush(custom3, 0);
  gemmini_fence();
}

static void axis_perf_reset_counters(void) {
  for (int custom = 0; custom < (int)AXIS_PERF_MEMBERS; ++custom) {
    counter_configure(custom, AXIS_PERF_COUNTER_LOOP,
                      LOOP_MATMUL_ACTIVE_CYCLES);
    counter_configure(custom, AXIS_PERF_COUNTER_EX, EXE_ACTIVE_CYCLE);
    counter_reset(custom);
  }
  gemmini_fence();
}

static bool axis_perf_run_one(gemmini_partition_axis_t axis) {
  shared_multi_matmul_job_t job;
  memset(&job, 0, sizeof(job));

  job.gemmini_list = AXIS_PERF_MASK;
  job.tile_id = 0;
  job.sp_addr_range = TOTAL_SPAD_ROWS;
  job.acc_addr_range = TOTAL_ACC_ROWS;
  job.dim_I = AXIS_PERF_I;
  job.dim_J = AXIS_PERF_J;
  job.dim_K = AXIS_PERF_K;
  job.A = &axis_perf_a[0][0];
  job.B = &axis_perf_b[0][0];
  job.D = NULL;
  job.C = &axis_perf_c[0][0];
  job.stride_A = AXIS_PERF_K;
  job.stride_B = AXIS_PERF_J;
  job.stride_D = 0;
  job.stride_C = AXIS_PERF_J;
  job.A_scale_factor = MVIN_SCALE_IDENTITY;
  job.B_scale_factor = MVIN_SCALE_IDENTITY;
  job.D_scale_factor = MVIN_SCALE_IDENTITY;
  job.act = NO_ACTIVATION;
  job.scale = ACC_SCALE_IDENTITY;
  job.repeating_bias = false;
  job.a_transpose = false;
  job.b_transpose = false;
  job.full_C = true;
  job.low_D = false;
  job.weightA = 1;
  job.dataflow = WEIGHT_STATIONARY;
  job.partition_axis = axis;

  axis_perf_flush_all();
  shared_multi_choose_tiling_factors(&job);
  shared_multi_tiled_matmul_job_init(&job);
  if (job.tile_I == 0 || job.tile_J == 0 || job.tile_K == 0 ||
      job.partition_status != SHARED_MULTI_PARTITION_OK) {
    printf("partition_axis_perf: axis=%s init failed status=%d\n",
           axis_perf_name(axis), (int)job.partition_status);
    return false;
  }

  /* Exclude flush/config/counter programming from the measured interval. */
  gemmini_fence();
  axis_perf_reset_counters();
  axis_perf_cpu_fence();

  const uint64_t start = axis_perf_read_cycles();
  size_t job_steps = 0;
  while (!job.done) {
    shared_multi_tiled_matmul_job_step(&job);
    ++job_steps;
  }
  gemmini_fence();
  const uint64_t end = axis_perf_read_cycles();
  axis_perf_cpu_fence();

  if (job.partition_status != SHARED_MULTI_PARTITION_OK) {
    printf("partition_axis_perf: axis=%s step failed status=%d\n",
           axis_perf_name(axis), (int)job.partition_status);
    return false;
  }
  if (AXIS_PERF_REQUIRE_MULTISTEP && job_steps <= 1u) {
    printf("partition_axis_perf: axis=%s expected multi-step job, got=%u\n",
           axis_perf_name(axis), (unsigned)job_steps);
    return false;
  }

  for (int custom = 0; custom < (int)AXIS_PERF_MEMBERS; ++custom)
    counter_snapshot_take(custom);
  gemmini_fence();

  const size_t active_members = gemmini_partition_effective_member_count(
      axis, AXIS_PERF_MEMBERS, job.tile_I, job.tile_J, job.tile_K);
  const uint64_t total_macs =
      (uint64_t)AXIS_PERF_I * AXIS_PERF_J * AXIS_PERF_K;
  const uint64_t ideal_cycles =
      total_macs / ((uint64_t)active_members * DIM * DIM);
  const uint64_t wall_cycles = end - start;

  printf("AXIS_PERF axis=%s B_owner=K dims=%ux%ux%u mask=0x%x "
         "tile=%ux%ux%u chunks=%ux%ux%u steps=%u "
         "wall_cycles=%llu ideal_cycles=%llu\n",
         axis_perf_name(axis), (unsigned)AXIS_PERF_I,
         (unsigned)AXIS_PERF_J, (unsigned)AXIS_PERF_K, AXIS_PERF_MASK,
         (unsigned)job.tile_I, (unsigned)job.tile_J, (unsigned)job.tile_K,
         (unsigned)job.I0, (unsigned)job.J0, (unsigned)job.K0,
         (unsigned)job_steps,
         (unsigned long long)wall_cycles,
         (unsigned long long)ideal_cycles);
  for (int custom = 0; custom < (int)AXIS_PERF_MEMBERS; ++custom) {
    printf("AXIS_PERF axis=%s B_owner=K gemmini=%d "
           "loop_cycles=%u ex_cycles=%u\n",
           axis_perf_name(axis), custom,
           counter_read(custom, AXIS_PERF_COUNTER_LOOP),
           counter_read(custom, AXIS_PERF_COUNTER_EX));
  }

  if (!axis_perf_check(axis))
    return false;
  printf("AXIS_PERF axis=%s B_owner=K correctness=PASS\n",
         axis_perf_name(axis));
  return true;
}

int main(void) {
  const gemmini_partition_axis_t axis =
      (gemmini_partition_axis_t)AXIS_PERF_PARTITION_AXIS;

  if (gemmini_partition_generated_axis_abi_version() !=
      GEMMINI_SHARED_PARTITION_ENCODING_VERSION) {
    printf("partition_axis_perf: SKIP (generated shared-axis ABI unavailable)\n");
    return 0;
  }
  if (AXIS_PERF_I == 0 || AXIS_PERF_J == 0 || AXIS_PERF_K == 0 ||
      (AXIS_PERF_I + DIM - 1u) / DIM < AXIS_PERF_MEMBERS ||
      (AXIS_PERF_J + DIM - 1u) / DIM < AXIS_PERF_MEMBERS ||
      (AXIS_PERF_K + DIM - 1u) / DIM < AXIS_PERF_MEMBERS) {
    printf("partition_axis_perf: SKIP (dimensions do not provide %u shards)\n",
           (unsigned)AXIS_PERF_MEMBERS);
    return 0;
  }
  if (axis != GEMMINI_PARTITION_AXIS_M &&
      axis != GEMMINI_PARTITION_AXIS_N &&
      axis != GEMMINI_PARTITION_AXIS_K) {
    printf("partition_axis_perf: invalid axis=%d\n", (int)axis);
    return 1;
  }

  axis_perf_initialize();
  axis_perf_cpu_fence();
  printf("AXIS_PERF start axis=%s B_owner=K dims=%ux%ux%u dim=%u "
         "members=%u mask=0x%x\n",
         axis_perf_name(axis), (unsigned)AXIS_PERF_I,
         (unsigned)AXIS_PERF_J, (unsigned)AXIS_PERF_K, (unsigned)DIM,
         (unsigned)AXIS_PERF_MEMBERS, (unsigned)AXIS_PERF_MASK);

  if (!axis_perf_run_one(axis))
    return 1;

  printf("partition_axis_perf: PASS axis=%s\n", axis_perf_name(axis));
  return 0;
}
