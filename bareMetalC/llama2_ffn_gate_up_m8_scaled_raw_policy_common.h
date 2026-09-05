// Common raw-descriptor replay for the scaled llama2-7B ffn_gate_up policy
// waveforms.  A wrapper selects exactly one WAVE_POLICY_* value before
// including this file.
//
// The logical shape is M=8, N=1376, K=512 (N and K are 1/8 of the original
// ffn_gate_up shape).  Four jobs use distinct A/B/C ranges, only B is page
// packed, and there is no operand initialization or CPU-gold calculation.
// Tiling, job cursors, validation, address calculation, and descriptor packing
// run before timing.  The timed path only replays the resulting LOOP_WS RoCC
// descriptors in the same tenant-round order as the full FireSim benchmark,
// followed by one final fence.

#ifndef LLAMA2_FFN_GATE_UP_M8_SCALED_RAW_POLICY_COMMON_H_
#define LLAMA2_FFN_GATE_UP_M8_SCALED_RAW_POLICY_COMMON_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils_all.h"

#define WAVE_POLICY_QUEUED_G4 1
#define WAVE_POLICY_2PLUS2 2
#define WAVE_POLICY_3PLUS1 3
#define WAVE_POLICY_2PLUS1PLUS1 4
#define WAVE_POLICY_1PLUS1PLUS1PLUS1 5

#ifndef WAVE_POLICY
#error "A scaled raw waveform wrapper must define WAVE_POLICY"
#endif

#if WAVE_POLICY < WAVE_POLICY_QUEUED_G4 || \
    WAVE_POLICY > WAVE_POLICY_1PLUS1PLUS1PLUS1
#error "Unknown scaled raw waveform policy"
#endif

#if !GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE
#error "The scaled raw waveform suite requires the shared partition descriptor ABI"
#endif

#define WAVE_M 8u
#define WAVE_N 1376u
#define WAVE_K 512u
#define WAVE_JOBS 4u
#define WAVE_AXIS GEMMINI_PARTITION_AXIS_K
#define WAVE_CACHE_LINE_BYTES 64u
#define WAVE_EVICTION_BYTES (8u * 1024u * 1024u)
#define WAVE_A_BYTES (WAVE_M * WAVE_K * sizeof(elem_t))
#define WAVE_C_BYTES (WAVE_M * WAVE_N * sizeof(acc_t))
#define WAVE_B_PAGES 384u
#define WAVE_B_BYTES (WAVE_B_PAGES * GEMMINI_PAGE_PACKED_PAGE_BYTES)

// Fixed target-DRAM ranges keep the ELF/loadmem image compact and stay inside
// the unmodified 256-MiB Verilator aperture.
#define WAVE_EVICTION_BASE UINT64_C(0x81000000)
#define WAVE_A_BASE        UINT64_C(0x81800000)
#define WAVE_C_BASE        UINT64_C(0x81900000)
#define WAVE_B0_BASE       UINT64_C(0x82000000)
#define WAVE_B1_BASE       UINT64_C(0x82400000)
#define WAVE_B2_BASE       UINT64_C(0x82800000)
#define WAVE_B3_BASE       UINT64_C(0x82c00000)
#define WAVE_B_SLOT_BYTES  UINT64_C(0x00400000)

#define WAVE_SP_QUARTER_ROWS (BANK_ROWS / 4u)
#define WAVE_ACC_QUARTER_ROWS ((ACC_ROWS / 2u) / 4u)
#define WAVE_ACC_GID_ROWS (ACC_ROWS / 8u)

#define WAVE_G1_TILE_I 1u
#define WAVE_G1_TILE_J 11u
#define WAVE_G1_TILE_K 11u
#define WAVE_G1_STEPS_PER_JOB 96u
#define WAVE_G2_TILE_I 1u
#define WAVE_G2_TILE_J 16u
#define WAVE_G2_TILE_K 16u
#define WAVE_G2_STEPS_PER_JOB 44u
#define WAVE_G3_TILE_I 1u
#define WAVE_G3_TILE_J 21u
#define WAVE_G3_TILE_K 18u
#define WAVE_G3_STEPS_PER_JOB 36u
#define WAVE_G4_TILE_I 1u
#define WAVE_G4_TILE_J 25u
#define WAVE_G4_TILE_K 20u
#define WAVE_G4_STEPS_PER_JOB 28u

enum {
  WAVE_CMD_SPADDR = 0,
  WAVE_CMD_PARTITION,
  WAVE_CMD_BOUNDS,
  WAVE_CMD_ADDRS_AB,
  WAVE_CMD_ADDRS_DC,
  WAVE_CMD_STRIDES_AB,
  WAVE_CMD_STRIDES_DC,
  WAVE_CMD_PAGE_OFFSETS,
  WAVE_MEMBER_COMMANDS,
};

typedef struct {
  uint64_t rs1;
  uint64_t rs2;
} wave_rocc_args_t;

typedef struct {
  wave_rocc_args_t command[WAVE_MEMBER_COMMANDS];
} wave_member_desc_t;

typedef struct {
  wave_rocc_args_t launch;
  wave_member_desc_t member[1];
} wave_group1_desc_t;

typedef struct {
  wave_rocc_args_t launch;
  wave_member_desc_t member[2];
} wave_group2_desc_t;

typedef struct {
  wave_rocc_args_t launch;
  wave_member_desc_t member[3];
} wave_group3_desc_t;

typedef struct {
  wave_rocc_args_t launch;
  wave_member_desc_t member[4];
} wave_group4_desc_t;

typedef struct {
  unsigned mask;
  unsigned members;
  unsigned q;
  unsigned first_job;
  unsigned queue_jobs;
  size_t tile_I;
  size_t tile_J;
  size_t tile_K;
  size_t steps_per_job;
} wave_tenant_spec_t;

typedef struct {
  uint64_t start;
  uint64_t issue_end;
  uint64_t end;
  bool consumed_exactly;
} wave_cycles_t;

#if WAVE_POLICY == WAVE_POLICY_QUEUED_G4
#define WAVE_POLICY_NAME "queued_g4"
#define WAVE_POLICY_TAG "QUEUED-G4"
#define WAVE_TOPOLOGY "masks=f queues=4"
#define WAVE_TILINGS "g4=1/25/20"
#define WAVE_GIDS "0/4"
#define WAVE_TENANT_COUNT 1u
#define WAVE_SCHEDULER_ROUNDS 112u
#define WAVE_GROUP_STEPS_0 112u
#define WAVE_GROUP_STEPS_1 0u
#define WAVE_GROUP_STEPS_2 0u
#define WAVE_GROUP_STEPS_3 0u
#define WAVE_TIMED_ROCC_COMMANDS (WAVE_GROUP_STEPS_0 * 33u)
static const wave_tenant_spec_t wave_tenant_specs[WAVE_TENANT_COUNT] = {
  {0xfu, 4u, 0u, 0u, 4u,
   WAVE_G4_TILE_I, WAVE_G4_TILE_J, WAVE_G4_TILE_K,
   WAVE_G4_STEPS_PER_JOB},
};
static wave_group4_desc_t wave_schedule0[WAVE_GROUP_STEPS_0]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
#elif WAVE_POLICY == WAVE_POLICY_2PLUS2
#define WAVE_POLICY_NAME "2+2"
#define WAVE_POLICY_TAG "2PLUS2"
#define WAVE_TOPOLOGY "masks=3/c queues=2/2"
#define WAVE_TILINGS "g2/g2=1/16/16"
#define WAVE_GIDS "0/4,2/6"
#define WAVE_TENANT_COUNT 2u
#define WAVE_SCHEDULER_ROUNDS 88u
#define WAVE_GROUP_STEPS_0 88u
#define WAVE_GROUP_STEPS_1 88u
#define WAVE_GROUP_STEPS_2 0u
#define WAVE_GROUP_STEPS_3 0u
#define WAVE_TIMED_ROCC_COMMANDS \
  ((WAVE_GROUP_STEPS_0 + WAVE_GROUP_STEPS_1) * 17u)
static const wave_tenant_spec_t wave_tenant_specs[WAVE_TENANT_COUNT] = {
  {0x3u, 2u, 0u, 0u, 2u,
   WAVE_G2_TILE_I, WAVE_G2_TILE_J, WAVE_G2_TILE_K,
   WAVE_G2_STEPS_PER_JOB},
  {0xcu, 2u, 2u, 2u, 2u,
   WAVE_G2_TILE_I, WAVE_G2_TILE_J, WAVE_G2_TILE_K,
   WAVE_G2_STEPS_PER_JOB},
};
static wave_group2_desc_t wave_schedule0[WAVE_GROUP_STEPS_0]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
static wave_group2_desc_t wave_schedule1[WAVE_GROUP_STEPS_1]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
#elif WAVE_POLICY == WAVE_POLICY_3PLUS1
#define WAVE_POLICY_NAME "3+1"
#define WAVE_POLICY_TAG "3PLUS1"
#define WAVE_TOPOLOGY "masks=7/8 queues=3/1"
#define WAVE_TILINGS "g3=1/21/18,g1=1/11/11"
#define WAVE_GIDS "0/4,3/7"
#define WAVE_TENANT_COUNT 2u
#define WAVE_SCHEDULER_ROUNDS 108u
#define WAVE_GROUP_STEPS_0 108u
#define WAVE_GROUP_STEPS_1 96u
#define WAVE_GROUP_STEPS_2 0u
#define WAVE_GROUP_STEPS_3 0u
#define WAVE_TIMED_ROCC_COMMANDS \
  (WAVE_GROUP_STEPS_0 * 25u + WAVE_GROUP_STEPS_1 * 9u)
static const wave_tenant_spec_t wave_tenant_specs[WAVE_TENANT_COUNT] = {
  {0x7u, 3u, 0u, 0u, 3u,
   WAVE_G3_TILE_I, WAVE_G3_TILE_J, WAVE_G3_TILE_K,
   WAVE_G3_STEPS_PER_JOB},
  {0x8u, 1u, 3u, 3u, 1u,
   WAVE_G1_TILE_I, WAVE_G1_TILE_J, WAVE_G1_TILE_K,
   WAVE_G1_STEPS_PER_JOB},
};
static wave_group3_desc_t wave_schedule0[WAVE_GROUP_STEPS_0]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
static wave_group1_desc_t wave_schedule1[WAVE_GROUP_STEPS_1]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
#elif WAVE_POLICY == WAVE_POLICY_2PLUS1PLUS1
#define WAVE_POLICY_NAME "2+1+1"
#define WAVE_POLICY_TAG "2PLUS1PLUS1"
#define WAVE_TOPOLOGY "masks=3/4/8 queues=2/1/1"
#define WAVE_TILINGS "g2=1/16/16,g1/g1=1/11/11"
#define WAVE_GIDS "0/4,2/6,3/7"
#define WAVE_TENANT_COUNT 3u
#define WAVE_SCHEDULER_ROUNDS 96u
#define WAVE_GROUP_STEPS_0 88u
#define WAVE_GROUP_STEPS_1 96u
#define WAVE_GROUP_STEPS_2 96u
#define WAVE_GROUP_STEPS_3 0u
#define WAVE_TIMED_ROCC_COMMANDS \
  (WAVE_GROUP_STEPS_0 * 17u + \
   (WAVE_GROUP_STEPS_1 + WAVE_GROUP_STEPS_2) * 9u)
static const wave_tenant_spec_t wave_tenant_specs[WAVE_TENANT_COUNT] = {
  {0x3u, 2u, 0u, 0u, 2u,
   WAVE_G2_TILE_I, WAVE_G2_TILE_J, WAVE_G2_TILE_K,
   WAVE_G2_STEPS_PER_JOB},
  {0x4u, 1u, 2u, 2u, 1u,
   WAVE_G1_TILE_I, WAVE_G1_TILE_J, WAVE_G1_TILE_K,
   WAVE_G1_STEPS_PER_JOB},
  {0x8u, 1u, 3u, 3u, 1u,
   WAVE_G1_TILE_I, WAVE_G1_TILE_J, WAVE_G1_TILE_K,
   WAVE_G1_STEPS_PER_JOB},
};
static wave_group2_desc_t wave_schedule0[WAVE_GROUP_STEPS_0]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
static wave_group1_desc_t wave_schedule1[WAVE_GROUP_STEPS_1]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
static wave_group1_desc_t wave_schedule2[WAVE_GROUP_STEPS_2]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
#else
#define WAVE_POLICY_NAME "1+1+1+1"
#define WAVE_POLICY_TAG "1PLUS1PLUS1PLUS1"
#define WAVE_TOPOLOGY "masks=1/2/4/8 queues=1/1/1/1"
#define WAVE_TILINGS "g1/g1/g1/g1=1/11/11"
#define WAVE_GIDS "0/4,1/5,2/6,3/7"
#define WAVE_TENANT_COUNT 4u
#define WAVE_SCHEDULER_ROUNDS 96u
#define WAVE_GROUP_STEPS_0 96u
#define WAVE_GROUP_STEPS_1 96u
#define WAVE_GROUP_STEPS_2 96u
#define WAVE_GROUP_STEPS_3 96u
#define WAVE_TIMED_ROCC_COMMANDS \
  ((WAVE_GROUP_STEPS_0 + WAVE_GROUP_STEPS_1 + \
    WAVE_GROUP_STEPS_2 + WAVE_GROUP_STEPS_3) * 9u)
static const wave_tenant_spec_t wave_tenant_specs[WAVE_TENANT_COUNT] = {
  {0x1u, 1u, 0u, 0u, 1u,
   WAVE_G1_TILE_I, WAVE_G1_TILE_J, WAVE_G1_TILE_K,
   WAVE_G1_STEPS_PER_JOB},
  {0x2u, 1u, 1u, 1u, 1u,
   WAVE_G1_TILE_I, WAVE_G1_TILE_J, WAVE_G1_TILE_K,
   WAVE_G1_STEPS_PER_JOB},
  {0x4u, 1u, 2u, 2u, 1u,
   WAVE_G1_TILE_I, WAVE_G1_TILE_J, WAVE_G1_TILE_K,
   WAVE_G1_STEPS_PER_JOB},
  {0x8u, 1u, 3u, 3u, 1u,
   WAVE_G1_TILE_I, WAVE_G1_TILE_J, WAVE_G1_TILE_K,
   WAVE_G1_STEPS_PER_JOB},
};
static wave_group1_desc_t wave_schedule0[WAVE_GROUP_STEPS_0]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
static wave_group1_desc_t wave_schedule1[WAVE_GROUP_STEPS_1]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
static wave_group1_desc_t wave_schedule2[WAVE_GROUP_STEPS_2]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
static wave_group1_desc_t wave_schedule3[WAVE_GROUP_STEPS_3]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
#endif

#define WAVE_TOTAL_STEPS                                                \
  (WAVE_GROUP_STEPS_0 + WAVE_GROUP_STEPS_1 +                           \
   WAVE_GROUP_STEPS_2 + WAVE_GROUP_STEPS_3)

#if WAVE_TENANT_COUNT == 1
#define WAVE_DESCRIPTOR_BYTES (sizeof(wave_schedule0))
#elif WAVE_TENANT_COUNT == 2
#define WAVE_DESCRIPTOR_BYTES \
  (sizeof(wave_schedule0) + sizeof(wave_schedule1))
#elif WAVE_TENANT_COUNT == 3
#define WAVE_DESCRIPTOR_BYTES                                      \
  (sizeof(wave_schedule0) + sizeof(wave_schedule1) +               \
   sizeof(wave_schedule2))
#else
#define WAVE_DESCRIPTOR_BYTES                                      \
  (sizeof(wave_schedule0) + sizeof(wave_schedule1) +               \
   sizeof(wave_schedule2) + sizeof(wave_schedule3))
#endif

// This is the only descriptor stream consumed in the timed interval.  Its
// launch-first, variable-size records and tenant-round ordering intentionally
// match llama2_matmul_concurrency_microbench.c.
static uint8_t wave_replay_schedule[WAVE_DESCRIPTOR_BYTES]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));

static volatile uint64_t wave_cache_sink;

static inline void wave_cpu_fence(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static inline uint64_t wave_read_cycles(void) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles) : : "memory");
  return cycles;
}

static elem_t *wave_a(unsigned job) {
  return (elem_t *)(uintptr_t)(WAVE_A_BASE +
      (uint64_t)job * UINT64_C(0x00010000));
}

static void *wave_c(unsigned job) {
  return (void *)(uintptr_t)(WAVE_C_BASE +
      (uint64_t)job * UINT64_C(0x00100000));
}

static const elem_t *wave_b(unsigned job) {
  static const uint64_t bases[WAVE_JOBS] = {
    WAVE_B0_BASE, WAVE_B1_BASE, WAVE_B2_BASE, WAVE_B3_BASE,
  };
  return (const elem_t *)(uintptr_t)bases[job];
}

static bool wave_ranges_are_valid(void) {
  if (WAVE_EVICTION_BASE + WAVE_EVICTION_BYTES > WAVE_A_BASE ||
      WAVE_A_BYTES > UINT64_C(0x00010000) ||
      WAVE_A_BASE + 3u * UINT64_C(0x00010000) + WAVE_A_BYTES > WAVE_C_BASE ||
      WAVE_C_BYTES > UINT64_C(0x00100000) ||
      WAVE_C_BASE + 3u * UINT64_C(0x00100000) + WAVE_C_BYTES > WAVE_B0_BASE ||
      gemmini_page_packed_b_page_count(WAVE_K, WAVE_N) != WAVE_B_PAGES ||
      WAVE_B_BYTES > WAVE_B_SLOT_BYTES)
    return false;
  for (unsigned job = 0; job + 1u < WAVE_JOBS; ++job) {
    const uintptr_t end = (uintptr_t)wave_b(job) + WAVE_B_BYTES;
    if (end > (uintptr_t)wave_b(job + 1u))
      return false;
  }
  return (uintptr_t)wave_b(WAVE_JOBS - 1u) + WAVE_B_BYTES <
      UINT64_C(0x90000000);
}

static void wave_touch_bytes(const void *data, size_t bytes, uint64_t *sink) {
  const volatile uint8_t *p = (const volatile uint8_t *)data;
  for (size_t offset = 0; offset < bytes; offset += WAVE_CACHE_LINE_BYTES)
    *sink += p[offset];
}

static void wave_touch_policy_schedules(uint64_t *sink) {
  wave_touch_bytes(
      wave_replay_schedule, sizeof(wave_replay_schedule), sink);
}

static void wave_condition_cache(void) {
  volatile uint8_t *eviction =
      (volatile uint8_t *)(uintptr_t)WAVE_EVICTION_BASE;
  uint64_t sink = wave_cache_sink;

  // First evict the 2-MiB L2 with an 8-MiB streaming pass.
  for (size_t offset = 0; offset < WAVE_EVICTION_BYTES;
       offset += WAVE_CACHE_LINE_BYTES)
    sink += eviction[offset];

  // Descriptor replay is the timed CPU work, so remove compulsory descriptor
  // misses without touching B or C.
  wave_touch_policy_schedules(&sink);

  // Match the full microbenchmark's four-A round-robin recent pass.  The
  // same-value stores preserve the uninitialized payload; B and C stay cold.
  for (size_t offset = 0; offset < WAVE_A_BYTES;
       offset += WAVE_CACHE_LINE_BYTES) {
    for (unsigned job = 0; job < WAVE_JOBS; ++job) {
      volatile uint8_t *a = (volatile uint8_t *)wave_a(job);
      const uint8_t value = a[offset];
      a[offset] = value;
    }
  }

  wave_cache_sink = sink;
  wave_cpu_fence();
}

static inline void wave_set_args(
    wave_rocc_args_t *args, uint64_t rs1, uint64_t rs2) {
  args->rs1 = rs1;
  args->rs2 = rs2;
}

static bool wave_spec_is_valid(const wave_tenant_spec_t *spec) {
  if (spec == NULL || spec->members == 0u || spec->members > 4u ||
      spec->q + spec->members > 4u || spec->queue_jobs == 0u ||
      spec->first_job + spec->queue_jobs > WAVE_JOBS)
    return false;
  const unsigned expected_mask =
      ((1u << spec->members) - 1u) << spec->q;
  return spec->mask == expected_mask &&
      spec->steps_per_job != 0u &&
      spec->tile_I != 0u && spec->tile_J != 0u && spec->tile_K != 0u;
}

static bool wave_prepare_job(
    shared_multi_matmul_job_t *job, unsigned job_index,
    const wave_tenant_spec_t *spec) {
  if (job == NULL || job_index >= WAVE_JOBS || !wave_spec_is_valid(spec))
    return false;

  memset(job, 0, sizeof(*job));
  job->gemmini_list = (int)spec->mask;
  job->tile_id = (int)job_index;  // Metadata only; launch gid is ACC-derived.
  job->sp_addr_start_stack = (size_t)spec->q * WAVE_SP_QUARTER_ROWS;
  job->sp_addr_end_stack = (size_t)spec->q * WAVE_SP_QUARTER_ROWS;
  job->acc_addr_start_stack = (size_t)spec->q * WAVE_ACC_QUARTER_ROWS;
  job->sp_addr_range =
      4u * (size_t)spec->members * WAVE_SP_QUARTER_ROWS;
  job->acc_addr_range =
      2u * (size_t)spec->members * WAVE_ACC_QUARTER_ROWS;
  job->dim_I = WAVE_M;
  job->dim_J = WAVE_N;
  job->dim_K = WAVE_K;
  job->A = wave_a(job_index);
  job->B = wave_b(job_index);
  job->D = NULL;
  job->C = wave_c(job_index);
  job->stride_A = WAVE_K;
  job->stride_B = GEMMINI_PAGE_PACKED_STRIDE(WAVE_N);
  job->stride_D = 0u;
  job->stride_C = WAVE_N;
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
  job->partition_axis = WAVE_AXIS;
  job->gemmini_num = (int)spec->members;

  const gemmini_tiling_request_t request = {
    .dim_I = WAVE_M,
    .dim_J = WAVE_N,
    .dim_K = WAVE_K,
    .dim = DIM,
    .gemmini_count = spec->members,
    .sp_addr_range = job->sp_addr_range,
    .acc_addr_range = job->acc_addr_range,
    .double_buffered = true,
    .act = NO_ACTIVATION,
  };
  const gemmini_tiling_factors_t tiling =
      gemmini_shared_multi_choose_tiling_axis(&request, WAVE_AXIS);
  const size_t operand_rows =
      (size_t)spec->members * WAVE_SP_QUARTER_ROWS;
  const size_t acc_rows =
      (size_t)spec->members * WAVE_ACC_QUARTER_ROWS;

  if (tiling.tile_I != spec->tile_I || tiling.tile_J != spec->tile_J ||
      tiling.tile_K != spec->tile_K ||
      !shared_multi_matmul_job_set_exact_tiling(
          job, tiling.tile_I, tiling.tile_J, tiling.tile_K) ||
      job->partition_status != SHARED_MULTI_PARTITION_OK ||
      job->sp_addr_A_stacked > operand_rows ||
      job->sp_addr_B_stacked > operand_rows ||
      job->acc_addr_stacked > acc_rows)
    return false;

  job->partition_status =
      shared_multi_partition_axis_status(job, job->partition_axis);
  if (job->partition_status != SHARED_MULTI_PARTITION_OK ||
      !shared_multi_matmul_job_init_state(job) || job->done || !job->no_bias)
    return false;

  return job->I0 * job->J0 * job->K0 == spec->steps_per_job &&
      !gemmini_page_packed_stride_is_packed(job->stride_A) &&
      gemmini_page_packed_stride_is_packed(job->stride_B) &&
      !gemmini_page_packed_stride_is_packed(job->stride_D) &&
      !gemmini_page_packed_stride_is_packed(job->stride_C);
}

static bool wave_prepare_one_step(
    shared_multi_matmul_job_t *job, const wave_tenant_spec_t *spec,
    wave_member_desc_t *members, wave_rocc_args_t *launch) {
  shared_multi_matmul_job_step_t step;
  if (job == NULL || spec == NULL || members == NULL || launch == NULL ||
      !shared_multi_matmul_job_plan_step(job, &step) ||
      step.active_gemmini_count != spec->members ||
      step.group_list != spec->mask)
    return false;

  const size_t plain_stride_A =
      gemmini_page_packed_stride_payload(job->stride_A);
  const size_t plain_stride_C =
      gemmini_page_packed_stride_payload(job->stride_C);
  const size_t sp_phase = (size_t)(job->inner_call_counter & 1);
  const size_t local_sp_addr_start = job->sp_addr_start_stack +
      sp_phase * (BANK_NUM * BANK_ROWS / 2u);
  const size_t local_sp_addr_end =
      (sp_phase + 1u) * (BANK_NUM * BANK_ROWS / 2u) -
      job->sp_addr_end_stack;
  const size_t acc_phase = job->lastK_toggle ? 0u : 1u;
  const size_t local_acc_addr_start = job->acc_addr_start_stack +
      acc_phase * (ACC_ROWS / 2u);

  // Use the full microbenchmark's absolute ACC-quarter identity.  This gives
  // gid=q+4*phase, independent of the tenant's allocation width.
  if (WAVE_ACC_GID_ROWS == 0u ||
      local_acc_addr_start % WAVE_ACC_GID_ROWS != 0u)
    return false;
  const size_t group_id = local_acc_addr_start / WAVE_ACC_GID_ROWS;
  if (group_id != (size_t)spec->q + 4u * acc_phase || group_id >= 8u)
    return false;

  const size_t A_row_offset = step.i0 * job->tile_I;
  const size_t A_col_offset = step.k0 * job->tile_K;
  const size_t B_row_offset = step.k0 * job->tile_K;
  const size_t B_col_offset = step.j0 * job->tile_J;
  const size_t D_row_offset = A_row_offset;
  const size_t D_col_offset = B_col_offset;
  const size_t C_row_offset = A_row_offset;
  const size_t C_col_offset = B_col_offset;

  if (!gemmini_loop_bounds_fields_valid(
          step.I, step.J, step.K,
          step.pad_I, step.pad_J, step.pad_K) ||
      !gemmini_page_offset_fields_valid(
          A_row_offset, A_col_offset, B_row_offset, B_col_offset,
          D_row_offset, D_col_offset, C_row_offset, C_col_offset) ||
      !gemmini_shared_partition_page_offsets_valid(
          WAVE_AXIS, step.K, 0u, step.I, 0u,
          step.I, step.J, step.K,
          false, true, false, false,
          A_row_offset, A_col_offset, B_row_offset, B_col_offset,
          D_row_offset, D_col_offset, C_row_offset, C_col_offset))
    return false;

  const uintptr_t a_outer = (uintptr_t)job->A +
      (step.i0 * job->tile_I * DIM * plain_stride_A +
       step.k0 * job->tile_K * DIM) * sizeof(elem_t);
  const uintptr_t b_outer = (uintptr_t)job->B;
  const uintptr_t c_outer = step.final_k_step
      ? (uintptr_t)job->C +
          (step.i0 * job->tile_I * DIM * plain_stride_C +
           step.j0 * job->tile_J * DIM) * job->sizeof_C
      : UINT64_C(0);

  for (size_t rank = 0; rank < spec->members; ++rank) {
    gemmini_partition_plan_t plan;
    if (!gemmini_partition_plan_member(
            WAVE_AXIS,
            step.I, step.J, step.K,
            step.pad_I, step.pad_J, step.pad_K,
            spec->members, rank, &plan) ||
        !gemmini_shared_partition_page_offsets_valid(
            WAVE_AXIS,
            plan.partition_extent, plan.partition_offset,
            plan.aux_extent, plan.aux_offset,
            step.I, step.J, step.K,
            false, true, false, false,
            A_row_offset, A_col_offset, B_row_offset, B_col_offset,
            D_row_offset, D_col_offset, C_row_offset, C_col_offset))
      return false;

    // K partitioning advances plain A by the member K offset.  Packed B keeps
    // the full-buffer base and carries outer/member offsets in funct=28/24.
    const uintptr_t a_local = a_outer +
        plan.partition_offset * DIM * sizeof(elem_t);
    const uintptr_t b_local = b_outer;
    const uintptr_t c_local = c_outer == 0u ? 0u : c_outer +
        plan.aux_offset * plain_stride_C * DIM * job->sizeof_C;
    wave_member_desc_t *member = &members[rank];

    wave_set_args(&member->command[WAVE_CMD_SPADDR],
        local_acc_addr_start,
        ((uint64_t)local_sp_addr_end << 16) | local_sp_addr_start);
    wave_set_args(&member->command[WAVE_CMD_PARTITION],
        gemmini_shared_partition_pack_rs1(
            WAVE_AXIS, plan.partition_offset, plan.aux_offset),
        gemmini_shared_partition_pack_rs2(
            plan.partition_extent, plan.aux_extent, plan.aux_pad));
    wave_set_args(&member->command[WAVE_CMD_BOUNDS],
        ((uint64_t)plan.partition_pad << 32) |
            ((uint64_t)step.pad_J << 16) | step.pad_I,
        ((uint64_t)step.K << 32) |
            ((uint64_t)step.J << 16) | step.I);
    wave_set_args(&member->command[WAVE_CMD_ADDRS_AB], a_local, b_local);
    wave_set_args(&member->command[WAVE_CMD_ADDRS_DC], 0u, c_local);
    wave_set_args(&member->command[WAVE_CMD_STRIDES_AB],
        job->stride_A, job->stride_B);
    wave_set_args(&member->command[WAVE_CMD_STRIDES_DC],
        job->stride_D, job->stride_C);
    wave_set_args(&member->command[WAVE_CMD_PAGE_OFFSETS],
        GEMMINI_PACK_PAGE_BLOCK_OFFSETS(
            A_row_offset, A_col_offset, B_row_offset, B_col_offset),
        GEMMINI_PACK_PAGE_BLOCK_OFFSETS(
            D_row_offset, D_col_offset, C_row_offset, C_col_offset));
  }

  wave_set_args(launch,
      ((uint64_t)step.group_list << 48) |
          ((uint64_t)group_id << 32) |
          ((uint64_t)job->act << 8) |
          ((uint64_t)job->low_D << 2) |
          ((uint64_t)job->full_C << 1) |
          (uint64_t)gemmini_partition_ex_accumulate(job->no_bias, step.k0),
      ((uint64_t)job->b_transpose << 1) |
          (uint64_t)job->a_transpose);

  shared_multi_matmul_job_complete_step(job);
  return job->partition_status == SHARED_MULTI_PARTITION_OK;
}

static bool wave_build_tenant_schedule(
    const wave_tenant_spec_t *spec, void *schedule,
    size_t descriptor_bytes, size_t schedule_steps) {
  if (!wave_spec_is_valid(spec) || schedule == NULL ||
      descriptor_bytes !=
          spec->members * sizeof(wave_member_desc_t) +
              sizeof(wave_rocc_args_t) ||
      schedule_steps != spec->queue_jobs * spec->steps_per_job)
    return false;

  size_t schedule_index = 0u;
  for (unsigned queued = 0; queued < spec->queue_jobs; ++queued) {
    shared_multi_matmul_job_t job;
    if (!wave_prepare_job(&job, spec->first_job + queued, spec))
      return false;

    const size_t first = schedule_index;
    while (!job.done) {
      if (schedule_index >= schedule_steps)
        return false;
      uint8_t *descriptor =
          (uint8_t *)schedule + schedule_index * descriptor_bytes;
      wave_rocc_args_t *launch = (wave_rocc_args_t *)(void *)descriptor;
      wave_member_desc_t *members =
          (wave_member_desc_t *)(void *)(launch + 1);
      if (!wave_prepare_one_step(&job, spec, members, launch))
        return false;
      ++schedule_index;
    }
    if (schedule_index - first != spec->steps_per_job)
      return false;
  }
  return schedule_index == schedule_steps;
}

static bool wave_prepare_policy_schedules(void) {
  bool built = false;
#if WAVE_POLICY == WAVE_POLICY_QUEUED_G4
  built = wave_build_tenant_schedule(
      &wave_tenant_specs[0], wave_schedule0,
      sizeof(wave_schedule0[0]), WAVE_GROUP_STEPS_0);
#elif WAVE_POLICY == WAVE_POLICY_2PLUS2 || \
      WAVE_POLICY == WAVE_POLICY_3PLUS1
  built = wave_build_tenant_schedule(
             &wave_tenant_specs[0], wave_schedule0,
             sizeof(wave_schedule0[0]), WAVE_GROUP_STEPS_0) &&
         wave_build_tenant_schedule(
             &wave_tenant_specs[1], wave_schedule1,
             sizeof(wave_schedule1[0]), WAVE_GROUP_STEPS_1);
#elif WAVE_POLICY == WAVE_POLICY_2PLUS1PLUS1
  built = wave_build_tenant_schedule(
             &wave_tenant_specs[0], wave_schedule0,
             sizeof(wave_schedule0[0]), WAVE_GROUP_STEPS_0) &&
         wave_build_tenant_schedule(
             &wave_tenant_specs[1], wave_schedule1,
             sizeof(wave_schedule1[0]), WAVE_GROUP_STEPS_1) &&
         wave_build_tenant_schedule(
             &wave_tenant_specs[2], wave_schedule2,
             sizeof(wave_schedule2[0]), WAVE_GROUP_STEPS_2);
#else
  built = wave_build_tenant_schedule(
             &wave_tenant_specs[0], wave_schedule0,
             sizeof(wave_schedule0[0]), WAVE_GROUP_STEPS_0) &&
         wave_build_tenant_schedule(
             &wave_tenant_specs[1], wave_schedule1,
             sizeof(wave_schedule1[0]), WAVE_GROUP_STEPS_1) &&
         wave_build_tenant_schedule(
             &wave_tenant_specs[2], wave_schedule2,
             sizeof(wave_schedule2[0]), WAVE_GROUP_STEPS_2) &&
         wave_build_tenant_schedule(
             &wave_tenant_specs[3], wave_schedule3,
             sizeof(wave_schedule3[0]), WAVE_GROUP_STEPS_3);
#endif
  if (!built)
    return false;

  uint8_t *cursor = wave_replay_schedule;
  const uint8_t *const end =
      wave_replay_schedule + sizeof(wave_replay_schedule);
#define WAVE_APPEND(record_)                                                \
  do {                                                                      \
    const size_t record_bytes_ = sizeof(record_);                           \
    if (cursor > end || record_bytes_ > (size_t)(end - cursor))             \
      return false;                                                         \
    memcpy(cursor, &(record_), record_bytes_);                              \
    cursor += record_bytes_;                                                \
  } while (0)

  // Assemble the same tenant-round stream used by the full FireSim runner.
#if WAVE_POLICY == WAVE_POLICY_QUEUED_G4
  for (size_t round = 0; round < WAVE_GROUP_STEPS_0; ++round)
    WAVE_APPEND(wave_schedule0[round]);
#elif WAVE_POLICY == WAVE_POLICY_2PLUS2
  for (size_t round = 0; round < WAVE_SCHEDULER_ROUNDS; ++round) {
    WAVE_APPEND(wave_schedule0[round]);
    WAVE_APPEND(wave_schedule1[round]);
  }
#elif WAVE_POLICY == WAVE_POLICY_3PLUS1
  for (size_t round = 0; round < WAVE_GROUP_STEPS_1; ++round) {
    WAVE_APPEND(wave_schedule0[round]);
    WAVE_APPEND(wave_schedule1[round]);
  }
  for (size_t round = WAVE_GROUP_STEPS_1;
       round < WAVE_GROUP_STEPS_0; ++round)
    WAVE_APPEND(wave_schedule0[round]);
#elif WAVE_POLICY == WAVE_POLICY_2PLUS1PLUS1
  for (size_t round = 0; round < WAVE_GROUP_STEPS_0; ++round) {
    WAVE_APPEND(wave_schedule0[round]);
    WAVE_APPEND(wave_schedule1[round]);
    WAVE_APPEND(wave_schedule2[round]);
  }
  for (size_t round = WAVE_GROUP_STEPS_0;
       round < WAVE_SCHEDULER_ROUNDS; ++round) {
    WAVE_APPEND(wave_schedule1[round]);
    WAVE_APPEND(wave_schedule2[round]);
  }
#else
  for (size_t round = 0; round < WAVE_SCHEDULER_ROUNDS; ++round) {
    WAVE_APPEND(wave_schedule0[round]);
    WAVE_APPEND(wave_schedule1[round]);
    WAVE_APPEND(wave_schedule2[round]);
    WAVE_APPEND(wave_schedule3[round]);
  }
#endif

#undef WAVE_APPEND
  return cursor == end;
}

#define WAVE_ISSUE_MEMBER(custom_num, member_ptr)                            \
  do {                                                                       \
    ROCC_INSTRUCTION_RS1_RS2(                                                \
        custom_num,                                                          \
        (member_ptr)->command[WAVE_CMD_SPADDR].rs1,                          \
        (member_ptr)->command[WAVE_CMD_SPADDR].rs2,                          \
        k_LOOP_WS_CONFIG_SPADDR);                                            \
    ROCC_INSTRUCTION_RS1_RS2(                                                \
        custom_num,                                                          \
        (member_ptr)->command[WAVE_CMD_PARTITION].rs1,                       \
        (member_ptr)->command[WAVE_CMD_PARTITION].rs2,                       \
        k_LOOP_WS_CONFIG_PARTITION_BOUNDS);                                  \
    ROCC_INSTRUCTION_RS1_RS2(                                                \
        custom_num,                                                          \
        (member_ptr)->command[WAVE_CMD_BOUNDS].rs1,                          \
        (member_ptr)->command[WAVE_CMD_BOUNDS].rs2,                          \
        k_LOOP_WS_CONFIG_BOUNDS);                                            \
    ROCC_INSTRUCTION_RS1_RS2(                                                \
        custom_num,                                                          \
        (member_ptr)->command[WAVE_CMD_ADDRS_AB].rs1,                        \
        (member_ptr)->command[WAVE_CMD_ADDRS_AB].rs2,                        \
        k_LOOP_WS_CONFIG_ADDRS_AB);                                          \
    ROCC_INSTRUCTION_RS1_RS2(                                                \
        custom_num,                                                          \
        (member_ptr)->command[WAVE_CMD_ADDRS_DC].rs1,                        \
        (member_ptr)->command[WAVE_CMD_ADDRS_DC].rs2,                        \
        k_LOOP_WS_CONFIG_ADDRS_DC);                                          \
    ROCC_INSTRUCTION_RS1_RS2(                                                \
        custom_num,                                                          \
        (member_ptr)->command[WAVE_CMD_STRIDES_AB].rs1,                      \
        (member_ptr)->command[WAVE_CMD_STRIDES_AB].rs2,                      \
        k_LOOP_WS_CONFIG_STRIDES_AB);                                        \
    ROCC_INSTRUCTION_RS1_RS2(                                                \
        custom_num,                                                          \
        (member_ptr)->command[WAVE_CMD_STRIDES_DC].rs1,                      \
        (member_ptr)->command[WAVE_CMD_STRIDES_DC].rs2,                      \
        k_LOOP_WS_CONFIG_STRIDES_DC);                                        \
    ROCC_INSTRUCTION_RS1_RS2(                                                \
        custom_num,                                                          \
        (member_ptr)->command[WAVE_CMD_PAGE_OFFSETS].rs1,                    \
        (member_ptr)->command[WAVE_CMD_PAGE_OFFSETS].rs2,                    \
        k_LOOP_WS_CONFIG_PAGE_OFFSETS);                                      \
  } while (0)

static inline __attribute__((always_inline)) const uint8_t *
wave_issue_step_descriptor(const uint8_t *cursor) {
  const wave_rocc_args_t *launch =
      (const wave_rocc_args_t *)(const void *)cursor;
  const wave_member_desc_t *member =
      (const wave_member_desc_t *)(const void *)(launch + 1);
  const uint64_t launch_rs1 = launch->rs1;
  const uint64_t launch_rs2 = launch->rs2;
  const unsigned group_list = (unsigned)((launch_rs1 >> 48) & 0xfu);

  switch (group_list) {
  case 0x1u:
    WAVE_ISSUE_MEMBER(custom0, &member[0]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom0, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 1);
  case 0x2u:
    WAVE_ISSUE_MEMBER(custom1, &member[0]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom1, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 1);
  case 0x3u:
    WAVE_ISSUE_MEMBER(custom0, &member[0]);
    WAVE_ISSUE_MEMBER(custom1, &member[1]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom1, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 2);
  case 0x4u:
    WAVE_ISSUE_MEMBER(custom2, &member[0]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom2, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 1);
  case 0x7u:
    WAVE_ISSUE_MEMBER(custom0, &member[0]);
    WAVE_ISSUE_MEMBER(custom1, &member[1]);
    WAVE_ISSUE_MEMBER(custom2, &member[2]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom2, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 3);
  case 0x8u:
    WAVE_ISSUE_MEMBER(custom3, &member[0]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom3, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 1);
  case 0xcu:
    WAVE_ISSUE_MEMBER(custom2, &member[0]);
    WAVE_ISSUE_MEMBER(custom3, &member[1]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom3, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 2);
  case 0xfu:
    WAVE_ISSUE_MEMBER(custom0, &member[0]);
    WAVE_ISSUE_MEMBER(custom1, &member[1]);
    WAVE_ISSUE_MEMBER(custom2, &member[2]);
    WAVE_ISSUE_MEMBER(custom3, &member[3]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom3, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 4);
  default:
    __builtin_unreachable();
  }
}

static bool wave_configure_hardware_once(void) {
  const size_t stride_A =
      gemmini_page_packed_a_dma_stride_bytes(WAVE_K);
  const size_t stride_B = gemmini_page_packed_b_dma_stride_bytes(
      GEMMINI_PAGE_PACKED_STRIDE(WAVE_N));
  const size_t stride_D =
      gemmini_page_packed_acc_dma_stride_bytes(0u, sizeof(acc_t));
  const size_t stride_C =
      gemmini_page_packed_acc_dma_stride_bytes(WAVE_N, sizeof(acc_t));
  if (stride_A != 1024u || stride_B != 512u || stride_D != 0u ||
      stride_C != 5504u)
    return false;

#define WAVE_CONFIGURE_ONE(custom_num)                                       \
  do {                                                                       \
    gemmini_extended_config_ex(                                              \
        custom_num, WEIGHT_STATIONARY, NO_ACTIVATION, 0, 1, false, false);   \
    gemmini_extended_config_st(                                              \
        custom_num, stride_C, NO_ACTIVATION, ACC_SCALE_IDENTITY);            \
    gemmini_extended3_config_ld(                                             \
        custom_num, stride_A, MVIN_SCALE_IDENTITY, false, 0);                \
    gemmini_extended3_config_ld(                                             \
        custom_num, stride_B, MVIN_SCALE_IDENTITY, false, 1);                \
    gemmini_extended3_config_ld(                                             \
        custom_num, stride_D, MVIN_SCALE_IDENTITY, false, 2);                \
  } while (0)

  // Every policy covers all four physical Gemminis.  Configure each exactly
  // once; logical job initialization emits no hardware CONFIG command.
  WAVE_CONFIGURE_ONE(custom0);
  WAVE_CONFIGURE_ONE(custom1);
  WAVE_CONFIGURE_ONE(custom2);
  WAVE_CONFIGURE_ONE(custom3);
#undef WAVE_CONFIGURE_ONE
  gemmini_fence();
  return true;
}

// The compact launch-first descriptor traversal and runtime mask dispatch are
// deliberately the same lean timed path as the full FireSim microbenchmark.
static __attribute__((noinline)) wave_cycles_t wave_replay_policy(void) {
  wave_cycles_t cycles;
  wave_cpu_fence();
  cycles.start = wave_read_cycles();
  const uint8_t *cursor = wave_replay_schedule;
  for (size_t step = 0; step < WAVE_TOTAL_STEPS; ++step)
    cursor = wave_issue_step_descriptor(cursor);

  cycles.issue_end = wave_read_cycles();
  gemmini_fence();
  cycles.end = wave_read_cycles();
  wave_cpu_fence();
  cycles.consumed_exactly =
      cursor == wave_replay_schedule + sizeof(wave_replay_schedule);
  return cycles;
}

static bool wave_geometry_is_valid(void) {
  return DIM == 8 && BANK_NUM == 4 && BANK_ROWS == 4096 && ACC_ROWS == 8192 &&
      WAVE_SP_QUARTER_ROWS == 1024u &&
      WAVE_ACC_QUARTER_ROWS == 1024u && WAVE_ACC_GID_ROWS == 1024u &&
      sizeof(wave_rocc_args_t) == 16u &&
      sizeof(wave_member_desc_t) == 128u &&
      sizeof(wave_group1_desc_t) == 144u &&
      sizeof(wave_group2_desc_t) == 272u &&
      sizeof(wave_group3_desc_t) == 400u &&
      sizeof(wave_group4_desc_t) == 528u &&
      gemmini_partition_generated_axis_abi_version() ==
          GEMMINI_SHARED_PARTITION_ENCODING_VERSION &&
      wave_ranges_are_valid();
}

int main(void) {
  if (!wave_geometry_is_valid()) {
    printf("WAVEFORM-%s-FAIL stage=geometry\n", WAVE_POLICY_TAG);
    return 1;
  }

  printf("WAVEFORM-%s-BEGIN policy=%s shape=ffn_gate_up_scaled_1over8 "
         "source_shape=8x11008x4096 M=%u N=%u K=%u jobs=4 axis=K "
         "page_packing=A0B1C0D0 data_fill=0 cpu_gold=0 "
         "cache=evict8MiB_then_descriptor_prepass_then_A_recent_rr "
         "issue_path=precomputed_raw_rocc absolute_gid=q_plus_4phase "
         "hw_config=once_per_physical_gemmini final_fences=1\n",
         WAVE_POLICY_TAG, WAVE_POLICY_NAME, WAVE_M, WAVE_N, WAVE_K);

  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
  gemmini_flush(custom3, 0);
  gemmini_fence();

  if (!wave_prepare_policy_schedules()) {
    printf("WAVEFORM-%s-FAIL stage=schedule_prepare\n", WAVE_POLICY_TAG);
    return 1;
  }
  if (!wave_configure_hardware_once()) {
    printf("WAVEFORM-%s-FAIL stage=hardware_config\n", WAVE_POLICY_TAG);
    return 1;
  }

  printf("WAVEFORM-%s-PREFLIGHT tenants=%u rounds=%u "
         "group_steps=%u/%u/%u/%u timed_rocc_commands=%u "
         "descriptor_bytes=%llu %s tilings=%s gids=%s\n",
         WAVE_POLICY_TAG, WAVE_TENANT_COUNT, WAVE_SCHEDULER_ROUNDS,
         WAVE_GROUP_STEPS_0, WAVE_GROUP_STEPS_1,
         WAVE_GROUP_STEPS_2, WAVE_GROUP_STEPS_3,
         WAVE_TIMED_ROCC_COMMANDS,
         (unsigned long long)sizeof(wave_replay_schedule),
         WAVE_TOPOLOGY, WAVE_TILINGS, WAVE_GIDS);

  wave_condition_cache();
  const wave_cycles_t cycles = wave_replay_policy();
  if (!cycles.consumed_exactly) {
    printf("WAVEFORM-%s-FAIL stage=replay_consumption\n", WAVE_POLICY_TAG);
    return 1;
  }
  printf("WAVEFORM-%s-RESULT start_cycle=%llu issue_cycles=%llu "
         "fence_cycles=%llu makespan_cycles=%llu rounds=%u "
         "timed_rocc_commands=%u cache_sink=%llu\n",
         WAVE_POLICY_TAG,
         (unsigned long long)cycles.start,
         (unsigned long long)(cycles.issue_end - cycles.start),
         (unsigned long long)(cycles.end - cycles.issue_end),
         (unsigned long long)(cycles.end - cycles.start),
         WAVE_SCHEDULER_ROUNDS, WAVE_TIMED_ROCC_COMMANDS,
         (unsigned long long)wave_cache_sink);
  printf("WAVEFORM-%s-PASS\n", WAVE_POLICY_TAG);
  return 0;
}

#endif  // LLAMA2_FFN_GATE_UP_M8_SCALED_RAW_POLICY_COMMON_H_
