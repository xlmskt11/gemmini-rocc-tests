// Targeted bare-metal waveform proxy for the FireSim schema-v3 record:
//   shape=ffn_gate_up, M=8, N=11008, K=4096, policy=2+2, epoch=970.
// N and K are both scaled by 1/8 to keep debug-waveform generation practical;
// the concurrency policy, partition axis, tiling, and local-memory layout stay
// unchanged.
//
// Four distinct operand/output address ranges are used. B alone is page
// packed. There is no operand initialization or CPU-gold calculation.
//
// All tiling, validation, address construction, and command-operand packing is
// completed before the timed interval. The timed path only replays the exact
// shared LOOP_WS RoCC command stream for the two 2-Gemmini groups.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils_all.h"

#define WAVE_M 8u
#define WAVE_N 1376u
#define WAVE_K 512u
#define WAVE_JOBS 4
#define WAVE_STEPS_PER_JOB 44u
#define WAVE_ROUNDS (2u * WAVE_STEPS_PER_JOB)
#define WAVE_MEMBERS_PER_GROUP 2u
#define WAVE_COMMANDS_PER_MEMBER 8u
#define WAVE_COMMANDS_PER_GROUP_STEP \
  (WAVE_MEMBERS_PER_GROUP * WAVE_COMMANDS_PER_MEMBER + 1u)
#define WAVE_TIMED_ROCC_COMMANDS \
  (WAVE_ROUNDS * 2u * WAVE_COMMANDS_PER_GROUP_STEP)
#define WAVE_CACHE_LINE_BYTES 64u
#define WAVE_EVICTION_BYTES (8u * 1024u * 1024u)
#define WAVE_A_BYTES (WAVE_M * WAVE_K * sizeof(elem_t))
#define WAVE_C_BYTES (WAVE_M * WAVE_N * sizeof(acc_t))
#define WAVE_B_PAGES 384u
#define WAVE_B_BYTES (WAVE_B_PAGES * GEMMINI_PAGE_PACKED_PAGE_BYTES)

#if !GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE
#error "The 2+2 waveform replay requires the shared partition descriptor ABI"
#endif

// Keep the ELF/loadmem image small: the disjoint buffers live at fixed target
// DRAM addresses instead of being emitted in BSS. The scaled operands are
// compactly placed inside the unmodified 256-MiB Verilator memory aperture.
#define WAVE_EVICTION_BASE UINT64_C(0x81000000)
#define WAVE_A_BASE        UINT64_C(0x81800000)
#define WAVE_C_BASE        UINT64_C(0x81900000)
#define WAVE_B0_BASE       UINT64_C(0x82000000)
#define WAVE_B1_BASE       UINT64_C(0x82400000)
#define WAVE_B2_BASE       UINT64_C(0x82800000)
#define WAVE_B3_BASE       UINT64_C(0x82c00000)
#define WAVE_B_SLOT_BYTES  UINT64_C(0x00400000)

typedef struct {
  uint64_t rs1;
  uint64_t rs2;
} wave_rocc_args_t;

typedef struct {
  wave_rocc_args_t spaddr;
  wave_rocc_args_t partition;
  wave_rocc_args_t bounds;
  wave_rocc_args_t addrs_ab;
  wave_rocc_args_t addrs_dc;
  wave_rocc_args_t strides_ab;
  wave_rocc_args_t strides_dc;
  wave_rocc_args_t page_offsets;
} wave_member_desc_t;

typedef struct {
  wave_member_desc_t member[WAVE_MEMBERS_PER_GROUP];
  wave_rocc_args_t launch;
} wave_group_step_desc_t;

// These schedules are metadata, not operands. Keeping them out of the stack
// avoids a roughly 48-KiB bare-metal stack allocation.
static wave_group_step_desc_t wave_group01_schedule[WAVE_ROUNDS]
    __attribute__((aligned(WAVE_CACHE_LINE_BYTES)));
static wave_group_step_desc_t wave_group23_schedule[WAVE_ROUNDS]
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

static elem_t *wave_a(int job) {
  return (elem_t *)(uintptr_t)(WAVE_A_BASE +
      (uint64_t)job * UINT64_C(0x00010000));
}

static void *wave_c(int job) {
  return (void *)(uintptr_t)(WAVE_C_BASE +
      (uint64_t)job * UINT64_C(0x00100000));
}

static const elem_t *wave_b(int job) {
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
  for (int job = 0; job < WAVE_JOBS - 1; ++job) {
    const uintptr_t end = (uintptr_t)wave_b(job) + WAVE_B_BYTES;
    if (end > (uintptr_t)wave_b(job + 1))
      return false;
  }
  return (uintptr_t)wave_b(WAVE_JOBS - 1) + WAVE_B_BYTES <
      UINT64_C(0x90000000);
}

static void wave_touch_schedule(
    const wave_group_step_desc_t schedule[WAVE_ROUNDS], uint64_t *sink) {
  const volatile uint8_t *bytes = (const volatile uint8_t *)schedule;
  for (size_t offset = 0; offset < sizeof(wave_group_step_desc_t) * WAVE_ROUNDS;
       offset += WAVE_CACHE_LINE_BYTES)
    *sink += bytes[offset];
}

static void wave_condition_cache(void) {
  volatile uint8_t *eviction =
      (volatile uint8_t *)(uintptr_t)WAVE_EVICTION_BASE;
  uint64_t sink = wave_cache_sink;
  for (size_t offset = 0; offset < WAVE_EVICTION_BYTES;
       offset += WAVE_CACHE_LINE_BYTES)
    sink += eviction[offset];

  // Descriptor replay must not measure compulsory metadata misses. Bring the
  // schedules back only after eviction. They occupy about 48 KiB of the 2-MiB
  // L2 and do not touch B or C.
  wave_touch_schedule(wave_group01_schedule, &sink);
  wave_touch_schedule(wave_group23_schedule, &sink);

  // Match the Linux microbenchmark's four-A round-robin recent pass. This is
  // deliberately last so A is newer than the descriptor metadata. B and C
  // remain untouched and cold.
  for (size_t offset = 0; offset < WAVE_A_BYTES;
       offset += WAVE_CACHE_LINE_BYTES) {
    for (int job = 0; job < WAVE_JOBS; ++job) {
      volatile uint8_t *a = (volatile uint8_t *)wave_a(job);
      const uint8_t value = a[offset];
      a[offset] = value;
    }
  }
  wave_cache_sink = sink;
  wave_cpu_fence();
}

// Prepare one independent software job only. Hardware CONFIG commands are
// intentionally emitted once per physical Gemmini by wave_configure_hardware.
static bool wave_prepare_job(shared_multi_matmul_job_t *job, int job_index) {
  const bool high_half = job_index >= 2;
  const size_t q = high_half ? 2u : 0u;

  memset(job, 0, sizeof(*job));
  job->gemmini_list = high_half ? 0xcu : 0x3u;
  job->tile_id = job_index;
  job->sp_addr_start_stack = q * BANK_ROWS / 4u;
  job->sp_addr_end_stack = q * BANK_ROWS / 4u;
  job->acc_addr_start_stack = q * (ACC_ROWS / 2u) / 4u;
  job->sp_addr_range = 2u * BANK_ROWS;
  job->acc_addr_range = ACC_ROWS / 2u;
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
  job->partition_axis = GEMMINI_PARTITION_AXIS_K;

  const gemmini_tiling_request_t request = {
    .dim_I = WAVE_M,
    .dim_J = WAVE_N,
    .dim_K = WAVE_K,
    .dim = DIM,
    .gemmini_count = WAVE_MEMBERS_PER_GROUP,
    .sp_addr_range = job->sp_addr_range,
    .acc_addr_range = job->acc_addr_range,
    .double_buffered = true,
    .act = NO_ACTIVATION,
  };
  const gemmini_tiling_factors_t tiling =
      gemmini_shared_multi_choose_tiling_axis(
          &request, GEMMINI_PARTITION_AXIS_K);
  job->gemmini_num = WAVE_MEMBERS_PER_GROUP;
  if (!shared_multi_matmul_job_set_exact_tiling(
          job, tiling.tile_I, tiling.tile_J, tiling.tile_K) ||
      job->partition_status != SHARED_MULTI_PARTITION_OK ||
      job->tile_I != 1u || job->tile_J != 16u || job->tile_K != 16u ||
      job->sp_addr_A_stacked != 128u ||
      job->sp_addr_B_stacked != 2048u ||
      job->acc_addr_stacked != 128u ||
      job->sp_addr_A_stacked > 2u * BANK_ROWS / 4u ||
      job->sp_addr_B_stacked > 2u * BANK_ROWS / 4u ||
      job->acc_addr_stacked > 2u * (ACC_ROWS / 2u) / 4u)
    return false;

  job->partition_status =
      shared_multi_partition_axis_status(job, job->partition_axis);
  if (job->partition_status != SHARED_MULTI_PARTITION_OK) {
    job->done = true;
    return false;
  }
  if (!shared_multi_matmul_job_init_state(job))
    return false;

  return job->I0 == 1u && job->J0 == 11u && job->K0 == 4u &&
      job->last_I == 1u && job->last_J == 12u && job->last_K == 16u &&
      job->padding_I == 0u && job->padding_J == 0u &&
      job->padding_K == 0u && job->no_bias &&
      !gemmini_page_packed_stride_is_packed(job->stride_A) &&
      gemmini_page_packed_stride_is_packed(job->stride_B) &&
      !gemmini_page_packed_stride_is_packed(job->stride_D) &&
      !gemmini_page_packed_stride_is_packed(job->stride_C) &&
      !job->done;
}

static bool wave_hardware_configs_match(
    const shared_multi_matmul_job_t *reference,
    const shared_multi_matmul_job_t *candidate) {
  return reference->stride_A == candidate->stride_A &&
      reference->stride_B == candidate->stride_B &&
      reference->stride_D == candidate->stride_D &&
      reference->stride_C == candidate->stride_C &&
      reference->A_scale_factor == candidate->A_scale_factor &&
      reference->B_scale_factor == candidate->B_scale_factor &&
      reference->D_scale_factor == candidate->D_scale_factor &&
      reference->dataflow == candidate->dataflow &&
      reference->act == candidate->act &&
      reference->scale == candidate->scale &&
      reference->repeating_bias == candidate->repeating_bias &&
      reference->a_transpose == candidate->a_transpose &&
      reference->b_transpose == candidate->b_transpose &&
      reference->low_D == candidate->low_D &&
      reference->sizeof_D == candidate->sizeof_D &&
      reference->sizeof_C == candidate->sizeof_C;
}

static bool wave_configure_hardware(
    const shared_multi_matmul_job_t jobs[WAVE_JOBS]) {
  for (int job = 1; job < WAVE_JOBS; ++job) {
    if (!wave_hardware_configs_match(&jobs[0], &jobs[job]))
      return false;
  }

  const shared_multi_matmul_job_t *job = &jobs[0];
  const size_t config_stride_A =
      gemmini_page_packed_a_dma_stride_bytes(job->stride_A);
  const size_t config_stride_B =
      gemmini_page_packed_b_dma_stride_bytes(job->stride_B);
  const size_t config_stride_D =
      gemmini_page_packed_acc_dma_stride_bytes(
          job->stride_D, job->sizeof_D);
  const size_t config_stride_C =
      gemmini_page_packed_acc_dma_stride_bytes(
          job->stride_C, job->sizeof_C);
  if (config_stride_A != 1024u || config_stride_B != 512u ||
      config_stride_D != 0u || config_stride_C != 5504u)
    return false;

#define WAVE_CONFIGURE_ONE(custom_)                                           \
  do {                                                                        \
    gemmini_extended_config_ex(                                               \
        custom_, job->dataflow, job->act & 3, 0, 1,                          \
        job->a_transpose, job->b_transpose);                                  \
    gemmini_extended_config_st(                                               \
        custom_, config_stride_C, job->act & 3, job->scale);                  \
    gemmini_extended3_config_ld(                                              \
        custom_, config_stride_A, job->A_scale_factor, false, 0);             \
    gemmini_extended3_config_ld(                                              \
        custom_, config_stride_B, job->B_scale_factor, false, 1);             \
    gemmini_extended3_config_ld(                                              \
        custom_, job->repeating_bias ? 0 : config_stride_D,                   \
        job->D_scale_factor, job->low_D, 2);                                  \
  } while (0)

  WAVE_CONFIGURE_ONE(custom0);
  WAVE_CONFIGURE_ONE(custom1);
  WAVE_CONFIGURE_ONE(custom2);
  WAVE_CONFIGURE_ONE(custom3);

#undef WAVE_CONFIGURE_ONE
  return true;
}

static wave_rocc_args_t wave_rocc_args(uint64_t rs1, uint64_t rs2) {
  const wave_rocc_args_t args = {.rs1 = rs1, .rs2 = rs2};
  return args;
}

static bool wave_build_job_schedule(
    shared_multi_matmul_job_t *job,
    wave_group_step_desc_t *schedule,
    size_t schedule_capacity,
    size_t *steps_written) {
  if (job == NULL || schedule == NULL || steps_written == NULL || job->done ||
      job->partition_axis != GEMMINI_PARTITION_AXIS_K ||
      job->gemmini_num != WAVE_MEMBERS_PER_GROUP ||
      (job->gemmini_list != 0x3 && job->gemmini_list != 0xc) ||
      job->dataflow != WEIGHT_STATIONARY || !job->no_bias ||
      job->a_transpose || job->b_transpose || !job->full_C || job->low_D ||
      gemmini_page_packed_stride_is_packed(job->stride_A) ||
      !gemmini_page_packed_stride_is_packed(job->stride_B) ||
      gemmini_page_packed_stride_is_packed(job->stride_D) ||
      gemmini_page_packed_stride_is_packed(job->stride_C))
    return false;

  const size_t plain_stride_A =
      gemmini_page_packed_stride_payload(job->stride_A);
  const size_t plain_stride_C =
      gemmini_page_packed_stride_payload(job->stride_C);
  size_t count = 0;

  while (!job->done) {
    if (count >= schedule_capacity)
      return false;

    shared_multi_matmul_job_step_t step;
    if (!shared_multi_matmul_job_plan_step(job, &step) ||
        step.active_gemmini_count != WAVE_MEMBERS_PER_GROUP ||
        step.group_list != (uint64_t)(unsigned int)job->gemmini_list)
      return false;

    const size_t i0 = step.i0;
    const size_t j0 = step.j0;
    const size_t k0 = step.k0;
    const int phase = job->inner_call_counter & 1;
    const size_t local_sp_start = phase == 0
        ? job->sp_addr_start_stack
        : job->sp_addr_start_stack + BANK_NUM * BANK_ROWS / 2u;
    const size_t local_sp_end = phase == 0
        ? BANK_NUM * BANK_ROWS / 2u - job->sp_addr_end_stack
        : BANK_NUM * BANK_ROWS - job->sp_addr_end_stack;
    const size_t local_acc_start = job->lastK_toggle
        ? job->acc_addr_start_stack
        : job->acc_addr_start_stack + ACC_ROWS / 2u;
    const size_t group_id_unit = ACC_ROWS / 4u;
    if (local_acc_start % group_id_unit != 0u)
      return false;
    const uint64_t group_id = local_acc_start / group_id_unit;
    if (group_id >= 4u)
      return false;

    const size_t A_row_offset = i0 * job->tile_I;
    const size_t A_col_offset = k0 * job->tile_K;
    const size_t B_row_offset = k0 * job->tile_K;
    const size_t B_col_offset = j0 * job->tile_J;
    const size_t D_row_offset = A_row_offset;
    const size_t D_col_offset = B_col_offset;
    const size_t C_row_offset = A_row_offset;
    const size_t C_col_offset = B_col_offset;
    if (!gemmini_shared_partition_page_offsets_valid(
            GEMMINI_PARTITION_AXIS_K,
            step.K, 0, step.I, 0,
            step.I, step.J, step.K,
            false, true, false, false,
            A_row_offset, A_col_offset, B_row_offset, B_col_offset,
            D_row_offset, D_col_offset, C_row_offset, C_col_offset))
      return false;

    const elem_t *a = job->A +
        i0 * job->tile_I * DIM * plain_stride_A +
        k0 * job->tile_K * DIM;
    const elem_t *b = job->B;
    void *out = step.final_k_step
        ? (void *)((uint8_t *)job->C +
            (i0 * job->tile_I * DIM * plain_stride_C +
             j0 * job->tile_J * DIM) * job->sizeof_C)
        : NULL;

    wave_group_step_desc_t *desc = &schedule[count];
    for (size_t rank = 0; rank < WAVE_MEMBERS_PER_GROUP; ++rank) {
      gemmini_partition_plan_t plan;
      if (!gemmini_partition_plan_member(
              GEMMINI_PARTITION_AXIS_K,
              step.I, step.J, step.K,
              step.pad_I, step.pad_J, step.pad_K,
              WAVE_MEMBERS_PER_GROUP, rank, &plan) ||
          plan.partition_extent != 8u ||
          plan.partition_offset != rank * 8u ||
          plan.aux_extent != (rank == 0 ? 1u : 0u) ||
          plan.aux_offset != rank ||
          plan.partition_pad != 0u || plan.aux_pad != 0u)
        return false;

      const size_t member_pad_K = plan.partition_pad;
      const elem_t *a_local = a + plan.partition_offset * DIM;
      // B is the only page-packed operand. Its pointer stays at the full-buffer
      // base; funct=24 supplies this member's K offset exactly once.
      const elem_t *b_local = b;
      void *c_local = out == NULL
          ? NULL
          : (void *)((uint8_t *)out +
              plan.aux_offset * plain_stride_C * DIM * job->sizeof_C);

      if (!gemmini_loop_bounds_fields_valid(
              step.I, step.J, step.K,
              step.pad_I, step.pad_J, member_pad_K) ||
          !gemmini_page_offset_fields_valid(
              A_row_offset, A_col_offset, B_row_offset, B_col_offset,
              D_row_offset, D_col_offset, C_row_offset, C_col_offset) ||
          !gemmini_shared_partition_page_offsets_valid(
              GEMMINI_PARTITION_AXIS_K,
              plan.partition_extent, plan.partition_offset,
              plan.aux_extent, plan.aux_offset,
              step.I, step.J, step.K,
              false, true, false, false,
              A_row_offset, A_col_offset, B_row_offset, B_col_offset,
              D_row_offset, D_col_offset, C_row_offset, C_col_offset))
        return false;

      wave_member_desc_t *member = &desc->member[rank];
      member->spaddr = wave_rocc_args(
          local_acc_start,
          ((uint64_t)local_sp_end << 16) | (uint64_t)local_sp_start);
      member->partition = wave_rocc_args(
          gemmini_shared_partition_pack_rs1(
              GEMMINI_PARTITION_AXIS_K,
              plan.partition_offset, plan.aux_offset),
          gemmini_shared_partition_pack_rs2(
              plan.partition_extent, plan.aux_extent, plan.aux_pad));
      member->bounds = wave_rocc_args(
          ((uint64_t)member_pad_K << 32) |
              ((uint64_t)step.pad_J << 16) | (uint64_t)step.pad_I,
          ((uint64_t)step.K << 32) |
              ((uint64_t)step.J << 16) | (uint64_t)step.I);
      member->addrs_ab = wave_rocc_args(
          (uint64_t)(uintptr_t)a_local,
          (uint64_t)(uintptr_t)b_local);
      member->addrs_dc = wave_rocc_args(
          0,
          (uint64_t)(uintptr_t)c_local);
      member->strides_ab = wave_rocc_args(job->stride_A, job->stride_B);
      member->strides_dc = wave_rocc_args(job->stride_D, job->stride_C);
      member->page_offsets = wave_rocc_args(
          GEMMINI_PACK_PAGE_BLOCK_OFFSETS(
              A_row_offset, A_col_offset, B_row_offset, B_col_offset),
          GEMMINI_PACK_PAGE_BLOCK_OFFSETS(
              D_row_offset, D_col_offset, C_row_offset, C_col_offset));
    }

    desc->launch = wave_rocc_args(
        ((uint64_t)step.group_list << 48) |
            (group_id << 32) |
            ((uint64_t)job->act << 8) |
            ((uint64_t)job->low_D << 2) |
            ((uint64_t)job->full_C << 1) |
            (uint64_t)gemmini_partition_ex_accumulate(job->no_bias, k0),
        ((uint64_t)job->b_transpose << 1) |
            (uint64_t)job->a_transpose);

    ++count;
    shared_multi_matmul_job_complete_step(job);
  }

  if (job->partition_status != SHARED_MULTI_PARTITION_OK ||
      count != WAVE_STEPS_PER_JOB)
    return false;
  *steps_written = count;
  return true;
}

#define WAVE_ISSUE_MEMBER(custom_, member_)                                  \
  do {                                                                        \
    const wave_member_desc_t *wave_member_ = (member_);                       \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_, wave_member_->spaddr.rs1, wave_member_->spaddr.rs2,          \
        k_LOOP_WS_CONFIG_SPADDR);                                             \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_, wave_member_->partition.rs1, wave_member_->partition.rs2,    \
        k_LOOP_WS_CONFIG_PARTITION_BOUNDS);                                   \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_, wave_member_->bounds.rs1, wave_member_->bounds.rs2,          \
        k_LOOP_WS_CONFIG_BOUNDS);                                             \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_, wave_member_->addrs_ab.rs1, wave_member_->addrs_ab.rs2,      \
        k_LOOP_WS_CONFIG_ADDRS_AB);                                           \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_, wave_member_->addrs_dc.rs1, wave_member_->addrs_dc.rs2,      \
        k_LOOP_WS_CONFIG_ADDRS_DC);                                           \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_, wave_member_->strides_ab.rs1,                                \
        wave_member_->strides_ab.rs2, k_LOOP_WS_CONFIG_STRIDES_AB);           \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_, wave_member_->strides_dc.rs1,                                \
        wave_member_->strides_dc.rs2, k_LOOP_WS_CONFIG_STRIDES_DC);           \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_, wave_member_->page_offsets.rs1,                              \
        wave_member_->page_offsets.rs2, k_LOOP_WS_CONFIG_PAGE_OFFSETS);       \
  } while (0)

static inline __attribute__((always_inline)) void wave_issue_group01(
    const wave_group_step_desc_t *desc) {
  WAVE_ISSUE_MEMBER(custom0, &desc->member[0]);
  WAVE_ISSUE_MEMBER(custom1, &desc->member[1]);
  ROCC_INSTRUCTION_RS1_RS2(
      custom1, desc->launch.rs1, desc->launch.rs2, k_LOOP_WS);
}

static inline __attribute__((always_inline)) void wave_issue_group23(
    const wave_group_step_desc_t *desc) {
  WAVE_ISSUE_MEMBER(custom2, &desc->member[0]);
  WAVE_ISSUE_MEMBER(custom3, &desc->member[1]);
  ROCC_INSTRUCTION_RS1_RS2(
      custom3, desc->launch.rs1, desc->launch.rs2, k_LOOP_WS);
}

#undef WAVE_ISSUE_MEMBER

int main(void) {
  if (DIM != 8 || BANK_NUM != 4 || BANK_ROWS != 4096 || ACC_ROWS != 8192 ||
      sizeof(wave_group_step_desc_t) != 272u ||
      gemmini_partition_generated_axis_abi_version() !=
          GEMMINI_SHARED_PARTITION_ENCODING_VERSION ||
      !wave_ranges_are_valid()) {
    printf("WAVEFORM-2PLUS2-FAIL stage=geometry\n");
    return 1;
  }

  printf("WAVEFORM-2PLUS2-BEGIN shape=ffn_gate_up_scaled_1over8 "
         "source_shape=8x11008x4096 M=%u N=%u K=%u "
         "masks=0x3/0xc queues=2/2 axis=K tile=1/16/16 "
         "group_id=acc_quarter_allocation "
         "page_packing=A0B1C0D0 "
         "cache=B_cold_A_recent_rr_C_cold_descriptors_L2_warm "
         "issue_path=precomputed_unchecked_rocc "
         "replay_order=g01_then_g23 hw_config=once_per_physical_gemmini\n",
         WAVE_M, WAVE_N, WAVE_K);

  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
  gemmini_flush(custom3, 0);
  gemmini_fence();

  shared_multi_matmul_job_t jobs[WAVE_JOBS];
  for (int job = 0; job < WAVE_JOBS; ++job) {
    if (!wave_prepare_job(&jobs[job], job)) {
      printf("WAVEFORM-2PLUS2-FAIL stage=job_prepare job=%d\n", job);
      return 1;
    }
  }

  size_t group01_steps = 0;
  size_t group23_steps = 0;
  for (int job = 0; job < WAVE_JOBS; ++job) {
    wave_group_step_desc_t *schedule = job < 2
        ? wave_group01_schedule
        : wave_group23_schedule;
    size_t *group_steps = job < 2 ? &group01_steps : &group23_steps;
    size_t job_steps = 0;
    if (!wave_build_job_schedule(
            &jobs[job], schedule + *group_steps,
            WAVE_ROUNDS - *group_steps, &job_steps)) {
      printf("WAVEFORM-2PLUS2-FAIL stage=descriptor_build job=%d\n", job);
      return 1;
    }
    *group_steps += job_steps;
  }
  if (group01_steps != WAVE_ROUNDS || group23_steps != WAVE_ROUNDS) {
    printf("WAVEFORM-2PLUS2-FAIL stage=schedule_size "
           "group01=%llu group23=%llu expected=%u\n",
           (unsigned long long)group01_steps,
           (unsigned long long)group23_steps, WAVE_ROUNDS);
    return 1;
  }

  if (!wave_configure_hardware(jobs)) {
    printf("WAVEFORM-2PLUS2-FAIL stage=hardware_config\n");
    return 1;
  }
  gemmini_fence();

  printf("WAVEFORM-2PLUS2-PREFLIGHT jobs=%u steps_per_job=%u "
         "group01_steps=%llu group23_steps=%llu commands_per_group_step=%u "
         "timed_rocc_commands=%u descriptor_bytes=%llu "
         "spad_group01=A0_2048_B6144_8192/A8192_10240_B14336_16384 "
         "spad_group23=A2048_4096_B4096_6144/A10240_12288_B12288_14336 "
         "acc_group01=0/4096 acc_group23=2048/6144 "
         "group_ids_group01=0/2 group_ids_group23=1/3\n",
         WAVE_JOBS, WAVE_STEPS_PER_JOB,
         (unsigned long long)group01_steps,
         (unsigned long long)group23_steps,
         WAVE_COMMANDS_PER_GROUP_STEP, WAVE_TIMED_ROCC_COMMANDS,
         (unsigned long long)(sizeof(wave_group01_schedule) +
                              sizeof(wave_group23_schedule)));

  wave_condition_cache();
  wave_cpu_fence();
  const uint64_t start = wave_read_cycles();

  // Deliberately keep the timed producer path to two descriptor replays and a
  // loop branch. LoopMatmul backpressure, not CPU planning, paces refill after
  // the two hardware loop slots fill. There is no intermediate fence.
  for (size_t round = 0; round < WAVE_ROUNDS; ++round) {
    wave_issue_group01(&wave_group01_schedule[round]);
    wave_issue_group23(&wave_group23_schedule[round]);
  }

  const uint64_t issue_end = wave_read_cycles();
  gemmini_fence();
  const uint64_t end = wave_read_cycles();
  wave_cpu_fence();

  printf("WAVEFORM-2PLUS2-RESULT start_cycle=%llu issue_cycles=%llu "
         "fence_cycles=%llu makespan_cycles=%llu rounds=%u "
         "timed_rocc_commands=%u tenant_issue_timestamps=not_sampled "
         "cache_sink=%llu\n",
         (unsigned long long)start,
         (unsigned long long)(issue_end - start),
         (unsigned long long)(end - issue_end),
         (unsigned long long)(end - start),
         WAVE_ROUNDS, WAVE_TIMED_ROCC_COMMANDS,
         (unsigned long long)wave_cache_sink);

  printf("WAVEFORM-2PLUS2-PASS\n");
  return 0;
}
