// Four-Gemmini, full-local-memory waveform proxy for the same four logical
// matmuls used by llama2_ffn_gate_up_m8_2plus2_waveform.c.
//
// Each logical matmul uses physical Gemmini 0/1/2/3 together (mask 0xf) and is
// allocated the complete shared scratchpad and accumulator. Consequently the
// four jobs form one queued stream: job0, job1, job2, job3. There is no fence
// at a job boundary and one final fence drains the complete stream.
//
// N and K are both scaled by 1/8 from the llama2-7B ffn_gate_up shape. B alone
// is page packed. Operand/output data are neither initialized nor checked.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_testutils_all.h"

#define WAVE_M 8u
#define WAVE_N 1376u
#define WAVE_K 512u
#define WAVE_JOBS 4u
#define WAVE_MEMBERS 4u
#define WAVE_MASK 0xfu
#define WAVE_STEPS_PER_JOB 28u
#define WAVE_TOTAL_STEPS (WAVE_JOBS * WAVE_STEPS_PER_JOB)
#define WAVE_CACHE_LINE_BYTES 64u
#define WAVE_EVICTION_BYTES (8u * 1024u * 1024u)
#define WAVE_A_BYTES (WAVE_M * WAVE_K * sizeof(elem_t))
#define WAVE_C_BYTES (WAVE_M * WAVE_N * sizeof(acc_t))
#define WAVE_B_PAGES 384u
#define WAVE_B_BYTES (WAVE_B_PAGES * GEMMINI_PAGE_PACKED_PAGE_BYTES)

#if !GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE
#error "The g4-full waveform replay requires the shared partition descriptor ABI"
#endif

// Keep the ELF/loadmem image small by placing operands at fixed target-DRAM
// addresses inside the unmodified 256-MiB Verilator memory aperture.
#define WAVE_EVICTION_BASE UINT64_C(0x81000000)
#define WAVE_A_BASE        UINT64_C(0x81800000)
#define WAVE_C_BASE        UINT64_C(0x81900000)
#define WAVE_B0_BASE       UINT64_C(0x82000000)
#define WAVE_B1_BASE       UINT64_C(0x82400000)
#define WAVE_B2_BASE       UINT64_C(0x82800000)
#define WAVE_B3_BASE       UINT64_C(0x82c00000)
#define WAVE_B_SLOT_BYTES  UINT64_C(0x00400000)

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
  wave_member_desc_t member[WAVE_MEMBERS];
  wave_rocc_args_t launch;
} wave_quad_desc_t;

typedef struct {
  uint64_t start;
  uint64_t issue_end;
  uint64_t end;
} wave_cycles_t;

static wave_quad_desc_t wave_schedule[WAVE_TOTAL_STEPS]
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
  if (gemmini_page_packed_b_page_count(WAVE_K, WAVE_N) != WAVE_B_PAGES ||
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

static void wave_condition_cache(void) {
  volatile uint8_t *eviction =
      (volatile uint8_t *)(uintptr_t)WAVE_EVICTION_BASE;
  uint64_t sink = wave_cache_sink;
  for (size_t offset = 0; offset < WAVE_EVICTION_BYTES;
       offset += WAVE_CACHE_LINE_BYTES)
    sink += eviction[offset];

  // The replay descriptors are CPU input, so make them resident before the
  // timed command-only loop. This does not touch B or C.
  volatile const uint8_t *schedule_bytes =
      (volatile const uint8_t *)(const void *)wave_schedule;
  for (size_t offset = 0; offset < sizeof(wave_schedule);
       offset += WAVE_CACHE_LINE_BYTES)
    sink += schedule_bytes[offset];

  // Match the Linux microbenchmark's four-A round-robin recent pass. This is
  // cache conditioning, not operand initialization; B/C deliberately stay
  // cold.
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

static bool wave_prepare_job(
    shared_multi_matmul_job_t *job, unsigned job_index) {
  memset(job, 0, sizeof(*job));
  job->gemmini_list = WAVE_MASK;
  // tile_id is software-only in this replay. The launch group ID is derived
  // from the ACC half below, because every outer-K step for one C tile writes
  // the same ACC allocation even while SPAD ping-pong alternates.
  job->tile_id = (int)job_index;
  job->sp_addr_start_stack = 0u;
  job->sp_addr_end_stack = 0u;
  job->acc_addr_start_stack = 0u;
  job->sp_addr_range = BANK_NUM * BANK_ROWS;
  job->acc_addr_range = ACC_ROWS;
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
  job->gemmini_num = WAVE_MEMBERS;

  const gemmini_tiling_request_t request = {
    .dim_I = WAVE_M,
    .dim_J = WAVE_N,
    .dim_K = WAVE_K,
    .dim = DIM,
    .gemmini_count = WAVE_MEMBERS,
    .sp_addr_range = job->sp_addr_range,
    .acc_addr_range = job->acc_addr_range,
    .double_buffered = true,
    .act = NO_ACTIVATION,
  };
  const gemmini_tiling_factors_t tiling =
      gemmini_shared_multi_choose_tiling_axis(
          &request, GEMMINI_PARTITION_AXIS_K);
  if (!shared_multi_matmul_job_set_exact_tiling(
          job, tiling.tile_I, tiling.tile_J, tiling.tile_K) ||
      job->partition_status != SHARED_MULTI_PARTITION_OK ||
      job->tile_I != 1u || job->tile_J != 25u || job->tile_K != 20u ||
      job->sp_addr_A_stacked != 160u ||
      job->sp_addr_B_stacked != 4000u ||
      job->acc_addr_stacked != 200u)
    return false;

  // Software state initialization is intentionally separated from Gemmini
  // configuration side effects. All jobs have identical hardware config.
  job->partition_status =
      shared_multi_partition_axis_status(job, job->partition_axis);
  return job->partition_status == SHARED_MULTI_PARTITION_OK &&
      shared_multi_matmul_job_init_state(job) && !job->done && job->no_bias &&
      job->I0 == 1u && job->J0 == 7u && job->K0 == 4u &&
      job->last_I == 1u && job->last_J == 22u && job->last_K == 4u &&
      job->padding_I == 0u && job->padding_J == 0u &&
      job->padding_K == 0u;
}

static inline void wave_set_args(
    wave_rocc_args_t *args, uint64_t rs1, uint64_t rs2) {
  args->rs1 = rs1;
  args->rs2 = rs2;
}

static bool wave_prepare_one_step(
    shared_multi_matmul_job_t *job, wave_quad_desc_t *descriptor) {
  shared_multi_matmul_job_step_t step;
  if (job == NULL || descriptor == NULL ||
      !shared_multi_matmul_job_plan_step(job, &step) ||
      step.active_gemmini_count != WAVE_MEMBERS ||
      step.group_list != WAVE_MASK)
    return false;

  const size_t plain_stride_A =
      gemmini_page_packed_stride_payload(job->stride_A);
  const size_t plain_stride_C =
      gemmini_page_packed_stride_payload(job->stride_C);
  const size_t t = (size_t)(job->inner_call_counter & 1);
  const size_t local_sp_addr_start =
      t == 0u ? 0u : BANK_NUM * BANK_ROWS / 2u;
  const size_t local_sp_addr_end =
      t == 0u ? BANK_NUM * BANK_ROWS / 2u : BANK_NUM * BANK_ROWS;
  const size_t local_acc_addr_start =
      job->lastK_toggle ? 0u : ACC_ROWS / 2u;
  const size_t local_group_id =
      local_acc_addr_start / (ACC_ROWS / 2u);

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

  if (!gemmini_shared_partition_page_offsets_valid(
          job->partition_axis,
          step.K, 0u, step.I, 0u,
          step.I, step.J, step.K,
          false, true, false, false,
          A_row_offset, A_col_offset,
          B_row_offset, B_col_offset,
          D_row_offset, D_col_offset,
          C_row_offset, C_col_offset))
    return false;

  for (size_t rank = 0; rank < WAVE_MEMBERS; ++rank) {
    gemmini_partition_plan_t plan;
    if (!gemmini_partition_plan_member(
            job->partition_axis,
            step.I, step.J, step.K,
            step.pad_I, step.pad_J, step.pad_K,
            WAVE_MEMBERS, rank, &plan) ||
        !gemmini_shared_partition_page_offsets_valid(
            job->partition_axis,
            plan.partition_extent, plan.partition_offset,
            plan.aux_extent, plan.aux_offset,
            step.I, step.J, step.K,
            false, true, false, false,
            A_row_offset, A_col_offset,
            B_row_offset, B_col_offset,
            D_row_offset, D_col_offset,
            C_row_offset, C_col_offset))
      return false;

    const size_t expected_partition_extent = step.K / WAVE_MEMBERS;
    if (step.K % WAVE_MEMBERS != 0u ||
        plan.partition_extent != expected_partition_extent ||
        plan.partition_offset != rank * expected_partition_extent ||
        plan.aux_extent != (rank == 0u ? 1u : 0u) ||
        plan.aux_offset != (rank == 0u ? 0u : 1u) ||
        plan.partition_pad != 0u ||
        plan.aux_pad != 0u)
      return false;

    // For K partitioning, A advances by the member's K offset. Page-packed B
    // keeps the full-buffer base and carries both outer/member coordinates via
    // funct=28 and funct=24, respectively. C's auxiliary-I offset matches the
    // established shared_multi_tiled_matmul_job_step() path.
    const uintptr_t a_local = a_outer +
        plan.partition_offset * DIM * sizeof(elem_t);
    const uintptr_t b_local = b_outer;
    const uintptr_t c_local = c_outer == 0u ? 0u : c_outer +
        plan.aux_offset * plain_stride_C * DIM * job->sizeof_C;
    const size_t member_pad_K = plan.partition_pad;
    wave_member_desc_t *member = &descriptor->member[rank];

    wave_set_args(&member->command[WAVE_CMD_SPADDR],
        local_acc_addr_start,
        ((uint64_t)local_sp_addr_end << 16) | local_sp_addr_start);
    wave_set_args(&member->command[WAVE_CMD_PARTITION],
        gemmini_shared_partition_pack_rs1(
            job->partition_axis, plan.partition_offset, plan.aux_offset),
        gemmini_shared_partition_pack_rs2(
            plan.partition_extent, plan.aux_extent, plan.aux_pad));
    wave_set_args(&member->command[WAVE_CMD_BOUNDS],
        ((uint64_t)member_pad_K << 32) |
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

  const bool ex_accumulate =
      gemmini_partition_ex_accumulate(job->no_bias, step.k0);
  wave_set_args(&descriptor->launch,
      ((uint64_t)step.group_list << 48) |
          ((uint64_t)local_group_id << 32) |
          ((uint64_t)job->act << 8) |
          ((uint64_t)job->low_D << 2) |
          ((uint64_t)job->full_C << 1) |
          (uint64_t)ex_accumulate,
      ((uint64_t)job->b_transpose << 1) |
          (uint64_t)job->a_transpose);

  shared_multi_matmul_job_complete_step(job);
  return job->partition_status == SHARED_MULTI_PARTITION_OK;
}

static bool wave_prepare_schedule(void) {
  shared_multi_matmul_job_t jobs[WAVE_JOBS];
  size_t schedule_index = 0u;
  for (unsigned job = 0; job < WAVE_JOBS; ++job) {
    if (!wave_prepare_job(&jobs[job], job))
      return false;
  }

  // Flatten the one-tenant queue outside timing. Job boundaries require no
  // CPU action in the replay loop and no intermediate Gemmini fence.
  for (unsigned job = 0; job < WAVE_JOBS; ++job) {
    const size_t first = schedule_index;
    while (!jobs[job].done) {
      if (schedule_index >= WAVE_TOTAL_STEPS ||
          !wave_prepare_one_step(&jobs[job],
                                 &wave_schedule[schedule_index]))
        return false;
      ++schedule_index;
    }
    if (schedule_index - first != WAVE_STEPS_PER_JOB)
      return false;
  }
  return schedule_index == WAVE_TOTAL_STEPS;
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

static inline __attribute__((always_inline)) void wave_issue_quad(
    const wave_quad_desc_t *descriptor) {
  WAVE_ISSUE_MEMBER(custom0, &descriptor->member[0]);
  WAVE_ISSUE_MEMBER(custom1, &descriptor->member[1]);
  WAVE_ISSUE_MEMBER(custom2, &descriptor->member[2]);
  WAVE_ISSUE_MEMBER(custom3, &descriptor->member[3]);
  ROCC_INSTRUCTION_RS1_RS2(
      custom3, descriptor->launch.rs1, descriptor->launch.rs2, k_LOOP_WS);
}

static bool wave_configure_hw_once(void) {
  const size_t stride_A =
      gemmini_page_packed_a_dma_stride_bytes(WAVE_K);
  const size_t stride_B = gemmini_page_packed_b_dma_stride_bytes(
      GEMMINI_PAGE_PACKED_STRIDE(WAVE_N));
  const size_t stride_D =
      gemmini_page_packed_acc_dma_stride_bytes(0u, sizeof(acc_t));
  const size_t stride_C = gemmini_page_packed_acc_dma_stride_bytes(
      WAVE_N, sizeof(acc_t));
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

  WAVE_CONFIGURE_ONE(custom0);
  WAVE_CONFIGURE_ONE(custom1);
  WAVE_CONFIGURE_ONE(custom2);
  WAVE_CONFIGURE_ONE(custom3);
#undef WAVE_CONFIGURE_ONE
  gemmini_fence();
  return true;
}

// Keep a named, non-inlined replay boundary so objdump can verify that the
// measured loop contains only descriptor loads, 33 RoCC issues, loop control,
// timestamps, and the one final fence.
static __attribute__((noinline)) wave_cycles_t wave_replay_schedule(void) {
  wave_cycles_t cycles;
  wave_cpu_fence();
  cycles.start = wave_read_cycles();
  for (size_t step = 0; step < WAVE_TOTAL_STEPS; ++step)
    wave_issue_quad(&wave_schedule[step]);
  cycles.issue_end = wave_read_cycles();
  gemmini_fence();
  cycles.end = wave_read_cycles();
  wave_cpu_fence();
  return cycles;
}

int main(void) {
  if (DIM != 8 || BANK_NUM != 4 || BANK_ROWS != 4096 || ACC_ROWS != 8192 ||
      gemmini_partition_generated_axis_abi_version() !=
          GEMMINI_SHARED_PARTITION_ENCODING_VERSION ||
      sizeof(wave_quad_desc_t) != 528u || !wave_ranges_are_valid()) {
    printf("WAVEFORM-G4-FULL-FAIL stage=geometry\n");
    return 1;
  }

  printf("WAVEFORM-G4-FULL-BEGIN shape=ffn_gate_up_scaled_1over8 "
         "source_shape=8x11008x4096 M=%u N=%u K=%u "
         "mask=0xf gemminis=4 jobs=4 schedule=queued4 "
         "spad=full acc=full axis=K tile=1/25/20 steps_per_job=28 "
         "group_id=acc_half "
         "local_sp=phase0_A[0,4096)_B[4096,8192)_"
         "phase1_A[8192,12288)_B[12288,16384) "
         "local_acc=phase0[0,4096)_phase1[4096,8192) "
         "page_packing=A0B1C0D0 cache=B_cold_A_recent_rr_C_cold\n",
         WAVE_M, WAVE_N, WAVE_K);

  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
  gemmini_flush(custom3, 0);
  gemmini_fence();

  if (!wave_prepare_schedule()) {
    printf("WAVEFORM-G4-FULL-FAIL stage=schedule_prepare\n");
    return 1;
  }
  if (!wave_configure_hw_once()) {
    printf("WAVEFORM-G4-FULL-FAIL stage=hardware_config\n");
    return 1;
  }
  wave_condition_cache();

  const wave_cycles_t cycles = wave_replay_schedule();
  printf("WAVEFORM-G4-FULL-RESULT start_cycle=%llu issue_cycles=%llu "
         "fence_cycles=%llu makespan_cycles=%llu steps=%u "
         "commands_per_step=33 cache_sink=%llu\n",
         (unsigned long long)cycles.start,
         (unsigned long long)(cycles.issue_end - cycles.start),
         (unsigned long long)(cycles.end - cycles.issue_end),
         (unsigned long long)(cycles.end - cycles.start),
         WAVE_TOTAL_STEPS,
         (unsigned long long)wave_cache_sink);
  printf("WAVEFORM-G4-FULL-PASS\n");
  return 0;
}
