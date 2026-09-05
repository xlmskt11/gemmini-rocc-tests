// See LICENSE for license details.

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct gemmini_command_trace {
  unsigned custom;
  unsigned funct;
  uint64_t rs1;
  uint64_t rs2;
};

static struct gemmini_command_trace gemmini_trace[128];
static unsigned gemmini_trace_count;

enum attention_event_kind {
  ATTENTION_EVENT_QK,
  ATTENTION_EVENT_PV,
  ATTENTION_EVENT_VPU_LAST,
};

enum {
  TEST_LOOP_PRODUCE_EVENT_ID_SHIFT = 3,
  TEST_LOOP_PRODUCE_EVENT_VALID_SHIFT = 6,
  TEST_LOOP_PRODUCE_EVENT_SEAL_SHIFT = 7,
  TEST_LOOP_WAIT_EVENT_ID_SHIFT = 35,
  TEST_LOOP_WAIT_EVENT_VALID_SHIFT = 38,
};

struct attention_event {
  enum attention_event_kind kind;
  unsigned sync_group_id;
  unsigned wait_event_id;
  unsigned produce_event_id;
  bool wait_event_valid;
  bool produce_event_valid;
  bool produce_event_seal;
};

static struct attention_event attention_events[64];
static unsigned attention_event_count;
static bool gemmini_command_is_qk[4];
static bool vpu_stage_active;
static uint64_t vpu_stage_metadata;

struct vpu_command_trace {
  uint32_t uop;
  uint64_t payload;
};

static struct vpu_command_trace vpu_trace[64];
static unsigned vpu_trace_count;
static bool capture_vpu_trace;

static void record_attention_event(enum attention_event_kind kind,
                                   unsigned sync_group_id,
                                   unsigned wait_event_id,
                                   bool wait_event_valid,
                                   unsigned produce_event_id,
                                   bool produce_event_valid,
                                   bool produce_event_seal) {
  assert(attention_event_count <
         (unsigned)(sizeof(attention_events) / sizeof(attention_events[0])));
  attention_events[attention_event_count++] =
      (struct attention_event){kind, sync_group_id, wait_event_id,
                               produce_event_id, wait_event_valid,
                               produce_event_valid, produce_event_seal};
}

static void record_gemmini_command(unsigned custom, uint64_t rs1,
                                    uint64_t rs2, unsigned funct) {
  if (gemmini_trace_count <
      (unsigned)(sizeof(gemmini_trace) / sizeof(gemmini_trace[0]))) {
    gemmini_trace[gemmini_trace_count].custom = custom;
    gemmini_trace[gemmini_trace_count].funct = funct;
    gemmini_trace[gemmini_trace_count].rs1 = rs1;
    gemmini_trace[gemmini_trace_count].rs2 = rs2;
    ++gemmini_trace_count;
  }
  /* Numeric literals are the LOOP_WS ABI values defined by the kernel below;
   * this recorder itself must be visible before that header is included. */
  if (funct == 10u)
    gemmini_command_is_qk[custom] = rs1 != 0u;
  if (funct == 8u && custom == 0u) {
    record_attention_event(gemmini_command_is_qk[custom]
                               ? ATTENTION_EVENT_QK
                               : ATTENTION_EVENT_PV,
                           (unsigned)((rs1 >> 32) & UINT64_C(0x7)),
                           (unsigned)((rs1 >>
                                       TEST_LOOP_WAIT_EVENT_ID_SHIFT) &
                                      UINT64_C(0x7)),
                           ((rs1 >> TEST_LOOP_WAIT_EVENT_VALID_SHIFT) & 1u) !=
                               0u,
                           (unsigned)((rs1 >>
                                       TEST_LOOP_PRODUCE_EVENT_ID_SHIFT) &
                                      UINT64_C(0x7)),
                           ((rs1 >> TEST_LOOP_PRODUCE_EVENT_VALID_SHIFT) & 1u) !=
                               0u,
                           ((rs1 >> TEST_LOOP_PRODUCE_EVENT_SEAL_SHIFT) & 1u) !=
                               0u);
  }
}

#define VPU_FA_GEMMINI_ISSUE(custom_, rs1_, rs2_, funct_)                    \
  record_gemmini_command((unsigned)(custom_), (uint64_t)(rs1_),              \
                          (uint64_t)(rs2_), (unsigned)(funct_))
#if defined(VPU_FA_TEST_MATRIX_PORTS) || defined(VPU_FA_TEST_PARTITION_AXIS_ABI) || \
    defined(VPU_FA_TEST_DIM) || defined(VPU_FA_TEST_ACC_ROWS) ||                \
    defined(VPU_FA_TEST_BANK_ROWS)
#include "include/gemmini_params.h"
#include "include/vpu_params.h"
#endif
#ifdef VPU_FA_TEST_MATRIX_PORTS
#undef VPU_MATRIX_PORTS
#define VPU_MATRIX_PORTS VPU_FA_TEST_MATRIX_PORTS
#endif
#ifdef VPU_FA_TEST_PARTITION_AXIS_ABI
#undef GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION
#define GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION                              \
  VPU_FA_TEST_PARTITION_AXIS_ABI
#endif
#ifdef VPU_FA_TEST_DIM
#undef DIM
#define DIM VPU_FA_TEST_DIM
#undef VPU_MATRIX_ROW_ELEMENTS
#define VPU_MATRIX_ROW_ELEMENTS VPU_FA_TEST_DIM
#endif
#ifdef VPU_FA_TEST_ACC_ROWS
#undef ACC_ROWS
#define ACC_ROWS VPU_FA_TEST_ACC_ROWS
#endif
#ifdef VPU_FA_TEST_BANK_ROWS
#undef BANK_ROWS
#define BANK_ROWS VPU_FA_TEST_BANK_ROWS
#endif
#define VPU_ENABLE_GEMMINI_FLASHATTENTION 1
#include "include/vpu_flashattention_kernel.h"

void vpu_host_issue(uint64_t transport, uint64_t payload) {
  if (capture_vpu_trace) {
    assert(vpu_trace_count <
           (unsigned)(sizeof(vpu_trace) / sizeof(vpu_trace[0])));
    vpu_trace[vpu_trace_count++] =
        (struct vpu_command_trace){(uint32_t)transport, payload};
  }
  const uint64_t event_metadata_mask =
      (UINT64_C(0x7) << VPU_EVENT_WAIT_ID_SHIFT) |
      (UINT64_C(1) << VPU_EVENT_WAIT_VALID_SHIFT) |
      (UINT64_C(0x7) << VPU_EVENT_PRODUCE_ID_SHIFT) |
      (UINT64_C(1) << VPU_EVENT_PRODUCE_VALID_SHIFT) |
      (UINT64_C(1) << VPU_EVENT_PRODUCE_SEAL_SHIFT);
  const uint64_t metadata = transport & event_metadata_mask;
  const bool event_command =
      ((transport >> VPU_EVENT_WAIT_VALID_SHIFT) & 1u) != 0u ||
      ((transport >> VPU_EVENT_PRODUCE_VALID_SHIFT) & 1u) != 0u;
  if (event_command) {
    if (!vpu_stage_active) {
      vpu_stage_active = true;
      vpu_stage_metadata = metadata;
    } else {
      assert(metadata == vpu_stage_metadata);
    }
  }
  if (((transport >> VPU_EVENT_STAGE_LAST_SHIFT) & 1u) != 0u) {
    assert(vpu_stage_active);
    record_attention_event(
        ATTENTION_EVENT_VPU_LAST, 0u,
        (unsigned)((transport >> VPU_EVENT_WAIT_ID_SHIFT) &
                   ((UINT64_C(1) << VPU_EVENT_ID_BITS) - 1u)),
        ((transport >> VPU_EVENT_WAIT_VALID_SHIFT) & 1u) != 0u,
        (unsigned)((transport >> VPU_EVENT_PRODUCE_ID_SHIFT) &
                   ((UINT64_C(1) << VPU_EVENT_ID_BITS) - 1u)),
        ((transport >> VPU_EVENT_PRODUCE_VALID_SHIFT) & 1u) != 0u,
        ((transport >> VPU_EVENT_PRODUCE_SEAL_SHIFT) & 1u) != 0u);
    vpu_stage_active = false;
  }
}

uint64_t vpu_host_issue_result(uint64_t transport, uint64_t payload) {
  (void)transport;
  (void)payload;
  return 0u;
}

static void reset_gemmini_trace(void) {
  gemmini_trace_count = 0u;
}

static void reset_attention_events(void) {
  attention_event_count = 0u;
  memset(gemmini_command_is_qk, 0, sizeof(gemmini_command_is_qk));
  vpu_stage_active = false;
  vpu_stage_metadata = 0u;
}

static void reset_vpu_trace(void) {
  vpu_trace_count = 0u;
  capture_vpu_trace = true;
}

static void assert_matmul_event(unsigned index, enum attention_event_kind kind,
                                unsigned sync_group_id,
                                unsigned wait_event_id, bool wait_valid,
                                unsigned produce_event_id, bool seal) {
  assert(attention_events[index].kind == kind);
  assert(attention_events[index].sync_group_id == sync_group_id);
  assert(attention_events[index].wait_event_id == wait_event_id);
  assert(attention_events[index].wait_event_valid == wait_valid);
  assert(attention_events[index].produce_event_id == produce_event_id);
  assert(attention_events[index].produce_event_valid);
  assert(attention_events[index].produce_event_seal == seal);
}

static void assert_vpu_stage_event(unsigned index, unsigned wait_event_id,
                                   unsigned produce_event_id,
                                   bool produce_valid) {
  assert(attention_events[index].kind == ATTENTION_EVENT_VPU_LAST);
  assert(attention_events[index].wait_event_valid);
  assert(attention_events[index].wait_event_id == wait_event_id);
  assert(attention_events[index].produce_event_valid == produce_valid);
  assert(attention_events[index].produce_event_id == produce_event_id);
  assert(attention_events[index].produce_event_seal == produce_valid);
}

static unsigned count_gemmini_commands(unsigned funct) {
  unsigned count = 0u;
  for (unsigned i = 0u; i < gemmini_trace_count; ++i)
    count += gemmini_trace[i].funct == funct;
  return count;
}

static const struct gemmini_command_trace *find_gemmini_command(
    unsigned funct) {
  const struct gemmini_command_trace *found = NULL;
  for (unsigned i = 0u; i < gemmini_trace_count; ++i) {
    if (gemmini_trace[i].funct != funct)
      continue;
    assert(found == NULL);
    found = &gemmini_trace[i];
  }
  assert(found != NULL);
  return found;
}

#if GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE
static const struct gemmini_command_trace *find_gemmini_command_for_custom(
    unsigned funct, unsigned custom) {
  const struct gemmini_command_trace *found = NULL;
  for (unsigned i = 0u; i < gemmini_trace_count; ++i) {
    if (gemmini_trace[i].funct != funct ||
        gemmini_trace[i].custom != custom)
      continue;
    assert(found == NULL);
    found = &gemmini_trace[i];
  }
  assert(found != NULL);
  return found;
}
#endif

static vpu_flashattention_config_t make_config(
    unsigned query_rows, unsigned gemmini_mask, bool page_packed) {
  const size_t width = 64u;
  const bool pack_query = page_packed && query_rows > 1u;
  return (vpu_flashattention_config_t){
      .queries = (const elem_t *)(uintptr_t)0x10000000u,
      .keys = (const elem_t *)(uintptr_t)0x20000000u,
      .values = (const elem_t *)(uintptr_t)0x30000000u,
      .output = (float *)(uintptr_t)0x40000000u,
      .query_rows = query_rows,
      .sequence = 256u,
      .q_dim = width,
      .k_dim = width,
      .value_dim = width,
      /* Deliberately differs from 1/sqrt(64) to exercise explicit scaling. */
      .score_scale = 0.15625f,
      .query_base = 0u,
      .query_stride = pack_query
          ? GEMMINI_PAGE_PACKED_STRIDE(width) : width,
      .key_stride = page_packed
          ? GEMMINI_PAGE_PACKED_STRIDE(width) : width,
      .value_stride = page_packed
          ? GEMMINI_PAGE_PACKED_STRIDE(width) : width,
      /* VPU H_STORE always emits the final FP32 result row-major. */
      .output_stride = width,
      .gemmini_mask = gemmini_mask,
      .qk_partition_axis = GEMMINI_PARTITION_AXIS_M,
      .pv_partition_axis = GEMMINI_PARTITION_AXIS_M,
      .causal_mask_workspace = (float *)(uintptr_t)0x50000000u,
      .causal_mask_workspace_elements =
          VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS,
  };
}

static void check_shared_accumulator_loop_ws_abi(void) {
  const vpu_flashattention_config_t config = make_config(16u, 0x1u, false);
  const unsigned value_tiles =
      vpu_fa_ceil_div((unsigned)config.value_dim, DIM);
  vpu_fa_matmul_job_t job;

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK,
      0u, 16u, 0u, 128u));
  assert(vpu_fa_matmul_job_next_k_tile(&job) == 0u);
  vpu_fa_matmul_job_set_events(&job, 5u, true, 3u, true);
  vpu_fa_matmul_job_init(&job);
  reset_gemmini_trace();
  assert(vpu_fa_matmul_job_step(&job, 0u, 0u));
  assert(job.core.done);
  assert(count_gemmini_commands(
             VPU_FA_K_LOOP_WS_CONFIG_PARTITION_BOUNDS) ==
         (GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE ? 1u : 0u));
  const struct gemmini_command_trace *spaddr =
      find_gemmini_command(VPU_FA_K_LOOP_WS_CONFIG_SPADDR);
  const struct gemmini_command_trace *addrs_ab =
      find_gemmini_command(VPU_FA_K_LOOP_WS_CONFIG_ADDRS_AB);
  const struct gemmini_command_trace *addrs_dc =
      find_gemmini_command(VPU_FA_K_LOOP_WS_CONFIG_ADDRS_DC);
  const struct gemmini_command_trace *run =
      find_gemmini_command(VPU_FA_K_LOOP_WS);
  assert(spaddr->custom == 0u);
  assert(spaddr->rs1 == VPU_FA_SCORE_BASE_ROW);
  assert(spaddr->rs1 < ACC_ROWS);
  assert((spaddr->rs2 & UINT64_C(0xffff)) == vpu_fa_spad_start(0u));
  assert(addrs_ab->rs1 != 0u);
  assert(addrs_dc->rs1 == 0u && addrs_dc->rs2 == 0u);
  assert((run->rs1 & UINT64_C(1)) == 0u);
  assert(((run->rs1 >> VPU_FA_LOOP_WAIT_EVENT_ID_SHIFT) & UINT64_C(0x7)) ==
         5u);
  assert((run->rs1 & VPU_FA_LOOP_WAIT_EVENT_VALID) != 0u);
  assert(((run->rs1 >> VPU_FA_LOOP_PRODUCE_EVENT_ID_SHIFT) &
          UINT64_C(0x7)) == 3u);
  assert((run->rs1 & VPU_FA_LOOP_PRODUCE_EVENT_VALID) != 0u);
  assert((run->rs1 & VPU_FA_LOOP_PRODUCE_EVENT_SEAL) != 0u);
  assert(((run->rs1 >> 48) & UINT64_C(0xf)) == UINT64_C(1));

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_PV,
      0u, 16u, 0u, 128u));
  vpu_fa_matmul_job_set_events(&job, 2u, true, 7u, true);
  vpu_fa_matmul_job_init(&job);
  reset_gemmini_trace();
  assert(vpu_fa_matmul_job_step(&job, 0u, 0u));
  assert(job.core.done);
  assert(count_gemmini_commands(
             VPU_FA_K_LOOP_WS_CONFIG_PARTITION_BOUNDS) ==
         (GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE ? 1u : 0u));
  spaddr = find_gemmini_command(VPU_FA_K_LOOP_WS_CONFIG_SPADDR);
  addrs_ab = find_gemmini_command(VPU_FA_K_LOOP_WS_CONFIG_ADDRS_AB);
  addrs_dc = find_gemmini_command(VPU_FA_K_LOOP_WS_CONFIG_ADDRS_DC);
  run = find_gemmini_command(VPU_FA_K_LOOP_WS);
  assert(spaddr->rs1 == vpu_fa_output_base_row(16u, value_tiles));
  assert(spaddr->rs1 < ACC_ROWS);
  assert((spaddr->rs2 & UINT64_C(0xffff)) == VPU_FA_SCORE_BASE_ROW);
  assert(addrs_ab->rs1 == 0u);
  assert(addrs_dc->rs1 == 0u && addrs_dc->rs2 == 0u);
  assert((run->rs1 & UINT64_C(1)) == 0u);
  assert(((run->rs1 >> VPU_FA_LOOP_WAIT_EVENT_ID_SHIFT) & UINT64_C(0x7)) ==
         2u);
  assert((run->rs1 & VPU_FA_LOOP_WAIT_EVENT_VALID) != 0u);
  assert(((run->rs1 >> VPU_FA_LOOP_PRODUCE_EVENT_ID_SHIFT) &
          UINT64_C(0x7)) == 7u);
  assert((run->rs1 & VPU_FA_LOOP_PRODUCE_EVENT_VALID) != 0u);
  assert((run->rs1 & VPU_FA_LOOP_PRODUCE_EVENT_SEAL) != 0u);

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_PV,
      0u, 16u, 128u, 128u));
  vpu_fa_matmul_job_init(&job);
  reset_gemmini_trace();
  assert(vpu_fa_matmul_job_step(&job, 0u, 0u));
  assert(job.core.done);
  run = find_gemmini_command(VPU_FA_K_LOOP_WS);
  addrs_ab = find_gemmini_command(VPU_FA_K_LOOP_WS_CONFIG_ADDRS_AB);
  addrs_dc = find_gemmini_command(VPU_FA_K_LOOP_WS_CONFIG_ADDRS_DC);
  assert(addrs_ab->rs1 == 0u);
  assert(addrs_dc->rs1 == 0u && addrs_dc->rs2 == 0u);
  assert((run->rs1 & UINT64_C(1)) != 0u);
  assert((run->rs1 & VPU_FA_LOOP_WAIT_EVENT_VALID) == 0u);
  assert((run->rs1 & VPU_FA_LOOP_PRODUCE_EVENT_VALID) == 0u);

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK,
      0u, 16u, 0u, 128u));
  vpu_fa_matmul_job_init(&job);
  reset_gemmini_trace();
  assert(vpu_fa_matmul_job_step(&job, 1u, 1u));
  spaddr = find_gemmini_command(VPU_FA_K_LOOP_WS_CONFIG_SPADDR);
  run = find_gemmini_command(VPU_FA_K_LOOP_WS);
  assert(spaddr->rs1 == VPU_FA_ACC_HALF_ROWS);
  assert((spaddr->rs2 & UINT64_C(0xffff)) == vpu_fa_spad_start(1u));
  assert(((run->rs1 >> 32) & UINT64_C(0x7)) == 1u);

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_PV,
      0u, 16u, 128u, 128u));
  vpu_fa_matmul_job_init(&job);
  reset_gemmini_trace();
  assert(vpu_fa_matmul_job_step(&job, 1u, 1u));
  spaddr = find_gemmini_command(VPU_FA_K_LOOP_WS_CONFIG_SPADDR);
  run = find_gemmini_command(VPU_FA_K_LOOP_WS);
  assert(spaddr->rs1 == vpu_fa_output_base_row(16u, value_tiles));
  assert((spaddr->rs2 & UINT64_C(0xffff)) == VPU_FA_ACC_HALF_ROWS);
  assert(((run->rs1 >> 32) & UINT64_C(0x7)) == 1u);
}

static void check_matmul_job_partition_axes(void) {
#if !GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE
  vpu_flashattention_config_t config = make_config(64u, 0x1u, false);
  vpu_fa_matmul_job_t job;

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK, 0u, 64u, 0u, 64u));
  assert(job.core.partition_axis == GEMMINI_PARTITION_AXIS_M);
  vpu_fa_matmul_job_init(&job);
  reset_gemmini_trace();
  assert(vpu_fa_matmul_job_step(&job, 0u, 0u));
  assert(count_gemmini_commands(
             VPU_FA_K_LOOP_WS_CONFIG_PARTITION_BOUNDS) == 0u);
  assert(count_gemmini_commands(VPU_FA_K_LOOP_WS_CONFIG_SPADDR) == 1u);
  assert(count_gemmini_commands(VPU_FA_K_LOOP_WS) == 1u);

  config.qk_partition_axis = GEMMINI_PARTITION_AXIS_N;
  assert(!vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK, 0u, 64u, 0u, 64u));
  config.qk_partition_axis = GEMMINI_PARTITION_AXIS_K;
  assert(!vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK, 0u, 64u, 0u, 64u));
#else
  for (unsigned packed = 0u; packed <= 1u; ++packed) {
    vpu_flashattention_config_t config =
        make_config(64u, 0xfu, packed != 0u);
    for (unsigned kind_value = VPU_FA_MATMUL_QK;
         kind_value <= VPU_FA_MATMUL_PV; ++kind_value) {
      const vpu_fa_matmul_kind_t kind =
          (vpu_fa_matmul_kind_t)kind_value;
      for (unsigned axis_value = GEMMINI_PARTITION_AXIS_M;
           axis_value <= GEMMINI_PARTITION_AXIS_K; ++axis_value) {
        const gemmini_partition_axis_t axis =
            (gemmini_partition_axis_t)axis_value;
        if (kind == VPU_FA_MATMUL_QK)
          config.qk_partition_axis = axis;
        else
          config.pv_partition_axis = axis;
        vpu_fa_matmul_job_t job;
        assert(vpu_fa_matmul_job_prepare(
            &job, &config, kind, 0u, 64u, 0u, 64u));
        assert(job.core.partition_axis == axis);
        vpu_fa_matmul_job_init(&job);
        assert(job.initialized && !job.core.done);

        reset_gemmini_trace();
        assert(vpu_fa_matmul_job_step(&job, 0u, 0u));
        assert(job.core.done);

        const unsigned active_count =
            (unsigned)gemmini_partition_effective_member_count(
                axis, job.core.gemmini_num,
                job.core.tile_I, job.core.tile_J, vpu_fa_matmul_job_total_k_tiles(&job));
        const unsigned group_list =
            (unsigned)gemmini_partition_take_members(
                job.core.gemmini_list, active_count);
        unsigned rank = 0u;
        for (unsigned custom = 0u; custom < 4u; ++custom) {
          if (((group_list >> custom) & 1u) == 0u)
            continue;

          gemmini_partition_plan_t plan;
          assert(gemmini_partition_plan_member(
              axis, job.core.tile_I, job.core.tile_J, vpu_fa_matmul_job_total_k_tiles(&job),
              0u, 0u, 0u, active_count, rank, &plan));
          const struct gemmini_command_trace *partition =
              find_gemmini_command_for_custom(
                  VPU_FA_K_LOOP_WS_CONFIG_PARTITION_BOUNDS, custom);
          assert(((partition->rs1 >> 32) & UINT64_C(0x3)) == axis_value);
          assert((partition->rs1 & UINT64_C(0xffff)) ==
                 plan.partition_offset);
          assert(((partition->rs1 >> 16) & UINT64_C(0xffff)) ==
                 plan.aux_offset);
          assert((partition->rs2 & UINT64_C(0xffff)) ==
                 plan.partition_extent);
          assert(((partition->rs2 >> 16) & UINT64_C(0xffff)) ==
                 plan.aux_pad);
          assert(((partition->rs2 >> 32) & UINT64_C(0xffff)) ==
                 plan.aux_extent);

          size_t a_i_offset = 0u;
          size_t a_k_offset = 0u;
          size_t b_k_offset = 0u;
          size_t b_j_offset = 0u;
          if (axis == GEMMINI_PARTITION_AXIS_M) {
            a_i_offset = plan.partition_offset;
            b_k_offset = plan.aux_offset;
          } else if (axis == GEMMINI_PARTITION_AXIS_N) {
            a_i_offset = plan.aux_offset;
            b_j_offset = plan.partition_offset;
          } else {
            a_k_offset = plan.partition_offset;
            b_k_offset = plan.partition_offset;
          }

          const struct gemmini_command_trace *addrs_ab =
              find_gemmini_command_for_custom(
                  VPU_FA_K_LOOP_WS_CONFIG_ADDRS_AB, custom);
          uintptr_t expected_a = 0u;
          uintptr_t expected_b;
          if (kind == VPU_FA_MATMUL_QK) {
            expected_a = (uintptr_t)config.queries;
            expected_b = (uintptr_t)config.keys;
            if (packed == 0u) {
              expected_a +=
                  (a_i_offset * DIM * config.query_stride +
                   a_k_offset * DIM) * sizeof(elem_t);
              expected_b +=
                  (b_j_offset * DIM * config.key_stride +
                   b_k_offset * DIM) * sizeof(elem_t);
            }
          } else {
            expected_b = (uintptr_t)config.values;
            if (packed == 0u) {
              expected_b +=
                  (b_k_offset * DIM * config.value_stride +
                   b_j_offset * DIM) * sizeof(elem_t);
            }
          }
          assert(addrs_ab->rs1 == expected_a);
          assert(addrs_ab->rs2 == expected_b);
          ++rank;
        }
        assert(rank == active_count);
      }
    }
  }

  vpu_flashattention_config_t config = make_config(64u, 0xfu, false);
  vpu_fa_matmul_job_t job;
  config.qk_partition_axis = (gemmini_partition_axis_t)3;
  assert(!vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK, 0u, 64u, 0u, 64u));
#endif
}

static void check_independent_qk_pv_axes(void) {
#if !GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE
  static float causal_mask[VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS];
  for (unsigned qk_value = GEMMINI_PARTITION_AXIS_M;
       qk_value <= GEMMINI_PARTITION_AXIS_K; ++qk_value) {
    for (unsigned pv_value = GEMMINI_PARTITION_AXIS_M;
         pv_value <= GEMMINI_PARTITION_AXIS_K; ++pv_value) {
      vpu_flashattention_config_t config = make_config(64u, 0x1u, false);
      config.sequence = 64u;
      config.causal_mask_workspace = causal_mask;
      config.qk_partition_axis = (gemmini_partition_axis_t)qk_value;
      config.pv_partition_axis = (gemmini_partition_axis_t)pv_value;
      vpu_flashattention_plan_t plan;
      const vpu_flashattention_status_t status =
          vpu_flashattention_make_plan(&config, &plan);
      const bool legacy_m_pair =
          qk_value == GEMMINI_PARTITION_AXIS_M &&
          pv_value == GEMMINI_PARTITION_AXIS_M;
      assert((status == VPU_FLASHATTENTION_OK) == legacy_m_pair);
      if (legacy_m_pair)
        assert(vpu_fa_preflight_jobs(&config, &plan) ==
               VPU_FLASHATTENTION_OK);
    }
  }
#else
  static float causal_mask[VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS];
  const unsigned members = VPU_MATRIX_PORTS < 4u ? VPU_MATRIX_PORTS : 4u;
  const unsigned valid_mask = (1u << members) - 1u;
  for (unsigned qk_value = GEMMINI_PARTITION_AXIS_M;
       qk_value <= GEMMINI_PARTITION_AXIS_K; ++qk_value) {
    for (unsigned pv_value = GEMMINI_PARTITION_AXIS_M;
         pv_value <= GEMMINI_PARTITION_AXIS_K; ++pv_value) {
      vpu_flashattention_config_t config =
          make_config(64u, valid_mask, false);
      config.sequence = 64u;
      config.qk_partition_axis = (gemmini_partition_axis_t)qk_value;
      config.pv_partition_axis = (gemmini_partition_axis_t)pv_value;
      vpu_flashattention_plan_t plan;
      const vpu_flashattention_status_t status =
          vpu_flashattention_make_plan(&config, &plan);
      if (status != VPU_FLASHATTENTION_OK)
        fprintf(stderr, "QK/PV axis plan failed: qk=%u pv=%u status=%u\n",
                qk_value, pv_value, (unsigned)status);
      assert(status == VPU_FLASHATTENTION_OK);
      assert(vpu_fa_preflight_jobs(&config, &plan) ==
             VPU_FLASHATTENTION_OK);

      if (qk_value == GEMMINI_PARTITION_AXIS_M) {
        assert(plan.q_tiles % members == 0u);
        assert(plan.qk_tiles >= members);
      } else if (qk_value == GEMMINI_PARTITION_AXIS_N) {
        assert(plan.kv_tiles % members == 0u);
        assert(plan.q_tiles >= members);
      } else {
        assert(plan.qk_tiles % members == 0u);
      }
      if (pv_value == GEMMINI_PARTITION_AXIS_M) {
        assert(plan.q_tiles % members == 0u);
        assert(plan.kv_tiles >= members);
      } else if (pv_value == GEMMINI_PARTITION_AXIS_N) {
        assert(vpu_fa_ceil_div((unsigned)config.value_dim, DIM) % members ==
               0u);
        assert(plan.q_tiles >= members);
      } else {
        assert(plan.kv_tiles % members == 0u);
      }
    }
  }

  vpu_flashattention_config_t config = make_config(64u, 0xfu, false);
  config.sequence = 64u;
  config.qk_partition_axis = GEMMINI_PARTITION_AXIS_N;
  config.pv_partition_axis = GEMMINI_PARTITION_AXIS_K;
  vpu_fa_matmul_job_t job;

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK, 0u, 64u, 0u, 64u));
  assert(job.core.partition_axis == GEMMINI_PARTITION_AXIS_N);
  vpu_fa_matmul_job_init(&job);
  reset_gemmini_trace();
  assert(vpu_fa_matmul_job_step(&job, 0u, 0u));
  assert(((find_gemmini_command_for_custom(
               VPU_FA_K_LOOP_WS_CONFIG_PARTITION_BOUNDS, 0u)->rs1 >> 32) &
          UINT64_C(0x3)) == GEMMINI_PARTITION_AXIS_N);

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_PV, 0u, 64u, 0u, 64u));
  assert(job.core.partition_axis == GEMMINI_PARTITION_AXIS_K);
  assert(job.core.tile_K == vpu_fa_matmul_job_total_k_tiles(&job));
  vpu_fa_matmul_job_init(&job);
  reset_gemmini_trace();
  assert(vpu_fa_matmul_job_step(&job, 0u, 0u));
  assert(((find_gemmini_command_for_custom(
               VPU_FA_K_LOOP_WS_CONFIG_PARTITION_BOUNDS, 0u)->rs1 >> 32) &
          UINT64_C(0x3)) == GEMMINI_PARTITION_AXIS_K);

  /* The public scheduler must preserve the independent axes as it builds
   * Virgo lookahead jobs, rather than silently replacing either with M. */
  config.gemmini_mask = valid_mask;
  config.causal_mask_workspace = causal_mask;
  reset_gemmini_trace();
  reset_attention_events();
  const vpu_flashattention_result_t result = vpu_flashattention_auto(&config);
  assert(result.status == VPU_FLASHATTENTION_OK);
  bool saw_qk_n = false;
  bool saw_pv_k = false;
  bool trace_is_qk[4] = {false, false, false, false};
  unsigned trace_axis[4] = {0u, 0u, 0u, 0u};
  for (unsigned i = 0u; i < gemmini_trace_count; ++i) {
    const unsigned custom = gemmini_trace[i].custom;
    if (gemmini_trace[i].funct == VPU_FA_K_LOOP_WS_CONFIG_ADDRS_AB) {
      trace_is_qk[custom] = gemmini_trace[i].rs1 != 0u;
    } else if (gemmini_trace[i].funct ==
               VPU_FA_K_LOOP_WS_CONFIG_PARTITION_BOUNDS) {
      trace_axis[custom] =
          (unsigned)((gemmini_trace[i].rs1 >> 32) & UINT64_C(0x3));
    } else if (gemmini_trace[i].funct == VPU_FA_K_LOOP_WS) {
      if (trace_is_qk[custom]) {
        assert(trace_axis[custom] == GEMMINI_PARTITION_AXIS_N);
        saw_qk_n = true;
      } else {
        assert(trace_axis[custom] == GEMMINI_PARTITION_AXIS_K);
        saw_pv_k = true;
      }
    }
  }
  assert(saw_qk_n && saw_pv_k);

  /* A final six-tile KV block remains exact: N and K use 2/2/1/1 rather
   * than rounding the tail down to the four-member main-tile quantum. */
  config.gemmini_mask = 0xfu;
  const unsigned tail_rows = 6u * DIM;
  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK, 0u, 64u, 0u, tail_rows));
  assert(job.core.tile_J == 6u);
  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_PV, 0u, 64u, 0u, tail_rows));
  assert(job.core.tile_K == 6u && vpu_fa_matmul_job_total_k_tiles(&job) == 6u);
  vpu_fa_matmul_job_init(&job);
  assert(vpu_fa_matmul_job_step(&job, 0u, 0u));
  assert(job.core.done);

  /* QK-K keeps the four-tile recurring K step and its natural 4+2 split. */
  config.qk_partition_axis = GEMMINI_PARTITION_AXIS_K;
  config.q_dim = tail_rows;
  config.k_dim = tail_rows;
  config.query_stride = tail_rows;
  config.key_stride = tail_rows;
  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK, 0u, 64u, 0u, 64u));
  assert(job.core.tile_K % 4u == 0u && job.core.tile_K < vpu_fa_matmul_job_total_k_tiles(&job));
  assert(vpu_fa_matmul_job_planned_steps(&job) == 2u);
  vpu_fa_matmul_job_init(&job);
  for (unsigned step = 0u; step < vpu_fa_matmul_job_planned_steps(&job); ++step) {
    assert(vpu_fa_matmul_job_step(&job, step & 1u, 0u));
    assert(job.core.done == (step + 1u == vpu_fa_matmul_job_planned_steps(&job)));
  }

  if (members > 1u) {
    config.gemmini_mask = valid_mask;
    config.pv_partition_axis = GEMMINI_PARTITION_AXIS_N;
    config.value_dim = (members + 1u) * DIM;
    config.value_stride = config.value_dim;
    config.output_stride = config.value_dim;
    vpu_flashattention_plan_t plan;
    assert(vpu_flashattention_make_plan(&config, &plan) ==
             VPU_FLASHATTENTION_NO_TILING);
  }
#endif
}

static void check_forced_block_and_qk_tile_k_overrides(void) {
#if DIM == 8 && VPU_MATRIX_PORTS == 4
  vpu_flashattention_config_t config = make_config(32u, 0xfu, false);
  config.sequence = 64u;
  config.query_base = 32u;
  config.q_dim = 32u;
  config.k_dim = 32u;
  config.value_dim = 32u;
  config.query_stride = 32u;
  config.key_stride = 32u;
  config.value_stride = 32u;
  config.output_stride = 32u;
  config.q_block_rows = 32u;
  config.kv_block_rows = 32u;
  config.qk_tile_k = 4u;

  for (unsigned qk_value = GEMMINI_PARTITION_AXIS_M;
       qk_value <= GEMMINI_PARTITION_AXIS_K; ++qk_value) {
    for (unsigned pv_value = GEMMINI_PARTITION_AXIS_M;
         pv_value <= GEMMINI_PARTITION_AXIS_K; ++pv_value) {
      config.qk_partition_axis = (gemmini_partition_axis_t)qk_value;
      config.pv_partition_axis = (gemmini_partition_axis_t)pv_value;

      vpu_flashattention_plan_t plan;
      assert(vpu_flashattention_make_plan(&config, &plan) ==
             VPU_FLASHATTENTION_OK);
      assert(plan.q_tiles == 4u);
      assert(plan.kv_tiles == 4u);
      assert(plan.qk_tiles == 4u);
      assert(plan.q_rows == 32u);
      assert(plan.kv_rows == 32u);
      assert(plan.qk_depth == 32u);
      assert(plan.gemmini_count == 4u);
      assert(vpu_fa_preflight_jobs(&config, &plan) ==
             VPU_FLASHATTENTION_OK);

      for (unsigned kind_value = VPU_FA_MATMUL_QK;
           kind_value <= VPU_FA_MATMUL_PV; ++kind_value) {
        const vpu_fa_matmul_kind_t kind =
            (vpu_fa_matmul_kind_t)kind_value;
        vpu_fa_matmul_job_t job;
        assert(vpu_fa_matmul_job_prepare(
            &job, &config, kind, 0u, plan.q_rows, 0u, plan.kv_rows));
        assert(job.core.tile_I == 4u);
        assert(job.core.tile_J == 4u);
        assert(job.core.tile_K == 4u);
        assert(vpu_fa_matmul_job_total_k_tiles(&job) == 4u);
        assert(job.core.gemmini_num == 4u);
        assert(gemmini_partition_effective_member_count(
                   job.core.partition_axis, job.core.gemmini_num,
                   job.core.tile_I, job.core.tile_J, job.core.tile_K) == 4u);
      }
    }
  }

  vpu_flashattention_plan_t plan;
  vpu_flashattention_config_t invalid = config;
  invalid.q_block_rows = 31u;
  assert(vpu_flashattention_make_plan(&invalid, &plan) ==
         VPU_FLASHATTENTION_INVALID_SHAPE);

  invalid = config;
  invalid.kv_block_rows = 31u;
  assert(vpu_flashattention_make_plan(&invalid, &plan) ==
         VPU_FLASHATTENTION_INVALID_SHAPE);

  invalid = config;
  invalid.q_block_rows = 40u;
  assert(vpu_flashattention_make_plan(&invalid, &plan) ==
         VPU_FLASHATTENTION_INVALID_SHAPE);

  invalid = config;
  invalid.kv_block_rows = 72u;
  assert(vpu_flashattention_make_plan(&invalid, &plan) ==
         VPU_FLASHATTENTION_INVALID_SHAPE);

  invalid = config;
  invalid.qk_tile_k = 5u;
  assert(vpu_flashattention_make_plan(&invalid, &plan) ==
         VPU_FLASHATTENTION_INVALID_SHAPE);

  invalid = config;
  invalid.qk_partition_axis = GEMMINI_PARTITION_AXIS_K;
  invalid.qk_tile_k = 2u;
  assert(vpu_flashattention_make_plan(&invalid, &plan) ==
         VPU_FLASHATTENTION_NO_TILING);

  invalid = make_config(128u, 0xfu, false);
  invalid.q_block_rows = 128u;
  invalid.kv_block_rows = 256u;
  assert(vpu_flashattention_make_plan(&invalid, &plan) ==
         VPU_FLASHATTENTION_NO_TILING);
#endif
}

static void check_score_scale_validation(void) {
  vpu_flashattention_config_t config = make_config(16u, 0x1u, false);
  vpu_flashattention_plan_t plan;

  assert(vpu_flashattention_make_plan(&config, &plan) ==
         VPU_FLASHATTENTION_OK);
  config.score_scale = 0.0f;
  assert(vpu_flashattention_make_plan(&config, &plan) ==
         VPU_FLASHATTENTION_INVALID_ARGUMENT);
  config.score_scale = -0.125f;
  assert(vpu_flashattention_make_plan(&config, &plan) ==
         VPU_FLASHATTENTION_INVALID_ARGUMENT);
  config.score_scale = NAN;
  assert(vpu_flashattention_make_plan(&config, &plan) ==
         VPU_FLASHATTENTION_INVALID_ARGUMENT);
  config.score_scale = INFINITY;
  assert(vpu_flashattention_make_plan(&config, &plan) ==
         VPU_FLASHATTENTION_INVALID_ARGUMENT);
}

static void check_packed_output_rejected(void) {
  vpu_flashattention_config_t config = make_config(16u, 0x1u, false);
  config.output_stride = GEMMINI_PAGE_PACKED_STRIDE(config.value_dim);
  vpu_flashattention_plan_t plan;
  assert(vpu_flashattention_make_plan(&config, &plan) ==
         VPU_FLASHATTENTION_INVALID_STRIDE);
}

static void check_all_query_tails(unsigned gemmini_mask, bool page_packed) {
  for (unsigned query_rows = 1u; query_rows <= 256u; ++query_rows) {
    const vpu_flashattention_config_t config =
        make_config(query_rows, gemmini_mask, page_packed);
    vpu_flashattention_plan_t plan;
    assert(vpu_flashattention_make_plan(&config, &plan) ==
           VPU_FLASHATTENTION_OK);
    assert(vpu_fa_preflight_jobs(&config, &plan) ==
           VPU_FLASHATTENTION_OK);
  }
}

static void check_dim8_four_member_tail(void) {
#if DIM == 8 && VPU_MATRIX_PORTS == 4
  vpu_flashattention_config_t config = make_config(176u, 0xfu, true);
  config.q_block_rows = 64u;
  vpu_flashattention_plan_t plan;
  vpu_fa_matmul_job_t job;

  assert(vpu_flashattention_make_plan(&config, &plan) ==
         VPU_FLASHATTENTION_OK);
  assert(plan.q_rows == 64u);
  assert(plan.kv_rows == 256u);
  assert(plan.qk_depth == 64u);
  assert(vpu_fa_tiling_fits(8u, 32u, 8u));
  assert(8u * 32u * DIM + 8u * 8u * DIM <= ACC_ROWS / 2u);
  assert(8u * 32u * DIM + DIM <= ACC_ROWS / 2u);
  assert(vpu_fa_output_base_row(plan.q_rows, 8u) ==
         ACC_ROWS / 2u - 8u * 8u * DIM);
  assert(VPU_FA_CAUSAL_MASK_BASE_ROW == ACC_ROWS - DIM);
  assert(vpu_fa_preflight_jobs(&config, &plan) ==
         VPU_FLASHATTENTION_OK);

  /* The final 48-row block is six DIM tiles.  It must remain six so the
   * existing quotient/remainder dispatch becomes 2/2/1/1. */
  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK,
      128u, 48u, 0u, 128u));
  assert(job.core.tile_I == 6u);
  assert(job.core.tile_J == 16u);
  assert(job.core.tile_K == 8u);
  assert(job.core.tile_I / 4u + (0u < job.core.tile_I % 4u) == 2u);
  assert(job.core.tile_I / 4u + (1u < job.core.tile_I % 4u) == 2u);
  assert(job.core.tile_I / 4u + (2u < job.core.tile_I % 4u) == 1u);
  assert(job.core.tile_I / 4u + (3u < job.core.tile_I % 4u) == 1u);

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_PV,
      128u, 48u, 0u, 128u));
  assert(job.core.tile_I == 6u);
  assert(job.core.tile_J == 8u);
  assert(job.core.tile_K == 16u);

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK,
      128u, 48u, 128u, 112u));
  assert(job.core.tile_I == 6u);
  assert(job.core.tile_J == 14u);
  assert(job.core.tile_K == 8u);

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_PV,
      128u, 48u, 128u, 112u));
  assert(job.core.tile_I == 6u);
  assert(job.core.tile_J == 8u);
  assert(job.core.tile_K == 14u);

  /* Exact tails are admitted only while their real I/J/K footprint fits the
   * double-buffered SPAD and accumulator partitions. */
  assert(!vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK,
      0u, 1024u, 0u, 128u));
#endif
}

static void check_acc_half_live_set_boundary(void) {
  vpu_flashattention_config_t config = make_config(240u, 0x1u, false);
  /* Keep the ACC live set, rather than the sequence length, as the limiting
   * resource after m/l/alpha split the FP state SRAM into thirds. */
  config.sequence = 512u;
  vpu_flashattention_plan_t plan;
  assert(vpu_flashattention_make_plan(&config, &plan) ==
         VPU_FLASHATTENTION_OK);

#if DIM == 8
  const unsigned expected_q_tiles = 10u;
  const unsigned expected_kv_tiles = 43u;
  const unsigned value_tiles = 8u;
#elif DIM == 16
  const unsigned expected_q_tiles = 5u;
  const unsigned expected_kv_tiles = 21u;
  const unsigned value_tiles = 4u;
#else
#error "vpu_flashattention_plan_test covers DIM=8 and DIM=16 fusion targets"
#endif

  assert(plan.q_tiles == expected_q_tiles);
  assert(plan.kv_tiles == expected_kv_tiles);
  assert(vpu_fa_acc_live_set_fits(expected_q_tiles, expected_kv_tiles,
                                  value_tiles));
  assert(!vpu_fa_acc_live_set_fits(expected_q_tiles,
                                   expected_kv_tiles + 1u, value_tiles));
  assert(vpu_fa_tiling_fits(expected_q_tiles, expected_kv_tiles, value_tiles));
  assert(!vpu_fa_tiling_fits(expected_q_tiles, expected_kv_tiles + 1u,
                             value_tiles));
  const size_t score_rows =
      (size_t)expected_q_tiles * expected_kv_tiles * DIM;
  const size_t output_rows =
      (size_t)expected_q_tiles * value_tiles * DIM;
  assert(score_rows + output_rows <= ACC_ROWS / 2u);
  assert(score_rows + DIM <= ACC_ROWS / 2u);
  assert((size_t)expected_q_tiles * (expected_kv_tiles + 1u) * DIM +
             output_rows > ACC_ROWS / 2u);
  assert(vpu_fa_output_base_row(plan.q_rows, value_tiles) ==
         ACC_ROWS / 2u - output_rows);
  assert(VPU_FA_CAUSAL_MASK_BASE_ROW == ACC_ROWS - DIM);

  /* DIM8 uses fragmentStride to fetch the two tile-major ACC rows in one
   * 16-lane request; lane width no longer disables cross-tile coalescing. */
  assert(!vpu_fa_cross_tile_coalescing_fits(0u));
  assert(vpu_fa_cross_tile_coalescing_fits(DIM));
  assert(vpu_fa_cross_tile_coalescing_fits(2u * DIM));
  assert(vpu_fa_cross_tile_coalescing_fits(VPU_VLEN));
  assert(!vpu_fa_cross_tile_coalescing_fits(VPU_VLEN + 1u));
}

static void check_two_chunk_fabric_stream(bool exp_stream,
                                          unsigned tail_columns) {
  const unsigned full_tiles =
      tail_columns == 0u ? 2u * VPU_VLEN / DIM : VPU_VLEN / DIM;
  const unsigned chunk_stride =
      (VPU_VLEN / DIM) * VPU_FA_MATRIX_TILE_ELEMENTS;
  const unsigned expected_trace_count =
      exp_stream ? (tail_columns == 0u ? 8u : 13u)
                 : (tail_columns == 0u ? 6u : 9u);

  assert(vpu_fa_score_two_chunk_chain(full_tiles, tail_columns, true));
  assert(!vpu_fa_score_two_chunk_chain(full_tiles, tail_columns, false));

  /* The enclosing row run establishes this state before LOOP_START.  Exclude
   * that setup command here so this trace is exactly the captured body. */
  capture_vpu_trace = false;
  vpu_fa_group_set_vector_stride(0u, VPU_FA_MATRIX_TILE_ELEMENTS);
  reset_vpu_trace();
  if (exp_stream)
    vpu_fa_group_score_exp_stream(0u, full_tiles, tail_columns, true);
  else
    vpu_fa_group_score_max_stream(0u, full_tiles, tail_columns, true);
  capture_vpu_trace = false;

  assert(vpu_trace_count == expected_trace_count);
  unsigned second_pointer_setups = 0u;
  unsigned full_vls = 0u;
  unsigned tail_vls = 0u;
  unsigned muls = 0u;
  unsigned subs = 0u;
  unsigned exps = 0u;
  unsigned reductions = 0u;
  unsigned reduction_modes[2] = {VPU_REDUCTION_SINGLE,
                                 VPU_REDUCTION_SINGLE};

  for (unsigned i = 0u; i < vpu_trace_count; ++i) {
    const uint32_t uop = vpu_trace[i].uop;
    const unsigned opcode = vpu_micro_opcode(uop);
    assert(opcode != VPU_OP_C_SET_VSTRIDE);
    assert(opcode != VPU_OP_C_LOOP_START);
    assert(opcode != VPU_OP_C_LOOP_END);
    if (opcode == VPU_OP_C_SET_VL) {
      full_vls += vpu_trace[i].payload == VPU_VLEN;
      tail_vls += vpu_trace[i].payload == tail_columns;
    } else if (opcode == VPU_OP_S_ADDI_INT &&
               vpu_micro_rd(uop) == VPU_FA_GP_VECTOR) {
      if (vpu_micro_rs1(uop) == VPU_FA_GP_SCORE_ROW) {
        assert(((uop >> 14) & VPU_ADDI_INT_IMM_MAX) == chunk_stride);
        ++second_pointer_setups;
      } else {
        assert(false);
      }
    }
    muls += opcode == VPU_OP_V_MUL_VF;
    subs += opcode == VPU_OP_V_SUB_VF;
    exps += opcode == VPU_OP_V_EXP_V;
    if (opcode == (exp_stream ? VPU_OP_V_RED_SUM : VPU_OP_V_RED_MAX)) {
      assert(reductions < 2u);
      reduction_modes[reductions++] = vpu_micro_funct1(uop);
    }
  }

  assert(full_vls == (tail_columns == 0u ? 1u
                                             : (exp_stream ? 3u : 2u)));
  assert(tail_vls == (tail_columns == 0u ? 0u
                                             : (exp_stream ? 3u : 2u)));
  assert(second_pointer_setups == 1u);
  assert(muls == (exp_stream ? 0u : 2u));
  assert(subs == (exp_stream ? 2u : 0u));
  assert(exps == (exp_stream ? 2u : 0u));
  assert(reductions == 2u);
  assert(reduction_modes[0] == VPU_REDUCTION_START);
  assert(reduction_modes[1] == VPU_REDUCTION_FINAL);

  /* Fabric phases must be contiguous even though pointer/VL controls may sit
   * between two vector commands. */
  unsigned first_reduce = vpu_trace_count;
  unsigned last_base = 0u;
  unsigned first_exp = vpu_trace_count;
  unsigned last_exp = 0u;
  for (unsigned i = 0u; i < vpu_trace_count; ++i) {
    const unsigned opcode = vpu_micro_opcode(vpu_trace[i].uop);
    if (opcode == VPU_OP_V_MUL_VF || opcode == VPU_OP_V_SUB_VF)
      last_base = i;
    if (opcode == VPU_OP_V_EXP_V) {
      if (first_exp == vpu_trace_count)
        first_exp = i;
      last_exp = i;
    }
    if ((opcode == VPU_OP_V_RED_MAX || opcode == VPU_OP_V_RED_SUM) &&
        first_reduce == vpu_trace_count)
      first_reduce = i;
  }
  assert(last_base < first_reduce);
  if (exp_stream) {
    assert(last_base < first_exp);
    assert(last_exp < first_reduce);
  }

  /* With two full chunks, two live GP addresses remove every control command
   * between same-fabric operations.  A tail has only the required VL update. */
  if (tail_columns == 0u) {
    if (exp_stream) {
      assert(vpu_micro_opcode(vpu_trace[2].uop) == VPU_OP_V_SUB_VF);
      assert(vpu_micro_opcode(vpu_trace[3].uop) == VPU_OP_V_SUB_VF);
      assert(vpu_micro_opcode(vpu_trace[4].uop) == VPU_OP_V_EXP_V);
      assert(vpu_micro_opcode(vpu_trace[5].uop) == VPU_OP_V_EXP_V);
      assert(vpu_micro_opcode(vpu_trace[6].uop) == VPU_OP_V_RED_SUM);
      assert(vpu_micro_opcode(vpu_trace[7].uop) == VPU_OP_V_RED_SUM);
    } else {
      assert(vpu_micro_opcode(vpu_trace[2].uop) == VPU_OP_V_MUL_VF);
      assert(vpu_micro_opcode(vpu_trace[3].uop) == VPU_OP_V_MUL_VF);
      assert(vpu_micro_opcode(vpu_trace[4].uop) == VPU_OP_V_RED_MAX);
      assert(vpu_micro_opcode(vpu_trace[5].uop) == VPU_OP_V_RED_MAX);
    }
  }
}

static void check_long_score_row_chunking(void) {
  assert(VPU_VLEN % DIM == 0u);
  const unsigned two_full_tiles = 2u * VPU_VLEN / DIM;
  assert(vpu_fa_matrix_row_run_vlen_chunks_bank_local(
      vpu_fa_score_base_from_slot(0u), two_full_tiles, 0u, DIM,
      2u * VPU_VLEN));
  assert(!vpu_fa_matrix_row_run_vlen_chunks_bank_local(
      VPU_ELEMENTS_PER_BANK - DIM, two_full_tiles, 0u, 1u,
      2u * VPU_VLEN));

  /* Both admitted two-chunk shapes use fabric batching: two full chunks, and
   * one full chunk followed by a real matrix-row tail. */
  check_two_chunk_fabric_stream(false, 0u);
  check_two_chunk_fabric_stream(true, 0u);
  check_two_chunk_fabric_stream(false, DIM - 1u);
  check_two_chunk_fabric_stream(true, DIM - 1u);

  /* Three chunks retain the compact legacy loop and standalone reductions. */
  const unsigned fallback_tiles = 3u * VPU_VLEN / DIM;
  assert(!vpu_fa_score_two_chunk_chain(fallback_tiles, 0u, true));
  for (unsigned stream = 0u; stream < 2u; ++stream) {
    reset_vpu_trace();
    if (stream == 0u)
      vpu_fa_group_score_max_stream(0u, fallback_tiles, 0u, true);
    else
      vpu_fa_group_score_exp_stream(0u, fallback_tiles, 0u, true);
    capture_vpu_trace = false;

    unsigned chunk_loops = 0u;
    unsigned chunk_loop_ends = 0u;
    unsigned reductions = 0u;
    unsigned stride_matrix = 0u;
    unsigned stride_zero = 0u;
    for (unsigned i = 0u; i < vpu_trace_count; ++i) {
      const uint32_t uop = vpu_trace[i].uop;
      const unsigned opcode = vpu_micro_opcode(uop);
      if (opcode == VPU_OP_C_LOOP_START &&
          vpu_micro_rd(uop) == VPU_FA_GP_TILE_LOOP)
        chunk_loops += ((uop >> 10) & VPU_LOOP_COUNT_MAX) == 3u;
      if (opcode == VPU_OP_C_LOOP_END &&
          vpu_micro_rd(uop) == VPU_FA_GP_TILE_LOOP)
        ++chunk_loop_ends;
      if (opcode == (stream == 0u ? VPU_OP_V_RED_MAX : VPU_OP_V_RED_SUM)) {
        ++reductions;
        assert(vpu_micro_funct1(uop) == VPU_REDUCTION_SINGLE);
      }
      if (opcode == VPU_OP_C_SET_VSTRIDE) {
        stride_matrix +=
            vpu_trace[i].payload == VPU_FA_MATRIX_TILE_ELEMENTS;
        stride_zero += vpu_trace[i].payload == 0u;
      }
    }
    assert(chunk_loops == 1u);
    assert(chunk_loop_ends == 1u);
    assert(reductions == 1u);
    assert(stride_matrix == 1u);
    assert(stride_zero == 1u);
  }
}

static void check_causal_prefix_geometry(void) {
  const unsigned kv_start = 4u * DIM;
  const unsigned kv_rows = 8u * DIM;
  assert(vpu_fa_causal_active_columns(
             kv_start, kv_rows, kv_start - DIM, DIM) == 0u);
  assert(vpu_fa_causal_active_columns(
             kv_start, kv_rows, kv_start + 2u * DIM, DIM) == 3u * DIM);
  assert(vpu_fa_causal_active_columns(
             kv_start, kv_rows, kv_start + kv_rows - 1u, 1u) == kv_rows);
  assert(vpu_fa_causal_row_run(2u * DIM, 0u, kv_start, kv_start) == DIM);
}

static void check_output_coalescing_bank_boundary(void) {
  const unsigned q_rows = 128u;
  const unsigned value_dim = 64u;
  const unsigned value_tiles = vpu_fa_ceil_div(value_dim, DIM);
  const uint32_t output_base = vpu_fa_output_base(q_rows, value_tiles);
  const size_t first_row_base =
      vpu_fa_matrix_address(output_base, value_tiles, 0u, 0u);
  const size_t last_column = value_dim - 1u;
  const size_t footprint =
      (last_column / DIM) * VPU_FA_MATRIX_TILE_ELEMENTS +
      last_column % DIM + 1u;
  const bool expected_first_row_local =
      first_row_base / VPU_ELEMENTS_PER_BANK ==
      (first_row_base + footprint - 1u) / VPU_ELEMENTS_PER_BANK;

  assert(vpu_fa_matrix_row_run_bank_local(
             output_base, value_tiles, 0u, DIM, value_dim) ==
         expected_first_row_local);
#if DIM == 8 && VPU_VSPAD_BANKS == 8
  /* With O ending exactly at the lower-half boundary, this 128x64 output
   * occupies bank 3 exactly instead of straddling the bank-2/3 boundary. */
  assert(expected_first_row_local);
  assert(vpu_fa_matrix_row_run_bank_local(
      output_base, value_tiles, DIM, DIM, value_dim));
#endif
}

static void check_virgo_attention_issue_order(void) {
  static float causal_mask[VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS];
  vpu_flashattention_config_t config = make_config(80u, 0x1u, false);
  config.sequence = 1200u;
  config.query_base = config.sequence - config.query_rows;
  config.causal_mask_workspace = causal_mask;

  vpu_flashattention_plan_t plan;
  assert(vpu_flashattention_make_plan(&config, &plan) ==
         VPU_FLASHATTENTION_OK);
  assert(plan.q_rows == config.query_rows);
  const unsigned blocks =
      vpu_fa_ceil_div((unsigned)config.sequence, plan.kv_rows);
  assert(blocks >= 3u);

  reset_gemmini_trace();
  reset_attention_events();
  const vpu_flashattention_result_t result =
      vpu_flashattention_auto(&config);
  assert(result.status == VPU_FLASHATTENTION_OK);
  assert(!vpu_stage_active);
  assert(result.stats.causal_blocks == blocks);
  /* Event ordering leaves synchronization groups dedicated to SPAD ping/pong,
   * so every QK and PV in this shape remains one step. */
  assert(result.stats.gemmini_job_steps == 2u * blocks);
  assert(attention_event_count == 4u * blocks);

  unsigned event = 0u;
  assert_matmul_event(event, ATTENTION_EVENT_QK, 0u, 0u, false,
                      vpu_fa_qk_event(0u), true);
  ++event;

  /* SM0 replays while the disjoint QK1 is admitted. */
  assert_matmul_event(event, ATTENTION_EVENT_QK,
                      vpu_fa_group_pair_base(1u), 0u, false,
                      vpu_fa_qk_event(1u), true);
  ++event;
  assert_vpu_stage_event(event, vpu_fa_qk_event(0u),
                         vpu_fa_softmax_event(0u), true);
  ++event;

  /* In steady state SMb overlaps PV(b-1), closes, and then Rb overlaps
   * QK(b+1). Events express each edge independently of sync groups. */
  for (unsigned block = 1u; block + 1u < blocks; ++block) {
    const unsigned previous_group = vpu_fa_group_pair_base(block - 1u);
    const unsigned next_group = vpu_fa_group_pair_base(block + 1u);
    assert_matmul_event(event, ATTENTION_EVENT_PV, previous_group,
                        vpu_fa_softmax_event(block - 1u), true,
                        vpu_fa_pv_event(block - 1u), true);
    ++event;
    assert_vpu_stage_event(event, vpu_fa_qk_event(block),
                           vpu_fa_softmax_event(block), true);
    ++event;
    assert_matmul_event(event, ATTENTION_EVENT_QK, next_group, 0u, false,
                        vpu_fa_qk_event(block + 1u), true);
    ++event;
    assert_vpu_stage_event(event, vpu_fa_pv_event(block - 1u), 0u, false);
    ++event;
  }

  const unsigned last_group = vpu_fa_group_pair_base(blocks - 1u);
  const unsigned penultimate_group = vpu_fa_group_pair_base(blocks - 2u);
  for (unsigned block = 0u; block < blocks; ++block)
    assert((vpu_fa_group_pair_base(block) & 1u) ==
           vpu_fa_block_memory_slot(block));
  assert_matmul_event(event, ATTENTION_EVENT_PV, penultimate_group,
                      vpu_fa_softmax_event(blocks - 2u), true,
                      vpu_fa_pv_event(blocks - 2u), true);
  ++event;
  assert_vpu_stage_event(event, vpu_fa_qk_event(blocks - 1u),
                         vpu_fa_softmax_event(blocks - 1u), true);
  ++event;
  /* With no next QK, R closes before the final PV owns normalization. */
  assert_vpu_stage_event(event, vpu_fa_pv_event(blocks - 2u), 0u, false);
  ++event;
  assert_matmul_event(event, ATTENTION_EVENT_PV, last_group,
                      vpu_fa_softmax_event(blocks - 1u), true,
                      vpu_fa_pv_event(blocks - 1u), true);
  ++event;
  assert_vpu_stage_event(event, vpu_fa_pv_event(blocks - 1u), 0u, false);
  ++event;
  assert(event == attention_event_count);
}

static void check_k_split_group_and_memory_slot_separation(void) {
  vpu_flashattention_config_t config = make_config(16u, 0x1u, false);
  const unsigned reduction_dim = 8u * DIM;
  config.q_dim = reduction_dim;
  config.k_dim = reduction_dim;
  config.query_stride = reduction_dim;
  config.key_stride = reduction_dim;
  config.qk_tile_k = 4u;

  vpu_fa_matmul_job_t job;
  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK,
      0u, 16u, 0u, 128u));
  assert(!vpu_fa_matmul_job_is_single_step(&job));
  const unsigned steps = vpu_fa_matmul_job_planned_steps(&job);
  assert(steps == 2u);
  vpu_fa_matmul_job_set_events(&job, 6u, true, 4u, true);

  reset_gemmini_trace();
  reset_attention_events();
  vpu_flashattention_stats_t stats = {0};
  static const unsigned expected_k_tiles[] = {4u, 4u};
  vpu_fa_matmul_job_init(&job);
  for (unsigned step = 0u; step < steps; ++step) {
    const unsigned before = vpu_fa_matmul_job_next_k_tile(&job);
    assert(vpu_fa_matmul_job_step(&job, 1u ^ (step & 1u), 1u));
    assert(vpu_fa_matmul_job_next_k_tile(&job) - before == expected_k_tiles[step]);
    ++stats.gemmini_job_steps;
  }
  assert(stats.gemmini_job_steps == steps);
  assert(attention_event_count == steps);

  unsigned spaddr_count = 0u;
  for (unsigned i = 0u; i < gemmini_trace_count; ++i) {
    if (gemmini_trace[i].funct != VPU_FA_K_LOOP_WS_CONFIG_SPADDR)
      continue;
    const unsigned expected_group = 1u ^ (spaddr_count & 1u);
    assert(gemmini_trace[i].rs1 == VPU_FA_ACC_HALF_ROWS);
    assert((gemmini_trace[i].rs2 & UINT64_C(0xffff)) ==
           vpu_fa_spad_start(expected_group));
    ++spaddr_count;
  }
  assert(spaddr_count == steps);

  for (unsigned step = 0u; step < steps; ++step) {
    const unsigned expected_group = 1u ^ (step & 1u);
    assert_matmul_event(step, ATTENTION_EVENT_QK, expected_group, 6u,
                        step == 0u, 4u, step + 1u == steps);
  }
  assert(attention_events[steps - 1u].sync_group_id == 0u);
}

int main(void) {
  for (unsigned members = 1u; members <= VPU_MATRIX_PORTS; ++members) {
    const unsigned mask = (1u << members) - 1u;
    check_all_query_tails(mask, false);
    check_all_query_tails(mask, true);
  }
  check_score_scale_validation();
  check_packed_output_rejected();
  check_shared_accumulator_loop_ws_abi();
  check_matmul_job_partition_axes();
  check_independent_qk_pv_axes();
  check_forced_block_and_qk_tile_k_overrides();
  check_acc_half_live_set_boundary();
  check_long_score_row_chunking();
  check_causal_prefix_geometry();
  check_output_coalescing_bank_boundary();
  check_dim8_four_member_tail();
  check_virgo_attention_issue_order();
  check_k_split_group_and_memory_slot_separation();
  return 0;
}
