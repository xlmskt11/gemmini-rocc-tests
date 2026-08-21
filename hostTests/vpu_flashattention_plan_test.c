// See LICENSE for license details.

#define VPU_ENABLE_GEMMINI_FLASHATTENTION 1
#include "include/vpu_flashattention_kernel.h"

#include <assert.h>
#include <stdint.h>

void vpu_host_issue(uint64_t transport, uint64_t payload) {
  (void)transport;
  (void)payload;
}

uint64_t vpu_host_issue_result(uint64_t transport, uint64_t payload) {
  (void)transport;
  (void)payload;
  return 0u;
}

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
      .causal_mask_workspace = (float *)(uintptr_t)0x50000000u,
      .causal_mask_workspace_elements =
          VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS,
  };
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
  const vpu_flashattention_config_t config = make_config(240u, 0xfu, true);
  vpu_flashattention_plan_t plan;
  vpu_fa_matmul_job_t job;

  assert(vpu_flashattention_make_plan(&config, &plan) ==
         VPU_FLASHATTENTION_OK);
  assert(plan.q_rows == 64u);
  assert(plan.kv_rows == 128u);
  assert(plan.qk_depth == 64u);
  assert(vpu_fa_preflight_jobs(&config, &plan) ==
         VPU_FLASHATTENTION_OK);

  /* The final 48-row block is six DIM tiles.  It must remain six so the
   * existing quotient/remainder dispatch becomes 2/2/1/1. */
  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK, 192u, 48u, 0u, 128u));
  assert(job.tile_i == 6u);
  assert(job.tile_j == 16u);
  assert(job.tile_k == 8u);
  assert(job.tile_i / 4u + (0u < job.tile_i % 4u) == 2u);
  assert(job.tile_i / 4u + (1u < job.tile_i % 4u) == 2u);
  assert(job.tile_i / 4u + (2u < job.tile_i % 4u) == 1u);
  assert(job.tile_i / 4u + (3u < job.tile_i % 4u) == 1u);

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_PV, 192u, 48u, 0u, 128u));
  assert(job.tile_i == 6u);
  assert(job.tile_j == 8u);
  assert(job.tile_k == 16u);

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK, 192u, 48u, 128u, 112u));
  assert(job.tile_i == 6u);
  assert(job.tile_j == 14u);
  assert(job.tile_k == 8u);

  assert(vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_PV, 192u, 48u, 128u, 112u));
  assert(job.tile_i == 6u);
  assert(job.tile_j == 8u);
  assert(job.tile_k == 14u);

  /* Exact tails are admitted only while their real I/J/K footprint fits the
   * double-buffered SPAD and accumulator partitions. */
  assert(!vpu_fa_matmul_job_prepare(
      &job, &config, VPU_FA_MATMUL_QK, 0u, 1024u, 0u, 128u));
#endif
}

int main(void) {
  for (unsigned members = 1u; members <= VPU_MATRIX_PORTS; ++members) {
    const unsigned mask = (1u << members) - 1u;
    check_all_query_tails(mask, false);
    check_all_query_tails(mask, true);
  }
  check_score_scale_validation();
  check_packed_output_rejected();
  check_dim8_four_member_tail();
  return 0;
}
