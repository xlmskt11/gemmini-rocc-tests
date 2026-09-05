// See LICENSE for license details.

#ifndef GEMMINI_TEST_PARTITION_AXIS_ABI_VERSION
#define GEMMINI_TEST_PARTITION_AXIS_ABI_VERSION 1
#endif
#define GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION                              \
  GEMMINI_TEST_PARTITION_AXIS_ABI_VERSION
#include "include/gemmini_tiling.h"

#include <assert.h>
#include <stdint.h>

typedef int8_t elem_t;
#define DIM 16
#define MAX_BLOCK_LEN 64
#define MAX_BYTES 64
#include "include/gemmini_page_packed.h"

static void test_command_packing(void) {
  assert(gemmini_partition_generated_axis_abi_version() ==
         GEMMINI_TEST_PARTITION_AXIS_ABI_VERSION);
  assert(GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE ==
         (GEMMINI_TEST_PARTITION_AXIS_ABI_VERSION == 1));
  const size_t partition_extent = 0x1234;
  const size_t partition_offset = 0x3456;
  const size_t aux_extent = 0x5678;
  const size_t aux_offset = 0x789a;
  const size_t aux_pad = 0x9abc;

  const uint64_t legacy_rs1 =
      ((uint64_t)aux_offset << 16) | partition_offset;
  const uint64_t legacy_rs2 =
      ((uint64_t)aux_extent << 32) |
      ((uint64_t)aux_pad << 16) |
      partition_extent;

  assert(gemmini_shared_partition_pack_rs1(
             GEMMINI_PARTITION_AXIS_M,
             partition_offset, aux_offset) == legacy_rs1);
  assert(gemmini_shared_partition_pack_rs2(
             partition_extent, aux_extent, aux_pad) == legacy_rs2);
  assert(gemmini_shared_partition_pack_rs1(
             GEMMINI_PARTITION_AXIS_N, 0, 0) == UINT64_C(1) << 32);
  assert(gemmini_shared_partition_pack_rs1(
             GEMMINI_PARTITION_AXIS_K, 0, 0) == UINT64_C(2) << 32);

  assert(gemmini_shared_partition_fields_valid(
      GEMMINI_PARTITION_AXIS_K, UINT16_MAX, UINT16_MAX,
      UINT16_MAX, UINT16_MAX, UINT16_MAX));
  assert(!gemmini_shared_partition_fields_valid(
      (gemmini_partition_axis_t)3, 1, 1, 1, 1, 1));
  assert(!gemmini_shared_partition_fields_valid(
      GEMMINI_PARTITION_AXIS_M, (size_t)UINT16_MAX + 1, 0, 0, 0, 0));

  assert(gemmini_partition_axis_supported_by_abi(
      GEMMINI_PARTITION_AXIS_M, 0));
  assert(!gemmini_partition_axis_supported_by_abi(
      GEMMINI_PARTITION_AXIS_N, 0));
  assert(!gemmini_partition_axis_supported_by_abi(
      GEMMINI_PARTITION_AXIS_K, 0));
  assert(gemmini_partition_axis_supported_by_abi(
      GEMMINI_PARTITION_AXIS_N, 1));
  assert(gemmini_partition_axis_supported_by_abi(
      GEMMINI_PARTITION_AXIS_K, 1));
}

static void test_masks_and_member_counts(void) {
  assert(gemmini_partition_take_members(UINT64_C(0xa), 0) == 0);
  assert(gemmini_partition_take_members(UINT64_C(0xa), 1) == UINT64_C(0x2));
  assert(gemmini_partition_take_members(UINT64_C(0xa), 2) == UINT64_C(0xa));

  assert(gemmini_partition_effective_member_count(
             GEMMINI_PARTITION_AXIS_M, 4, 1, 99, 3) == 3);
  assert(gemmini_partition_effective_member_count(
             GEMMINI_PARTITION_AXIS_N, 4, 99, 2, 99) == 4);
  assert(gemmini_partition_effective_member_count(
             GEMMINI_PARTITION_AXIS_N, 4, 1, 2, 99) == 2);
  assert(gemmini_partition_effective_member_count(
             GEMMINI_PARTITION_AXIS_K, 4, 99, 99, 1) == 1);
}

static void test_axis_plans(void) {
  gemmini_partition_plan_t plan;

  assert(gemmini_partition_plan_member(
      GEMMINI_PARTITION_AXIS_M,
      7, 10, 5, 1, 2, 3, 3, 2, &plan));
  assert(plan.partition_extent == 2);
  assert(plan.partition_offset == 5);
  assert(plan.partition_pad == 1);
  assert(plan.aux_extent == 1);
  assert(plan.aux_offset == 4);
  assert(plan.aux_pad == 3);

  assert(gemmini_partition_plan_member(
      GEMMINI_PARTITION_AXIS_N,
      7, 10, 5, 1, 2, 3, 3, 1, &plan));
  assert(plan.partition_extent == 3);
  assert(plan.partition_offset == 4);
  assert(plan.partition_pad == 0);
  assert(plan.aux_extent == 2);
  assert(plan.aux_offset == 3);
  assert(plan.aux_pad == 0);

  assert(gemmini_partition_plan_member(
      GEMMINI_PARTITION_AXIS_K,
      7, 10, 5, 1, 2, 3, 3, 2, &plan));
  assert(plan.partition_extent == 1);
  assert(plan.partition_offset == 4);
  assert(plan.partition_pad == 3);
  assert(plan.aux_extent == 2);
  assert(plan.aux_offset == 5);
  assert(plan.aux_pad == 1);
  assert(gemmini_partition_d_memory_i_offset(
             plan.aux_offset, false) == 5);
  assert(gemmini_partition_d_memory_i_offset(
             plan.aux_offset, true) == 0);

  /* Zero-extent followers must not inherit the global tail padding. */
  assert(gemmini_partition_plan_member(
      GEMMINI_PARTITION_AXIS_M,
      1, 1, 4, 7, 0, 0, 4, 0, &plan));
  assert(plan.partition_extent == 1 && plan.partition_pad == 7);
  assert(gemmini_partition_plan_member(
      GEMMINI_PARTITION_AXIS_M,
      1, 1, 4, 7, 0, 0, 4, 3, &plan));
  assert(plan.partition_extent == 0 && plan.partition_pad == 0);
}

static void test_accumulate_contract(void) {
  assert(!gemmini_partition_ex_accumulate(true, 0));
  assert(gemmini_partition_ex_accumulate(true, 1));
  assert(gemmini_partition_ex_accumulate(false, 0));
  assert(gemmini_partition_ex_accumulate(false, 1));
}

static void test_page_offset_preflight(void) {
  assert(gemmini_loop_bounds_fields_valid(
      UINT16_MAX, UINT16_MAX, UINT16_MAX,
      UINT16_MAX, UINT16_MAX, UINT16_MAX));
  assert(!gemmini_loop_bounds_fields_valid(
      (size_t)UINT16_MAX + 1, 1, 1, 0, 0, 0));
  assert(gemmini_page_offset_fields_valid(
      UINT16_MAX, 0, 0, 0, 0, 0, 0, 0));
  assert(!gemmini_page_offset_fields_valid(
      (size_t)UINT16_MAX + 1, 0, 0, 0, 0, 0, 0, 0));

  assert(gemmini_page_block_range_fits_u16(UINT16_MAX, 0, 1));
  assert(!gemmini_page_block_range_fits_u16(UINT16_MAX, 0, 2));
  assert(gemmini_page_block_range_fits_u16(UINT16_MAX - 1, 1, 1));
  assert(!gemmini_page_block_range_fits_u16(UINT16_MAX - 1, 1, 2));

  /* N owns B/D/C columns: outer 65534 + member offset 1 + extent 1 fits. */
  assert(gemmini_shared_partition_page_offsets_valid(
      GEMMINI_PARTITION_AXIS_N,
      1, 1, 1, 0, 1, 2, 1,
      false, true, true, true,
      0, 0, 0, UINT16_MAX - 1,
      0, UINT16_MAX - 1, 0, UINT16_MAX - 1));
  assert(!gemmini_shared_partition_page_offsets_valid(
      GEMMINI_PARTITION_AXIS_N,
      2, 1, 1, 0, 1, 3, 1,
      false, true, true, true,
      0, 0, 0, UINT16_MAX - 1,
      0, UINT16_MAX - 1, 0, UINT16_MAX - 1));
}

static void test_axis_tiling(void) {
  const gemmini_tiling_request_t request = {
      .dim_I = 64,
      .dim_J = 64,
      .dim_K = 64,
      .dim = 16,
      .gemmini_count = 4,
      .sp_addr_range = 8192,
      .acc_addr_range = 2048,
      .double_buffered = true,
      .act = 0,
  };

  const gemmini_tiling_factors_t m_factors =
      gemmini_shared_multi_choose_tiling_axis(
          &request, GEMMINI_PARTITION_AXIS_M);
  const gemmini_tiling_factors_t n_factors =
      gemmini_shared_multi_choose_tiling_axis(
          &request, GEMMINI_PARTITION_AXIS_N);
  const gemmini_tiling_factors_t k_factors =
      gemmini_shared_multi_choose_tiling_axis(
          &request, GEMMINI_PARTITION_AXIS_K);
  assert(m_factors.tile_I >= 4 && m_factors.tile_I % 4 == 0);
  assert(m_factors.tile_K >= 4);
  assert(n_factors.tile_J >= 4 && n_factors.tile_J % 4 == 0);
  assert(n_factors.tile_I >= 4);
  assert(k_factors.tile_K >= 4 && k_factors.tile_K % 4 == 0);

  const gemmini_tiling_request_t roomy = {
      .dim_I = 512,
      .dim_J = 512,
      .dim_K = 512,
      .dim = 16,
      .gemmini_count = 4,
      .sp_addr_range = 16384,
      .acc_addr_range = 4096,
      .double_buffered = true,
      .act = 0,
  };
  const gemmini_tiling_factors_t roomy_n =
      gemmini_shared_multi_choose_tiling_axis(
          &roomy, GEMMINI_PARTITION_AXIS_N);
  const gemmini_tiling_factors_t roomy_k =
      gemmini_shared_multi_choose_tiling_axis(
          &roomy, GEMMINI_PARTITION_AXIS_K);
  assert(roomy_n.tile_J % 4 == 0 && roomy_n.tile_I >= 4);
  assert(roomy_k.tile_K % 4 == 0);

  /* An empty mask is rejected by the job initializer, but the chooser runs
   * first in the legacy call path and must not stall on a zero quantum. */
  gemmini_tiling_request_t empty = request;
  empty.gemmini_count = 0;
  const gemmini_tiling_factors_t empty_m =
      gemmini_shared_multi_choose_tiling_axis(
          &empty, GEMMINI_PARTITION_AXIS_M);
  assert(empty_m.tile_I != 0 && empty_m.tile_J != 0 && empty_m.tile_K != 0);
}

int main(void) {
  test_command_packing();
  test_masks_and_member_counts();
  test_axis_plans();
  test_accumulate_contract();
  test_page_offset_preflight();
  test_axis_tiling();
  return 0;
}
