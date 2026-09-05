// See LICENSE for license details.

/* Compile this fixture with GEMMINI_TEST_PARTITION_AXIS_ABI_VERSION=0 and 1
 * and inspect the emitted RISC-V instructions.  It deliberately exercises
 * both generic shared LOOP_WS entry points without requiring a simulator. */
#ifndef GEMMINI_TEST_PARTITION_AXIS_ABI_VERSION
#define GEMMINI_TEST_PARTITION_AXIS_ABI_VERSION 1
#endif

#include "include/gemmini_params.h"
#undef GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION
#define GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION                              \
  GEMMINI_TEST_PARTITION_AXIS_ABI_VERSION
#include "include/gemmini_all.h"

void gemmini_shared_loop_abi_emit(void) {
  shared_gemmini_loop_ws_axis(
      custom0, 0x1, 0, 0, TOTAL_SPAD_ROWS / 2, 0,
      GEMMINI_PARTITION_AXIS_M, 1, 1, 0, 0, 0,
      1, 1, 1, 0, 0, 0,
      (const elem_t *)(uintptr_t)0x1000,
      (const elem_t *)(uintptr_t)0x2000, NULL, NULL,
      1, 1, 0, 1, false, false, false, false, false, NO_ACTIVATION);
}

void gemmini_shared_loop_page_abi_emit(void) {
  shared_gemmini_loop_ws_with_page_offsets_axis(
      custom0, 0x1, 1, TOTAL_SPAD_ROWS / 2, TOTAL_SPAD_ROWS, ACC_ROWS / 2,
      GEMMINI_PARTITION_AXIS_M, 1, 1, 0, 0, 0,
      1, 1, 1, 0, 0, 0,
      (const elem_t *)(uintptr_t)0x3000,
      (const elem_t *)(uintptr_t)0x4000, NULL, NULL,
      1, 1, 0, 1, false, false, false, false, false, NO_ACTIVATION,
      0, 0, 0, 0, 0, 0, 0, 0);
}
