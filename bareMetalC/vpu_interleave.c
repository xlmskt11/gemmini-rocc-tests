// See LICENSE for license details.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "include/gemmini_params.h"
#include "rocc-software/src/xcustom.h"
#include "include/vpu.h"

/*
 * Full-SoC routing/concurrency smoke test.
 *
 * The four Gemminis share the comparison configuration's external scratchpad,
 * so each one uses a disjoint physical bank. Their load/store command streams
 * are interleaved with a VPU load/execute/store stream sharing custom0. A
 * routing regression which sends funct64 to Gemmini0, or sends ordinary
 * custom0 commands to the VPU, therefore normally shows up as a bad result or
 * a failure to complete.
 *
 * This smoke intentionally uses short unicast MVIN/MVOUT streams so all four
 * opcode routes can run concurrently without depending on the pre-existing
 * SharedExtMem_4 output-stationary execute path. Multi-Gemmini execute is
 * covered by static_tiled_matmul_2group_test; funct8/funct15 ready/valid
 * atomicity is covered by RoccCommandRouterSpec.
 */

#define GEMMINI_COUNT 4u
#define VPU_SMOKE_VL 17u

#define custom0 0
#define custom1 1
#define custom2 2
#define custom3 3

/* funct7 must remain preprocessor constants for xcustom.h stringification. */
#define GEMMINI_SMOKE_CONFIG 0
#define GEMMINI_SMOKE_MVIN 2
#define GEMMINI_SMOKE_MVOUT 3
#define GEMMINI_SMOKE_FLUSH 7

#define GEMMINI_SMOKE_CONFIG_LD 1
#define GEMMINI_SMOKE_CONFIG_ST 2

/* The 4x16 comparison configs use FP32 load/accumulator scale fields. */
#define GEMMINI_SMOKE_FP32_ONE_BITS 0x3f800000u
#define GEMMINI_SMOKE_ADDR(index_) ((uint32_t)(index_) * BANK_ROWS)

#define GEMMINI_SMOKE_ISSUE(custom_, rs1_, rs2_, funct_)                  \
  ROCC_INSTRUCTION_0_R_R(custom_, (uint64_t)(rs1_), (uint64_t)(rs2_),    \
                         funct_)

#define GEMMINI_SMOKE_MATRIX_OPERAND(local_address_)                      \
  (((uint64_t)DIM << (ADDR_LEN + 16)) | ((uint64_t)DIM << ADDR_LEN) |     \
   (uint32_t)(local_address_))

#define GEMMINI_FLUSH_ONE(custom_)                                        \
  GEMMINI_SMOKE_ISSUE(custom_, 0, 0, GEMMINI_SMOKE_FLUSH)

#if VPU_VLEN < VPU_SMOKE_VL
#error "vpu_interleave requires VPU_VLEN >= 17"
#endif

#if ADDR_LEN != 32
#error "vpu_interleave operand packing requires ADDR_LEN == 32"
#endif

#if !defined(HAS_MVIN_SCALE) || !defined(ACC_SCALE_T_IS_FLOAT)
#error "vpu_interleave requires the FP32 Gemmini scale-field configuration"
#endif

enum {
  VPU_GP_SLOT = 0,
  VPU_GP_OFFSET = 1,
  VPU_FP_TWO = 0,
  VPU_FP_ONE = 1,
  VPU_H_INPUT = 0,
  VPU_H_OUTPUT = 1,
};

static elem_t gemmini_input[GEMMINI_COUNT][DIM][DIM]
    __attribute__((aligned(64)));
static elem_t gemmini_output[GEMMINI_COUNT][DIM][DIM]
    __attribute__((aligned(64)));

static vpu_storage_t vpu_input[VPU_SMOKE_VL + 2]
    __attribute__((aligned(64)));
static vpu_storage_t vpu_output[VPU_SMOKE_VL + 2]
    __attribute__((aligned(64)));

static inline void gemmini_smoke_memory_fence(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

#define GEMMINI_LOAD_ONE(custom_, index_)                                  \
  do {                                                                     \
    const uint64_t config_ld =                                             \
        ((uint64_t)GEMMINI_SMOKE_FP32_ONE_BITS << 32) |                    \
        ((uint64_t)DIM << 16) | ((uint64_t)1 << 8) |                       \
        GEMMINI_SMOKE_CONFIG_LD;                                           \
    const uint64_t config_st = GEMMINI_SMOKE_CONFIG_ST;                    \
    const uint64_t config_st_rs2 =                                         \
        ((uint64_t)GEMMINI_SMOKE_FP32_ONE_BITS << 32) |                    \
        (DIM * sizeof(elem_t));                                             \
    GEMMINI_SMOKE_ISSUE(custom_, config_ld, DIM * sizeof(elem_t),          \
                         GEMMINI_SMOKE_CONFIG);                             \
    GEMMINI_SMOKE_ISSUE(custom_, config_st, config_st_rs2,                 \
                         GEMMINI_SMOKE_CONFIG);                             \
    GEMMINI_SMOKE_ISSUE(                                                   \
        custom_, (uintptr_t)&gemmini_input[(index_)][0][0],                \
        GEMMINI_SMOKE_MATRIX_OPERAND(GEMMINI_SMOKE_ADDR(index_)),           \
        GEMMINI_SMOKE_MVIN);                                               \
  } while (0)

#define GEMMINI_STORE_ONE(custom_, index_)                                 \
  do {                                                                     \
    GEMMINI_SMOKE_ISSUE(                                                   \
        custom_, (uintptr_t)&gemmini_output[(index_)][0][0],               \
        GEMMINI_SMOKE_MATRIX_OPERAND(GEMMINI_SMOKE_ADDR(index_)),           \
        GEMMINI_SMOKE_MVOUT);                                              \
  } while (0)

static void initialize_data(void) {
  for (size_t row = 0; row < DIM; ++row) {
    for (size_t col = 0; col < DIM; ++col) {
      for (size_t accelerator = 0; accelerator < GEMMINI_COUNT;
           ++accelerator) {
        const int value =
            (int)((accelerator * 5u + row * 3u + col) % 15u) - 7;
        gemmini_input[accelerator][row][col] = (elem_t)value;
        gemmini_output[accelerator][row][col] = (elem_t)0x55;
      }
    }
  }

  const vpu_storage_t guard = vpu_float_to_storage(123.0f);
  vpu_input[0] = guard;
  vpu_input[VPU_SMOKE_VL + 1] = guard;
  vpu_output[0] = guard;
  vpu_output[VPU_SMOKE_VL + 1] = guard;
  for (size_t i = 0; i < VPU_SMOKE_VL; ++i) {
    vpu_input[i + 1] = vpu_float_to_storage((float)((int)i - 8));
    vpu_output[i + 1] = vpu_float_to_storage(-321.0f);
  }
}

static int verify_gemmini(void) {
  for (size_t accelerator = 0; accelerator < GEMMINI_COUNT; ++accelerator) {
    for (size_t row = 0; row < DIM; ++row) {
      for (size_t col = 0; col < DIM; ++col) {
        const elem_t expected = gemmini_input[accelerator][row][col];
        const elem_t actual = gemmini_output[accelerator][row][col];
        if (actual != expected) {
          printf("Gemmini%u mismatch at [%u,%u]: got=%d expected=%d\n",
                 (unsigned)accelerator, (unsigned)row, (unsigned)col,
                 (int)actual, (int)expected);
          return 1;
        }
      }
    }
  }
  return 0;
}

static int verify_vpu(uint64_t status) {
  if ((status & VPU_STATUS_ERROR_MASK) != 0) {
    printf("VPU interleave status error: 0x%llx\n",
           (unsigned long long)status);
    return 1;
  }

  const vpu_storage_t guard = vpu_float_to_storage(123.0f);
  if (vpu_output[0] != guard || vpu_output[VPU_SMOKE_VL + 1] != guard) {
    printf("VPU interleave store overwrote a guard element\n");
    return 1;
  }

  for (size_t i = 0; i < VPU_SMOKE_VL; ++i) {
    const float input = vpu_storage_to_float(vpu_input[i + 1]);
    const float expected = 2.0f * input + 1.0f;
    const float actual = vpu_storage_to_float(vpu_output[i + 1]);
    if (vpu_float_to_bits(actual) != vpu_float_to_bits(expected)) {
      printf("VPU interleave mismatch at %u: got=0x%x expected=0x%x\n",
             (unsigned)i, vpu_float_to_bits(actual),
             vpu_float_to_bits(expected));
      return 1;
    }
  }
  return 0;
}

int main(void) {
  initialize_data();
  vpu_cpu_memory_fence();

  /* Exercise every unicast route before filling any accelerator queue. */
  GEMMINI_FLUSH_ONE(custom0);
  GEMMINI_FLUSH_ONE(custom1);
  GEMMINI_FLUSH_ONE(custom2);
  GEMMINI_FLUSH_ONE(custom3);

  vpu_clear_status(VPU_CLEAR_ALL);
  vpu_set_vl(VPU_SMOKE_VL);
  vpu_write_gp(VPU_GP_SLOT, VPU_PING_INPUT_ADDR);
  vpu_write_gp(VPU_GP_OFFSET, 0);
  vpu_write_fp(VPU_FP_TWO, 2.0f);
  vpu_write_fp(VPU_FP_ONE, 1.0f);
  vpu_write_h(VPU_H_INPUT, (uintptr_t)&vpu_input[1]);
  vpu_write_h(VPU_H_OUTPUT, (uintptr_t)&vpu_output[1]);

  /*
   * Interleave custom0/funct64 traffic between custom0..3 Gemmini traffic.
   * The VPU scoreboard keeps the in-place operations behind the DMA load,
   * while the independent Gemmini queues continue to make progress.
   */
  GEMMINI_LOAD_ONE(custom0, 0);
  vpu_h_prefetch_v(VPU_GP_SLOT, VPU_GP_OFFSET, VPU_H_INPUT);

  GEMMINI_LOAD_ONE(custom1, 1);
  vpu_v_mul_vf(VPU_GP_SLOT, VPU_GP_SLOT, VPU_FP_TWO);

  GEMMINI_LOAD_ONE(custom2, 2);
  vpu_v_add_vf(VPU_GP_SLOT, VPU_GP_SLOT, VPU_FP_ONE);

  GEMMINI_LOAD_ONE(custom3, 3);
  vpu_h_store_v(VPU_GP_SLOT, VPU_GP_OFFSET, VPU_H_OUTPUT);

  /*
   * Do not leave dependent MVOUTs resident while C_FENCE waits for the VPU.
   * In the shared-reservation configuration that unnecessarily extends the
   * Gemmini store dependencies across the whole VPU drain and can trip the
   * reservation-station no-progress watchdog.  The load phase has already
   * exercised concurrent custom0/funct64 and custom0..3 traffic; drain it
   * before allocating the four dependent stores, as the existing Gemmini
   * multi-accelerator smoke tests do at command-stream boundaries.
   */
  const uint64_t vpu_status = vpu_fence();
  gemmini_smoke_memory_fence();

  GEMMINI_STORE_ONE(custom0, 0);
  GEMMINI_STORE_ONE(custom1, 1);
  GEMMINI_STORE_ONE(custom2, 2);
  GEMMINI_STORE_ONE(custom3, 3);

  gemmini_smoke_memory_fence();

  if (verify_vpu(vpu_status) || verify_gemmini()) {
    return 1;
  }

  printf("Gemmini0..3 + VPU interleave smoke test passed\n");
  return 0;
}
