// See LICENSE for license details.

/*
 * Minimal end-to-end contract test for a cooperative K split:
 *
 *   two Gemminis reduce A*B into one shared accumulator tile,
 *   the unique global-last writer leaves that FP32 tile in shared ACC SRAM,
 *   a grouped VPU program adds one in place, and
 *   VPU H_STORE returns the tile for comparison with a CPU reference.
 *
 * This source intentionally uses only the existing fusion LOOP_WS and VPU
 * command APIs. It is a runnable test only for a generated configuration with
 * at least two logical Gemmini endpoints and shared-partition ABI version 1.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "include/gemmini_params.h"
#define VPU_ENABLE_GEMMINI_FLASHATTENTION 1
#include "vpu_kernels.h"

#define K_CTO_MEMBERS 2u
#define K_CTO_GROUP_MASK ((1u << K_CTO_MEMBERS) - 1u)
#define K_CTO_GROUP_ID 0u
#define K_CTO_I_TILES 1u
#define K_CTO_J_TILES 1u
#define K_CTO_K_TILES K_CTO_MEMBERS
#define K_CTO_I (K_CTO_I_TILES * DIM)
#define K_CTO_J (K_CTO_J_TILES * DIM)
#define K_CTO_K (K_CTO_K_TILES * DIM)

enum {
  K_CTO_GP_ROW = 0,
  K_CTO_GP_HOST_OFFSET = 1,
  K_CTO_GP_ROW_LOOP = 15,
  K_CTO_FP_ONE = 0,
  K_CTO_H_OUTPUT = 0,
};

static elem_t input_a[K_CTO_I][K_CTO_K] __attribute__((aligned(64)));
static elem_t input_b[K_CTO_K][K_CTO_J] __attribute__((aligned(64)));
static float hardware_output[K_CTO_I][K_CTO_J]
    __attribute__((aligned(64)));
static float cpu_reference[K_CTO_I][K_CTO_J]
    __attribute__((aligned(64)));

static inline elem_t k_cto_encode(float value) {
  return (elem_t)vpu_float_to_bf16(value);
}

static inline float k_cto_decode(elem_t value) {
  return vpu_bf16_to_float((uint16_t)value);
}

static void k_cto_initialize(void) {
  for (size_t i = 0; i < K_CTO_I; ++i) {
    for (size_t k = 0; k < K_CTO_K; ++k) {
      input_a[i][k] = k_cto_encode((float)((int)((i + 2u * k) % 5u) - 2));
    }
  }
  for (size_t k = 0; k < K_CTO_K; ++k) {
    for (size_t j = 0; j < K_CTO_J; ++j) {
      input_b[k][j] = k_cto_encode((float)((int)((3u * k + j) % 5u) - 2));
    }
  }

  for (size_t i = 0; i < K_CTO_I; ++i) {
    for (size_t j = 0; j < K_CTO_J; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < K_CTO_K; ++k)
        sum += k_cto_decode(input_a[i][k]) * k_cto_decode(input_b[k][j]);
      cpu_reference[i][j] = sum + 1.0f;
      hardware_output[i][j] = -12345.0f;
    }
  }
}

static void k_cto_configure_gemminis(void) {
  for (unsigned custom = 0; custom < K_CTO_MEMBERS; ++custom) {
    vpu_fa_gemmini_flush(custom);
    vpu_fa_config_ex(custom, 1u, false);
    vpu_fa_config_st(custom);
    vpu_fa_config_ld(custom, K_CTO_K * sizeof(elem_t), 0u);
    vpu_fa_config_ld(custom, K_CTO_J * sizeof(elem_t), 1u);
    vpu_fa_config_ld(custom, 0u, 2u);
  }
}

static bool k_cto_issue_matmul(void) {
  for (unsigned custom = 0; custom < K_CTO_MEMBERS; ++custom) {
    gemmini_partition_plan_t plan;
    if (!gemmini_partition_plan_member(
            GEMMINI_PARTITION_AXIS_K, K_CTO_I_TILES, K_CTO_J_TILES,
            K_CTO_K_TILES, 0u, 0u, 0u, K_CTO_MEMBERS, custom, &plan))
      return false;

    const elem_t *const a_local =
        &input_a[0][plan.partition_offset * DIM];
    const elem_t *const b_local =
        &input_b[plan.partition_offset * DIM][0];

    VPU_FA_SHARED_LOOP_WS_AXIS(
        custom, K_CTO_GROUP_MASK, K_CTO_GROUP_ID,
        0u, VPU_FA_SPAD_END, VPU_FA_SCORE_BASE_ROW,
        GEMMINI_PARTITION_AXIS_K,
        plan.partition_extent, plan.aux_extent, plan.aux_pad,
        plan.partition_offset, plan.aux_offset,
        K_CTO_I_TILES, K_CTO_J_TILES, K_CTO_K_TILES,
        0u, 0u, 0u,
        a_local, b_local, NULL, NULL,
        K_CTO_K, K_CTO_J, 0u, 0u,
        false, false, false, false,
        false, VPU_FA_NO_ACTIVATION,
        0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
        true);
  }
  return true;
}

static void k_cto_issue_grouped_vpu_add_one(void) {
  vpu_fa_group_set_vector_stride(K_CTO_GROUP_ID, 0u);
  vpu_fa_group_write_fp(K_CTO_GROUP_ID, K_CTO_FP_ONE, 1.0f);
  vpu_fa_group_write_gp(K_CTO_GROUP_ID, K_CTO_GP_ROW,
                        VPU_FA_SCORE_BASE);
  vpu_fa_group_set_vl(K_CTO_GROUP_ID, DIM);
  vpu_fa_group_loop_start(K_CTO_GROUP_ID, K_CTO_GP_ROW_LOOP, DIM);
  vpu_fa_group_vector(K_CTO_GROUP_ID, VPU_OP_V_ADD_VF,
                      K_CTO_GP_ROW, K_CTO_GP_ROW, K_CTO_FP_ONE);
  vpu_fa_group_addi_gp(K_CTO_GROUP_ID, K_CTO_GP_ROW,
                       K_CTO_GP_ROW, DIM);
  vpu_fa_group_loop_end(K_CTO_GROUP_ID, K_CTO_GP_ROW_LOOP);
  vpu_fa_group_terminator(K_CTO_GROUP_ID);
}

static void k_cto_enqueue_output_store(void) {
  vpu_write_h(K_CTO_H_OUTPUT, (uintptr_t)&hardware_output[0][0]);
  vpu_set_vl(DIM);
  for (unsigned row = 0; row < DIM; ++row) {
    vpu_write_gp(K_CTO_GP_ROW,
                 VPU_FA_SCORE_BASE + row * DIM);
    vpu_write_gp(K_CTO_GP_HOST_OFFSET, row * DIM);
    vpu_h_store_v(K_CTO_GP_ROW, K_CTO_GP_HOST_OFFSET, K_CTO_H_OUTPUT);
  }
}

static int k_cto_run(void) {
  k_cto_initialize();
  vpu_publish_cpu_writes();
  vpu_clear_status(VPU_CLEAR_ALL);
  k_cto_configure_gemminis();

  if (!k_cto_issue_matmul()) {
    printf("vpu_k_partition_cto_smoke: partition planning failed\n");
    return 1;
  }
  k_cto_issue_grouped_vpu_add_one();
  k_cto_enqueue_output_store();

  const uint64_t status = vpu_fence();
  if ((status & VPU_STATUS_ERROR_MASK) != 0u) {
    printf("vpu_k_partition_cto_smoke: VPU status=0x%llx\n",
           (unsigned long long)status);
    return 1;
  }

  for (size_t i = 0; i < K_CTO_I; ++i) {
    for (size_t j = 0; j < K_CTO_J; ++j) {
      if (vpu_float_to_bits(hardware_output[i][j]) !=
          vpu_float_to_bits(cpu_reference[i][j])) {
        printf("vpu_k_partition_cto_smoke: mismatch i=%u j=%u "
               "got=0x%x expected=0x%x\n",
               (unsigned)i, (unsigned)j,
               vpu_float_to_bits(hardware_output[i][j]),
               vpu_float_to_bits(cpu_reference[i][j]));
        return 1;
      }
    }
  }

  printf("vpu_k_partition_cto_smoke: PASS (K split, shared ACC, VPU +1)\n");
  return 0;
}

int main(void) {
#if !defined(GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION) || \
    GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION != 1
  printf("vpu_k_partition_cto_smoke: SKIP (shared-axis ABI unavailable)\n");
  return 0;
#else
  if (VPU_MATRIX_PORTS < K_CTO_MEMBERS) {
    printf("vpu_k_partition_cto_smoke: SKIP (need at least two logical "
           "Gemmini endpoints)\n");
    return 0;
  }
  return k_cto_run();
#endif
}
