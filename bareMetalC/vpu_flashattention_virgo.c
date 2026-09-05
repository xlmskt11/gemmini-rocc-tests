// See LICENSE for license details.

/*
 * Three-KV-block numerical regression for the software-only Virgo-style
 * FlashAttention schedule.  The blocks use different score patterns and
 * maxima of 2ln(2), 4ln(2), and 6ln(2), exercising both ACC halves, half-zero
 * reuse, and online output rescaling.  Their exact power-of-two weights give
 * a compact reference without the slow scalar expf implementation.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "include/gemmini_params.h"
#define VPU_ENABLE_GEMMINI_FLASHATTENTION 1
#include "vpu_kernels.h"

#ifndef FA_QK_PARTITION_AXIS
#define FA_QK_PARTITION_AXIS GEMMINI_PARTITION_AXIS_M
#endif
#ifndef FA_PV_PARTITION_AXIS
#define FA_PV_PARTITION_AXIS GEMMINI_PARTITION_AXIS_M
#endif

enum {
  FA_VIRGO_QUERY_ROWS = 128,
  FA_VIRGO_SEQUENCE = 600,
  FA_VIRGO_Q_DIM = 1,
  FA_VIRGO_VALUE_DIM = 1,
  FA_VIRGO_QUERY_BASE = FA_VIRGO_SEQUENCE - FA_VIRGO_QUERY_ROWS,
};

static elem_t queries[FA_VIRGO_QUERY_ROWS][FA_VIRGO_Q_DIM]
    __attribute__((aligned(64)));
static elem_t keys[FA_VIRGO_SEQUENCE][FA_VIRGO_Q_DIM]
    __attribute__((aligned(64)));
static elem_t values[FA_VIRGO_SEQUENCE][FA_VIRGO_VALUE_DIM]
    __attribute__((aligned(64)));
static float output[FA_VIRGO_QUERY_ROWS][FA_VIRGO_VALUE_DIM]
    __attribute__((aligned(64)));
static float expected_output[FA_VIRGO_QUERY_ROWS]
    __attribute__((aligned(64)));
static float causal_mask_workspace[VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS]
    __attribute__((aligned(64)));

static float fa_virgo_abs(float value) { return value < 0.0f ? -value : value; }

static int fa_virgo_finite(float value) {
  return (vpu_float_to_bits(value) & UINT32_C(0x7f800000)) !=
         UINT32_C(0x7f800000);
}

static int fa_virgo_high_score(size_t block, size_t key) {
  switch (block % 3u) {
  case 0u:
    return (key & 1u) != 0u;
  case 1u:
    return (key & 1u) == 0u;
  default:
    return (key & 2u) != 0u;
  }
}

static void fa_virgo_initialize(const vpu_flashattention_plan_t *plan) {
  float weighted_sum = 0.0f;
  float weight_sum = 0.0f;

  for (size_t row = 0; row < FA_VIRGO_QUERY_ROWS; ++row)
    queries[row][0] = (elem_t)vpu_float_to_bf16(1.0f);

  for (size_t key = 0; key < FA_VIRGO_SEQUENCE; ++key) {
    const size_t block = key / plan->kv_rows;
    const int high_score = fa_virgo_high_score(block, key);
    const int residue = (int)((key * 37u) % 251u);
    const float high_key = (float)(2u * (block + 1u));
    keys[key][0] =
        (elem_t)vpu_float_to_bf16(high_score ? high_key : 0.0f);
    values[key][0] =
        (elem_t)vpu_float_to_bf16((float)(residue - 125) * 0.0078125f);

    const float weight =
        high_score ? (float)(1u << (2u * (block + 1u))) : 1.0f;
    weighted_sum +=
        weight * vpu_bf16_to_float((uint16_t)values[key][0]);
    weight_sum += weight;
    if (key >= FA_VIRGO_QUERY_BASE)
      expected_output[key - FA_VIRGO_QUERY_BASE] =
          weighted_sum / weight_sum;
  }
}

int main(void) {
  const vpu_flashattention_config_t config = {
      .queries = &queries[0][0],
      .keys = &keys[0][0],
      .values = &values[0][0],
      .output = &output[0][0],
      .query_rows = FA_VIRGO_QUERY_ROWS,
      .sequence = FA_VIRGO_SEQUENCE,
      .q_dim = FA_VIRGO_Q_DIM,
      .k_dim = FA_VIRGO_Q_DIM,
      .value_dim = FA_VIRGO_VALUE_DIM,
      .score_scale = 0x1.62e43p-1f,
      .query_base = FA_VIRGO_QUERY_BASE,
      .query_stride = FA_VIRGO_Q_DIM,
      .key_stride = FA_VIRGO_Q_DIM,
      .value_stride = FA_VIRGO_VALUE_DIM,
      .output_stride = FA_VIRGO_VALUE_DIM,
      .gemmini_mask = (1u << VPU_MATRIX_PORTS) - 1u,
      .qk_partition_axis = FA_QK_PARTITION_AXIS,
      .pv_partition_axis = FA_PV_PARTITION_AXIS,
      .causal_mask_workspace = causal_mask_workspace,
      .causal_mask_workspace_elements =
          VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS,
  };

  vpu_flashattention_plan_t plan;
  const vpu_flashattention_status_t plan_status =
      vpu_flashattention_make_plan(&config, &plan);
  if (plan_status != VPU_FLASHATTENTION_OK) {
    printf("VIRGO_ATTN plan failed status=%u\n", (unsigned)plan_status);
    return 1;
  }

  const size_t kv_blocks =
      (FA_VIRGO_SEQUENCE + plan.kv_rows - 1u) / plan.kv_rows;
  printf("VIRGO_ATTN plan q_rows=%u kv_rows=%u kv_blocks=%u\n",
         plan.q_rows, plan.kv_rows, (unsigned)kv_blocks);
  if (plan.q_rows != FA_VIRGO_QUERY_ROWS || kv_blocks != 3u) {
    printf("VIRGO_ATTN expected one query block and three KV blocks\n");
    return 1;
  }
  fa_virgo_initialize(&plan);

  const vpu_flashattention_result_t result =
      vpu_flashattention_auto(&config);
  if (result.status != VPU_FLASHATTENTION_OK ||
      (result.vpu_status & VPU_STATUS_ERROR_MASK) != 0u ||
      result.stats.qk_jobs != kv_blocks || result.stats.pv_jobs != kv_blocks ||
      result.stats.causal_blocks != kv_blocks) {
    printf("VIRGO_ATTN kernel failed status=%u vpu_status=0x%llx "
           "fault_addr=0x%llx fault_info=0x%llx\n",
           (unsigned)result.status, (unsigned long long)result.vpu_status,
           (unsigned long long)vpu_read_fault_address(),
           (unsigned long long)vpu_read_fault_info());
    return 1;
  }

  int errors = 0;
  float max_error = 0.0f;
  for (size_t row = 0; row < FA_VIRGO_QUERY_ROWS; ++row) {
    const float expected = expected_output[row];
    const float error = fa_virgo_abs(output[row][0] - expected);
    if (error > max_error)
      max_error = error;
    if (!fa_virgo_finite(output[row][0]) || error > 1.0e-5f) {
      if (errors < 12) {
        printf("VIRGO_ATTN mismatch row=%u got=0x%08x expected=0x%08x\n",
               (unsigned)row, vpu_float_to_bits(output[row][0]),
               vpu_float_to_bits(expected));
      }
      ++errors;
    }
  }

  printf("VIRGO_ATTN %s errors=%d max_error=0x%08x "
         "qk_jobs=%llu pv_jobs=%llu status=0x%llx\n",
         errors == 0 ? "PASS" : "FAIL", errors,
         vpu_float_to_bits(max_error),
         (unsigned long long)result.stats.qk_jobs,
         (unsigned long long)result.stats.pv_jobs,
         (unsigned long long)result.vpu_status);
  return errors == 0 ? 0 : 1;
}
