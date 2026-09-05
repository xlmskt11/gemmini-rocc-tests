// See LICENSE for license details.

/*
 * Numerical and scheduling driver for the reusable Gemmini--VPU causal
 * FlashAttention kernel in vpu_flashattention_kernel.h.  This file owns only
 * test vectors, the CPU references, and reporting; the hardware schedule is
 * selected from the runtime config passed to vpu_flashattention_auto().
 */

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* elem_t, DIM, and the BF16/FP32 Gemmini ABI must be visible before the
 * opt-in FlashAttention portion of vpu_kernels.h is parsed. */
#include "include/gemmini_params.h"
#ifndef VPU_ENABLE_GEMMINI_FLASHATTENTION
#define VPU_ENABLE_GEMMINI_FLASHATTENTION 1
#endif
#include "vpu_kernels.h"

#ifndef FA_QUERY_ROWS
#define FA_QUERY_ROWS 16u
#endif
#ifndef FA_SEQUENCE
#define FA_SEQUENCE 64u
#endif
#ifndef FA_HEAD_DIM
#define FA_HEAD_DIM 32u
#endif
#ifndef FA_Q_DIM
#define FA_Q_DIM FA_HEAD_DIM
#endif
#ifndef FA_K_DIM
#define FA_K_DIM FA_Q_DIM
#endif
#ifndef FA_V_DIM
#define FA_V_DIM FA_Q_DIM
#endif
#ifndef FA_QUERY_BASE
#define FA_QUERY_BASE \
  ((FA_SEQUENCE) >= (FA_QUERY_ROWS) ? (FA_SEQUENCE) - (FA_QUERY_ROWS) : 0u)
#endif
#ifndef FA_GEMMINI_MASK
#define FA_GEMMINI_MASK ((1u << VPU_MATRIX_PORTS) - 1u)
#endif
#ifndef FA_QK_PARTITION_AXIS
#define FA_QK_PARTITION_AXIS GEMMINI_PARTITION_AXIS_M
#endif
#ifndef FA_PV_PARTITION_AXIS
#define FA_PV_PARTITION_AXIS GEMMINI_PARTITION_AXIS_M
#endif
#ifndef FA_SCORE_SCALE
/* Nonstandard positive scale exercises the runtime scale path. */
#define FA_SCORE_SCALE 0.15625f
#endif
#ifndef FA_CHECK_REFERENCE
#define FA_CHECK_REFERENCE 0
#endif

static elem_t queries[FA_QUERY_ROWS][FA_Q_DIM]
    __attribute__((aligned(64)));
static elem_t keys[FA_SEQUENCE][FA_K_DIM]
    __attribute__((aligned(64)));
static elem_t values[FA_SEQUENCE][FA_V_DIM]
    __attribute__((aligned(64)));
static float hardware_output[FA_QUERY_ROWS][FA_V_DIM]
    __attribute__((aligned(64)));

/* The reusable kernel intentionally owns no static DMA source.  Supplying
 * this workspace makes its ownership/lifetime explicit and keeps its
 * asynchronous mask prefetch valid through the final fence. */
static float causal_mask_workspace[VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS]
    __attribute__((aligned(64)));

#if FA_CHECK_REFERENCE
static float fused_reference[FA_QUERY_ROWS][FA_V_DIM]
    __attribute__((aligned(64)));
static float mathematical_gold[FA_QUERY_ROWS][FA_V_DIM]
    __attribute__((aligned(64)));
static float reference_scores[FA_SEQUENCE]
    __attribute__((aligned(64)));
static float reference_probabilities[FA_SEQUENCE]
    __attribute__((aligned(64)));
static float reference_vector[FA_V_DIM]
    __attribute__((aligned(64)));

static float fa_input_value(unsigned kind, unsigned row, unsigned column) {
  const int centered = (int)((row * (7u + kind * 2u) +
                              column * (3u + kind) + kind * 5u) % 29u) - 14;
  const float scale = kind == 2u ? 0.03125f : 0.0625f;
  return (float)centered * scale;
}

static inline elem_t fa_encode_bf16(float value) {
  return (elem_t)vpu_float_to_bf16(value);
}

static inline float fa_decode_bf16(elem_t value) {
  return vpu_bf16_to_float((uint16_t)value);
}

static inline float fa_round_bf16(float value) {
  return vpu_bf16_to_float(vpu_float_to_bf16(value));
}

static void fa_initialize_inputs(void) {
  for (size_t row = 0; row < FA_QUERY_ROWS; ++row) {
    for (size_t column = 0; column < FA_Q_DIM; ++column) {
      queries[row][column] = fa_encode_bf16(
          fa_input_value(0, FA_QUERY_BASE + (unsigned)row,
                         (unsigned)column));
    }
  }
  for (size_t row = 0; row < FA_SEQUENCE; ++row) {
    for (size_t column = 0; column < FA_K_DIM; ++column) {
      keys[row][column] = fa_encode_bf16(
          fa_input_value(1, (unsigned)row, (unsigned)column));
    }
    for (size_t column = 0; column < FA_V_DIM; ++column) {
      values[row][column] = fa_encode_bf16(
          fa_input_value(2, (unsigned)row, (unsigned)column));
    }
  }
}

static float fa_dot_qk(size_t query_row, size_t key_row) {
  float sum = 0.0f;
  for (size_t depth = 0; depth < FA_Q_DIM; ++depth) {
    sum = fmaf(fa_decode_bf16(queries[query_row][depth]),
               fa_decode_bf16(keys[key_row][depth]), sum);
  }
  return sum * FA_SCORE_SCALE;
}

/* The fused reference deliberately follows the selected KV blocking and
 * rounds P to BF16 before PV, matching the matrix bridge.  The mathematical
 * reference evaluates the ordinary stable-softmax expression separately. */
static void fa_compute_references(
    const vpu_flashattention_plan_t *plan) {
  for (size_t query = 0; query < FA_QUERY_ROWS; ++query) {
    float online_m = -INFINITY;
    float online_l = 0.0f;
    memset(reference_vector, 0, sizeof(reference_vector));

    for (size_t kv_start = 0; kv_start < FA_SEQUENCE;
         kv_start += plan->kv_rows) {
      const size_t kv_rows = FA_SEQUENCE - kv_start < plan->kv_rows
          ? FA_SEQUENCE - kv_start : plan->kv_rows;
      float block_max = -INFINITY;
      for (size_t key_offset = 0; key_offset < kv_rows; ++key_offset) {
        const size_t key = kv_start + key_offset;
        reference_scores[key_offset] =
            key <= FA_QUERY_BASE + query ? fa_dot_qk(query, key) : -INFINITY;
        block_max = fmaxf(block_max, reference_scores[key_offset]);
      }

      const float new_m = fmaxf(online_m, block_max);
      const float alpha = kv_start == 0 ? 0.0f : expf(online_m - new_m);
      float p_sum = 0.0f;
      for (size_t key_offset = 0; key_offset < kv_rows; ++key_offset) {
        reference_probabilities[key_offset] =
            expf(reference_scores[key_offset] - new_m);
        p_sum += reference_probabilities[key_offset];
      }

      online_l = online_l * alpha + p_sum;
      for (size_t column = 0; column < FA_V_DIM; ++column) {
        reference_vector[column] *= alpha;
        float pv = 0.0f;
        for (size_t key_offset = 0; key_offset < kv_rows; ++key_offset) {
          pv = fmaf(fa_round_bf16(reference_probabilities[key_offset]),
                    fa_decode_bf16(values[kv_start + key_offset][column]),
                    pv);
        }
        reference_vector[column] += pv;
      }
      online_m = new_m;
    }

    for (size_t column = 0; column < FA_V_DIM; ++column) {
      fused_reference[query][column] = reference_vector[column] / online_l;
    }

    float max_score = -INFINITY;
    for (size_t key = 0; key < FA_SEQUENCE; ++key) {
      reference_scores[key] = key <= FA_QUERY_BASE + query
          ? fa_dot_qk(query, key) : -INFINITY;
      max_score = fmaxf(max_score, reference_scores[key]);
    }
    float denominator = 0.0f;
    memset(reference_vector, 0, sizeof(reference_vector));
    for (size_t key = 0; key < FA_SEQUENCE; ++key) {
      const float probability = expf(reference_scores[key] - max_score);
      denominator += probability;
      for (size_t column = 0; column < FA_V_DIM; ++column) {
        reference_vector[column] = fmaf(
            probability, fa_decode_bf16(values[key][column]),
            reference_vector[column]);
      }
    }
    for (size_t column = 0; column < FA_V_DIM; ++column) {
      mathematical_gold[query][column] =
          reference_vector[column] / denominator;
    }
  }
}

static bool fa_close(float actual, float expected, float rtol, float atol) {
  return isfinite(actual) &&
      fabsf(actual - expected) <= atol + rtol * fabsf(expected);
}

static int fa_check_results(void) {
  int errors = 0;
  float max_fused_error = 0.0f;
  float max_gold_error = 0.0f;
  for (size_t row = 0; row < FA_QUERY_ROWS; ++row) {
    for (size_t column = 0; column < FA_V_DIM; ++column) {
      const float actual = hardware_output[row][column];
      const float fused_error = fabsf(actual - fused_reference[row][column]);
      const float gold_error =
          fabsf(actual - mathematical_gold[row][column]);
      max_fused_error = fmaxf(max_fused_error, fused_error);
      max_gold_error = fmaxf(max_gold_error, gold_error);
      if (!fa_close(actual, fused_reference[row][column], 8.0e-3f,
                    5.0e-4f) ||
          !fa_close(actual, mathematical_gold[row][column], 1.5e-2f,
                    1.0e-3f)) {
        if (errors < 12) {
          /* The bare-metal printf intentionally has no floating-point
           * formatter; print exact IEEE-754 payloads instead of a literal
           * unsupported %g conversion. */
          printf("mismatch row=%u col=%u got=0x%08x "
                 "fused_ref=0x%08x gold=0x%08x\n",
                 (unsigned)row, (unsigned)column,
                 vpu_float_to_bits(actual),
                 vpu_float_to_bits(fused_reference[row][column]),
                 vpu_float_to_bits(mathematical_gold[row][column]));
        }
        ++errors;
      }
    }
  }
  printf("FLASH_ATTN max_abs_error_bits fused_bf16=0x%08x "
         "mathematical=0x%08x\n",
         vpu_float_to_bits(max_fused_error),
         vpu_float_to_bits(max_gold_error));
  return errors;
}
#endif  // FA_CHECK_REFERENCE

int main(void) {
  const vpu_flashattention_config_t config = {
      .queries = &queries[0][0],
      .keys = &keys[0][0],
      .values = &values[0][0],
      .output = &hardware_output[0][0],
      .query_rows = FA_QUERY_ROWS,
      .sequence = FA_SEQUENCE,
      .q_dim = FA_Q_DIM,
      .k_dim = FA_K_DIM,
      .value_dim = FA_V_DIM,
      .score_scale = FA_SCORE_SCALE,
      .query_base = FA_QUERY_BASE,
      .query_stride = FA_Q_DIM,
      .key_stride = FA_K_DIM,
      .value_stride = FA_V_DIM,
      .output_stride = FA_V_DIM,
      .gemmini_mask = FA_GEMMINI_MASK,
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
    printf("FLASH_ATTN plan failed status=%u\n", (unsigned)plan_status);
    return 1;
  }

  printf("FLASH_ATTN q=[%u,%u)x%u k=%ux%u v=%ux%u mask=0x%x\n",
         (unsigned)FA_QUERY_BASE,
         (unsigned)(FA_QUERY_BASE + FA_QUERY_ROWS), (unsigned)FA_Q_DIM,
         (unsigned)FA_SEQUENCE, (unsigned)FA_K_DIM,
         (unsigned)FA_SEQUENCE, (unsigned)FA_V_DIM, plan.gemmini_mask);
  printf("FLASH_ATTN tiler tile_I=%u tile_J=%u tile_K=%u "
         "blocks=query:%u kv:%u depth:%u elements\n",
         plan.q_tiles, plan.kv_tiles, plan.qk_tiles,
         plan.q_rows, plan.kv_rows, plan.qk_depth);

#if FA_CHECK_REFERENCE
  fa_initialize_inputs();
  fa_compute_references(&plan);
#endif

  const vpu_flashattention_result_t result =
      vpu_flashattention_auto(&config);
  printf("FLASH_ATTN scheduler causal_blocks=%llu rectangular_blocks=%llu "
         "skipped_future_blocks=%llu\n",
         (unsigned long long)result.stats.causal_blocks,
         (unsigned long long)result.stats.rectangular_blocks,
         (unsigned long long)result.stats.skipped_future_blocks);
  printf("FLASH_ATTN hwloop qk_regions=%llu qk_rows=%llu "
         "normalize_regions=%llu normalize_rows=%llu\n",
         (unsigned long long)result.stats.qk_row_loop_regions,
         (unsigned long long)result.stats.qk_row_loop_rows,
         (unsigned long long)result.stats.normalize_row_loop_regions,
         (unsigned long long)result.stats.normalize_row_loop_rows);
  printf("FLASH_ATTN mask_hwloop regions=%llu rows=%llu "
         "diagonal_vectors=%llu future_vectors=%llu\n",
         (unsigned long long)result.stats.mask_row_loop_regions,
         (unsigned long long)result.stats.mask_row_loop_rows,
         (unsigned long long)result.stats.mask_diagonal_vectors,
         (unsigned long long)result.stats.mask_future_vectors);

  if (result.status != VPU_FLASHATTENTION_OK ||
      (result.vpu_status & VPU_STATUS_ERROR_MASK) != 0) {
    printf("FLASH_ATTN kernel failed status=%u vpu_status=0x%llx "
           "fault_addr=0x%llx fault_info=0x%llx\n",
           (unsigned)result.status,
           (unsigned long long)result.vpu_status,
           (unsigned long long)vpu_read_fault_address(),
           (unsigned long long)vpu_read_fault_info());
    return 1;
  }

#if FA_CHECK_REFERENCE
  const int errors = fa_check_results();
  printf("FLASH_ATTN %s errors=%d status=0x%llx\n",
         errors == 0 ? "PASS" : "FAIL", errors,
         (unsigned long long)result.vpu_status);
  return errors == 0 ? 0 : 1;
#else
  printf("FLASH_ATTN DONE reference=skipped status=0x%llx\n",
         (unsigned long long)result.vpu_status);
  return 0;
#endif
}
