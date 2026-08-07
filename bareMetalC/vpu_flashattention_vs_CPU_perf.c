// See LICENSE for license details.

/*
 * Scalar Rocket CPU versus Gemmini--VPU causal FlashAttention benchmark.
 *
 * CPU timing covers a conventional stable causal-attention calculation over
 * BF16 Q/K/V inputs with FP32 FMA accumulation and libm expf.  Fusion timing
 * covers the complete public API call, including planning/preflight, causal
 * mask preparation, CPU-write publication, Gemmini flush/configuration,
 * grouped QK/VPU/PV work, output DMA, and the final VPU fence.
 *
 * Input generation, UART output, result comparison, checksums, and VPU perf
 * counter reads are deliberately outside both timed regions.
 */

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/gemmini_params.h"
#ifndef VPU_ENABLE_GEMMINI_FLASHATTENTION
#define VPU_ENABLE_GEMMINI_FLASHATTENTION 1
#endif
#include "vpu_kernels.h"

#ifndef FA_QUERY_ROWS
#define FA_QUERY_ROWS 256u
#endif
#ifndef FA_SEQUENCE
#define FA_SEQUENCE 256u
#endif
#ifndef FA_HEAD_DIM
#define FA_HEAD_DIM 64u
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
#define FA_GEMMINI_MASK 0xfu
#endif

#ifndef FA_PERF_WARMUPS
#define FA_PERF_WARMUPS 0u
#endif
#ifndef FA_PERF_REPEATS
#define FA_PERF_REPEATS 1u
#endif
#ifndef FA_PERF_CHECK
#define FA_PERF_CHECK 0
#endif

#if FA_QUERY_ROWS == 0 || FA_SEQUENCE == 0 || FA_Q_DIM == 0 || \
    FA_K_DIM == 0 || FA_V_DIM == 0
#error "FlashAttention performance dimensions must be nonzero"
#endif
#if FA_PERF_REPEATS < 1
#error "FA_PERF_REPEATS must be at least one"
#endif
#if FA_PERF_WARMUPS < 0
#error "FA_PERF_WARMUPS cannot be negative"
#endif
#if FA_PERF_CHECK != 0 && FA_PERF_CHECK != 1
#error "FA_PERF_CHECK must be zero or one"
#endif

static elem_t perf_queries[FA_QUERY_ROWS][FA_Q_DIM]
    __attribute__((aligned(64)));
static elem_t perf_keys[FA_SEQUENCE][FA_K_DIM]
    __attribute__((aligned(64)));
static elem_t perf_values[FA_SEQUENCE][FA_V_DIM]
    __attribute__((aligned(64)));
static float perf_cpu_output[FA_QUERY_ROWS][FA_V_DIM]
    __attribute__((aligned(64)));
static float perf_fusion_output[FA_QUERY_ROWS][FA_V_DIM]
    __attribute__((aligned(64)));
static float perf_score_scratch[FA_SEQUENCE]
    __attribute__((aligned(64)));
static float perf_causal_mask_workspace[
    VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS]
    __attribute__((aligned(64)));

/* Consuming both outputs through this externally visible object prevents an
 * optimizer from proving that a timed scalar calculation is disposable. */
static volatile uint64_t perf_checksum_sink;

static inline uint64_t perf_read_cycles(void) {
  uint64_t cycles;
  __asm__ volatile("rdcycle %0" : "=r"(cycles) : : "memory");
  return cycles;
}

static inline elem_t perf_encode_bf16(float value) {
  return (elem_t)vpu_float_to_bf16(value);
}

static inline float perf_decode_bf16(elem_t value) {
  return vpu_bf16_to_float((uint16_t)value);
}

static float perf_input_value(unsigned kind, size_t row, size_t column) {
  const int centered =
      (int)((row * (7u + kind * 2u) + column * (3u + kind) +
             kind * 5u) % 29u) - 14;
  const float scale = kind == 2u ? 0.03125f : 0.0625f;
  return (float)centered * scale;
}

static void perf_initialize_inputs(void) {
  for (size_t row = 0; row < FA_QUERY_ROWS; ++row) {
    for (size_t column = 0; column < FA_Q_DIM; ++column) {
      perf_queries[row][column] = perf_encode_bf16(
          perf_input_value(0u, FA_QUERY_BASE + row, column));
    }
  }
  for (size_t row = 0; row < FA_SEQUENCE; ++row) {
    for (size_t column = 0; column < FA_K_DIM; ++column) {
      perf_keys[row][column] =
          perf_encode_bf16(perf_input_value(1u, row, column));
    }
    for (size_t column = 0; column < FA_V_DIM; ++column) {
      perf_values[row][column] =
          perf_encode_bf16(perf_input_value(2u, row, column));
    }
  }
  memset(perf_cpu_output, 0, sizeof(perf_cpu_output));
  memset(perf_fusion_output, 0, sizeof(perf_fusion_output));
  memset(perf_causal_mask_workspace, 0,
         sizeof(perf_causal_mask_workspace));
}

/*
 * Conventional scalar causal attention.  It computes every visible QK score
 * once, retains those scores in a one-row scratch buffer, accumulates
 * unnormalized exp(score-max)*V, and normalizes O at the end.  This is the
 * mathematical reference, not a cycle-by-cycle model of blockwise VPU EXP or
 * the BF16 P conversion at the Gemmini matrix bridge.
 */
__attribute__((noinline))
static void perf_cpu_causal_attention(
    const vpu_flashattention_config_t *config, float *output) {
  const float score_scale = 1.0f / sqrtf((float)config->q_dim);

  for (size_t query = 0; query < config->query_rows; ++query) {
    const size_t global_query = config->query_base + query;
    const size_t visible_keys = global_query + 1u < config->sequence
        ? global_query + 1u : config->sequence;
    const elem_t *query_row =
        config->queries + query * config->query_stride;
    float row_max = -INFINITY;

    for (size_t key = 0; key < visible_keys; ++key) {
      const elem_t *key_row = config->keys + key * config->key_stride;
      float dot = 0.0f;
      for (size_t depth = 0; depth < config->q_dim; ++depth) {
        dot = fmaf(perf_decode_bf16(query_row[depth]),
                   perf_decode_bf16(key_row[depth]), dot);
      }
      const float score = dot * score_scale;
      perf_score_scratch[key] = score;
      row_max = fmaxf(row_max, score);
    }

    float *output_row = output + query * config->output_stride;
    for (size_t column = 0; column < config->value_dim; ++column)
      output_row[column] = 0.0f;

    float denominator = 0.0f;
    for (size_t key = 0; key < visible_keys; ++key) {
      const float probability = expf(perf_score_scratch[key] - row_max);
      const elem_t *value_row =
          config->values + key * config->value_stride;
      denominator += probability;
      for (size_t column = 0; column < config->value_dim; ++column) {
        output_row[column] =
            fmaf(probability, perf_decode_bf16(value_row[column]),
                 output_row[column]);
      }
    }

    const float reciprocal_denominator = 1.0f / denominator;
    for (size_t column = 0; column < config->value_dim; ++column)
      output_row[column] *= reciprocal_denominator;
  }
}

static uint64_t perf_output_checksum(const float *output,
                                     size_t rows, size_t columns,
                                     size_t stride) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < columns; ++column) {
      const uint32_t bits = vpu_float_to_bits(output[row * stride + column]);
      hash ^= bits;
      hash *= UINT64_C(1099511628211);
    }
  }
  return hash;
}

typedef struct {
  uint64_t whole;
  uint64_t thousandths;
  bool valid;
} perf_ratio_t;

static perf_ratio_t perf_ratio_thousandths(uint64_t numerator,
                                           uint64_t denominator) {
  perf_ratio_t ratio = {0u, 0u, denominator != 0u};
  if (denominator == 0u) return ratio;
  ratio.whole = numerator / denominator;
  ratio.thousandths =
      (((numerator % denominator) * 1000u) + denominator / 2u) /
      denominator;
  if (ratio.thousandths == 1000u) {
    ++ratio.whole;
    ratio.thousandths = 0u;
  }
  return ratio;
}

static uint64_t perf_rounded_average(uint64_t total, uint64_t count) {
  return total / count +
      ((total % count) * 2u >= count ? 1u : 0u);
}

static void perf_print_ratio(const char *name, perf_ratio_t ratio,
                             const char *suffix) {
  if (!ratio.valid) {
    printf("%s=N/A", name);
    return;
  }
  printf("%s=%llu.%03llu%s", name,
         (unsigned long long)ratio.whole,
         (unsigned long long)ratio.thousandths, suffix);
}

#if FA_PERF_CHECK
static int perf_check_results(const vpu_flashattention_config_t *config,
                              float *max_abs_error) {
  int errors = 0;
  const float *cpu_output = &perf_cpu_output[0][0];
  const float *fusion_output = &perf_fusion_output[0][0];
  *max_abs_error = 0.0f;
  for (size_t row = 0; row < config->query_rows; ++row) {
    for (size_t column = 0; column < config->value_dim; ++column) {
      const float expected =
          cpu_output[row * config->output_stride + column];
      const float actual =
          fusion_output[row * config->output_stride + column];
      const float error = fabsf(actual - expected);
      *max_abs_error = fmaxf(*max_abs_error, error);
      if (!isfinite(actual) ||
          error > 1.0e-3f + 1.5e-2f * fabsf(expected)) {
        if (errors < 12) {
          printf("FLASH_ATTN_PERF mismatch row=%u col=%u "
                 "cpu_bits=0x%x fusion_bits=0x%x abs_error_bits=0x%x\n",
                 (unsigned)row, (unsigned)column,
                 vpu_float_to_bits(expected), vpu_float_to_bits(actual),
                 vpu_float_to_bits(error));
        }
        ++errors;
      }
    }
  }
  return errors;
}
#endif

typedef struct {
  uint64_t cycles;
  uint64_t busy_cycles;
  uint64_t sfu_busy_cycles;
  uint64_t dma_read_bytes;
  uint64_t dma_write_bytes;
  uint64_t dma_exec_overlap_cycles;
  uint64_t bank_conflict_stall_cycles;
  uint64_t hazard_stall_cycles;
} perf_vpu_counters_t;

/* Snapshot immediately after the final timed auto() call.  Read the free-
 * running VPU cycle counter first; it is a VPU-local diagnostic and is not a
 * replacement for the rdcycle fusion E2E measurement. */
static perf_vpu_counters_t perf_snapshot_vpu_counters(void) {
  perf_vpu_counters_t counters;
  counters.cycles = vpu_read_perf(VPU_PERF_CYCLES);
  counters.busy_cycles = vpu_read_perf(VPU_PERF_BUSY_CYCLES);
  counters.sfu_busy_cycles = vpu_read_perf(VPU_PERF_SFU_BUSY_CYCLES);
  counters.dma_read_bytes = vpu_read_perf(VPU_PERF_DMA_READ_BYTES);
  counters.dma_write_bytes = vpu_read_perf(VPU_PERF_DMA_WRITE_BYTES);
  counters.dma_exec_overlap_cycles =
      vpu_read_perf(VPU_PERF_DMA_EXEC_OVERLAP_CYCLES);
  counters.bank_conflict_stall_cycles =
      vpu_read_perf(VPU_PERF_BANK_CONFLICT_STALL_CYCLES);
  counters.hazard_stall_cycles =
      vpu_read_perf(VPU_PERF_HAZARD_STALL_CYCLES);
  return counters;
}

static int perf_check_fusion_result(
    const char *phase, unsigned iteration,
    const vpu_flashattention_result_t *result) {
  if (result->status == VPU_FLASHATTENTION_OK &&
      (result->vpu_status & VPU_STATUS_ERROR_MASK) == 0u)
    return 0;

  printf("FLASH_ATTN_PERF %s iteration=%u failed "
         "status=%u vpu_status=0x%llx fault_addr=0x%llx "
         "fault_info=0x%llx\n",
         phase, iteration, (unsigned)result->status,
         (unsigned long long)result->vpu_status,
         (unsigned long long)vpu_read_fault_address(),
         (unsigned long long)vpu_read_fault_info());
  return 1;
}

int main(void) {
  const vpu_flashattention_config_t config = {
      .queries = &perf_queries[0][0],
      .keys = &perf_keys[0][0],
      .values = &perf_values[0][0],
      .output = &perf_fusion_output[0][0],
      .query_rows = FA_QUERY_ROWS,
      .sequence = FA_SEQUENCE,
      .q_dim = FA_Q_DIM,
      .k_dim = FA_K_DIM,
      .value_dim = FA_V_DIM,
      .query_base = FA_QUERY_BASE,
      .query_stride = FA_Q_DIM,
      .key_stride = FA_K_DIM,
      .value_stride = FA_V_DIM,
      .output_stride = FA_V_DIM,
      .gemmini_mask = FA_GEMMINI_MASK,
      .causal_mask_workspace = perf_causal_mask_workspace,
      .causal_mask_workspace_elements =
          VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS,
  };

  vpu_flashattention_plan_t plan;
  const vpu_flashattention_status_t plan_status =
      vpu_flashattention_make_plan(&config, &plan);
  if (plan_status != VPU_FLASHATTENTION_OK) {
    printf("FLASH_ATTN_PERF plan failed status=%u\n",
           (unsigned)plan_status);
    return 1;
  }

  perf_initialize_inputs();
  printf("FLASH_ATTN_PERF metric=target_cycles "
         "cpu=scalar_bf16_input_fp32_fma_libm "
         "fusion=auto_e2e_plan_flush_qk_vpu_pv_store_fence "
         "cache_state=post_input_cpu_warm\n");
  printf("FLASH_ATTN_PERF q=[%u,%u)x%u k=%ux%u v=%ux%u "
         "gemmini_mask=0x%x warmups=%u repeats=%u check=%u\n",
         (unsigned)config.query_base,
         (unsigned)(config.query_base + config.query_rows),
         (unsigned)config.q_dim, (unsigned)config.sequence,
         (unsigned)config.k_dim, (unsigned)config.sequence,
         (unsigned)config.value_dim, config.gemmini_mask,
         (unsigned)FA_PERF_WARMUPS, (unsigned)FA_PERF_REPEATS,
         (unsigned)FA_PERF_CHECK);
  printf("FLASH_ATTN_PERF plan tile_I=%u tile_J=%u tile_K=%u "
         "q_block_rows=%u kv_block_rows=%u qk_depth=%u\n",
         plan.q_tiles, plan.kv_tiles, plan.qk_tiles,
         plan.q_rows, plan.kv_rows, plan.qk_depth);

  vpu_flashattention_result_t fusion_result;
  memset(&fusion_result, 0, sizeof(fusion_result));

#if FA_PERF_WARMUPS > 0
  for (unsigned iteration = 0u; iteration < FA_PERF_WARMUPS; ++iteration) {
    perf_cpu_causal_attention(&config, &perf_cpu_output[0][0]);
    perf_checksum_sink ^= perf_output_checksum(
        &perf_cpu_output[0][0], config.query_rows,
        config.value_dim, config.output_stride);
    fusion_result = vpu_flashattention_auto(&config);
    if (perf_check_fusion_result("warmup", iteration, &fusion_result))
      return 1;
    perf_checksum_sink ^= perf_output_checksum(
        &perf_fusion_output[0][0], config.query_rows,
        config.value_dim, config.output_stride);
  }
#endif

  uint64_t cpu_min_cycles = UINT64_MAX;
  uint64_t cpu_total_cycles = 0u;
  uint64_t fusion_min_cycles = UINT64_MAX;
  uint64_t fusion_total_cycles = 0u;
  perf_vpu_counters_t last_vpu_counters;
  memset(&last_vpu_counters, 0, sizeof(last_vpu_counters));

  for (unsigned iteration = 0u; iteration < FA_PERF_REPEATS; ++iteration) {
    const uint64_t cpu_start = perf_read_cycles();
    perf_cpu_causal_attention(&config, &perf_cpu_output[0][0]);
    const uint64_t cpu_cycles = perf_read_cycles() - cpu_start;
    if (cpu_cycles < cpu_min_cycles) cpu_min_cycles = cpu_cycles;
    cpu_total_cycles += cpu_cycles;
    perf_checksum_sink ^= perf_output_checksum(
        &perf_cpu_output[0][0], config.query_rows,
        config.value_dim, config.output_stride);

    const uint64_t fusion_start = perf_read_cycles();
    fusion_result = vpu_flashattention_auto(&config);
    const uint64_t fusion_cycles = perf_read_cycles() - fusion_start;
    if (perf_check_fusion_result("measure", iteration, &fusion_result))
      return 1;
    if (fusion_cycles < fusion_min_cycles)
      fusion_min_cycles = fusion_cycles;
    fusion_total_cycles += fusion_cycles;
    if (iteration + 1u == FA_PERF_REPEATS)
      last_vpu_counters = perf_snapshot_vpu_counters();
  }

  perf_checksum_sink ^= perf_output_checksum(
      &perf_fusion_output[0][0], config.query_rows,
      config.value_dim, config.output_stride);

  const uint64_t cpu_average_cycles = perf_rounded_average(
      cpu_total_cycles, FA_PERF_REPEATS);
  const uint64_t fusion_average_cycles = perf_rounded_average(
      fusion_total_cycles, FA_PERF_REPEATS);
  const perf_ratio_t min_speedup =
      perf_ratio_thousandths(cpu_min_cycles, fusion_min_cycles);
  const perf_ratio_t average_speedup =
      perf_ratio_thousandths(cpu_average_cycles, fusion_average_cycles);
  const perf_ratio_t cpu_cycles_per_query =
      perf_ratio_thousandths(cpu_min_cycles, config.query_rows);
  const perf_ratio_t fusion_cycles_per_query =
      perf_ratio_thousandths(fusion_min_cycles, config.query_rows);

  printf("FLASH_ATTN_PERF cpu_cycles_min=%llu cpu_cycles_avg=%llu "
         "fusion_e2e_cycles_min=%llu fusion_e2e_cycles_avg=%llu ",
         (unsigned long long)cpu_min_cycles,
         (unsigned long long)cpu_average_cycles,
         (unsigned long long)fusion_min_cycles,
         (unsigned long long)fusion_average_cycles);
  perf_print_ratio("cpu_over_fusion_min", min_speedup, "x");
  printf(" ");
  perf_print_ratio("cpu_over_fusion_avg", average_speedup, "x");
  printf("\n");

  printf("FLASH_ATTN_PERF ");
  perf_print_ratio("cpu_cycles_per_query_min", cpu_cycles_per_query, "");
  printf(" ");
  perf_print_ratio("fusion_e2e_cycles_per_query_min",
                   fusion_cycles_per_query, "");
  printf("\n");

  printf("FLASH_ATTN_PERF scheduler qk_jobs=%llu pv_jobs=%llu "
         "gemmini_job_steps=%llu causal_blocks=%llu "
         "rectangular_blocks=%llu skipped_future_blocks=%llu\n",
         (unsigned long long)fusion_result.stats.qk_jobs,
         (unsigned long long)fusion_result.stats.pv_jobs,
         (unsigned long long)fusion_result.stats.gemmini_job_steps,
         (unsigned long long)fusion_result.stats.causal_blocks,
         (unsigned long long)fusion_result.stats.rectangular_blocks,
         (unsigned long long)fusion_result.stats.skipped_future_blocks);
  printf("FLASH_ATTN_PERF hwloop qk_regions=%llu qk_rows=%llu "
         "mask_regions=%llu mask_rows=%llu diagonal_vectors=%llu "
         "future_vectors=%llu normalize_regions=%llu normalize_rows=%llu\n",
         (unsigned long long)fusion_result.stats.qk_row_loop_regions,
         (unsigned long long)fusion_result.stats.qk_row_loop_rows,
         (unsigned long long)fusion_result.stats.mask_row_loop_regions,
         (unsigned long long)fusion_result.stats.mask_row_loop_rows,
         (unsigned long long)fusion_result.stats.mask_diagonal_vectors,
         (unsigned long long)fusion_result.stats.mask_future_vectors,
         (unsigned long long)fusion_result.stats.normalize_row_loop_regions,
         (unsigned long long)fusion_result.stats.normalize_row_loop_rows);

  printf("FLASH_ATTN_PERF counter_scope=vpu_last_iteration_only_not_e2e "
         "vpu_counter_cycles=%llu vpu_busy_cycles=%llu "
         "sfu_busy_cycles=%llu dma_read_bytes=%llu dma_write_bytes=%llu "
         "dma_exec_overlap_cycles=%llu bank_conflict_stall_cycles=%llu "
         "hazard_stall_cycles=%llu\n",
         (unsigned long long)last_vpu_counters.cycles,
         (unsigned long long)last_vpu_counters.busy_cycles,
         (unsigned long long)last_vpu_counters.sfu_busy_cycles,
         (unsigned long long)last_vpu_counters.dma_read_bytes,
         (unsigned long long)last_vpu_counters.dma_write_bytes,
         (unsigned long long)last_vpu_counters.dma_exec_overlap_cycles,
         (unsigned long long)last_vpu_counters.bank_conflict_stall_cycles,
         (unsigned long long)last_vpu_counters.hazard_stall_cycles);

  const uint64_t cpu_checksum = perf_output_checksum(
      &perf_cpu_output[0][0], config.query_rows,
      config.value_dim, config.output_stride);
  const uint64_t fusion_checksum = perf_output_checksum(
      &perf_fusion_output[0][0], config.query_rows,
      config.value_dim, config.output_stride);

#if FA_PERF_CHECK
  float max_abs_error;
  const int errors = perf_check_results(&config, &max_abs_error);
  printf("FLASH_ATTN_PERF result=%s errors=%d max_abs_error_bits=0x%x "
         "cpu_checksum=0x%llx fusion_checksum=0x%llx "
         "sink=0x%llx status=0x%llx\n",
         errors == 0 ? "PASS" : "FAIL", errors,
         vpu_float_to_bits(max_abs_error),
         (unsigned long long)cpu_checksum,
         (unsigned long long)fusion_checksum,
         (unsigned long long)perf_checksum_sink,
         (unsigned long long)fusion_result.vpu_status);
  return errors == 0 ? 0 : 1;
#else
  printf("FLASH_ATTN_PERF result=UNCHECKED "
         "cpu_checksum=0x%llx fusion_checksum=0x%llx "
         "sink=0x%llx status=0x%llx\n",
         (unsigned long long)cpu_checksum,
         (unsigned long long)fusion_checksum,
         (unsigned long long)perf_checksum_sink,
         (unsigned long long)fusion_result.vpu_status);
  return 0;
#endif
}
