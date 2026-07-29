// See LICENSE for license details.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "include/vpu.h"
#include "include/vpu_kernels.h"
#include "include/vpu_testutils.h"

#define VPU_TEST_MAX_ELEMENTS 2048u
#define VPU_TEST_STORAGE_ELEMENTS (VPU_TEST_MAX_ELEMENTS + 2u)

#ifndef VPU_BENCHMARK_LENGTH
#define VPU_BENCHMARK_LENGTH VPU_TEST_MAX_ELEMENTS
#endif

#if VPU_BENCHMARK_LENGTH > VPU_TEST_MAX_ELEMENTS
#error "VPU_BENCHMARK_LENGTH exceeds the statically allocated test buffers"
#endif

/*
 * Payloads start one storage element past a cache-line-aligned allocation.
 * This tests element-aligned/cache-line-unaligned DMA while the surrounding
 * and post-tail sentinels catch an incorrect final beat mask.
 */
static vpu_storage_t input_storage[VPU_TEST_STORAGE_ELEMENTS]
    __attribute__((aligned(64)));
static vpu_storage_t aux_storage[VPU_TEST_STORAGE_ELEMENTS]
    __attribute__((aligned(64)));
static vpu_storage_t output_storage[VPU_TEST_STORAGE_ELEMENTS]
    __attribute__((aligned(64)));
static float input_reference[VPU_TEST_MAX_ELEMENTS]
    __attribute__((aligned(64)));
static float aux_reference[VPU_TEST_MAX_ELEMENTS]
    __attribute__((aligned(64)));
static float expected[VPU_TEST_MAX_ELEMENTS] __attribute__((aligned(64)));

/*
 * The FP32 correctness reference is also the scalar CPU baseline.  Keep the
 * cycle reads in this file so the benchmark does not depend on Gemmini test
 * headers, and use a memory clobber to keep the compiler from moving array
 * accesses across either boundary.  Rocket is an in-order core; the VPU side
 * additionally drains DMA and makes its stores visible in vpu_fence().
 */
#if VPU_STORAGE_KIND == VPU_STORAGE_FP32 && defined(__riscv)
#define VPU_CYCLE_BENCHMARK 1
#else
#define VPU_CYCLE_BENCHMARK 0
#endif

#if VPU_CYCLE_BENCHMARK
#define VPU_BENCHMARK_ALIGNMENT 64u
#define VPU_BENCHMARK_OUTPUT_ELEMENTS (VPU_TEST_MAX_ELEMENTS + 1u)

/*
 * Correctness deliberately uses the +1-element payloads above.  Performance
 * instead reuses the already-aligned FP32 reference inputs and writes into a
 * separate aligned allocation, so timing does not destroy or depend on the
 * unaligned-path sentinels.
 */
static vpu_storage_t benchmark_output[VPU_BENCHMARK_OUTPUT_ELEMENTS]
    __attribute__((aligned(VPU_BENCHMARK_ALIGNMENT)));

static int benchmark_check_alignment(void) {
  const uintptr_t addresses = (uintptr_t)input_reference |
                              (uintptr_t)aux_reference |
                              (uintptr_t)benchmark_output;
  if ((addresses & (VPU_BENCHMARK_ALIGNMENT - 1u)) != 0u) {
    printf("VPU benchmark buffers are not %u-byte aligned\n",
           (unsigned)VPU_BENCHMARK_ALIGNMENT);
    return 1;
  }
  return 0;
}

static inline uint64_t benchmark_read_cycles(void) {
  uint64_t cycles;
  __asm__ volatile("rdcycle %0" : "=r"(cycles) : : "memory");
  return cycles;
}

/*
 * Throughput-only lower bounds for the emitted fine-grained micro-ops.
 * A full-width ALU or reduction consumes VPU_NLANES elements/cycle. EXP uses
 * VPU_SFU_LANES while reciprocal partitions the shared FMA array into
 * VPU_RECIPROCAL_LANES four-stage pipelines. Each scalar operation consumes
 * one issue slot. Pipeline fill/drain, dependencies, SRAM/DMA, command
 * transport, tail padding, and fences are deliberately excluded; those costs
 * are visible in the measured VPU E2E number.
 */
static uint64_t benchmark_groups(size_t elements, uint64_t lanes) {
  return ((uint64_t)elements + lanes - 1u) / lanes;
}

static uint64_t benchmark_ideal_rmsnorm_cycles(size_t elements) {
  if (elements == 0) {
    return 0;
  }
  const uint64_t lane_groups = benchmark_groups(elements, VPU_NLANES);
  /* square + RED_SUM + normalize + weight, then MUL/ADD/SQRT/RECI scalar. */
  return 4u * lane_groups + 4u;
}

static uint64_t benchmark_ideal_silu_cycles(size_t elements) {
  const uint64_t lane_groups = benchmark_groups(elements, VPU_NLANES);
  const uint64_t exp_groups = benchmark_groups(elements, VPU_SFU_LANES);
  const uint64_t reciprocal_groups =
      benchmark_groups(elements, VPU_RECIPROCAL_LANES);
  /* reverse SUB + ADD + MUL, plus EXP + RECI. */
  return 3u * lane_groups + exp_groups + reciprocal_groups;
}

static uint64_t benchmark_ideal_swiglu_cycles(size_t elements) {
  const uint64_t lane_groups = benchmark_groups(elements, VPU_NLANES);
  const uint64_t exp_groups = benchmark_groups(elements, VPU_SFU_LANES);
  const uint64_t reciprocal_groups =
      benchmark_groups(elements, VPU_RECIPROCAL_LANES);
  /* reverse SUB + ADD + two MULs, plus EXP + RECI. */
  return 4u * lane_groups + exp_groups + reciprocal_groups;
}

static uint64_t benchmark_ideal_softmax_cycles(
    size_t elements, struct vpu_tiled_plan plan) {
  if (elements == 0) {
    return 0;
  }
  const size_t overflow = vpu_tiled_plan_overflow_elements(plan);
  const uint64_t lane_groups = benchmark_groups(elements, VPU_NLANES);
  const uint64_t sfu_groups = benchmark_groups(elements, VPU_SFU_LANES);
  const uint64_t overflow_lane_groups =
      benchmark_groups(overflow, VPU_NLANES);
  const uint64_t overflow_sfu_groups =
      benchmark_groups(overflow, VPU_SFU_LANES);
  /*
   * RED_MAX + SUB + RED_SUM + MUL cover the full row, as does its first EXP.
   * Only the non-resident suffix repeats SUB + EXP in the output pass.
   */
  return 4u * lane_groups + sfu_groups + overflow_lane_groups +
         overflow_sfu_groups + 1u;
}

struct benchmark_perf {
  uint64_t dma_read_bytes;
  uint64_t dma_write_bytes;
  uint64_t dma_exec_overlap_cycles;
};

static struct benchmark_perf benchmark_read_perf_counters(void) {
  struct benchmark_perf perf;
  perf.dma_read_bytes = vpu_read_perf(VPU_PERF_DMA_READ_BYTES);
  perf.dma_write_bytes = vpu_read_perf(VPU_PERF_DMA_WRITE_BYTES);
  perf.dma_exec_overlap_cycles =
      vpu_read_perf(VPU_PERF_DMA_EXEC_OVERLAP_CYCLES);
  return perf;
}

static int benchmark_check_dma_traffic(
    const char *operation, size_t n, struct vpu_tiled_plan plan,
    struct benchmark_perf perf) {
  const uint64_t expected_read_bytes =
      (uint64_t)vpu_tiled_plan_read_elements(plan) * VPU_STORAGE_BYTES;
  const uint64_t expected_write_bytes = (uint64_t)n * VPU_STORAGE_BYTES;
  if (perf.dma_read_bytes != expected_read_bytes ||
      perf.dma_write_bytes != expected_write_bytes) {
    printf("%s[%u] DMA traffic mismatch: read=%llu expected=%llu "
           "write=%llu expected=%llu\n",
           operation, (unsigned)n,
           (unsigned long long)perf.dma_read_bytes,
           (unsigned long long)expected_read_bytes,
           (unsigned long long)perf.dma_write_bytes,
           (unsigned long long)expected_write_bytes);
    return 1;
  }
  return 0;
}

struct benchmark_ratio {
  uint64_t whole;
  uint64_t fraction;
};

static struct benchmark_ratio benchmark_ratio_thousandths(
    uint64_t numerator, uint64_t denominator) {
  struct benchmark_ratio ratio;
  ratio.whole = numerator / denominator;
  ratio.fraction =
      (((numerator % denominator) * 1000u) + denominator / 2u) /
      denominator;
  if (ratio.fraction == 1000u) {
    ++ratio.whole;
    ratio.fraction = 0;
  }
  return ratio;
}

static void report_benchmark(const char *operation, size_t n,
                             uint64_t cpu_cycles,
                             uint64_t vpu_e2e_cycles,
                             uint64_t ideal_cycles,
                             struct vpu_tiled_plan plan,
                             struct benchmark_perf perf) {
  /* Avoid making slow simulated UART output dominate the regression. */
  if (n != VPU_BENCHMARK_LENGTH) {
    return;
  }
  const size_t resident_elements =
      vpu_tiled_plan_resident_elements(plan);
  const unsigned resident_banks_used = plan.resident_tiles == 0
      ? 0u
      : (unsigned)((plan.resident_tiles + VPU_SLOTS_PER_BANK - 1u) /
                   VPU_SLOTS_PER_BANK);
  if (n == 0 || vpu_e2e_cycles == 0 || ideal_cycles == 0) {
    printf("VPU_BENCH op=%s length=%u schedule=%s "
           "resident_tiles=%u/%u resident_elements=%u/%u "
           "resident_banks_used=%u/%u workspace_banks=%u "
           "cpu_scalar_cycles=N/A "
           "vpu_e2e_cycles=%llu ideal_useful_compute_cycles=%llu "
           "dma_read_bytes=%llu dma_write_bytes=%llu "
           "dma_exec_overlap_cycles=%llu "
           "cpu_over_vpu=N/A vpu_e2e_over_ideal=N/A\n",
           operation, (unsigned)n, vpu_tiled_schedule_name(plan.schedule),
           (unsigned)plan.resident_tiles, (unsigned)plan.tile_count,
           (unsigned)resident_elements, (unsigned)n,
           resident_banks_used, plan.resident_banks, plan.workspace_banks,
           (unsigned long long)vpu_e2e_cycles,
           (unsigned long long)ideal_cycles,
           (unsigned long long)perf.dma_read_bytes,
           (unsigned long long)perf.dma_write_bytes,
           (unsigned long long)perf.dma_exec_overlap_cycles);
    return;
  }

  const struct benchmark_ratio speedup =
      benchmark_ratio_thousandths(cpu_cycles, vpu_e2e_cycles);
  const struct benchmark_ratio e2e_over_ideal =
      benchmark_ratio_thousandths(vpu_e2e_cycles, ideal_cycles);
  printf("VPU_BENCH op=%s length=%u schedule=%s "
         "resident_tiles=%u/%u resident_elements=%u/%u "
         "resident_banks_used=%u/%u workspace_banks=%u "
         "cpu_scalar_cycles=%llu "
         "vpu_e2e_cycles=%llu ideal_useful_compute_cycles=%llu "
         "dma_read_bytes=%llu dma_write_bytes=%llu "
         "dma_exec_overlap_cycles=%llu "
         "cpu_over_vpu=%llu.%03llux vpu_e2e_over_ideal=%llu.%03llux\n",
         operation, (unsigned)n, vpu_tiled_schedule_name(plan.schedule),
         (unsigned)plan.resident_tiles, (unsigned)plan.tile_count,
         (unsigned)resident_elements, (unsigned)n,
         resident_banks_used, plan.resident_banks, plan.workspace_banks,
         (unsigned long long)cpu_cycles,
         (unsigned long long)vpu_e2e_cycles,
         (unsigned long long)ideal_cycles,
         (unsigned long long)perf.dma_read_bytes,
         (unsigned long long)perf.dma_write_bytes,
         (unsigned long long)perf.dma_exec_overlap_cycles,
         (unsigned long long)speedup.whole,
         (unsigned long long)speedup.fraction,
         (unsigned long long)e2e_over_ideal.whole,
         (unsigned long long)e2e_over_ideal.fraction);
}
#endif

#if VPU_STORAGE_KIND == VPU_STORAGE_BF16
#define VPU_TEST_QNAN_RAW 0x7fc0u
#define VPU_TEST_SNAN_RAW 0x7f81u
#define VPU_TEST_POS_INF_RAW 0x7f80u
#define VPU_TEST_NEG_INF_RAW 0xff80u
#define VPU_TEST_NEG_ZERO_RAW 0x8000u
#define VPU_TEST_MIN_SUBNORMAL_RAW 0x0001u
static uint32_t storage_raw(vpu_storage_t value) {
  return value;
}
static vpu_storage_t storage_from_raw(uint32_t value) {
  return (vpu_storage_t)value;
}
#else
#define VPU_TEST_QNAN_RAW 0x7fc00000u
#define VPU_TEST_SNAN_RAW 0x7f800001u
#define VPU_TEST_POS_INF_RAW 0x7f800000u
#define VPU_TEST_NEG_INF_RAW 0xff800000u
#define VPU_TEST_NEG_ZERO_RAW 0x80000000u
#define VPU_TEST_MIN_SUBNORMAL_RAW 0x00000001u
static uint32_t storage_raw(vpu_storage_t value) {
  return vpu_float_to_bits(value);
}
static vpu_storage_t storage_from_raw(uint32_t value) {
  return vpu_bits_to_float(value);
}
#endif

static vpu_storage_t guard_value(void) {
  return vpu_float_to_storage(-123.0f);
}

#if VPU_CYCLE_BENCHMARK
static void prepare_benchmark_output(void) {
  const vpu_storage_t guard = guard_value();
  for (size_t i = 0; i < VPU_BENCHMARK_OUTPUT_ELEMENTS; ++i) {
    benchmark_output[i] = guard;
  }
}
#endif

static void prepare_vectors(size_t n) {
  const vpu_storage_t guard = guard_value();
  for (size_t i = 0; i < VPU_TEST_STORAGE_ELEMENTS; ++i) {
    input_storage[i] = guard;
    aux_storage[i] = guard;
    output_storage[i] = guard;
  }
  for (size_t i = 0; i < n; ++i) {
    const float input = ((int)(i % 29u) - 14) * 0.125f;
    const float auxiliary = 0.75f + (float)(i % 11u) * 0.03125f;
    input_storage[i + 1u] = vpu_float_to_storage(input);
    aux_storage[i + 1u] = vpu_float_to_storage(auxiliary);
    input_reference[i] = vpu_storage_to_float(input_storage[i + 1u]);
    aux_reference[i] = vpu_storage_to_float(aux_storage[i + 1u]);
  }
}

static void refresh_references(size_t n) {
  for (size_t i = 0; i < n; ++i) {
    input_reference[i] = vpu_storage_to_float(input_storage[i + 1u]);
    aux_reference[i] = vpu_storage_to_float(aux_storage[i + 1u]);
  }
}

static int check_guards(const char *name, size_t n) {
  const vpu_storage_t guard = guard_value();
  if (input_storage[0] != guard || aux_storage[0] != guard ||
      output_storage[0] != guard ||
      input_storage[VPU_TEST_MAX_ELEMENTS + 1u] != guard ||
      aux_storage[VPU_TEST_MAX_ELEMENTS + 1u] != guard ||
      output_storage[VPU_TEST_MAX_ELEMENTS + 1u] != guard) {
    printf("%s[%u] corrupted a surrounding guard\n", name, (unsigned)n);
    return 1;
  }
  for (size_t i = n; i < VPU_TEST_MAX_ELEMENTS; ++i) {
    if (input_storage[i + 1u] != guard ||
        aux_storage[i + 1u] != guard ||
        output_storage[i + 1u] != guard) {
      printf("%s[%u] touched a tail guard at %u (input=%u aux=%u "
             "output=%u)\n", name, (unsigned)n, (unsigned)i,
             input_storage[i + 1u] != guard,
             aux_storage[i + 1u] != guard,
             output_storage[i + 1u] != guard);
      return 1;
    }
  }
  return 0;
}

static int check_output_values(const char *name, size_t n,
                               const vpu_storage_t *actual_values) {
#if VPU_STORAGE_KIND == VPU_STORAGE_BF16
  const float relative_tolerance = 1.0e-2f;
  const float absolute_tolerance = 1.0e-3f;
#else
  const float relative_tolerance = 2.0e-4f;
  const float absolute_tolerance = 2.0e-6f;
#endif

  for (size_t i = 0; i < n; ++i) {
    const float actual = vpu_storage_to_float(actual_values[i]);
    if (!vpu_ref_close(actual, expected[i], relative_tolerance,
                       absolute_tolerance)) {
      printf("%s[%u] mismatch at %u: got=0x%x expected=0x%x\n", name,
             (unsigned)n, (unsigned)i, vpu_float_to_bits(actual),
             vpu_float_to_bits(expected[i]));
      return 1;
    }
  }
  return 0;
}

static int finish_and_check(const char *name, size_t n, uint64_t status) {
  if ((status & VPU_STATUS_ERROR_MASK) != 0) {
    printf("%s[%u]: VPU status error 0x%llx\n", name, (unsigned)n,
           (unsigned long long)status);
    return 1;
  }
  if (check_guards(name, n)) {
    return 1;
  }
  return check_output_values(name, n, &output_storage[1]);
}

#if VPU_CYCLE_BENCHMARK
static int finish_benchmark_and_check(const char *name, size_t n,
                                      uint64_t status) {
  if ((status & VPU_STATUS_ERROR_MASK) != 0) {
    printf("%s[%u]: VPU status error 0x%llx\n", name, (unsigned)n,
           (unsigned long long)status);
    return 1;
  }

  const vpu_storage_t guard = guard_value();
  for (size_t i = n; i < VPU_BENCHMARK_OUTPUT_ELEMENTS; ++i) {
    if (benchmark_output[i] != guard) {
      printf("%s[%u] touched aligned benchmark tail guard at %u\n", name,
             (unsigned)n, (unsigned)i);
      return 1;
    }
  }
  return check_output_values(name, n, benchmark_output);
}
#endif

static int check_softmax_distribution(const char *name, size_t n,
                                      const vpu_storage_t *actual_values);

static int test_rmsnorm(size_t n, int final_norm) {
  const float epsilon = final_norm ? 1.0e-6f : 1.0e-5f;
  const char *name = final_norm ? "final_norm" : "rmsnorm";
  prepare_vectors(n);
#if VPU_CYCLE_BENCHMARK
  uint64_t cpu_cycles = 0;
#endif
  if (n != 0) {
#if VPU_CYCLE_BENCHMARK
    const uint64_t cpu_start = benchmark_read_cycles();
#endif
    vpu_ref_rmsnorm(input_reference, aux_reference, expected, n, epsilon);
#if VPU_CYCLE_BENCHMARK
    cpu_cycles = benchmark_read_cycles() - cpu_start;
#endif
  }
  vpu_clear_status(VPU_CLEAR_ERRORS);
  const uint64_t status = final_norm
      ? vpu_tiled_final_norm_auto(&input_storage[1], &aux_storage[1],
                                  &output_storage[1], n, epsilon)
      : vpu_tiled_rmsnorm_auto(&input_storage[1], &aux_storage[1],
                               &output_storage[1], n, epsilon);
  if (finish_and_check(name, n, status)) {
    return 1;
  }
#if VPU_CYCLE_BENCHMARK
  if (n == VPU_BENCHMARK_LENGTH) {
    const char *benchmark_name = final_norm ? "final_norm_aligned"
                                            : "rmsnorm_aligned";
    const struct vpu_tiled_plan plan = vpu_tiled_plan_for(
        final_norm ? VPU_TILED_KERNEL_FINAL_NORM
                   : VPU_TILED_KERNEL_RMSNORM,
        n);
    prepare_benchmark_output();
    vpu_clear_status(VPU_CLEAR_ALL);
    const uint64_t vpu_start = benchmark_read_cycles();
    const uint64_t benchmark_status = final_norm
        ? vpu_tiled_final_norm_auto(input_reference, aux_reference,
                                    benchmark_output, n, epsilon)
        : vpu_tiled_rmsnorm_auto(input_reference, aux_reference,
                                 benchmark_output, n, epsilon);
    const uint64_t vpu_e2e_cycles = benchmark_read_cycles() - vpu_start;
    const struct benchmark_perf perf = benchmark_read_perf_counters();
    if (finish_benchmark_and_check(benchmark_name, n, benchmark_status)) {
      return 1;
    }
    if (benchmark_check_dma_traffic(name, n, plan, perf)) {
      return 1;
    }
    report_benchmark(name, n, cpu_cycles, vpu_e2e_cycles,
                     benchmark_ideal_rmsnorm_cycles(n), plan, perf);
  }
#endif
  return 0;
}

static int test_silu(size_t n) {
  prepare_vectors(n);
#if VPU_CYCLE_BENCHMARK
  uint64_t cpu_cycles = 0;
#endif
  if (n != 0) {
#if VPU_CYCLE_BENCHMARK
    const uint64_t cpu_start = benchmark_read_cycles();
#endif
    vpu_ref_silu(input_reference, expected, n);
#if VPU_CYCLE_BENCHMARK
    cpu_cycles = benchmark_read_cycles() - cpu_start;
#endif
  }
  vpu_clear_status(VPU_CLEAR_ERRORS);
  const uint64_t status = vpu_tiled_silu_auto(
      &input_storage[1], &output_storage[1], n);
  if (finish_and_check("silu", n, status)) {
    return 1;
  }
#if VPU_CYCLE_BENCHMARK
  if (n == VPU_BENCHMARK_LENGTH) {
    const struct vpu_tiled_plan plan =
        vpu_tiled_plan_for(VPU_TILED_KERNEL_SILU, n);
    prepare_benchmark_output();
    vpu_clear_status(VPU_CLEAR_ALL);
    const uint64_t vpu_start = benchmark_read_cycles();
    const uint64_t benchmark_status = vpu_tiled_silu_auto(
        input_reference, benchmark_output, n);
    const uint64_t vpu_e2e_cycles = benchmark_read_cycles() - vpu_start;
    const struct benchmark_perf perf = benchmark_read_perf_counters();
    if (finish_benchmark_and_check("silu_aligned", n, benchmark_status)) {
      return 1;
    }
    if (benchmark_check_dma_traffic("silu", n, plan, perf)) {
      return 1;
    }
    report_benchmark("silu", n, cpu_cycles, vpu_e2e_cycles,
                     benchmark_ideal_silu_cycles(n), plan, perf);
  }
#endif
  return 0;
}

static int test_swiglu(size_t n) {
  prepare_vectors(n);
#if VPU_CYCLE_BENCHMARK
  uint64_t cpu_cycles = 0;
#endif
  if (n != 0) {
#if VPU_CYCLE_BENCHMARK
    const uint64_t cpu_start = benchmark_read_cycles();
#endif
    vpu_ref_swiglu(input_reference, aux_reference, expected, n);
#if VPU_CYCLE_BENCHMARK
    cpu_cycles = benchmark_read_cycles() - cpu_start;
#endif
  }
  vpu_clear_status(VPU_CLEAR_ERRORS);
  const uint64_t status = vpu_tiled_swiglu_auto(
      &input_storage[1], &aux_storage[1], &output_storage[1], n);
  if (finish_and_check("swiglu", n, status)) {
    return 1;
  }
#if VPU_CYCLE_BENCHMARK
  if (n == VPU_BENCHMARK_LENGTH) {
    const struct vpu_tiled_plan plan =
        vpu_tiled_plan_for(VPU_TILED_KERNEL_SWIGLU, n);
    prepare_benchmark_output();
    vpu_clear_status(VPU_CLEAR_ALL);
    const uint64_t vpu_start = benchmark_read_cycles();
    const uint64_t benchmark_status = vpu_tiled_swiglu_auto(
        input_reference, aux_reference, benchmark_output, n);
    const uint64_t vpu_e2e_cycles = benchmark_read_cycles() - vpu_start;
    const struct benchmark_perf perf = benchmark_read_perf_counters();
    if (finish_benchmark_and_check("swiglu_aligned", n, benchmark_status)) {
      return 1;
    }
    if (benchmark_check_dma_traffic("swiglu", n, plan, perf)) {
      return 1;
    }
    report_benchmark("swiglu", n, cpu_cycles, vpu_e2e_cycles,
                     benchmark_ideal_swiglu_cycles(n), plan, perf);
  }
#endif
  return 0;
}

static int test_softmax(size_t n) {
  prepare_vectors(n);
#if VPU_CYCLE_BENCHMARK
  uint64_t cpu_cycles = 0;
#endif
  if (n != 0) {
#if VPU_CYCLE_BENCHMARK
    const uint64_t cpu_start = benchmark_read_cycles();
#endif
    vpu_ref_softmax(input_reference, expected, n);
#if VPU_CYCLE_BENCHMARK
    cpu_cycles = benchmark_read_cycles() - cpu_start;
#endif
  }
  vpu_clear_status(VPU_CLEAR_ERRORS);
  const uint64_t status = vpu_tiled_softmax_auto(
      &input_storage[1], &output_storage[1], n);
  if (finish_and_check("softmax", n, status)) {
    return 1;
  }
  if (n != 0 &&
      check_softmax_distribution("softmax", n, &output_storage[1])) {
    return 1;
  }
#if VPU_CYCLE_BENCHMARK
  if (n == VPU_BENCHMARK_LENGTH) {
    const struct vpu_tiled_plan plan =
        vpu_tiled_plan_for(VPU_TILED_KERNEL_SOFTMAX, n);
    prepare_benchmark_output();
    vpu_clear_status(VPU_CLEAR_ALL);
    const uint64_t vpu_start = benchmark_read_cycles();
    const uint64_t benchmark_status = vpu_tiled_softmax_auto(
        input_reference, benchmark_output, n);
    const uint64_t vpu_e2e_cycles = benchmark_read_cycles() - vpu_start;
    const struct benchmark_perf perf = benchmark_read_perf_counters();
    if (finish_benchmark_and_check("softmax_aligned", n,
                                   benchmark_status)) {
      return 1;
    }
    if (n != 0 && check_softmax_distribution(
            "softmax_aligned", n, benchmark_output)) {
      return 1;
    }
    if (benchmark_check_dma_traffic("softmax", n, plan, perf)) {
      return 1;
    }
    report_benchmark("softmax", n, cpu_cycles, vpu_e2e_cycles,
                     benchmark_ideal_softmax_cycles(n, plan), plan, perf);
  }
#endif
  return 0;
}

static int check_finite_outputs(const char *name, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const float value = vpu_storage_to_float(output_storage[i + 1u]);
    if (!isfinite(value)) {
      printf("%s[%u] produced non-finite output at %u: 0x%x\n", name,
             (unsigned)n, (unsigned)i, vpu_float_to_bits(value));
      return 1;
    }
  }
  return 0;
}

static int check_canonical_nan_outputs(const char *name, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (storage_raw(output_storage[i + 1u]) != VPU_TEST_QNAN_RAW) {
      printf("%s[%u] did not canonicalize NaN at %u: raw=0x%x\n", name,
             (unsigned)n, (unsigned)i,
             storage_raw(output_storage[i + 1u]));
      return 1;
    }
  }
  return 0;
}

static int check_softmax_distribution(const char *name, size_t n,
                                      const vpu_storage_t *actual_values) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float value = vpu_storage_to_float(actual_values[i]);
    if (!isfinite(value) || value < 0.0f) {
      printf("%s[%u] produced invalid probability at %u: 0x%x\n", name,
             (unsigned)n, (unsigned)i, vpu_float_to_bits(value));
      return 1;
    }
    sum += value;
  }
#if VPU_STORAGE_KIND == VPU_STORAGE_BF16
  const float tolerance = 1.0e-2f;
#else
  const float tolerance = 2.0e-4f;
#endif
  if (!vpu_ref_close(sum, 1.0f, 0.0f, tolerance)) {
    printf("%s[%u] probability sum mismatch: 0x%x\n", name,
           (unsigned)n, vpu_float_to_bits(sum));
    return 1;
  }
  return 0;
}

static size_t stress_length(void) {
  const size_t requested = 2u * (size_t)VPU_VLEN + 17u;
  return requested < VPU_TEST_MAX_ELEMENTS ? requested
                                           : VPU_TEST_MAX_ELEMENTS;
}

/*
 * These finite cases use values large enough to expose an unstabilized EXP or
 * an overflowing low-precision accumulation, while remaining inside the
 * documented nonlinear approximation interval after range reduction.  The
 * reference is populated only after storage rounding, which is essential for
 * the BF16-storage build.
 */
static int test_finite_rmsnorm_stress(int final_norm) {
  const size_t n = stress_length();
  const char *name = final_norm ? "final_norm_finite_stress"
                                : "rmsnorm_finite_stress";
  const float epsilon = final_norm ? 1.0e-6f : 1.0e-5f;
  prepare_vectors(n);
  for (size_t i = 0; i < n; ++i) {
    float input;
    switch (i % 10u) {
      case 0: input_storage[i + 1u] = storage_from_raw(VPU_TEST_NEG_ZERO_RAW);
              input = 0.0f; break;
      case 1: input = 0.0f; break;
      case 2: input_storage[i + 1u] =
                  storage_from_raw(VPU_TEST_MIN_SUBNORMAL_RAW);
              input = 0.0f; break;
      case 3: input_storage[i + 1u] = storage_from_raw(
                  VPU_TEST_MIN_SUBNORMAL_RAW | VPU_TEST_NEG_ZERO_RAW);
              input = 0.0f; break;
      case 4: input = 16384.0f; break;
      case 5: input = -16384.0f; break;
      case 6: input = 0.0009765625f; break;
      case 7: input = -0.0009765625f; break;
      case 8: input = 127.5f; break;
      default: input = -127.5f; break;
    }
    if ((i % 10u) != 0u && (i % 10u) != 2u && (i % 10u) != 3u) {
      input_storage[i + 1u] = vpu_float_to_storage(input);
    }
    const float weight = (i % 10u) == 0u
        ? 1.0f
        : ((i & 1u) ? -0.75f : 1.25f);
    aux_storage[i + 1u] = vpu_float_to_storage(weight);
  }
  refresh_references(n);
  vpu_ref_rmsnorm(input_reference, aux_reference, expected, n, epsilon);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  const uint64_t status = final_norm
      ? vpu_tiled_final_norm_auto(&input_storage[1], &aux_storage[1],
                                  &output_storage[1], n, epsilon)
      : vpu_tiled_rmsnorm_auto(&input_storage[1], &aux_storage[1],
                               &output_storage[1], n, epsilon);
  if (finish_and_check(name, n, status) || check_finite_outputs(name, n)) {
    return 1;
  }
  if (storage_raw(output_storage[1]) != VPU_TEST_NEG_ZERO_RAW) {
    printf("%s did not preserve the sign of -0 input\n", name);
    return 1;
  }
  return 0;
}

static void prepare_activation_stress(size_t n) {
  prepare_vectors(n);
  for (size_t i = 0; i < n; ++i) {
    switch (i % 12u) {
      case 0:
        input_storage[i + 1u] = storage_from_raw(VPU_TEST_NEG_ZERO_RAW);
        break;
      case 1:
        input_storage[i + 1u] = storage_from_raw(0u);
        break;
      case 2:
        input_storage[i + 1u] =
            storage_from_raw(VPU_TEST_MIN_SUBNORMAL_RAW);
        break;
      case 3:
        input_storage[i + 1u] = storage_from_raw(
            VPU_TEST_MIN_SUBNORMAL_RAW | VPU_TEST_NEG_ZERO_RAW);
        break;
      case 4: input_storage[i + 1u] = vpu_float_to_storage(-10.0f); break;
      case 5: input_storage[i + 1u] = vpu_float_to_storage(10.0f); break;
      case 6: input_storage[i + 1u] = vpu_float_to_storage(-8.0f); break;
      case 7: input_storage[i + 1u] = vpu_float_to_storage(8.0f); break;
      case 8: input_storage[i + 1u] = vpu_float_to_storage(-4.0f); break;
      case 9: input_storage[i + 1u] = vpu_float_to_storage(4.0f); break;
      case 10: input_storage[i + 1u] = vpu_float_to_storage(-0.125f); break;
      default: input_storage[i + 1u] = vpu_float_to_storage(0.125f); break;
    }
    const float up = (i % 12u) == 0u
        ? 4096.0f
        : ((i & 1u) ? -4096.0f : 2048.0f);
    aux_storage[i + 1u] = vpu_float_to_storage(up);
  }
  refresh_references(n);
}

static int test_finite_silu_stress(void) {
  const size_t n = stress_length();
  const char *name = "silu_finite_stress";
  prepare_activation_stress(n);
  vpu_ref_silu(input_reference, expected, n);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  const uint64_t status = vpu_tiled_silu_auto(
      &input_storage[1], &output_storage[1], n);
  if (finish_and_check(name, n, status) || check_finite_outputs(name, n)) {
    return 1;
  }
  if (storage_raw(output_storage[1]) != VPU_TEST_NEG_ZERO_RAW) {
    printf("%s did not preserve the sign of -0 input\n", name);
    return 1;
  }
  return 0;
}

static int test_finite_swiglu_stress(void) {
  const size_t n = stress_length();
  const char *name = "swiglu_finite_stress";
  prepare_activation_stress(n);
  vpu_ref_swiglu(input_reference, aux_reference, expected, n);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  const uint64_t status = vpu_tiled_swiglu_auto(
      &input_storage[1], &aux_storage[1], &output_storage[1], n);
  if (finish_and_check(name, n, status) || check_finite_outputs(name, n)) {
    return 1;
  }
  if (storage_raw(output_storage[1]) != VPU_TEST_NEG_ZERO_RAW) {
    printf("%s did not preserve the sign of -0 gate\n", name);
    return 1;
  }
  return 0;
}

static int test_finite_softmax_stress(void) {
  const size_t n = stress_length();
  const char *name = "softmax_finite_stress";
  prepare_vectors(n);
  for (size_t i = 0; i < n; ++i) {
    /* All rounded BF16 values remain in [246, 256], so x-max is [-10, 0]. */
    input_storage[i + 1u] =
        vpu_float_to_storage(256.0f - 2.0f * (float)(i % 6u));
  }
  refresh_references(n);
  vpu_ref_softmax(input_reference, expected, n);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  const uint64_t status = vpu_tiled_softmax_auto(
      &input_storage[1], &output_storage[1], n);
  if (finish_and_check(name, n, status)) {
    return 1;
  }
  return check_softmax_distribution(name, n, &output_storage[1]);
}

static int test_finite_stress(void) {
  return test_finite_rmsnorm_stress(0) ||
         test_finite_rmsnorm_stress(1) ||
         test_finite_silu_stress() ||
         test_finite_swiglu_stress() ||
         test_finite_softmax_stress();
}

static int test_rmsnorm_nonfinite(void) {
  enum { COUNT = 17 };
  const char *name = "rmsnorm_nan";
  prepare_vectors(COUNT);
  input_storage[3] = storage_from_raw(VPU_TEST_QNAN_RAW);
  input_storage[4] = storage_from_raw(VPU_TEST_SNAN_RAW);
  refresh_references(COUNT);
  vpu_ref_rmsnorm(input_reference, aux_reference, expected, COUNT, 1.0e-5f);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  uint64_t status = vpu_tiled_rmsnorm_auto(
      &input_storage[1], &aux_storage[1], &output_storage[1], COUNT, 1.0e-5f);
  if (finish_and_check(name, COUNT, status) ||
      check_canonical_nan_outputs(name, COUNT)) {
    return 1;
  }
  if ((status & VPU_STATUS_FFLAG_NV) == 0) {
    printf("%s did not report NV for signaling NaN\n", name);
    return 1;
  }

  name = "final_norm_infinity";
  prepare_vectors(COUNT);
  input_storage[6] = storage_from_raw(VPU_TEST_POS_INF_RAW);
  refresh_references(COUNT);
  vpu_ref_rmsnorm(input_reference, aux_reference, expected, COUNT, 1.0e-6f);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  status = vpu_tiled_final_norm_auto(
      &input_storage[1], &aux_storage[1], &output_storage[1], COUNT, 1.0e-6f);
  if (finish_and_check(name, COUNT, status)) {
    return 1;
  }
  if ((status & VPU_STATUS_FFLAG_NV) == 0 ||
      storage_raw(output_storage[6]) != VPU_TEST_QNAN_RAW) {
    printf("%s did not implement Inf*0 as canonical invalid NaN\n", name);
    return 1;
  }
  for (size_t i = 0; i < COUNT; ++i) {
    if (i != 5u && vpu_storage_to_float(output_storage[i + 1u]) != 0.0f) {
      printf("%s expected a zero finite-input result at %u\n", name,
             (unsigned)i);
      return 1;
    }
  }
  return 0;
}

static int test_silu_nonfinite(void) {
  enum { COUNT = 12 };
  const char *name = "silu_special";
  prepare_vectors(COUNT);
  input_storage[1] = storage_from_raw(VPU_TEST_QNAN_RAW);
  input_storage[2] = storage_from_raw(VPU_TEST_SNAN_RAW);
  input_storage[3] = storage_from_raw(VPU_TEST_POS_INF_RAW);
  input_storage[4] = storage_from_raw(VPU_TEST_NEG_INF_RAW);
  input_storage[5] = storage_from_raw(0u);
  input_storage[6] = storage_from_raw(VPU_TEST_NEG_ZERO_RAW);
  input_storage[7] = storage_from_raw(VPU_TEST_MIN_SUBNORMAL_RAW);
  input_storage[8] = storage_from_raw(VPU_TEST_MIN_SUBNORMAL_RAW |
                                      VPU_TEST_NEG_ZERO_RAW);
  refresh_references(COUNT);
  vpu_ref_silu(input_reference, expected, COUNT);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  const uint64_t status = vpu_tiled_silu_auto(
      &input_storage[1], &output_storage[1], COUNT);
  if (finish_and_check(name, COUNT, status)) {
    return 1;
  }
  if ((status & VPU_STATUS_FFLAG_NV) == 0 ||
      storage_raw(output_storage[1]) != VPU_TEST_QNAN_RAW ||
      storage_raw(output_storage[2]) != VPU_TEST_QNAN_RAW ||
      storage_raw(output_storage[3]) != VPU_TEST_POS_INF_RAW ||
      storage_raw(output_storage[4]) != VPU_TEST_QNAN_RAW ||
      storage_raw(output_storage[5]) != 0u ||
      storage_raw(output_storage[6]) != VPU_TEST_NEG_ZERO_RAW ||
      storage_raw(output_storage[7]) != 0u ||
      storage_raw(output_storage[8]) != VPU_TEST_NEG_ZERO_RAW) {
    printf("%s NaN/Inf/subnormal/signed-zero mismatch\n", name);
    return 1;
  }
  return 0;
}

static int test_swiglu_nonfinite(void) {
  enum { COUNT = 12 };
  const char *name = "swiglu_special";
  prepare_vectors(COUNT);
  input_storage[1] = storage_from_raw(VPU_TEST_QNAN_RAW);
  input_storage[2] = storage_from_raw(VPU_TEST_SNAN_RAW);
  input_storage[3] = storage_from_raw(VPU_TEST_POS_INF_RAW);
  input_storage[4] = storage_from_raw(VPU_TEST_NEG_INF_RAW);
  input_storage[5] = storage_from_raw(0u);
  input_storage[6] = storage_from_raw(VPU_TEST_NEG_ZERO_RAW);
  input_storage[7] = storage_from_raw(VPU_TEST_MIN_SUBNORMAL_RAW);
  input_storage[8] = storage_from_raw(VPU_TEST_MIN_SUBNORMAL_RAW |
                                      VPU_TEST_NEG_ZERO_RAW);
  input_storage[9] = vpu_float_to_storage(1.0f);
  input_storage[10] = vpu_float_to_storage(-1.0f);
  aux_storage[1] = vpu_float_to_storage(1.0f);
  aux_storage[2] = vpu_float_to_storage(1.0f);
  aux_storage[3] = vpu_float_to_storage(2.0f);
  aux_storage[4] = vpu_float_to_storage(2.0f);
  aux_storage[5] = storage_from_raw(VPU_TEST_POS_INF_RAW);
  aux_storage[6] = vpu_float_to_storage(1.0f);
  aux_storage[7] = vpu_float_to_storage(1.0f);
  aux_storage[8] = vpu_float_to_storage(1.0f);
  aux_storage[9] = storage_from_raw(VPU_TEST_POS_INF_RAW);
  aux_storage[10] = storage_from_raw(VPU_TEST_NEG_INF_RAW);
  refresh_references(COUNT);
  vpu_ref_swiglu(input_reference, aux_reference, expected, COUNT);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  const uint64_t status = vpu_tiled_swiglu_auto(
      &input_storage[1], &aux_storage[1], &output_storage[1], COUNT);
  if (finish_and_check(name, COUNT, status)) {
    return 1;
  }
  if ((status & VPU_STATUS_FFLAG_NV) == 0 ||
      storage_raw(output_storage[1]) != VPU_TEST_QNAN_RAW ||
      storage_raw(output_storage[2]) != VPU_TEST_QNAN_RAW ||
      storage_raw(output_storage[3]) != VPU_TEST_POS_INF_RAW ||
      storage_raw(output_storage[4]) != VPU_TEST_QNAN_RAW ||
      storage_raw(output_storage[5]) != VPU_TEST_QNAN_RAW ||
      storage_raw(output_storage[6]) != VPU_TEST_NEG_ZERO_RAW ||
      storage_raw(output_storage[7]) != 0u ||
      storage_raw(output_storage[8]) != VPU_TEST_NEG_ZERO_RAW ||
      storage_raw(output_storage[9]) != VPU_TEST_POS_INF_RAW ||
      storage_raw(output_storage[10]) != VPU_TEST_POS_INF_RAW) {
    printf("%s NaN/Inf/subnormal/signed-zero mismatch\n", name);
    return 1;
  }
  return 0;
}

static int test_softmax_special(void) {
  enum { COUNT = 17 };
  const char *name = "softmax_zero_subnormal_neginf";
  prepare_vectors(COUNT);
  for (size_t i = 0; i < COUNT; ++i) {
    switch (i % 4u) {
      case 0: input_storage[i + 1u] = storage_from_raw(0u); break;
      case 1:
        input_storage[i + 1u] = storage_from_raw(VPU_TEST_NEG_ZERO_RAW);
        break;
      case 2:
        input_storage[i + 1u] =
            storage_from_raw(VPU_TEST_MIN_SUBNORMAL_RAW);
        break;
      default:
        input_storage[i + 1u] = storage_from_raw(
            VPU_TEST_MIN_SUBNORMAL_RAW | VPU_TEST_NEG_ZERO_RAW);
        break;
    }
  }
  input_storage[COUNT] = storage_from_raw(VPU_TEST_NEG_INF_RAW);
  refresh_references(COUNT);
  vpu_ref_softmax(input_reference, expected, COUNT);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  uint64_t status = vpu_tiled_softmax_auto(
      &input_storage[1], &output_storage[1], COUNT);
  if (finish_and_check(name, COUNT, status) ||
      check_softmax_distribution(name, COUNT, &output_storage[1])) {
    return 1;
  }
  if (storage_raw(output_storage[COUNT]) != 0u) {
    printf("%s did not map -Inf to a +0 probability\n", name);
    return 1;
  }

  name = "softmax_nan";
  prepare_vectors(COUNT);
  input_storage[4] = storage_from_raw(VPU_TEST_QNAN_RAW);
  input_storage[5] = storage_from_raw(VPU_TEST_SNAN_RAW);
  refresh_references(COUNT);
  vpu_ref_softmax(input_reference, expected, COUNT);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  status = vpu_tiled_softmax_auto(
      &input_storage[1], &output_storage[1], COUNT);
  if (finish_and_check(name, COUNT, status) ||
      check_canonical_nan_outputs(name, COUNT)) {
    return 1;
  }
  if ((status & VPU_STATUS_FFLAG_NV) == 0) {
    printf("%s did not report NV for signaling NaN\n", name);
    return 1;
  }

  /* Stable softmax's literal IEEE sequence is invalid for +Inf-(+Inf). */
  name = "softmax_posinf_ieee";
  prepare_vectors(COUNT);
  input_storage[8] = storage_from_raw(VPU_TEST_POS_INF_RAW);
  refresh_references(COUNT);
  vpu_ref_softmax(input_reference, expected, COUNT);
  vpu_clear_status(VPU_CLEAR_ERRORS);
  status = vpu_tiled_softmax_auto(
      &input_storage[1], &output_storage[1], COUNT);
  if (finish_and_check(name, COUNT, status) ||
      check_canonical_nan_outputs(name, COUNT)) {
    return 1;
  }
  if ((status & VPU_STATUS_FFLAG_NV) == 0) {
    printf("%s did not report NV for Inf-Inf\n", name);
    return 1;
  }
  return 0;
}

static int test_kernel_special_values(void) {
  return test_rmsnorm_nonfinite() || test_silu_nonfinite() ||
         test_swiglu_nonfinite() || test_softmax_special();
}

/*
 * Exercise the architecturally important SFU special values through the full
 * DMA -> SRAM -> execute -> DMA path. Unit tests check ULP accuracy; this
 * bare-metal case checks canonical NaN, infinity/zero signs, subnormal input,
 * and sticky NV/DZ propagation in both storage configurations.
 */
static uint64_t run_special_unary(int reciprocal) {
  enum { SPECIAL_COUNT = 12 };
  prepare_vectors(SPECIAL_COUNT);
  input_storage[1] = storage_from_raw(VPU_TEST_QNAN_RAW);
  input_storage[2] = storage_from_raw(VPU_TEST_SNAN_RAW);
  input_storage[3] = storage_from_raw(VPU_TEST_POS_INF_RAW);
  input_storage[4] = storage_from_raw(VPU_TEST_NEG_INF_RAW);
  input_storage[5] = storage_from_raw(0u);
  input_storage[6] = storage_from_raw(VPU_TEST_NEG_ZERO_RAW);
  input_storage[7] = storage_from_raw(VPU_TEST_MIN_SUBNORMAL_RAW);
  input_storage[8] = storage_from_raw(VPU_TEST_MIN_SUBNORMAL_RAW |
                                      VPU_TEST_NEG_ZERO_RAW);
  input_storage[9] = vpu_float_to_storage(1.0f);
  input_storage[10] = vpu_float_to_storage(-1.0f);
  input_storage[11] = vpu_float_to_storage(10.0f);
  input_storage[12] = vpu_float_to_storage(-10.0f);

  vpu_clear_status(VPU_CLEAR_ERRORS);
  vpu_stream_configure_spad();
  vpu_stream_configure_memory(&input_storage[1], &input_storage[1],
                              &output_storage[1]);
  vpu_write_gp(VPU_STREAM_GP_PING_OFFSET, 0);
  vpu_set_vl(SPECIAL_COUNT);
  vpu_h_prefetch_v(VPU_STREAM_GP_PING_INPUT,
                   VPU_STREAM_GP_PING_OFFSET, VPU_STREAM_H_INPUT);
  if (reciprocal) {
    vpu_v_reci_v(VPU_STREAM_GP_PING_OUTPUT,
                 VPU_STREAM_GP_PING_INPUT);
  } else {
    vpu_v_exp_v(VPU_STREAM_GP_PING_OUTPUT,
                VPU_STREAM_GP_PING_INPUT);
  }
  vpu_h_store_v(VPU_STREAM_GP_PING_OUTPUT,
                VPU_STREAM_GP_PING_OFFSET, VPU_STREAM_H_OUTPUT);
  return vpu_fence();
}

static int check_special_common(const char *name, uint64_t status) {
  if ((status & VPU_STATUS_ERROR_MASK) != 0) {
    printf("%s special-value status error 0x%llx\n", name,
           (unsigned long long)status);
    return 1;
  }
  if ((status & VPU_STATUS_FFLAG_NV) == 0) {
    printf("%s did not report NV for signaling NaN\n", name);
    return 1;
  }
  if (storage_raw(output_storage[1]) != VPU_TEST_QNAN_RAW ||
      storage_raw(output_storage[2]) != VPU_TEST_QNAN_RAW) {
    printf("%s did not produce canonical qNaN\n", name);
    return 1;
  }
  return check_guards(name, 12);
}

static int test_special_values(void) {
  uint64_t status = run_special_unary(0);
  if (check_special_common("exp", status)) {
    return 1;
  }
  if (storage_raw(output_storage[3]) != VPU_TEST_POS_INF_RAW ||
      storage_raw(output_storage[4]) != 0u ||
      storage_raw(output_storage[5]) != storage_raw(vpu_float_to_storage(1.0f)) ||
      storage_raw(output_storage[6]) != storage_raw(vpu_float_to_storage(1.0f))) {
    printf("exp special infinity/zero result mismatch\n");
    return 1;
  }

  status = run_special_unary(1);
  if (check_special_common("reciprocal", status)) {
    return 1;
  }
  if ((status & VPU_STATUS_FFLAG_DZ) == 0) {
    printf("reciprocal did not report DZ for signed zero\n");
    return 1;
  }
  if (storage_raw(output_storage[3]) != 0u ||
      storage_raw(output_storage[4]) != VPU_TEST_NEG_ZERO_RAW ||
      storage_raw(output_storage[5]) != VPU_TEST_POS_INF_RAW ||
      storage_raw(output_storage[6]) != VPU_TEST_NEG_INF_RAW ||
      storage_raw(output_storage[7]) != VPU_TEST_POS_INF_RAW ||
      storage_raw(output_storage[8]) != VPU_TEST_NEG_INF_RAW) {
    printf("reciprocal special infinity/zero/subnormal result mismatch\n");
    return 1;
  }
  return 0;
}

static int planner_check_resident_address(
    const char *name, struct vpu_tiled_plan plan) {
  if (plan.resident_tiles == 0) {
    printf("%s planner unexpectedly has no resident tiles\n", name);
    return 1;
  }
  const unsigned address =
      vpu_tiled_resident_address(plan, plan.resident_tiles - 1u);
  const uint64_t end = (uint64_t)address + VPU_VLEN;
  if ((address % VPU_VLEN) != 0u || end > VPU_VSPAD_ELEMENTS) {
    printf("%s planner last resident address is invalid: address=%u "
           "end=%llu vspad_elements=%u\n",
           name, address, (unsigned long long)end,
           (unsigned)VPU_VSPAD_ELEMENTS);
    return 1;
  }
  return 0;
}

static int planner_expect(const char *name, struct vpu_tiled_plan plan,
                          enum vpu_tiled_schedule schedule,
                          size_t tile_count, size_t resident_tiles,
                          unsigned resident_banks,
                          unsigned workspace_banks) {
  if (plan.schedule != schedule || plan.tile_count != tile_count ||
      plan.resident_tiles != resident_tiles ||
      plan.resident_banks != resident_banks ||
      plan.workspace_banks != workspace_banks) {
    printf("%s planner mismatch: schedule=%s tiles=%u resident_tiles=%u "
           "resident_banks=%u workspace_banks=%u\n",
           name, vpu_tiled_schedule_name(plan.schedule),
           (unsigned)plan.tile_count, (unsigned)plan.resident_tiles,
           plan.resident_banks, plan.workspace_banks);
    return 1;
  }
  if (resident_tiles != 0 && planner_check_resident_address(name, plan)) {
    return 1;
  }
  return 0;
}

/* Exercise capacity transitions without simulating the corresponding rows. */
static int test_tiled_planner(void) {
  const size_t slots = VPU_SLOTS_PER_BANK;
  const size_t rms_full_tiles =
      ((size_t)VPU_VSPAD_BANKS - 2u) * slots;
  const size_t rms_partial_tiles =
      ((size_t)VPU_VSPAD_BANKS - 3u) * slots;
  const size_t softmax_full_tiles =
      ((size_t)VPU_VSPAD_BANKS - 1u) * slots;
  const size_t softmax_partial_tiles =
      ((size_t)VPU_VSPAD_BANKS - 2u) * slots;

  struct vpu_tiled_plan plan = vpu_tiled_plan_for(
      VPU_TILED_KERNEL_RMSNORM, rms_full_tiles * VPU_VLEN);
  if (planner_expect("rmsnorm-full", plan,
                     VPU_TILED_SCHEDULE_RESIDENT_FULL,
                     rms_full_tiles, rms_full_tiles,
                     VPU_VSPAD_BANKS - 2u, 2u)) {
    return 1;
  }

  plan = vpu_tiled_plan_for(
      VPU_TILED_KERNEL_RMSNORM, (rms_full_tiles + 1u) * VPU_VLEN);
  if (planner_expect("rmsnorm-partial", plan,
                     VPU_TILED_SCHEDULE_RESIDENT_PARTIAL,
                     rms_full_tiles + 1u, rms_partial_tiles,
                     VPU_VSPAD_BANKS - 3u, 3u)) {
    return 1;
  }

  plan = vpu_tiled_plan_for(
      VPU_TILED_KERNEL_SOFTMAX, softmax_full_tiles * VPU_VLEN);
  if (planner_expect("softmax-full", plan,
                     VPU_TILED_SCHEDULE_RESIDENT_FULL,
                     softmax_full_tiles, softmax_full_tiles,
                     VPU_VSPAD_BANKS - 1u, 1u)) {
    return 1;
  }

  plan = vpu_tiled_plan_for(
      VPU_TILED_KERNEL_SOFTMAX,
      (softmax_full_tiles + 1u) * VPU_VLEN);
  if (planner_expect("softmax-partial", plan,
                     VPU_TILED_SCHEDULE_RESIDENT_PARTIAL,
                     softmax_full_tiles + 1u, softmax_partial_tiles,
                     VPU_VSPAD_BANKS - 2u, 2u)) {
    return 1;
  }

  const size_t streaming_elements =
      (softmax_full_tiles + 1u) * VPU_VLEN;
  plan = vpu_tiled_plan_for(VPU_TILED_KERNEL_SILU, streaming_elements);
  if (planner_expect("silu-streaming", plan,
                     VPU_TILED_SCHEDULE_STREAMING,
                     softmax_full_tiles + 1u, 0, 0,
                     VPU_VSPAD_BANKS)) {
    return 1;
  }
  plan = vpu_tiled_plan_for(VPU_TILED_KERNEL_SWIGLU, streaming_elements);
  return planner_expect("swiglu-streaming", plan,
                        VPU_TILED_SCHEDULE_STREAMING,
                        softmax_full_tiles + 1u, 0, 0,
                        VPU_VSPAD_BANKS);
}

static int test_length(size_t n) {
  printf("VPU nonlinear length %u\n", (unsigned)n);
  return test_rmsnorm(n, 0) || test_rmsnorm(n, 1) || test_silu(n) ||
         test_swiglu(n) || test_softmax(n);
}

int main(void) {
  if (test_tiled_planner()) {
    return 1;
  }
#if VPU_CYCLE_BENCHMARK
  if (benchmark_check_alignment()) {
    return 1;
  }
  printf("VPU_BENCH metric=target_cycles cpu=scalar_fp32_libm "
         "vpu=e2e_config_dma_execute_store_fence buffers=aligned64\n");
  printf("VPU_BENCH ideal_model=useful_arithmetic_issue_lower_bound "
         "length=%u nlanes=%u exp_lanes=%u reciprocal_lanes=%u "
         "excludes=pipeline_dependency_reduction_fold_scalar_latency_"
         "tail_padding_sram_dma_tlb_rocc_fence\n",
         (unsigned)VPU_BENCHMARK_LENGTH, (unsigned)VPU_NLANES,
         (unsigned)VPU_SFU_LANES, (unsigned)VPU_RECIPROCAL_LANES);
#endif
  static const size_t required_lengths[] = {
      0, 1, 15, 16, 17, 127, 128, 129, 2048,
      2048
  };
  for (size_t i = 0;
       i < sizeof(required_lengths) / sizeof(required_lengths[0]); ++i) {
    if (test_length(required_lengths[i])) {
      return 1;
    }
  }
  if (test_finite_stress()) {
    return 1;
  }
  if (test_kernel_special_values()) {
    return 1;
  }
  if (test_special_values()) {
    return 1;
  }

  printf("VPU multi-tile RMSNorm/final-norm, SiLU, SwiGLU, and softmax "
         "finite-stress and kernel/SFU special-value tests passed\n");
  return 0;
}
