// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_BAREMETALC_VPU_VS_CPU_PERF_COMMON_H_
#define GEMMINI_ROCC_TESTS_BAREMETALC_VPU_VS_CPU_PERF_COMMON_H_

/*
 * Common driver for the one-operation-per-binary Rocket scalar versus VPU
 * performance programs.  CPU timing contains only the scalar reference body.
 * VPU timing contains the complete synchronous *_auto() call: command setup,
 * DMA, execution, store, and the final fence.  Input preparation, warm-up,
 * checking, counters, checksums, and UART output are outside both intervals.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "include/vpu.h"
#include "include/vpu_kernels.h"
#include "include/vpu_testutils.h"

#ifndef VPU_PERF_ELEMENTS
#define VPU_PERF_ELEMENTS 2048u
#endif
#ifndef VPU_PERF_ITERATIONS
#define VPU_PERF_ITERATIONS 1u
#endif
#ifndef VPU_PERF_WARMUP_ITERATIONS
#define VPU_PERF_WARMUP_ITERATIONS 0u
#endif
#ifndef VPU_PERF_CHECK
#define VPU_PERF_CHECK 1
#endif
#ifndef VPU_PERF_RMSNORM_EPSILON
#define VPU_PERF_RMSNORM_EPSILON 1.0e-5f
#endif
#ifndef VPU_PERF_FINAL_NORM_EPSILON
#define VPU_PERF_FINAL_NORM_EPSILON 1.0e-6f
#endif
#ifndef VPU_PERF_ROPE_ROWS
#define VPU_PERF_ROPE_ROWS 2048u
#endif
#ifndef VPU_PERF_ROPE_DIM
#define VPU_PERF_ROPE_DIM 2u
#endif
#ifndef VPU_PERF_ROPE_LAYOUT
#define VPU_PERF_ROPE_LAYOUT VPU_ROPE_INTERLEAVED
#endif
#ifndef VPU_PERF_PERMUTE_GROUPS
#define VPU_PERF_PERMUTE_GROUPS 2048u
#endif
#ifndef VPU_PERF_PERMUTE_GROUP_ELEMENTS
#define VPU_PERF_PERMUTE_GROUP_ELEMENTS 2u
#endif
#ifndef VPU_PERF_IN_PLACE
#define VPU_PERF_IN_PLACE 0
#endif
#ifndef VPU_PERF_BUFFER_ELEMENTS
#define VPU_PERF_BUFFER_ELEMENTS VPU_PERF_ELEMENTS
#endif
#ifndef VPU_VSPAD_SUBBANKS
#define VPU_VSPAD_SUBBANKS 1u
#endif

#if VPU_PERF_BUFFER_ELEMENTS < 1
#error "VPU performance buffers require at least one element"
#endif
#if VPU_PERF_ITERATIONS < 1
#error "VPU_PERF_ITERATIONS must be at least one"
#endif
#if VPU_PERF_CHECK != 0 && VPU_PERF_CHECK != 1
#error "VPU_PERF_CHECK must be zero or one"
#endif
#if VPU_PERF_IN_PLACE != 0 && VPU_PERF_IN_PLACE != 1
#error "VPU_PERF_IN_PLACE must be zero or one"
#endif

#define VPU_PERF_ALIGNMENT 64u
#define VPU_PERF_ALLOCATED_ELEMENTS (VPU_PERF_BUFFER_ELEMENTS + 1u)

static vpu_storage_t vpu_perf_input[VPU_PERF_ALLOCATED_ELEMENTS]
    __attribute__((aligned(VPU_PERF_ALIGNMENT)));
static vpu_storage_t vpu_perf_aux0[VPU_PERF_ALLOCATED_ELEMENTS]
    __attribute__((aligned(VPU_PERF_ALIGNMENT)));
static vpu_storage_t vpu_perf_aux1[VPU_PERF_ALLOCATED_ELEMENTS]
    __attribute__((aligned(VPU_PERF_ALIGNMENT)));
static vpu_storage_t vpu_perf_output[VPU_PERF_ALLOCATED_ELEMENTS]
    __attribute__((aligned(VPU_PERF_ALIGNMENT)));

static float vpu_perf_input_fp32[VPU_PERF_ALLOCATED_ELEMENTS]
    __attribute__((aligned(VPU_PERF_ALIGNMENT)));
static float vpu_perf_aux0_fp32[VPU_PERF_ALLOCATED_ELEMENTS]
    __attribute__((aligned(VPU_PERF_ALIGNMENT)));
static float vpu_perf_aux1_fp32[VPU_PERF_ALLOCATED_ELEMENTS]
    __attribute__((aligned(VPU_PERF_ALIGNMENT)));
static float vpu_perf_expected[VPU_PERF_ALLOCATED_ELEMENTS]
    __attribute__((aligned(VPU_PERF_ALIGNMENT)));
static float vpu_perf_cpu_work[VPU_PERF_ALLOCATED_ELEMENTS]
    __attribute__((aligned(VPU_PERF_ALIGNMENT)));
static vpu_index_t vpu_perf_indices[VPU_VLEN]
    __attribute__((aligned(VPU_PERF_ALIGNMENT)));

static volatile uint64_t vpu_perf_checksum_sink;

struct vpu_perf_problem {
  size_t elements;
  size_t rows;
  size_t width;
  float epsilon;
  enum vpu_tiled_kernel tiled_kernel;
  int has_tiled_plan;
  enum vpu_rope_layout rope_layout;
  int in_place;
};

typedef void (*vpu_perf_prepare_fn)(const struct vpu_perf_problem *);
typedef void (*vpu_perf_cpu_fn)(const struct vpu_perf_problem *);
typedef uint64_t (*vpu_perf_vpu_fn)(const struct vpu_perf_problem *);
typedef uint64_t (*vpu_perf_ideal_fn)(
    const struct vpu_perf_problem *, struct vpu_tiled_plan);

struct vpu_perf_counters {
  uint64_t diagnostic_cycles;
  uint64_t busy_cycles;
  uint64_t dma_read_bytes;
  uint64_t dma_write_bytes;
  uint64_t dma_exec_overlap_cycles;
  uint64_t bank_conflict_stall_cycles;
  uint64_t hazard_stall_cycles;
  uint64_t sfu_busy_cycles;
};

struct vpu_perf_ratio {
  uint64_t whole;
  uint64_t thousandths;
  int valid;
};

static inline uint64_t vpu_perf_read_cycles(void) {
#if defined(__riscv)
  uint64_t cycles;
  __asm__ volatile("rdcycle %0" : "=r"(cycles) : : "memory");
  return cycles;
#else
  return 0u;
#endif
}

static inline void vpu_perf_compiler_barrier(void) {
  __asm__ volatile("" : : : "memory");
}

static inline vpu_storage_t vpu_perf_guard(void) {
  return vpu_float_to_storage(-123.0f);
}

static inline void vpu_perf_prepare_standard(
    const struct vpu_perf_problem *problem) {
  const vpu_storage_t guard = vpu_perf_guard();
  for (size_t i = 0; i < problem->elements; ++i) {
    const float input = ((int)(i % 29u) - 14) * 0.125f;
    const float auxiliary = 0.75f + (float)(i % 11u) * 0.03125f;
    vpu_perf_input[i] = vpu_float_to_storage(input);
    vpu_perf_aux0[i] = vpu_float_to_storage(auxiliary);
    vpu_perf_aux1[i] = vpu_float_to_storage(0.0f);
    vpu_perf_output[i] = guard;
    vpu_perf_input_fp32[i] = vpu_storage_to_float(vpu_perf_input[i]);
    vpu_perf_aux0_fp32[i] = vpu_storage_to_float(vpu_perf_aux0[i]);
    vpu_perf_aux1_fp32[i] = 0.0f;
    vpu_perf_expected[i] = 0.0f;
    vpu_perf_cpu_work[i] = vpu_perf_input_fp32[i];
  }
  const size_t guard_index = problem->elements;
  vpu_perf_input[guard_index] = guard;
  vpu_perf_aux0[guard_index] = guard;
  vpu_perf_aux1[guard_index] = guard;
  vpu_perf_output[guard_index] = guard;
  vpu_perf_input_fp32[guard_index] = -123.0f;
  vpu_perf_aux0_fp32[guard_index] = -123.0f;
  vpu_perf_aux1_fp32[guard_index] = -123.0f;
  vpu_perf_expected[guard_index] = -123.0f;
  vpu_perf_cpu_work[guard_index] = -123.0f;
}

static inline void vpu_perf_prepare_rope(
    const struct vpu_perf_problem *problem) {
  vpu_perf_prepare_standard(problem);
  const size_t half = problem->width / 2u;
  for (size_t element = 0; element < problem->width; ++element) {
    const size_t frequency =
        problem->rope_layout == VPU_ROPE_INTERLEAVED
            ? element / 2u : element % half;
    const float angle = ((int)(frequency % 23u) - 11) * 0.03125f;
    vpu_perf_aux0[element] = vpu_float_to_storage(cosf(angle));
    vpu_perf_aux1[element] = vpu_float_to_storage(sinf(angle));
    vpu_perf_aux0_fp32[element] =
        vpu_storage_to_float(vpu_perf_aux0[element]);
    vpu_perf_aux1_fp32[element] =
        vpu_storage_to_float(vpu_perf_aux1[element]);
  }
}

static inline void vpu_perf_prepare_permute(
    const struct vpu_perf_problem *problem) {
  vpu_perf_prepare_standard(problem);
  for (size_t element = 0; element < problem->width; ++element)
    vpu_perf_indices[element] =
        (vpu_index_t)(problem->width - 1u - element);
}

static inline uint64_t vpu_perf_checksum_storage(
    const vpu_storage_t *values, size_t elements) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (size_t i = 0; i < elements; ++i) {
    hash ^= (uint64_t)vpu_float_to_bits(vpu_storage_to_float(values[i]));
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static inline uint64_t vpu_perf_checksum_fp32(const float *values,
                                              size_t elements) {
  uint64_t hash = UINT64_C(1469598103934665603);
  for (size_t i = 0; i < elements; ++i) {
    hash ^= (uint64_t)vpu_float_to_bits(values[i]);
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static inline struct vpu_perf_ratio vpu_perf_ratio_thousandths(
    uint64_t numerator, uint64_t denominator) {
  struct vpu_perf_ratio ratio = {0u, 0u, denominator != 0u};
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

static inline void vpu_perf_print_ratio(const char *name,
                                        struct vpu_perf_ratio ratio,
                                        const char *suffix) {
  if (!ratio.valid) {
    printf("%s=N/A", name);
    return;
  }
  printf("%s=%llu.%03llu%s", name,
         (unsigned long long)ratio.whole,
         (unsigned long long)ratio.thousandths, suffix);
}

static inline uint64_t vpu_perf_groups(size_t elements, uint64_t lanes) {
  return ((uint64_t)elements + lanes - 1u) / lanes;
}

static inline uint64_t vpu_perf_ideal_rmsnorm(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  (void)plan;
  return 4u * vpu_perf_groups(problem->elements, VPU_NLANES) + 4u;
}

static inline uint64_t vpu_perf_ideal_final_norm(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  return vpu_perf_ideal_rmsnorm(problem, plan);
}

static inline uint64_t vpu_perf_ideal_relu(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  (void)plan;
  return vpu_perf_groups(problem->elements, VPU_NLANES);
}

static inline uint64_t vpu_perf_ideal_sigmoid(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  (void)plan;
  return 2u * vpu_perf_groups(problem->elements, VPU_NLANES) +
      vpu_perf_groups(problem->elements, VPU_SFU_LANES) +
      vpu_perf_groups(problem->elements, VPU_RECIPROCAL_LANES);
}

static inline uint64_t vpu_perf_ideal_tanh(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  (void)plan;
  return 5u * vpu_perf_groups(problem->elements, VPU_NLANES) +
      vpu_perf_groups(problem->elements, VPU_SFU_LANES) +
      vpu_perf_groups(problem->elements, VPU_RECIPROCAL_LANES);
}

static inline uint64_t vpu_perf_ideal_gelu(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  (void)plan;
  return 4u * vpu_perf_groups(problem->elements, VPU_NLANES) +
      vpu_perf_groups(problem->elements, VPU_SFU_LANES) +
      vpu_perf_groups(problem->elements, VPU_RECIPROCAL_LANES);
}

static inline uint64_t vpu_perf_ideal_silu(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  (void)plan;
  return 3u * vpu_perf_groups(problem->elements, VPU_NLANES) +
      vpu_perf_groups(problem->elements, VPU_SFU_LANES) +
      vpu_perf_groups(problem->elements, VPU_RECIPROCAL_LANES);
}

static inline uint64_t vpu_perf_ideal_swiglu(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  (void)plan;
  return 4u * vpu_perf_groups(problem->elements, VPU_NLANES) +
      vpu_perf_groups(problem->elements, VPU_SFU_LANES) +
      vpu_perf_groups(problem->elements, VPU_RECIPROCAL_LANES);
}

static inline uint64_t vpu_perf_ideal_softmax(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  const size_t overflow = vpu_tiled_plan_overflow_elements(plan);
  return 4u * vpu_perf_groups(problem->elements, VPU_NLANES) +
      vpu_perf_groups(problem->elements, VPU_SFU_LANES) +
      vpu_perf_groups(overflow, VPU_NLANES) +
      vpu_perf_groups(overflow, VPU_SFU_LANES) + 1u;
}

static inline uint64_t vpu_perf_ideal_rope(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  (void)plan;
  const uint64_t words = vpu_perf_groups(problem->width, VPU_NLANES);
  const uint64_t shift = problem->rope_layout == VPU_ROPE_INTERLEAVED
      ? 1u : (uint64_t)problem->width / 2u;
  const uint64_t low_source_words = shift < problem->width
      ? vpu_perf_groups(problem->width - shift, VPU_NLANES) : 0u;
  const uint64_t high_source_words = shift < problem->width
      ? words - shift / VPU_NLANES : 0u;

  /* Each of the two slide commands spends prepare+write on every SRAM word
   * and read-request+read-response on words which have an in-range source.
   * The remaining SUB/MUL/MUL/ADD commands have a one-word/cycle issue lower
   * bound.  Pipeline drains, dispatch, and SRAM backpressure remain excluded. */
  const uint64_t slide_cycles =
      4u * words + 2u * (low_source_words + high_source_words);
  const uint64_t arithmetic_cycles = 4u * words;
  return (uint64_t)problem->rows *
      (slide_cycles + arithmetic_cycles);
}

static inline uint64_t vpu_perf_ideal_permute(
    const struct vpu_perf_problem *problem, struct vpu_tiled_plan plan) {
  (void)plan;
  uint64_t group_cycles = 0u;
  for (size_t offset = 0u; offset < problem->width;
       offset += VPU_NLANES) {
    const size_t remaining = problem->width - offset;
    const uint64_t active = remaining < VPU_NLANES
        ? remaining : VPU_NLANES;
    /* The correctness-first gather FSM reads one index word, then serializes
     * one source-word request/response per active destination lane.  The
     * fixed term covers index request/response, end-of-lane fold, and write. */
    group_cycles += 4u + VPU_NLANES + 2u * active;
  }
  return (uint64_t)problem->rows * group_cycles;
}

static inline struct vpu_perf_counters vpu_perf_snapshot_counters(void) {
  struct vpu_perf_counters counters;
  counters.diagnostic_cycles = vpu_read_perf(VPU_PERF_CYCLES);
  counters.busy_cycles = vpu_read_perf(VPU_PERF_BUSY_CYCLES);
  counters.dma_read_bytes = vpu_read_perf(VPU_PERF_DMA_READ_BYTES);
  counters.dma_write_bytes = vpu_read_perf(VPU_PERF_DMA_WRITE_BYTES);
  counters.dma_exec_overlap_cycles =
      vpu_read_perf(VPU_PERF_DMA_EXEC_OVERLAP_CYCLES);
  counters.bank_conflict_stall_cycles =
      vpu_read_perf(VPU_PERF_BANK_CONFLICT_STALL_CYCLES);
  counters.hazard_stall_cycles =
      vpu_read_perf(VPU_PERF_HAZARD_STALL_CYCLES);
  counters.sfu_busy_cycles = vpu_read_perf(VPU_PERF_SFU_BUSY_CYCLES);
  return counters;
}

static inline int vpu_perf_validate_problem(
    const char *operation, const struct vpu_perf_problem *problem) {
  if (problem->elements == 0u ||
      problem->elements > VPU_PERF_BUFFER_ELEMENTS ||
      problem->rows == 0u || problem->width == 0u ||
      problem->rows > SIZE_MAX / problem->width ||
      problem->rows * problem->width != problem->elements) {
    printf("NONLINEAR_PERF op=%s invalid_shape elements=%llu rows=%llu "
           "width=%llu capacity=%llu\n", operation,
           (unsigned long long)problem->elements,
           (unsigned long long)problem->rows,
           (unsigned long long)problem->width,
           (unsigned long long)VPU_PERF_BUFFER_ELEMENTS);
    return 1;
  }
  if ((problem->rope_layout != VPU_ROPE_INTERLEAVED &&
       problem->rope_layout != VPU_ROPE_NEOX) ||
      (strcmp(operation, "rope") == 0 &&
       (problem->width > VPU_VLEN || (problem->width & 1u) != 0u)) ||
      (strcmp(operation, "permute") == 0 && problem->width > VPU_VLEN)) {
    printf("NONLINEAR_PERF op=%s unsupported_layout rows=%llu width=%llu "
           "layout=%u vlen=%u\n", operation,
           (unsigned long long)problem->rows,
           (unsigned long long)problem->width,
           (unsigned)problem->rope_layout, (unsigned)VPU_VLEN);
    return 1;
  }
  return 0;
}

static inline int vpu_perf_check_result(
    const char *operation, const struct vpu_perf_problem *problem,
    uint32_t *max_error_bits) {
  const float *expected = problem->in_place
      ? vpu_perf_cpu_work : vpu_perf_expected;
  const vpu_storage_t *actual = problem->in_place
      ? vpu_perf_input : vpu_perf_output;
#if VPU_STORAGE_KIND == VPU_STORAGE_BF16
  const float relative_tolerance = 1.0e-2f;
  const float absolute_tolerance = 1.0e-3f;
#else
  const float relative_tolerance = 2.0e-4f;
  const float absolute_tolerance = 2.0e-6f;
#endif
  float max_error = 0.0f;
  int errors = 0;
  for (size_t i = 0; i < problem->elements; ++i) {
    const float got = vpu_storage_to_float(actual[i]);
    const float error = fabsf(got - expected[i]);
    max_error = fmaxf(max_error, error);
    if (!vpu_ref_close(got, expected[i], relative_tolerance,
                       absolute_tolerance)) {
      if (errors < 8) {
        printf("NONLINEAR_PERF mismatch op=%s index=%llu got_bits=0x%x "
               "expected_bits=0x%x error_bits=0x%x\n", operation,
               (unsigned long long)i, vpu_float_to_bits(got),
               vpu_float_to_bits(expected[i]), vpu_float_to_bits(error));
      }
      ++errors;
    }
  }
  const vpu_storage_t guard = vpu_perf_guard();
  if (vpu_perf_input[problem->elements] != guard ||
      vpu_perf_aux0[problem->elements] != guard ||
      vpu_perf_aux1[problem->elements] != guard ||
      vpu_perf_output[problem->elements] != guard) {
    printf("NONLINEAR_PERF op=%s tail_guard_corrupted\n", operation);
    ++errors;
  }
  *max_error_bits = vpu_float_to_bits(max_error);
  return errors;
}

static inline int vpu_perf_run(
    const char *operation, struct vpu_perf_problem problem,
    vpu_perf_prepare_fn prepare, vpu_perf_cpu_fn cpu,
    vpu_perf_vpu_fn vpu, vpu_perf_ideal_fn ideal) {
  if (vpu_perf_validate_problem(operation, &problem)) return 1;
  if (problem.in_place != 0 && strcmp(operation, "rope") != 0 &&
      strcmp(operation, "permute") != 0) {
    printf("NONLINEAR_PERF op=%s in_place_not_supported\n", operation);
    return 1;
  }

  struct vpu_tiled_plan plan;
  memset(&plan, 0, sizeof(plan));
  if (problem.has_tiled_plan)
    plan = vpu_tiled_plan_for(problem.tiled_kernel, problem.elements);

  prepare(&problem);
#if VPU_PERF_WARMUP_ITERATIONS > 0
  for (unsigned iteration = 0u;
       iteration < VPU_PERF_WARMUP_ITERATIONS; ++iteration) {
    cpu(&problem);
    vpu_perf_compiler_barrier();
    const uint64_t warm_status = vpu(&problem);
    if ((warm_status & VPU_STATUS_ERROR_MASK) != 0u) {
      printf("NONLINEAR_PERF op=%s warmup=%u status_error=0x%llx\n",
             operation, iteration, (unsigned long long)warm_status);
      return 1;
    }
  }
#endif

  /* Reset in-place operands and all outputs after warm-up. */
  prepare(&problem);

  const uint64_t cpu_start = vpu_perf_read_cycles();
  for (unsigned iteration = 0u; iteration < VPU_PERF_ITERATIONS;
       ++iteration) {
    cpu(&problem);
    vpu_perf_compiler_barrier();
  }
  const uint64_t cpu_total_cycles = vpu_perf_read_cycles() - cpu_start;

  (void)vpu_fence();
  vpu_clear_status(VPU_CLEAR_ALL);
  (void)vpu_fence();
  uint64_t status = 0u;
  const uint64_t vpu_start = vpu_perf_read_cycles();
  for (unsigned iteration = 0u; iteration < VPU_PERF_ITERATIONS;
       ++iteration) {
    status = vpu(&problem);
    if ((status & VPU_STATUS_ERROR_MASK) != 0u) break;
  }
  const uint64_t vpu_total_cycles = vpu_perf_read_cycles() - vpu_start;
  const struct vpu_perf_counters counters = vpu_perf_snapshot_counters();

  if ((status & VPU_STATUS_ERROR_MASK) != 0u) {
    printf("NONLINEAR_PERF op=%s result=ERROR status=0x%llx "
           "fault_addr=0x%llx fault_info=0x%llx\n", operation,
           (unsigned long long)status,
           (unsigned long long)vpu_read_fault_address(),
           (unsigned long long)vpu_read_fault_info());
    return 1;
  }

  const uint64_t cpu_cycles_per_iteration =
      cpu_total_cycles / VPU_PERF_ITERATIONS;
  const uint64_t vpu_cycles_per_iteration =
      vpu_total_cycles / VPU_PERF_ITERATIONS;
  const uint64_t ideal_cycles = ideal(&problem, plan);
  const struct vpu_perf_ratio speedup = vpu_perf_ratio_thousandths(
      cpu_total_cycles, vpu_total_cycles);
  const struct vpu_perf_ratio e2e_over_ideal =
      vpu_perf_ratio_thousandths(vpu_cycles_per_iteration, ideal_cycles);

  const float *cpu_result = problem.in_place
      ? vpu_perf_cpu_work : vpu_perf_expected;
  const vpu_storage_t *vpu_result = problem.in_place
      ? vpu_perf_input : vpu_perf_output;
  const uint64_t cpu_checksum =
      vpu_perf_checksum_fp32(cpu_result, problem.elements);
  const uint64_t vpu_checksum =
      vpu_perf_checksum_storage(vpu_result, problem.elements);
  vpu_perf_checksum_sink ^= cpu_checksum ^ vpu_checksum;

  printf("NONLINEAR_PERF metric=target_cycles cpu=rocket_scalar_fp32 "
         "vpu=auto_e2e_config_dma_execute_store_fence "
         "op=%s elements=%llu rows=%llu width=%llu iterations=%u "
         "warmups=%u in_place=%u storage=%s\n", operation,
         (unsigned long long)problem.elements,
         (unsigned long long)problem.rows,
         (unsigned long long)problem.width,
         (unsigned)VPU_PERF_ITERATIONS,
         (unsigned)VPU_PERF_WARMUP_ITERATIONS,
         (unsigned)problem.in_place,
#if VPU_STORAGE_KIND == VPU_STORAGE_BF16
         "bf16"
#else
         "fp32"
#endif
  );
  printf("NONLINEAR_PERF hardware vlen=%u nlanes=%u sfu_lanes=%u "
         "reciprocal_lanes=%u vsram_kib=%u banks=%u subbanks=%u "
         "dma_bits=%u dma_max_bytes=%u dma_in_flight=%u "
         "nonlinear_chunk_rows=%u physical_batch_rows=%u\n",
         (unsigned)VPU_VLEN, (unsigned)VPU_NLANES,
         (unsigned)VPU_SFU_LANES, (unsigned)VPU_RECIPROCAL_LANES,
         (unsigned)VPU_VSPAD_KIB, (unsigned)VPU_VSPAD_BANKS,
         (unsigned)VPU_VSPAD_SUBBANKS, (unsigned)VPU_DMA_BUS_BITS,
         (unsigned)VPU_DMA_MAX_BYTES, (unsigned)VPU_DMA_MAX_IN_FLIGHT,
         (unsigned)vpu_stream_batch_capacity(),
         (unsigned)vpu_stream_physical_batch_capacity());
  printf("NONLINEAR_PERF cycles cpu_total=%llu vpu_e2e_total=%llu "
         "cpu_per_iteration=%llu vpu_e2e_per_iteration=%llu "
         "ideal_compute_per_iteration=%llu ",
         (unsigned long long)cpu_total_cycles,
         (unsigned long long)vpu_total_cycles,
         (unsigned long long)cpu_cycles_per_iteration,
         (unsigned long long)vpu_cycles_per_iteration,
         (unsigned long long)ideal_cycles);
  vpu_perf_print_ratio("cpu_over_vpu", speedup, "x");
  printf(" ");
  vpu_perf_print_ratio("vpu_e2e_over_ideal", e2e_over_ideal, "x");
  printf("\n");
  printf("NONLINEAR_PERF schedule=%s resident_tiles=%llu/%llu "
         "resident_banks=%u workspace_banks=%u\n",
         problem.has_tiled_plan
             ? vpu_tiled_schedule_name(plan.schedule) : "streaming_auto",
         (unsigned long long)(problem.has_tiled_plan
             ? plan.resident_tiles : 0u),
         (unsigned long long)(problem.has_tiled_plan
             ? plan.tile_count : 0u),
         problem.has_tiled_plan ? plan.resident_banks : 0u,
         problem.has_tiled_plan ? plan.workspace_banks : VPU_VSPAD_BANKS);
  printf("NONLINEAR_PERF counter_scope=aggregate_iterations_not_e2e "
         "diagnostic_vpu_cycles=%llu busy_cycles=%llu "
         "sfu_busy_cycles=%llu dma_read_bytes=%llu dma_write_bytes=%llu "
         "dma_exec_overlap_cycles=%llu bank_conflict_stall_cycles=%llu "
         "hazard_stall_cycles=%llu\n",
         (unsigned long long)counters.diagnostic_cycles,
         (unsigned long long)counters.busy_cycles,
         (unsigned long long)counters.sfu_busy_cycles,
         (unsigned long long)counters.dma_read_bytes,
         (unsigned long long)counters.dma_write_bytes,
         (unsigned long long)counters.dma_exec_overlap_cycles,
         (unsigned long long)counters.bank_conflict_stall_cycles,
         (unsigned long long)counters.hazard_stall_cycles);

#if VPU_PERF_CHECK
  uint32_t max_error_bits = 0u;
  const int errors =
      vpu_perf_check_result(operation, &problem, &max_error_bits);
  printf("NONLINEAR_PERF result=%s errors=%d max_abs_error_bits=0x%x "
         "cpu_checksum=0x%llx vpu_checksum=0x%llx sink=0x%llx "
         "status=0x%llx\n", errors == 0 ? "PASS" : "FAIL", errors,
         max_error_bits, (unsigned long long)cpu_checksum,
         (unsigned long long)vpu_checksum,
         (unsigned long long)vpu_perf_checksum_sink,
         (unsigned long long)status);
  return errors == 0 ? 0 : 1;
#else
  printf("NONLINEAR_PERF result=UNCHECKED cpu_checksum=0x%llx "
         "vpu_checksum=0x%llx sink=0x%llx status=0x%llx\n",
         (unsigned long long)cpu_checksum,
         (unsigned long long)vpu_checksum,
         (unsigned long long)vpu_perf_checksum_sink,
         (unsigned long long)status);
  return 0;
#endif
}

#endif  // GEMMINI_ROCC_TESTS_BAREMETALC_VPU_VS_CPU_PERF_COMMON_H_
