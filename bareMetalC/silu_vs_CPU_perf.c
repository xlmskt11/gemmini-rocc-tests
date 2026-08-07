// See LICENSE for license details.

#include "vpu_vs_cpu_perf_common.h"

static void silu_prepare(const struct vpu_perf_problem *problem) {
  vpu_perf_prepare_standard(problem);
}

static void silu_cpu(const struct vpu_perf_problem *problem) {
  vpu_ref_silu(vpu_perf_input_fp32, vpu_perf_expected, problem->elements);
}

static uint64_t silu_vpu(const struct vpu_perf_problem *problem) {
  return vpu_silu_auto(vpu_perf_input, vpu_perf_output, problem->elements);
}

static uint64_t silu_ideal(const struct vpu_perf_problem *problem,
                           struct vpu_tiled_plan plan) {
  return vpu_perf_ideal_silu(problem, plan);
}

int main(void) {
  const struct vpu_perf_problem problem = {
      .elements = VPU_PERF_ELEMENTS,
      .rows = 1,
      .width = VPU_PERF_ELEMENTS,
      .tiled_kernel = VPU_TILED_KERNEL_SILU,
      .has_tiled_plan = 1,
  };
  return vpu_perf_run("silu", problem, silu_prepare, silu_cpu, silu_vpu,
                      silu_ideal);
}
