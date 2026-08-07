// See LICENSE for license details.

#include "vpu_vs_cpu_perf_common.h"

static void swiglu_prepare(const struct vpu_perf_problem *problem) {
  vpu_perf_prepare_standard(problem);
}

static void swiglu_cpu(const struct vpu_perf_problem *problem) {
  vpu_ref_swiglu(vpu_perf_input_fp32, vpu_perf_aux0_fp32,
                  vpu_perf_expected, problem->elements);
}

static uint64_t swiglu_vpu(const struct vpu_perf_problem *problem) {
  return vpu_swiglu_auto(vpu_perf_input, vpu_perf_aux0, vpu_perf_output,
                         problem->elements);
}

static uint64_t swiglu_ideal(const struct vpu_perf_problem *problem,
                             struct vpu_tiled_plan plan) {
  return vpu_perf_ideal_swiglu(problem, plan);
}

int main(void) {
  const struct vpu_perf_problem problem = {
      .elements = VPU_PERF_ELEMENTS,
      .rows = 1,
      .width = VPU_PERF_ELEMENTS,
      .tiled_kernel = VPU_TILED_KERNEL_SWIGLU,
      .has_tiled_plan = 1,
  };
  return vpu_perf_run("swiglu", problem, swiglu_prepare, swiglu_cpu,
                      swiglu_vpu, swiglu_ideal);
}
