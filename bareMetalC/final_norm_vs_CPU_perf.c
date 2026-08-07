// See LICENSE for license details.

#include "vpu_vs_cpu_perf_common.h"

static void final_norm_prepare(const struct vpu_perf_problem *problem) {
  vpu_perf_prepare_standard(problem);
}

static void final_norm_cpu(const struct vpu_perf_problem *problem) {
  vpu_ref_rmsnorm(vpu_perf_input_fp32, vpu_perf_aux0_fp32,
                  vpu_perf_expected, problem->elements, problem->epsilon);
}

static uint64_t final_norm_vpu(const struct vpu_perf_problem *problem) {
  return vpu_final_norm_auto(vpu_perf_input, vpu_perf_aux0, vpu_perf_output,
                             problem->elements, problem->epsilon);
}

static uint64_t final_norm_ideal(const struct vpu_perf_problem *problem,
                                 struct vpu_tiled_plan plan) {
  return vpu_perf_ideal_final_norm(problem, plan);
}

int main(void) {
  const struct vpu_perf_problem problem = {
      .elements = VPU_PERF_ELEMENTS,
      .rows = 1,
      .width = VPU_PERF_ELEMENTS,
      .epsilon = VPU_PERF_FINAL_NORM_EPSILON,
      .tiled_kernel = VPU_TILED_KERNEL_FINAL_NORM,
      .has_tiled_plan = 1,
  };
  return vpu_perf_run("final_norm", problem, final_norm_prepare,
                      final_norm_cpu, final_norm_vpu, final_norm_ideal);
}
