// See LICENSE for license details.

#include "vpu_vs_cpu_perf_common.h"

static void rmsnorm_prepare(const struct vpu_perf_problem *problem) {
  vpu_perf_prepare_standard(problem);
}

static void rmsnorm_cpu(const struct vpu_perf_problem *problem) {
  vpu_ref_rmsnorm(vpu_perf_input_fp32, vpu_perf_aux0_fp32,
                  vpu_perf_expected, problem->elements, problem->epsilon);
}

static uint64_t rmsnorm_vpu(const struct vpu_perf_problem *problem) {
  return vpu_rmsnorm_auto(vpu_perf_input, vpu_perf_aux0, vpu_perf_output,
                          problem->elements, problem->epsilon);
}

static uint64_t rmsnorm_ideal(const struct vpu_perf_problem *problem,
                              struct vpu_tiled_plan plan) {
  return vpu_perf_ideal_rmsnorm(problem, plan);
}

int main(void) {
  const struct vpu_perf_problem problem = {
      .elements = VPU_PERF_ELEMENTS,
      .rows = 1,
      .width = VPU_PERF_ELEMENTS,
      .epsilon = VPU_PERF_RMSNORM_EPSILON,
      .tiled_kernel = VPU_TILED_KERNEL_RMSNORM,
      .has_tiled_plan = 1,
  };
  return vpu_perf_run("rmsnorm", problem, rmsnorm_prepare, rmsnorm_cpu,
                      rmsnorm_vpu, rmsnorm_ideal);
}
