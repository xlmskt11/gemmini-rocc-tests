// See LICENSE for license details.

#include "vpu_vs_cpu_perf_common.h"

static void gelu_prepare(const struct vpu_perf_problem *problem) {
  vpu_perf_prepare_standard(problem);
}

static void gelu_cpu(const struct vpu_perf_problem *problem) {
  vpu_ref_gelu(vpu_perf_input_fp32, vpu_perf_expected, problem->elements);
}

static uint64_t gelu_vpu(const struct vpu_perf_problem *problem) {
  return vpu_gelu_auto(vpu_perf_input, vpu_perf_output, problem->elements);
}

static uint64_t gelu_ideal(const struct vpu_perf_problem *problem,
                           struct vpu_tiled_plan plan) {
  return vpu_perf_ideal_gelu(problem, plan);
}

int main(void) {
  const struct vpu_perf_problem problem = {
      .elements = VPU_PERF_ELEMENTS,
      .rows = 1,
      .width = VPU_PERF_ELEMENTS,
  };
  return vpu_perf_run("gelu", problem, gelu_prepare, gelu_cpu, gelu_vpu,
                      gelu_ideal);
}
