// See LICENSE for license details.

#include "vpu_vs_cpu_perf_common.h"

static void sigmoid_prepare(const struct vpu_perf_problem *problem) {
  vpu_perf_prepare_standard(problem);
}

static void sigmoid_cpu(const struct vpu_perf_problem *problem) {
  vpu_ref_sigmoid(vpu_perf_input_fp32, vpu_perf_expected,
                  problem->elements);
}

static uint64_t sigmoid_vpu(const struct vpu_perf_problem *problem) {
  return vpu_sigmoid_auto(vpu_perf_input, vpu_perf_output,
                          problem->elements);
}

static uint64_t sigmoid_ideal(const struct vpu_perf_problem *problem,
                              struct vpu_tiled_plan plan) {
  return vpu_perf_ideal_sigmoid(problem, plan);
}

int main(void) {
  const struct vpu_perf_problem problem = {
      .elements = VPU_PERF_ELEMENTS,
      .rows = 1,
      .width = VPU_PERF_ELEMENTS,
  };
  return vpu_perf_run("sigmoid", problem, sigmoid_prepare, sigmoid_cpu,
                      sigmoid_vpu, sigmoid_ideal);
}
