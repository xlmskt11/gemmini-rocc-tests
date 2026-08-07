// See LICENSE for license details.

#define VPU_PERF_BUFFER_ELEMENTS \
  ((VPU_PERF_PERMUTE_GROUPS) * (VPU_PERF_PERMUTE_GROUP_ELEMENTS))

#include "vpu_vs_cpu_perf_common.h"

static void permute_prepare(const struct vpu_perf_problem *problem) {
  vpu_perf_prepare_permute(problem);
}

static void permute_cpu(const struct vpu_perf_problem *problem) {
  float *output = problem->in_place
      ? vpu_perf_cpu_work : vpu_perf_expected;
  const float *input = problem->in_place
      ? vpu_perf_cpu_work : vpu_perf_input_fp32;
  vpu_ref_permute(input, vpu_perf_indices, output,
                  problem->rows, problem->width);
}

static uint64_t permute_vpu(const struct vpu_perf_problem *problem) {
  return vpu_permute_auto(vpu_perf_input, vpu_perf_indices,
                          problem->in_place ? vpu_perf_input
                                            : vpu_perf_output,
                          problem->rows, problem->width);
}

static uint64_t permute_ideal(const struct vpu_perf_problem *problem,
                              struct vpu_tiled_plan plan) {
  return vpu_perf_ideal_permute(problem, plan);
}

int main(void) {
  const struct vpu_perf_problem problem = {
      .elements = (size_t)VPU_PERF_PERMUTE_GROUPS *
                  (size_t)VPU_PERF_PERMUTE_GROUP_ELEMENTS,
      .rows = VPU_PERF_PERMUTE_GROUPS,
      .width = VPU_PERF_PERMUTE_GROUP_ELEMENTS,
      .in_place = VPU_PERF_IN_PLACE,
  };
  return vpu_perf_run("permute", problem, permute_prepare, permute_cpu,
                      permute_vpu, permute_ideal);
}
