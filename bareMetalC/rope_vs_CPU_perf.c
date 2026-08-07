// See LICENSE for license details.

#define VPU_PERF_BUFFER_ELEMENTS \
  ((VPU_PERF_ROPE_ROWS) * (VPU_PERF_ROPE_DIM))

#include "vpu_vs_cpu_perf_common.h"

#ifndef VPU_PERF_ROPE_LAYOUT
#define VPU_PERF_ROPE_LAYOUT VPU_ROPE_NEOX
#endif

static void rope_prepare(const struct vpu_perf_problem *problem) {
  vpu_perf_prepare_rope(problem);
}

static void rope_cpu(const struct vpu_perf_problem *problem) {
  float *output = problem->in_place
      ? vpu_perf_cpu_work : vpu_perf_expected;
  const float *input = problem->in_place
      ? vpu_perf_cpu_work : vpu_perf_input_fp32;
  vpu_ref_rope(input, vpu_perf_aux0_fp32,
               vpu_perf_aux1_fp32, output, problem->rows,
               problem->width, problem->rope_layout);
}

static uint64_t rope_vpu(const struct vpu_perf_problem *problem) {
  return vpu_rope_auto(vpu_perf_input, vpu_perf_aux0, vpu_perf_aux1,
                       problem->in_place ? vpu_perf_input : vpu_perf_output,
                       problem->rows, problem->width,
                       problem->rope_layout);
}

static uint64_t rope_ideal(const struct vpu_perf_problem *problem,
                           struct vpu_tiled_plan plan) {
  return vpu_perf_ideal_rope(problem, plan);
}

int main(void) {
  const struct vpu_perf_problem problem = {
      .elements = (size_t)VPU_PERF_ROPE_ROWS *
                  (size_t)VPU_PERF_ROPE_DIM,
      .rows = VPU_PERF_ROPE_ROWS,
      .width = VPU_PERF_ROPE_DIM,
      .rope_layout = VPU_PERF_ROPE_LAYOUT,
      .in_place = VPU_PERF_IN_PLACE,
  };
  return vpu_perf_run("rope", problem, rope_prepare, rope_cpu, rope_vpu,
                      rope_ideal);
}
