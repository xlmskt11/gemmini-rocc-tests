// See LICENSE for license details.

#include "include/gemmini_tiling.h"

#include <assert.h>
#include <stddef.h>

typedef struct {
  size_t dim_I;
  size_t dim_J;
  size_t dim_K;
  size_t gemmini_count;
  size_t sp_addr_range;
  size_t acc_addr_range;
  bool double_buffered;
  int act;
  size_t expected_I;
  size_t expected_J;
  size_t expected_K;
} tiling_case_t;

int main(void) {
  static const tiling_case_t cases[] = {
      {16, 64, 32, 2, 8192, 2048, true, 0, 1, 4, 2},
      {17, 65, 33, 4, 8192, 2048, true, 0, 2, 5, 3},
      {2048, 2048, 2048, 4, 8192, 2048, true, 0, 8, 8, 16},
      {128, 2048, 64, 4, 8192, 256, true, 0, 4, 2, 4},
      {32, 128, 64, 2, 8192, 2048, false, 0, 2, 8, 4},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    gemmini_tiling_request_t request;
    request.dim_I = cases[i].dim_I;
    request.dim_J = cases[i].dim_J;
    request.dim_K = cases[i].dim_K;
    request.dim = 16;
    request.gemmini_count = cases[i].gemmini_count;
    request.sp_addr_range = cases[i].sp_addr_range;
    request.acc_addr_range = cases[i].acc_addr_range;
    request.double_buffered = cases[i].double_buffered;
    request.act = cases[i].act;

    const gemmini_tiling_factors_t factors =
        gemmini_shared_multi_choose_tiling(&request);
    assert(factors.tile_I == cases[i].expected_I);
    assert(factors.tile_J == cases[i].expected_J);
    assert(factors.tile_K == cases[i].expected_K);
    assert(factors.dim_I_padded >= request.dim_I);
    assert(factors.dim_J_padded >= request.dim_J);
    assert(factors.dim_K_padded >= request.dim_K);
    assert(factors.dim_I_padded % request.dim == 0);
    assert(factors.dim_J_padded % request.dim == 0);
    assert(factors.dim_K_padded % request.dim == 0);
  }

  return 0;
}
