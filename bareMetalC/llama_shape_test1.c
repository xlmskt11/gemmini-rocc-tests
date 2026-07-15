// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils_all.h"
// #include "include/gemmini_nn.h"

#define PROFILE false
#define profile_data_num 20000

#define MULTI true
#define gemmini_configuration 3

#define MAT_DIM_I 192
#define MAT_DIM_J 512
#define MAT_DIM_K 2048
#define RAND rand()
#define FAST true
#define NO_BIAS true
#define FULL_BIAS_WIDTH true
#define FULL_C_WIDTH true
#define REPEATING_BIAS false
#define CHECK false

#define PACKED_A false
#define PACKED_B false
#define PACKED_C false
#define PACKED_D false

#define q_type(p) (p >> 62)
#define start(p) ((p >> 31) & ((1UL << 31) - 1))
#define end(p) (p & ((1UL << 31) - 1))

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

#if FULL_C_WIDTH
typedef acc_t CACC_T;
#else
typedef elem_t CACC_T;
#endif

void print_gemmini_use(unsigned mask)
{
  int idx[4], n = 0;

  // bit0→gemmini0, bit1→gemmini1, bit2→gemmini2, bit3→gemmini3
  for (int g = 0; g < 4; ++g)
  {
    if (mask & (1u << g))
      idx[n++] = g;
  }

  printf("Use %d Gemmini", n);
  if (n > 0)
  {
    printf(" ");
    for (int i = 0; i < n; ++i)
    {
      printf("%d%s", idx[i], (i == n - 1) ? "" : ", ");
    }
  }
  printf(" with DIM: %d\n", DIM);
}

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J], ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k]*B[k][c];
    }
}

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J]) {
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J], acc_scale_t scale) {
  for (size_t r = 0; r < MAT_DIM_I; r++)                             
    for (size_t c = 0; c < MAT_DIM_J; c++) {
      // Scale element
      full_t scaled = ACC_SCALE(full[r][c], scale);

      // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
} 


int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

#if PROFILE
    printf("Set profiler address\n");
    static uint64_t P[total_gemmini_num][profile_data_num] row_align(1);
#if MULTI
    gemmini_profiler(custom0, (uint64_t *)P[0]);
    gemmini_profiler(custom1, (uint64_t *)P[1]);
    gemmini_profiler(custom2, (uint64_t *)P[2]);
#endif
    gemmini_profiler(custom3, (uint64_t *)P[3]);
#endif

#if MULTI
  print_gemmini_use(gemmini_configuration);
#else
  printf("Use Single Gemmini: %d\n", DIM);
#endif
    printf("MAT_DIM_I: %d\n", MAT_DIM_I);
    printf("MAT_DIM_J: %d\n", MAT_DIM_J);
    printf("MAT_DIM_K: %d\n", MAT_DIM_K);

    printf("Flush All Gemmini TLB of stale virtual addresses\n");
#if MULTI
    gemmini_flush(custom0, 0);
    gemmini_flush(custom1, 0);
    gemmini_flush(custom2, 0);
#endif
    gemmini_flush(custom3, 0);

    printf("Initialize our input and output matrices in main memory\n");
    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
    static CACC_T full_C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
    static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN_ACC);

#if !FAST && CHECK
    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];
#endif

    // printf("Init A\n");
    // for (size_t i = 0; i < MAT_DIM_I; ++i)
    // {
    //   for (size_t j = 0; j < MAT_DIM_K; ++j)
    //   {
    //     full_A[i][j] = RAND % 2;
    //   }
    // }

    // printf("Init D\n");
    // for (size_t i = 0; i < MAT_DIM_I; ++i) {
    //   for (size_t j = 0; j < MAT_DIM_J; ++j) {
    //     full_D[i][j] = NO_BIAS ? 0 : RAND % 2;
    //   }
    // }

#if FAST
  // identity matrix
  // printf("Init B\n");
  // for (size_t i = 0; i < MAT_DIM_K; ++i) {
  //   for (size_t j = 0; j < MAT_DIM_J; ++j) {
  //     full_B[i][j] = i == j;
  //   }
  // }
#else
  printf("Init B\n");
  for (size_t i = 0; i < MAT_DIM_K; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j) {
      full_B[i][j] = RAND % 2;
    }
  }
#if CHECK
  printf("Calculate Output\n");
  full_matmul(full_A, full_B, full_D, gold_full);
  full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);
#endif
#endif
  int total_cycles = 0;
  printf("Do Gemmini tiled matmul process\n");
#if MULTI
  shared_multi_matmul_job_t jm;
  memset(&jm, 0, sizeof(jm));
  jm.tile_id = 0;
  jm.gemmini_list = gemmini_configuration;
  jm.sp_addr_start_stack = 0;
  jm.sp_addr_end_stack = 0;
  jm.acc_addr_start_stack = 0;
  jm.sp_addr_range = TOTAL_SPAD_ROWS;
  jm.acc_addr_range = TOTAL_ACC_ROWS;
  jm.dim_I = MAT_DIM_I;
  jm.dim_J = MAT_DIM_J;
  jm.dim_K = MAT_DIM_K;
  jm.A = (elem_t *)full_A;
  jm.B = (elem_t *)full_B;
  jm.D = NO_BIAS ? NULL : &full_D[0][0];
  jm.C = (CACC_T *)full_C;
  jm.stride_A = PACKED_A ? GEMMINI_PAGE_PACKED_STRIDE(MAT_DIM_K) : MAT_DIM_K;
  jm.stride_B = PACKED_B ? GEMMINI_PAGE_PACKED_STRIDE(MAT_DIM_J) : MAT_DIM_J;
  jm.stride_D = NO_BIAS ? 0 : (PACKED_D ? GEMMINI_PAGE_PACKED_STRIDE(MAT_DIM_J) : MAT_DIM_J);
  jm.stride_C = PACKED_C ? GEMMINI_PAGE_PACKED_STRIDE(MAT_DIM_J) : MAT_DIM_J;
  jm.A_scale_factor = MVIN_SCALE_IDENTITY;
  jm.B_scale_factor = MVIN_SCALE_IDENTITY;
  jm.D_scale_factor = MVIN_SCALE_IDENTITY;
  jm.act = NO_ACTIVATION;
  jm.scale = ACC_SCALE_IDENTITY;
  jm.bert_scale = 0;
  jm.repeating_bias = REPEATING_BIAS;
  jm.a_transpose = false;
  jm.b_transpose = false;
  jm.full_C = FULL_C_WIDTH;
  jm.low_D = !FULL_BIAS_WIDTH;
  jm.weightA = 1;
  jm.dataflow = WEIGHT_STATIONARY;
  shared_multi_choose_tiling_factors(&jm);
  printf("Tiling factors: tile_I=%d, tile_J=%d, tile_K=%d\n", jm.tile_I, jm.tile_J, jm.tile_K);
  shared_multi_tiled_matmul_job_init(&jm);

  uint64_t matmul_start = read_cycles();
  while (!jm.done)
  {
    shared_multi_tiled_matmul_job_step(&jm);
  }

  gemmini_fence();
  uint64_t matmul_end = read_cycles();

#if CHECK
  printf("Check \"Out\" matrix\n");

#if FAST
  if (!is_equal_dynamic(full_C, full_A, MAT_DIM_I, MAT_DIM_J))
  {
    printf("Incorrect output matrix!\n");
    printf("C:\n");
    printMatrix_dynamic(full_C, MAT_DIM_I, MAT_DIM_J);
    printf("A:\n");
    printMatrix_dynamic(full_A, MAT_DIM_I, MAT_DIM_J);
    printf("\n");

    exit(1);
  }
#else
  if (!is_equal_dynamic(full_C, gold, MAT_DIM_I, MAT_DIM_J))
  {
    printf("Incorrect output matrix!\n");
    printf("C:\n");
    printMatrix_dynamic(full_C, MAT_DIM_I, MAT_DIM_J);
    printf("Gold:\n");
    printMatrix_dynamic(gold, MAT_DIM_I, MAT_DIM_J);
    printf("\n");

    exit(1);
  }
#endif
  printf("Output matrix came out as expected\n");
#endif

#else
  // uint64_t matmul_start = read_cycles();
  // tiled_matmul_auto(custom3, MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
  //                   (elem_t *)full_A, (elem_t *)full_B, NO_BIAS ? NULL : &full_D[0][0], (CACC_T *)full_C,
  //                   MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
  //                   MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
  //                   NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
  //                   false, false,
  //                   false, !FULL_BIAS_WIDTH,
  //                   1,
  //                   WS);
  // gemmini_fence();
  // uint64_t matmul_end = read_cycles();
  const int custom_num = custom3;
  tiled_matmul_single_job_t js;
  memset(&js, 0, sizeof(js));
  js.custom_num = custom_num;
  js.dim_I = MAT_DIM_I;
  js.dim_J = MAT_DIM_J;
  js.dim_K = MAT_DIM_K;
  js.A = (elem_t *)full_A;
  js.B = (elem_t *)full_B;
  js.D = NO_BIAS ? NULL : &full_D[0][0];
  js.C = (CACC_T *)full_C;
  js.stride_A = PACKED_A ? GEMMINI_PAGE_PACKED_STRIDE(MAT_DIM_K) : MAT_DIM_K;
  js.stride_B = PACKED_B ? GEMMINI_PAGE_PACKED_STRIDE(MAT_DIM_J) : MAT_DIM_J;
  js.stride_D = NO_BIAS ? 0 : (PACKED_D ? GEMMINI_PAGE_PACKED_STRIDE(MAT_DIM_J) : MAT_DIM_J);
  js.stride_C = PACKED_C ? GEMMINI_PAGE_PACKED_STRIDE(MAT_DIM_J) : MAT_DIM_J;
  js.A_scale_factor = MVIN_SCALE_IDENTITY;
  js.B_scale_factor = MVIN_SCALE_IDENTITY;
  js.D_scale_factor = MVIN_SCALE_IDENTITY;
  js.act = NO_ACTIVATION;
  js.scale = ACC_SCALE_IDENTITY;
  js.bert_scale = 0;
  js.repeating_bias = REPEATING_BIAS;
  js.a_transpose = false;
  js.b_transpose = false;
  js.full_C = FULL_C_WIDTH;
  js.low_D = !FULL_BIAS_WIDTH;
  js.weightA = 1;
  js.dataflow = WEIGHT_STATIONARY;
  const tiled_matmul_auto_factors_t tiling = tiled_matmul_auto_reduced_bank_conflict_factors(
      MAT_DIM_I,
      MAT_DIM_J,
      MAT_DIM_K,
      NO_ACTIVATION,
      WS);
  js.tile_I = tiling.tile_I;
  js.tile_J = tiling.tile_J;
  js.tile_K = tiling.tile_K;
  printf("Tiling factors: tile_I=%d, tile_J=%d, tile_K=%d\n", js.tile_I, js.tile_J, js.tile_K);
  tiled_matmul_single_job_init(&js);
  uint64_t matmul_start = read_cycles();
  while (!js.done)
  {
    tiled_matmul_single_job_step(&js);
  }
  gemmini_fence();
  uint64_t matmul_end = read_cycles();
#endif

  printf("Total Matmul cycle: %d\n", matmul_end - matmul_start);

#if PROFILE
  for (int i = 0; i < total_gemmini_num; i++)
  {
    for (int j = 0; j < profile_data_num; j++)
    {
      if (P[i][j] == 0)
      {
        break;
      }
      printf("%d, %d, %d, %d\n", i, q_type(P[i][j]), start(P[i][j]), end(P[i][j]));
    }
  }
#endif

  exit(0);
}
