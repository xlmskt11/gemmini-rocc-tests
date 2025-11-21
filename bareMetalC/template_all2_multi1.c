// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils_all.h"
// #include "include/gemmini_nn.h"

#define gemmini_configuration 15
#define MAT_DIM_I 192
#define MAT_DIM_J 256
#define MAT_DIM_K 512

#define profile_data_num 30

#define NO_BIAS false
#define FULL_BIAS_WIDTH true
#define REPEATING_BIAS false

#define RAND rand()
#define FAST true
#define CHECK false
#define FENCE true
#define PROFILE false
#define MULTI false

#define q_type(p) (p >> 62)
#define start(p) ((p >> 31) & ((1 << 31) - 1))
#define end(p) (p & ((1 << 31) - 1))

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
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
    static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
    static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
    static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

#if !FAST && CHECK
    static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
    static elem_t gold[MAT_DIM_I][MAT_DIM_J];
#endif

    printf("Init A\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i)
    {
      for (size_t j = 0; j < MAT_DIM_K; ++j)
      {
        full_A[i][j] = RAND % 2;
      }
    }

    printf("Init D\n");
    for (size_t i = 0; i < MAT_DIM_I; ++i) {
      for (size_t j = 0; j < MAT_DIM_J; ++j) {
        full_D[i][j] = NO_BIAS ? 0 : RAND % 2;
        // full_D[i][j] = NO_BIAS ? 0 : 5;
      }
    }

#if FAST
  // identity matrix
  printf("Init B\n");
  for (size_t i = 0; i < MAT_DIM_K; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j) {
      full_B[i][j] = i == j;
    }
  }
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
  int tile_id = 1;
  printf("Do Gemmini tiled matmul process\n");
  uint64_t matmul_start = read_cycles();
  // shared_multi_tiled_matmul_auto(gemmini_configuration, tile_id,
  //                                0, 0,
  //                                MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
  //                                (elem_t *)full_A, (elem_t *)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t *)full_C,
  //                                MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
  //                                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
  //                                NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
  //                                false, false,
  //                                false, !FULL_BIAS_WIDTH,
  //                                1,
  //                                WS);
  // shared_multi_tiled_matmul_auto_test(gemmini_configuration, tile_id,
  //                                0, 0,
  //                                BANK_NUM * BANK_ROWS / 2, ACC_ROWS / 2,
  //                                MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
  //                                (elem_t *)full_A, (elem_t *)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t *)full_C,
  //                                MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
  //                                MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
  //                                NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
  //                                false, false,
  //                                false, !FULL_BIAS_WIDTH,
  //                                1,
  //                                WS);
#if MULTI
  // 1) 각 matmul에 대해 tiling factor 자동 계산
  shared_multi_matmul_job_t j0;

  size_t spad_start_addr = 0;
  size_t acc_start_addr = 0;
  size_t spad_rows_used_0, acc_rows_used_0;
  size_t tile_I0, tile_J0, tile_K0;
  shared_multi_choose_tiling_factors(
      gemmini_configuration,
      spad_start_addr, acc_start_addr,
      TOTAL_SPAD_ROWS, TOTAL_ACC_ROWS,
      MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
      NO_ACTIVATION, WS,
      &tile_I0, &tile_J0, &tile_K0,
      &spad_rows_used_0, &acc_rows_used_0);

  shared_multi_tiled_matmul_job_init(
      &j0,
      gemmini_configuration, tile_id,
      spad_start_addr, acc_start_addr,
      spad_rows_used_0, acc_rows_used_0,
      MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
      (elem_t *)full_A, (elem_t *)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t *)full_C,
      MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      tile_I0, tile_J0, tile_K0,
      NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
      false, false,
      false, !FULL_BIAS_WIDTH,
      1,
      WEIGHT_STATIONARY);

  while (!j0.done)
  {
    shared_multi_tiled_matmul_job_step(&j0);
  }
#else
  tiled_matmul_auto(custom3, MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
                    (elem_t *)full_A, (elem_t *)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t *)full_C,
                    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    1,
                    WS);
#endif
#if FENCE
  gemmini_fence();
#endif

  uint64_t matmul_end = read_cycles();
  printf("Matmul cycle: %d\n", matmul_end - matmul_start);

#if CHECK
  printf("Check \"Out\" matrix\n");

#if FAST
  if (!is_equal_dynamic(full_C, full_A, MAT_DIM_I, MAT_DIM_J)) {
    printf("Incorrect output matrix!\n");
    printf("C:\n");
    printMatrix_dynamic(full_C, MAT_DIM_I, MAT_DIM_J);
    printf("A:\n");
    printMatrix_dynamic(full_A, MAT_DIM_I, MAT_DIM_J);
    printf("\n");

    exit(1);
  }
#else
  if (!is_equal_dynamic(full_C, gold, MAT_DIM_I, MAT_DIM_J)) {
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

