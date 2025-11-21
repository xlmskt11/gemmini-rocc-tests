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

#define gemmini_configuration_m0 1
#define gemmini_configuration_m1 14

#define MAT_DIM_I_m0 96
#define MAT_DIM_J_m0 96
#define MAT_DIM_K_m0 96

#define MAT_DIM_I_m1 192
#define MAT_DIM_J_m1 192
#define MAT_DIM_K_m1 192

#define profile_data_num 30

#define NO_BIAS true
#define FULL_BIAS_WIDTH true
#define REPEATING_BIAS false

#define RAND rand()
#define FAST true
#define CHECK true
#define FENCE true
#define PROFILE false
#define MULTI true

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

void Init_input_matrix(elem_t * M, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            M[i * cols + j] = RAND % 2;
        }
    }
}

void Init_bias_matrix(ACC_T * M, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            M[i * cols + j] = NO_BIAS ? 0 : RAND % 2;
        }
    }
}

void Init_weight_matrix(elem_t * M, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            M[i * cols + j] = FAST ? i == j : RAND % 2;
        }
    }
}

void full_matmul_dynamic(elem_t * A, elem_t * B, ACC_T * D, full_t * C_full, size_t I, size_t J, size_t K)
{
  for (size_t i=0; i<I; i++) {
    for (size_t j=0; j<J; j++) {
      C_full[i*J + j] = D[i*J + j];
      for (size_t k=0; k<K; k++) {
        C_full[i*J + j] += A[i*K + k] * B[k*J + j];
      }
    }
  }
}

void full_printMatrix_dynamic(elem_t * m, size_t rows, size_t cols)
{
  for (size_t i = 0; i < rows; ++i)
  {
    for (size_t j = 0; j < cols; ++j)
      printf("%d ", m[i * cols + j]);
    printf("\n");
  }
}

int full_is_equal_dynamic(elem_t * x, elem_t * y, size_t rows, size_t cols)
{
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      if (x[i * cols + j] != y[i * cols + j])
        return 0;
  return 1;
}

void full_matscale_dynamic(full_t * full, elem_t * out, size_t rows, size_t cols, acc_scale_t scale)
{
  for (size_t r = 0; r < rows; r++)
    for (size_t c = 0; c < cols; c++)
    {
      // Scale element
      full_t scaled = ACC_SCALE(full[r * cols + c], scale);

    // Saturate and cast element
#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r * cols + c] = elem;
#else
      out[r * cols + c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
}

int main()
{
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
  {
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

  print_gemmini_use(gemmini_configuration_m0);
  printf("MAT_DIM_I_m0: %d\n", MAT_DIM_I_m0);
  printf("MAT_DIM_J_m0: %d\n", MAT_DIM_J_m0);
  printf("MAT_DIM_K_m0: %d\n", MAT_DIM_K_m0);

  print_gemmini_use(gemmini_configuration_m1);
  printf("MAT_DIM_I_m1: %d\n", MAT_DIM_I_m1);
  printf("MAT_DIM_J_m1: %d\n", MAT_DIM_J_m1);
  printf("MAT_DIM_K_m1: %d\n", MAT_DIM_K_m1);

  printf("Flush All Gemmini TLB of stale virtual addresses\n");
#if MULTI
  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
#endif
  gemmini_flush(custom3, 0);

  printf("Initialize our input and output matrices in main memory\n");
  static elem_t full_A_m0[MAT_DIM_I_m0][MAT_DIM_K_m0] row_align(1);
  static elem_t full_B_m0[MAT_DIM_K_m0][MAT_DIM_J_m0] row_align(1);
  static elem_t full_C_m0[MAT_DIM_I_m0][MAT_DIM_J_m0] row_align(1);
  static ACC_T full_D_m0[MAT_DIM_I_m0][MAT_DIM_J_m0] row_align_acc(1);

  static elem_t full_A_m1[MAT_DIM_I_m1][MAT_DIM_K_m1] row_align(1);
  static elem_t full_B_m1[MAT_DIM_K_m1][MAT_DIM_J_m1] row_align(1);
  static elem_t full_C_m1[MAT_DIM_I_m1][MAT_DIM_J_m1] row_align(1);
  static ACC_T full_D_m1[MAT_DIM_I_m1][MAT_DIM_J_m1] row_align_acc(1);

#if !FAST && CHECK
  static full_t gold_full_m0[MAT_DIM_I_m0][MAT_DIM_J_m0];
  static elem_t gold_m0[MAT_DIM_I_m0][MAT_DIM_J_m0];
  static full_t gold_full_m1[MAT_DIM_I_m1][MAT_DIM_J_m1];
  static elem_t gold_m1[MAT_DIM_I_m1][MAT_DIM_J_m1];
#endif

  printf("Init A\n");
  Init_input_matrix(&full_A_m0[0][0], MAT_DIM_I_m0, MAT_DIM_K_m0);
  Init_input_matrix(&full_A_m1[0][0], MAT_DIM_I_m1, MAT_DIM_K_m1);

  printf("Init D\n");
  Init_bias_matrix(&full_D_m0[0][0], MAT_DIM_I_m0, MAT_DIM_J_m0);
  Init_bias_matrix(&full_D_m1[0][0], MAT_DIM_I_m1, MAT_DIM_J_m1);

  // identity matrix
  printf("Init B\n");
  Init_weight_matrix(&full_B_m0[0][0], MAT_DIM_K_m0, MAT_DIM_J_m0);
  Init_weight_matrix(&full_B_m1[0][0], MAT_DIM_K_m1, MAT_DIM_J_m1);

#if !FAST && CHECK
  printf("Calculate Output\n");
  full_matmul_dynamic(&full_A_m0[0][0], &full_B_m0[0][0], &full_D_m0[0][0], &gold_full_m0[0][0], MAT_DIM_I_m0, MAT_DIM_J_m0, MAT_DIM_K_m0);
  full_matmul_dynamic(&full_A_m1[0][0], &full_B_m1[0][0], &full_D_m1[0][0], &gold_full_m1[0][0], MAT_DIM_I_m1, MAT_DIM_J_m1, MAT_DIM_K_m1);

  full_matscale_dynamic(&gold_full_m0[0][0], &gold_m0[0][0], MAT_DIM_I_m0, MAT_DIM_J_m0, ACC_SCALE_IDENTITY);
  full_matscale_dynamic(&gold_full_m1[0][0], &gold_m1[0][0], MAT_DIM_I_m1, MAT_DIM_J_m1, ACC_SCALE_IDENTITY);
#endif
  int tile_id_m0 = 1;
  int tile_id_m1 = 2;
  printf("Do Gemmini tiled matmul process\n");
  uint64_t matmul_start = read_cycles();
  // shared_multi_tiled_matmul_auto_test(gemmini_configuration_m0, tile_id_m0,
  //                                     0, 0,
  //                                     BANK_NUM * BANK_ROWS / 4, ACC_ROWS / 4,
  //                                     MAT_DIM_I_m0, MAT_DIM_J_m0, MAT_DIM_K_m0,
  //                                     (elem_t *)full_A_m0, (elem_t *)full_B_m0, NO_BIAS ? NULL : &full_D_m0[0][0], (elem_t *)full_C_m0,
  //                                     MAT_DIM_K_m0, MAT_DIM_J_m0, MAT_DIM_J_m0, MAT_DIM_J_m0,
  //                                     MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
  //                                     NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
  //                                     false, false,
  //                                     false, !FULL_BIAS_WIDTH,
  //                                     1,
  //                                     WS);
  // shared_multi_tiled_matmul_auto_test(gemmini_configuration_m1, tile_id_m1,
  //                                     BANK_NUM * BANK_ROWS / 4, ACC_ROWS / 4,
  //                                     BANK_NUM * BANK_ROWS * 3 / 4, ACC_ROWS * 3 / 4,
  //                                     MAT_DIM_I_m1, MAT_DIM_J_m1, MAT_DIM_K_m1,
  //                                     (elem_t *)full_A_m1, (elem_t *)full_B_m1, NO_BIAS ? NULL : &full_D_m1[0][0], (elem_t *)full_C_m1,
  //                                     MAT_DIM_K_m1, MAT_DIM_J_m1, MAT_DIM_J_m1, MAT_DIM_J_m1,
  //                                     MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
  //                                     NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
  //                                     false, false,
  //                                     false, !FULL_BIAS_WIDTH,
  //                                     1,
  //                                     WS);

#if MULTI
  // 1) 각 matmul에 대해 tiling factor 자동 계산
  shared_multi_matmul_job_t j0, j1;

  size_t spad_start_addr = 0;
  size_t acc_start_addr = 0;
  size_t spad_rows_used_0, acc_rows_used_0;
  size_t tile_I0, tile_J0, tile_K0;
  size_t spad_rows_used_1, acc_rows_used_1;
  size_t tile_I1, tile_J1, tile_K1;
  shared_multi_choose_tiling_factors(
      gemmini_configuration_m0,
      spad_start_addr, acc_start_addr,
      TOTAL_SPAD_ROWS / 4, TOTAL_ACC_ROWS / 4,
      MAT_DIM_I_m0, MAT_DIM_J_m0, MAT_DIM_K_m0,
      NO_ACTIVATION, WS,
      &tile_I0, &tile_J0, &tile_K0,
      &spad_rows_used_0, &acc_rows_used_0);

  shared_multi_tiled_matmul_job_init(
      &j0,
      gemmini_configuration_m0, tile_id_m0,
      spad_start_addr, acc_start_addr,
      spad_rows_used_0, acc_rows_used_0,
      MAT_DIM_I_m0, MAT_DIM_J_m0, MAT_DIM_K_m0,
      (elem_t *)full_A_m0, (elem_t *)full_B_m0, NO_BIAS ? NULL : &full_D_m0[0][0], (elem_t *)full_C_m0,
      MAT_DIM_K_m0, MAT_DIM_J_m0, MAT_DIM_J_m0, MAT_DIM_J_m0,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      tile_I0, tile_J0, tile_K0,
      NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
      false, false,
      false, !FULL_BIAS_WIDTH,
      1,
      WEIGHT_STATIONARY);

  spad_start_addr += spad_rows_used_0;
  acc_start_addr += acc_rows_used_0;

  shared_multi_choose_tiling_factors(
      gemmini_configuration_m1,
      spad_start_addr, acc_start_addr,
      TOTAL_SPAD_ROWS - spad_start_addr, TOTAL_ACC_ROWS - acc_start_addr,
      MAT_DIM_I_m1, MAT_DIM_J_m1, MAT_DIM_K_m1,
      NO_ACTIVATION, WS,
      &tile_I1, &tile_J1, &tile_K1,
      &spad_rows_used_1, &acc_rows_used_1);

  shared_multi_tiled_matmul_job_init(
      &j1,
      gemmini_configuration_m1, tile_id_m1,
      spad_start_addr, acc_start_addr,
      spad_rows_used_1, acc_rows_used_1,
      MAT_DIM_I_m1, MAT_DIM_J_m1, MAT_DIM_K_m1,
      (elem_t *)full_A_m1, (elem_t *)full_B_m1, NO_BIAS ? NULL : &full_D_m1[0][0], (elem_t *)full_C_m1,
      MAT_DIM_K_m1, MAT_DIM_J_m1, MAT_DIM_J_m1, MAT_DIM_J_m1,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      tile_I1, tile_J1, tile_K1,
      NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
      false, false,
      false, !FULL_BIAS_WIDTH,
      1,
      WEIGHT_STATIONARY);

  spad_start_addr += spad_rows_used_1;
  acc_start_addr += acc_rows_used_1;

  while (!j0.done || !j1.done)
  {
    if (!j0.done)
      shared_multi_tiled_matmul_job_step(&j0);
    if (!j1.done)
      shared_multi_tiled_matmul_job_step(&j1);
  }

#else
  tiled_matmul_auto(custom3, MAT_DIM_I_m0, MAT_DIM_J_m0, MAT_DIM_K_m0,
                    (elem_t *)full_A_m0, (elem_t *)full_B_m0, NO_BIAS ? NULL : &full_D_m0[0][0], (elem_t *)full_C_m0,
                    MAT_DIM_K_m0, MAT_DIM_J_m0, MAT_DIM_J_m0, MAT_DIM_J_m0,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    1,
                    WS);
  tiled_matmul_auto(custom3, MAT_DIM_I_m1, MAT_DIM_J_m1, MAT_DIM_K_m1,
                    (elem_t *)full_A_m1, (elem_t *)full_B_m1, NO_BIAS ? NULL : &full_D_m1[0][0], (elem_t *)full_C_m1,
                    MAT_DIM_K_m1, MAT_DIM_J_m1, MAT_DIM_J_m1, MAT_DIM_J_m1,
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

#if MULTI
  printf("total spad_rows reserved: %d\n", spad_start_addr);
  printf("total acc_rows reserved: %d\n\n", acc_start_addr);

  printf("scratchpad row utilization: %d%%\n", (spad_start_addr * 100) / TOTAL_SPAD_ROWS);
  printf("accumulator row utilization: %d%%\n\n", (acc_start_addr * 100) / TOTAL_ACC_ROWS);
#endif

  printf("Matmul cycle: %d\n", matmul_end - matmul_start);

#if CHECK
  printf("Check \"Out\" matrix\n");

#if FAST
  if (!is_equal_dynamic(full_C_m0, full_A_m0, MAT_DIM_I_m0, MAT_DIM_J_m0))
  {
    printf("Incorrect output matrix!\n");
    printf("C_m0:\n");
    printMatrix_dynamic(full_C_m0, MAT_DIM_I_m0, MAT_DIM_J_m0);
    printf("A_m0:\n");
    printMatrix_dynamic(full_A_m0, MAT_DIM_I_m0, MAT_DIM_J_m0);
    printf("\n");

    exit(1);
  }

  if (!is_equal_dynamic(full_C_m1, full_A_m1, MAT_DIM_I_m1, MAT_DIM_J_m1))
  {
    printf("Incorrect output matrix!\n");
    printf("C_m1:\n");
    printMatrix_dynamic(full_C_m1, MAT_DIM_I_m1, MAT_DIM_J_m1);
    printf("A_m1:\n");
    printMatrix_dynamic(full_A_m1, MAT_DIM_I_m1, MAT_DIM_J_m1);
    printf("\n");

    exit(1);
  }
#else
  if (!is_equal_dynamic(full_C_m0, gold_m0, MAT_DIM_I_m0, MAT_DIM_J_m0))
  {
    printf("Incorrect output matrix!\n");
    printf("C_m0:\n");
    printMatrix_dynamic(full_C_m0, MAT_DIM_I_m0, MAT_DIM_J_m0);
    printf("Gold_m0:\n");
    printMatrix_dynamic(gold_m0, MAT_DIM_I_m0, MAT_DIM_J_m0);
    printf("\n");

    exit(1);
  }
  if (!is_equal_dynamic(full_C_m1, gold_m1, MAT_DIM_I_m1, MAT_DIM_J_m1))
  {
    printf("Incorrect output matrix!\n");
    printf("C_m1:\n");
    printMatrix_dynamic(full_C_m1, MAT_DIM_I_m1, MAT_DIM_J_m1);
    printf("Gold_m1:\n");
    printMatrix_dynamic(gold_m1, MAT_DIM_I_m1, MAT_DIM_J_m1);
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
