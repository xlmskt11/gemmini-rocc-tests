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
#define profile_data_num 5000

#define MULTI true
#define gemmini_configuration_m0 3
#define gemmini_configuration_m1 12

#define MAT_DIM_I 256
#define MAT_DIM_J 256
#define MAT_DIM_K 256
#define RAND rand()
#define FAST false
#define NO_BIAS false
#define FULL_BIAS_WIDTH true
#define REPEATING_BIAS false
#define CHECK false

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

  // bit0 -> gemmini0, bit1 -> gemmini1, bit2 -> gemmini2, bit3 -> gemmini3
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

void full_matmul(elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J],
                 ACC_T D[MAT_DIM_I][MAT_DIM_J], full_t C_full[MAT_DIM_I][MAT_DIM_J])
{
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++)
    {
      C_full[r][c] = D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; k++)
        C_full[r][c] += A[r][k] * B[k][c];
    }
}

void full_printMatrix(elem_t m[MAT_DIM_I][MAT_DIM_J])
{
  for (size_t i = 0; i < MAT_DIM_I; ++i)
  {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      printf("%d ", m[i][j]);
    printf("\n");
  }
}

int full_is_equal(elem_t x[MAT_DIM_I][MAT_DIM_J], elem_t y[MAT_DIM_I][MAT_DIM_J])
{
  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      if (x[i][j] != y[i][j])
        return 0;
  return 1;
}

void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J], elem_t out[MAT_DIM_I][MAT_DIM_J],
                   acc_scale_t scale)
{
  for (size_t r = 0; r < MAT_DIM_I; r++)
    for (size_t c = 0; c < MAT_DIM_J; c++)
    {
      full_t scaled = ACC_SCALE(full[r][c], scale);

#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled; // TODO should we also saturate when using floats?
#endif
    }
}

static void init_input_matrix(elem_t M[MAT_DIM_I][MAT_DIM_K])
{
  for (size_t i = 0; i < MAT_DIM_I; ++i)
  {
    for (size_t j = 0; j < MAT_DIM_K; ++j)
    {
      M[i][j] = RAND % 2;
    }
  }
}

static void init_bias_matrix(ACC_T M[MAT_DIM_I][MAT_DIM_J])
{
  for (size_t i = 0; i < MAT_DIM_I; ++i)
  {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
    {
      M[i][j] = NO_BIAS ? 0 : RAND % 2;
    }
  }
}

static void init_weight_matrix(elem_t M[MAT_DIM_K][MAT_DIM_J])
{
#if FAST
  for (size_t i = 0; i < MAT_DIM_K; ++i)
  {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
    {
      M[i][j] = i == j;
    }
  }
#else
  for (size_t i = 0; i < MAT_DIM_K; ++i)
  {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
    {
      M[i][j] = RAND % 2;
    }
  }
#endif
}

static void init_matmul_job(shared_multi_matmul_job_t *job, int tile_id, int gemmini_list,
                            elem_t A[MAT_DIM_I][MAT_DIM_K], elem_t B[MAT_DIM_K][MAT_DIM_J],
                            ACC_T D[MAT_DIM_I][MAT_DIM_J], elem_t C[MAT_DIM_I][MAT_DIM_J],
                            size_t sp_addr_start_stack, size_t sp_addr_end_stack, size_t acc_addr_start_stack,
                                size_t tile_I,
                            size_t tile_J, size_t tile_K)
{
  memset(job, 0, sizeof(*job));
  job->tile_id = tile_id;
  job->gemmini_list = gemmini_list;
  job->sp_addr_start_stack = sp_addr_start_stack;
  job->sp_addr_end_stack = sp_addr_end_stack;
  job->acc_addr_start_stack = acc_addr_start_stack;
  job->sp_addr_range = TOTAL_SPAD_ROWS;
  job->acc_addr_range = TOTAL_ACC_ROWS;
  job->dim_I = MAT_DIM_I;
  job->dim_J = MAT_DIM_J;
  job->dim_K = MAT_DIM_K;
  job->A = (elem_t *)A;
  job->B = (elem_t *)B;
  job->D = NO_BIAS ? NULL : &D[0][0];
  job->C = (elem_t *)C;
  job->stride_A = MAT_DIM_K;
  job->stride_B = MAT_DIM_J;
  job->stride_D = MAT_DIM_J;
  job->stride_C = MAT_DIM_J;
  job->A_scale_factor = MVIN_SCALE_IDENTITY;
  job->B_scale_factor = MVIN_SCALE_IDENTITY;
  job->D_scale_factor = MVIN_SCALE_IDENTITY;
  job->act = NO_ACTIVATION;
  job->scale = ACC_SCALE_IDENTITY;
  job->bert_scale = 0;
  job->repeating_bias = REPEATING_BIAS;
  job->a_transpose = false;
  job->b_transpose = false;
  job->full_C = false;
  job->low_D = !FULL_BIAS_WIDTH;
  job->weightA = 1;
  job->dataflow = WEIGHT_STATIONARY;

  shared_multi_choose_tiling_factors_static(job, tile_I, tile_J, tile_K);
  shared_multi_tiled_matmul_job_init(job);
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

#if MULTI
  printf("Matmul 0 group:\n");
  print_gemmini_use(gemmini_configuration_m0);
  printf("Matmul 1 group:\n");
  print_gemmini_use(gemmini_configuration_m1);
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
  static elem_t full_A_m0[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
  static elem_t full_B_m0[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
  static elem_t full_C_m0[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
  static ACC_T full_D_m0[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN_ACC);

  static elem_t full_A_m1[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
  static elem_t full_B_m1[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
  static elem_t full_C_m1[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
  static ACC_T full_D_m1[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN_ACC);

#if !FAST && CHECK
  static full_t gold_full_m0[MAT_DIM_I][MAT_DIM_J];
  static elem_t gold_m0[MAT_DIM_I][MAT_DIM_J];
  static full_t gold_full_m1[MAT_DIM_I][MAT_DIM_J];
  static elem_t gold_m1[MAT_DIM_I][MAT_DIM_J];
#endif

  printf("Init A\n");
  init_input_matrix(full_A_m0);
  init_input_matrix(full_A_m1);

  printf("Init D\n");
  init_bias_matrix(full_D_m0);
  init_bias_matrix(full_D_m1);

  printf("Init B\n");
  init_weight_matrix(full_B_m0);
  init_weight_matrix(full_B_m1);

#if !FAST && CHECK
  printf("Calculate Output\n");
  full_matmul(full_A_m0, full_B_m0, full_D_m0, gold_full_m0);
  full_matmul(full_A_m1, full_B_m1, full_D_m1, gold_full_m1);
  full_matscale(gold_full_m0, gold_m0, ACC_SCALE_IDENTITY);
  full_matscale(gold_full_m1, gold_m1, ACC_SCALE_IDENTITY);
#endif

  shared_multi_matmul_job_t j0, j1;
  printf("Do Gemmini tiled matmul process\n");
#if MULTI
  const size_t tile_I = 4;
  const size_t tile_J = 4;
  const size_t tile_K = 16;

  init_matmul_job(&j0, 1, gemmini_configuration_m0,
                  full_A_m0, full_B_m0, full_D_m0, full_C_m0,
                  0, 0, 0,
                  tile_I, tile_J, tile_K);
  init_matmul_job(&j1, 2, gemmini_configuration_m1,
                  full_A_m1, full_B_m1, full_D_m1, full_C_m1,
                  j0.sp_addr_A_stacked, j0.sp_addr_B_stacked, j0.acc_addr_stacked,
                  tile_I, tile_J, tile_K);

  uint64_t matmul_start = read_cycles();
  while (!j0.done || !j1.done)
  {
    if (!j0.done)
      shared_multi_tiled_matmul_job_step(&j0);
    if (!j1.done)
      shared_multi_tiled_matmul_job_step(&j1);
  }

  gemmini_fence();

  uint64_t matmul_end = read_cycles();

#if CHECK
  printf("Check \"Out\" matrix\n");

#if FAST
  if (!is_equal_dynamic(full_C_m0, full_A_m0, MAT_DIM_I, MAT_DIM_J))
  {
    printf("Incorrect output matrix!\n");
    printf("C_m0:\n");
    printMatrix_dynamic(full_C_m0, MAT_DIM_I, MAT_DIM_J);
    printf("A_m0:\n");
    printMatrix_dynamic(full_A_m0, MAT_DIM_I, MAT_DIM_J);
    printf("\n");

    exit(1);
  }

  if (!is_equal_dynamic(full_C_m1, full_A_m1, MAT_DIM_I, MAT_DIM_J))
  {
    printf("Incorrect output matrix!\n");
    printf("C_m1:\n");
    printMatrix_dynamic(full_C_m1, MAT_DIM_I, MAT_DIM_J);
    printf("A_m1:\n");
    printMatrix_dynamic(full_A_m1, MAT_DIM_I, MAT_DIM_J);
    printf("\n");

    exit(1);
  }
#else
  if (!is_equal_dynamic(full_C_m0, gold_m0, MAT_DIM_I, MAT_DIM_J))
  {
    printf("Incorrect output matrix!\n");
    printf("C_m0:\n");
    printMatrix_dynamic(full_C_m0, MAT_DIM_I, MAT_DIM_J);
    printf("Gold_m0:\n");
    printMatrix_dynamic(gold_m0, MAT_DIM_I, MAT_DIM_J);
    printf("\n");

    exit(1);
  }

  if (!is_equal_dynamic(full_C_m1, gold_m1, MAT_DIM_I, MAT_DIM_J))
  {
    printf("Incorrect output matrix!\n");
    printf("C_m1:\n");
    printMatrix_dynamic(full_C_m1, MAT_DIM_I, MAT_DIM_J);
    printf("Gold_m1:\n");
    printMatrix_dynamic(gold_m1, MAT_DIM_I, MAT_DIM_J);
    printf("\n");

    exit(1);
  }
#endif
  printf("Output matrix came out as expected\n");
#endif

#else
  uint64_t matmul_start = read_cycles();
  tiled_matmul_auto(custom3, MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
                    (elem_t *)full_A_m0, (elem_t *)full_B_m0, NO_BIAS ? NULL : &full_D_m0[0][0], (elem_t *)full_C_m0,
                    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    1,
                    WS);
  tiled_matmul_auto(custom3, MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
                    (elem_t *)full_A_m1, (elem_t *)full_B_m1, NO_BIAS ? NULL : &full_D_m1[0][0], (elem_t *)full_C_m1,
                    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    1,
                    WS);
  gemmini_fence();
  uint64_t matmul_end = read_cycles();
#endif

  printf("Total Matmul cycle: %llu\n", (unsigned long long)(matmul_end - matmul_start));

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
