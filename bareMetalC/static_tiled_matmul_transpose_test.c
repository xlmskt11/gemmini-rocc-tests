// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils_all.h"

#define MULTI true
#define gemmini_configuration 15

#define MAT_DIM_I (DIM * 4 + 3)
#define MAT_DIM_J (DIM * 2 + 2)
#define MAT_DIM_K (DIM * 3 + 5)

#define TILE_I 8
#define TILE_J 4
#define TILE_K 8

#define NO_BIAS true
#define FULL_BIAS_WIDTH true
#define REPEATING_BIAS false

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

static elem_t full_A[MAT_DIM_I][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
static elem_t full_At[MAT_DIM_K][MAT_DIM_I] row_align(MAX_BLOCK_LEN);
static elem_t full_B[MAT_DIM_K][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static elem_t full_Bt[MAT_DIM_J][MAT_DIM_K] row_align(MAX_BLOCK_LEN);
static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN_ACC);
static full_t gold_full[MAT_DIM_I][MAT_DIM_J];
static elem_t gold[MAT_DIM_I][MAT_DIM_J];

static void print_gemmini_use(unsigned mask)
{
  int idx[4], n = 0;

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
      printf("%d%s", idx[i], (i == n - 1) ? "" : ", ");
  }
  printf(" with DIM: %d\n", DIM);
}

static inline elem_t pattern_a(size_t i, size_t k)
{
  return (elem_t)(((i * 7 + k * 5 + 1) & 1) ? 1 : 0);
}

static inline elem_t pattern_b(size_t k, size_t j)
{
  return (elem_t)(((k * 11 + j * 3 + 2) & 1) ? 1 : 0);
}

static void init_matrices(void)
{
  for (size_t i = 0; i < MAT_DIM_I; ++i)
  {
    for (size_t k = 0; k < MAT_DIM_K; ++k)
    {
      const elem_t value = pattern_a(i, k);
      full_A[i][k] = value;
      full_At[k][i] = value;
    }
  }

  for (size_t k = 0; k < MAT_DIM_K; ++k)
  {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
    {
      const elem_t value = pattern_b(k, j);
      full_B[k][j] = value;
      full_Bt[j][k] = value;
    }
  }

  for (size_t i = 0; i < MAT_DIM_I; ++i)
    for (size_t j = 0; j < MAT_DIM_J; ++j)
      full_D[i][j] = NO_BIAS ? 0 : (ACC_T)(((i + j) % 3) - 1);
}

static void reference_matmul(void)
{
  for (size_t r = 0; r < MAT_DIM_I; ++r)
  {
    for (size_t c = 0; c < MAT_DIM_J; ++c)
    {
      gold_full[r][c] = full_D[r][c];
      for (size_t k = 0; k < MAT_DIM_K; ++k)
        gold_full[r][c] += full_A[r][k] * full_B[k][c];
    }
  }
}

static void full_matscale(full_t full[MAT_DIM_I][MAT_DIM_J],
                          elem_t out[MAT_DIM_I][MAT_DIM_J],
                          acc_scale_t scale)
{
  for (size_t r = 0; r < MAT_DIM_I; ++r)
  {
    for (size_t c = 0; c < MAT_DIM_J; ++c)
    {
      full_t scaled = ACC_SCALE(full[r][c], scale);

#ifndef ELEM_T_IS_FLOAT
      full_t elem = scaled > elem_t_max ? elem_t_max : (scaled < elem_t_min ? elem_t_min : scaled);
      out[r][c] = elem;
#else
      out[r][c] = scaled;
#endif
    }
  }
}

static void flush_all_gemmini(void)
{
#if MULTI
  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
#endif
  gemmini_flush(custom3, 0);
}

static void check_result(const char *case_name)
{
  for (size_t i = 0; i < MAT_DIM_I; ++i)
  {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
    {
      if (full_C[i][j] != gold[i][j])
      {
        printf("Mismatch in %s at (%d, %d)\n", case_name, (int)i, (int)j);
        printf("C[%d][%d] = %d, gold = %d\n", (int)i, (int)j, full_C[i][j], gold[i][j]);
        printf("C:\n");
        printMatrix_dynamic(full_C, MAT_DIM_I, MAT_DIM_J);
        printf("Gold:\n");
        printMatrix_dynamic(gold, MAT_DIM_I, MAT_DIM_J);
        exit(1);
      }
    }
  }
}

static uint64_t run_case(const char *case_name,
                         const elem_t *a,
                         const elem_t *b,
                         size_t stride_A,
                         size_t stride_B,
                         bool a_transpose,
                         bool b_transpose,
                         int tile_id)
{
  shared_multi_matmul_job_t job;

  memset(&job, 0, sizeof(job));
  memset(full_C, 0, sizeof(full_C));

  flush_all_gemmini();

  job.tile_id = tile_id;
  job.gemmini_list = gemmini_configuration;
  job.sp_addr_start_stack = 0;
  job.sp_addr_end_stack = 0;
  job.acc_addr_start_stack = 0;
  job.sp_addr_range = TOTAL_SPAD_ROWS;
  job.acc_addr_range = TOTAL_ACC_ROWS;
  job.dim_I = MAT_DIM_I;
  job.dim_J = MAT_DIM_J;
  job.dim_K = MAT_DIM_K;
  job.A = (elem_t *)a;
  job.B = (elem_t *)b;
  job.D = NO_BIAS ? NULL : &full_D[0][0];
  job.C = (elem_t *)full_C;
  job.stride_A = stride_A;
  job.stride_B = stride_B;
  job.stride_D = MAT_DIM_J;
  job.stride_C = MAT_DIM_J;
  job.A_scale_factor = MVIN_SCALE_IDENTITY;
  job.B_scale_factor = MVIN_SCALE_IDENTITY;
  job.D_scale_factor = MVIN_SCALE_IDENTITY;
  job.act = NO_ACTIVATION;
  job.scale = ACC_SCALE_IDENTITY;
  job.bert_scale = 0;
  job.repeating_bias = REPEATING_BIAS;
  job.a_transpose = a_transpose;
  job.b_transpose = b_transpose;
  job.full_C = false;
  job.low_D = !FULL_BIAS_WIDTH;
  job.weightA = 1;
  job.dataflow = WEIGHT_STATIONARY;

  printf("Run %s: stride_A=%d stride_B=%d a_transpose=%d b_transpose=%d\n",
         case_name, (int)stride_A, (int)stride_B, a_transpose, b_transpose);

  shared_multi_choose_tiling_factors_static(&job, TILE_I, TILE_J, TILE_K);

  const uint64_t start = read_cycles();
  shared_multi_tiled_matmul_job_init(&job);
  while (!job.done)
    shared_multi_tiled_matmul_job_step(&job);
  gemmini_fence();
  const uint64_t end = read_cycles();

  check_result(case_name);
  printf("%s passed in %d cycles\n", case_name, (int)(end - start));

  return end - start;
}

int main(void)
{
#ifndef BAREMETAL
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
  {
    perror("mlockall failed");
    exit(1);
  }
#endif

#if MULTI
  print_gemmini_use(gemmini_configuration);
#else
  printf("Use Single Gemmini: %d\n", DIM);
#endif
  printf("MAT_DIM_I: %d\n", MAT_DIM_I);
  printf("MAT_DIM_J: %d\n", MAT_DIM_J);
  printf("MAT_DIM_K: %d\n", MAT_DIM_K);

  init_matrices();
  reference_matmul();
  full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);

  uint64_t total_cycles = 0;
  total_cycles += run_case("matmul_a_transpose",
                           (elem_t *)full_At, (elem_t *)full_B,
                           MAT_DIM_I, MAT_DIM_J,
                           true, false, 1);

  total_cycles += run_case("matmul_b_transpose",
                           (elem_t *)full_A, (elem_t *)full_Bt,
                           MAT_DIM_K, MAT_DIM_K,
                           false, true, 2);

  printf("All transpose matmul cases passed\n");
  printf("Total cycles: %d\n", (int)total_cycles);

  exit(0);
}
