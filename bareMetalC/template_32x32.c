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
//#include "include/gemmini_nn.h"

#define MAT_DIM_I 128
#define MAT_DIM_J 128
#define MAT_DIM_K 128

#define profile_data_num 8192

#define NO_BIAS true
#define FULL_BIAS_WIDTH true
#define REPEATING_BIAS true

#define RAND rand()
#define FAST true
#define CHECK true
#define FENCE true
#define PROFILE false

#define q_type(p) (p >> 62)
#define start(p) ((p >> 31) & ((1 << 31) - 1))
#define end(p) (p & ((1 << 31) - 1))

#if FULL_BIAS_WIDTH
typedef acc_t ACC_T;
#else
typedef elem_t ACC_T;
#endif

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
    static uint64_t P[profile_data_num] row_align(1);
    gemmini_profiler(custom3, (uint64_t *)P);
#endif

    printf("One Gemmini with DIM: %d\n", DIM);
    printf("MAT_DIM_I: %d\n", MAT_DIM_I);
    printf("MAT_DIM_J: %d\n", MAT_DIM_J);
    printf("MAT_DIM_K: %d\n", MAT_DIM_K);

    printf("Flush All Gemmini TLB of stale virtual addresses\n");
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
    }
  }
#if FAST
  // identity matrix
  printf("Init B\n");
  for (size_t i = 0; i < MAT_DIM_K; ++i)
  {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
    {
      full_B[i][j] = i == j;
    }
  }
#else
  printf("Init B\n");
  for (size_t i = 0; i < MAT_DIM_K; ++i)
  {
    for (size_t j = 0; j < MAT_DIM_J; ++j)
    {
      full_B[i][j] = RAND % 2;
    }
  }
#if CHECK
  printf("Calculate Output\n");
  full_matmul(full_A, full_B, full_D, gold_full);
  full_matscale(gold_full, gold, ACC_SCALE_IDENTITY);
#endif
#endif

  printf("Do Gemmini tiled matmul process\n");
  uint64_t matmul_start = read_cycles();
  tiled_matmul_auto(custom3, MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)full_A, (elem_t*)full_B, NO_BIAS ? NULL : &full_D[0][0], (elem_t*)full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            false, false,
            false, !FULL_BIAS_WIDTH,
	          1,
            WS);
#if FENCE
  gemmini_fence();
#endif
  uint64_t matmul_end = read_cycles();
  printf("Matmul cycle: %d\n", matmul_end - matmul_start);

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

#if PROFILE
  for (int j = 0; j < profile_data_num; j++)
  {
    if (P[j] == 0)
    {
      break;
    }
    printf("%d, %d, %d, %d\n", 0, q_type(P[j]), start(P[j]), end(P[j]));
  }
#endif

  exit(0);
}

