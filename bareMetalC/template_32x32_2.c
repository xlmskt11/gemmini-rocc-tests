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

#define MAT_DIM_I 32
#define MAT_DIM_J 32
#define MAT_DIM_K 32

#define NO_BIAS true
#define FULL_BIAS_WIDTH true
#define REPEATING_BIAS true

#define FAST true
#define CHECK true
#define RAND rand()

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
  printf("One Gemmini with DIM: %d\n", DIM);
  printf("MAT_DIM_I: %d\n", MAT_DIM_I);
  printf("MAT_DIM_J: %d\n", MAT_DIM_J);
  printf("MAT_DIM_K: %d\n", MAT_DIM_K);

  printf("Flush All Gemmini TLB of stale virtual addresses\n");
  gemmini_flush(custom3, 0);

  printf("Initialize our input and output matrices in main memory\n");
  static elem_t first_full_A[MAT_DIM_I][MAT_DIM_K] row_align(1);
  static elem_t first_full_B[MAT_DIM_K][MAT_DIM_J] row_align(1);
  static elem_t first_full_C[MAT_DIM_I][MAT_DIM_J] row_align(1);
  static ACC_T first_full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(1);

  static elem_t second_full_B[MAT_DIM_J][MAT_DIM_K] row_align(1);
  static elem_t second_full_C[MAT_DIM_I][MAT_DIM_K] row_align(1);
  static ACC_T second_full_D[MAT_DIM_I][MAT_DIM_K] row_align_acc(1);

  static full_t first_gold_full[MAT_DIM_I][MAT_DIM_J];
  static full_t second_gold_full[MAT_DIM_I][MAT_DIM_K];
  static elem_t first_gold[MAT_DIM_I][MAT_DIM_J];
  static elem_t second_gold[MAT_DIM_I][MAT_DIM_K];

  printf("Init first A\n");
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_K; ++j) {
      first_full_A[i][j] = RAND % 2;
    }
  }
  printf("Init first D\n");
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j) {
      first_full_D[i][j] = NO_BIAS ? 0 : RAND % 2;
    }
  }
  printf("Init second D\n");
  for (size_t i = 0; i < MAT_DIM_I; ++i) {
    for (size_t j = 0; j < MAT_DIM_K; ++j) {
      second_full_D[i][j] = NO_BIAS ? 0 : RAND % 2;
    }
  }
#if FAST == 1
  printf("Init first B\n");
  for (size_t i = 0; i < MAT_DIM_K; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j) {
      first_full_B[i][j] = i == j;
    }
  }
  printf("Init second B\n");
  for (size_t i = 0; i < MAT_DIM_J; ++i) {
    for (size_t j = 0; j < MAT_DIM_K; ++j) {
      second_full_B[i][j] = i == j;
    }
  }
#else
  printf("Init first B\n");
  for (size_t i = 0; i < MAT_DIM_K; ++i) {
    for (size_t j = 0; j < MAT_DIM_J; ++j) {
      first_full_B[i][j] = RAND % 2;
    }
  }
  printf("Init second B\n");
  for (size_t i = 0; i < MAT_DIM_J; ++i) {
    for (size_t j = 0; j < MAT_DIM_K; ++j) {
      second_full_B[i][j] = RAND % 2;
    }
  }

#if CHECK == 1
  printf("Calculate Output\n");
  full_matmul(first_full_A, first_full_B, first_full_D, first_gold_full);
  full_matscale(first_gold_full, first_gold, ACC_SCALE_IDENTITY);
  full_matmul(first_gold, second_full_B, second_full_D, second_gold_full);
  full_matscale(second_gold_full, second_gold, ACC_SCALE_IDENTITY);
#endif
#endif

  printf("Do first Gemmini tiled matmul process\n");
  uint64_t matmul_start = read_cycles();
  tiled_matmul_auto(custom3, MAT_DIM_I, MAT_DIM_J, MAT_DIM_K,
            (elem_t*)first_full_A, (elem_t*)first_full_B, NO_BIAS ? NULL : &first_full_D[0][0], (elem_t*)first_full_C,
            MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
            false, false,
            false, !FULL_BIAS_WIDTH,
	          1,
            WS);
  gemmini_fence();
  // uint64_t first_matmul_end = read_cycles();
  // printf("First matmul cycle: %d\n", first_matmul_end-first_matmul_start);

  // printf("Do second Gemmini tiled matmul process\n");
  // uint64_t second_matmul_start = read_cycles();
  tiled_matmul_auto(custom3, MAT_DIM_I, MAT_DIM_K, MAT_DIM_J,
    (elem_t*)first_full_C, (elem_t*)second_full_B, NO_BIAS ? NULL : &second_full_D[0][0], (elem_t*)second_full_C,
    MAT_DIM_J, MAT_DIM_K, MAT_DIM_K, MAT_DIM_K,
    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
    false, false,
    false, !FULL_BIAS_WIDTH,
    1,
    WS);
  gemmini_fence();
  uint64_t matmul_end = read_cycles();
  // printf("Second matmul cycle: %d\n", second_matmul_end-second_matmul_start);

#if CHECK == 1
  printf("Check \"Out\" matrix\n");
  // if (!full_is_equal(full_C, gold)) {
  //   printf("Incorrect output matrix!\n");
  //   printf("C:\n");
  //   full_printMatrix(full_C);
  //   printf("Gold:\n");
  //   full_printMatrix(gold);
  //   printf("\n");

  //   exit(1);
  // }

#if FAST == 1
  if (!is_equal_dynamic(second_full_C, first_full_A, MAT_DIM_I, MAT_DIM_K)) {
    printf("Incorrect output matrix!\n");
    printf("C:\n");
    printMatrix_dynamic(second_full_C, MAT_DIM_I, MAT_DIM_K);
    printf("A:\n");
    printMatrix_dynamic(first_full_A, MAT_DIM_I, MAT_DIM_K);
    printf("\n");

    exit(1);
  }
#else
  if (!is_equal_dynamic(second_full_C, second_gold, MAT_DIM_I, MAT_DIM_K)) {
    printf("Incorrect output matrix!\n");
    printf("C:\n");
    printMatrix_dynamic(second_full_C, MAT_DIM_I, MAT_DIM_K);
    printf("Gold:\n");
    printMatrix_dynamic(second_gold, MAT_DIM_I, MAT_DIM_K);
    printf("\n");

    exit(1);
  }
#endif
  printf("Output matrix came out as expected\n");
#endif

  // printf("Total cycle: %d\n", first_matmul_end-first_matmul_start+second_matmul_end-second_matmul_start);
  printf("Total cycle: %d\n", matmul_end-matmul_start);

  exit(0);
}

