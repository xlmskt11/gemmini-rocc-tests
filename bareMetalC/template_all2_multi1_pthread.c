// See LICENSE for license details.

#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils_all.h"
// #include "include/gemmini_nn.h"

#define PROFILE false
#define profile_data_num 30

#define MULTI true
#define gemmini_configuration 15

#define MAT_DIM_I 256
#define MAT_DIM_J 256
#define MAT_DIM_K 256
#define RAND rand()
#define FAST true
#define NO_BIAS true
#define FULL_BIAS_WIDTH true
#define REPEATING_BIAS false
#define CHECK true
#define WARMUP_DIM 32

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

static void warmup_matmul(elem_t *A, elem_t *B, elem_t *C) {
  printf("Warm-up matmul\n");
  gemmini_flush(custom3, 0);
  tiled_matmul_auto(custom0, WARMUP_DIM, WARMUP_DIM, WARMUP_DIM,
                    A, B, NULL, C,
                    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    1,
                    WS);
  tiled_matmul_auto(custom1, WARMUP_DIM, WARMUP_DIM, WARMUP_DIM,
                    A, B, NULL, C,
                    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    1,
                    WS);
  tiled_matmul_auto(custom2, WARMUP_DIM, WARMUP_DIM, WARMUP_DIM,
                    A, B, NULL, C,
                    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    1,
                    WS);
  tiled_matmul_auto(custom3, WARMUP_DIM, WARMUP_DIM, WARMUP_DIM,
                    A, B, NULL, C,
                    MAT_DIM_K, MAT_DIM_J, MAT_DIM_J, MAT_DIM_J,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, REPEATING_BIAS,
                    false, false,
                    false, !FULL_BIAS_WIDTH,
                    1,
                    WS);
  gemmini_fence();
}

static int run_matmul(void) {
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
    static elem_t full_C[MAT_DIM_I][MAT_DIM_J] row_align(MAX_BLOCK_LEN);
    static ACC_T full_D[MAT_DIM_I][MAT_DIM_J] row_align_acc(MAX_BLOCK_LEN_ACC);

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

  warmup_matmul((elem_t *)full_A, (elem_t *)full_B, (elem_t *)full_C);

  shared_multi_matmul_job_t j0;
  int total_cycles = 0;
  printf("Do Gemmini tiled matmul process\n");
#if MULTI
  const size_t sp_addr_range = BANK_NUM * BANK_ROWS;
  const size_t acc_addr_range = ACC_ROWS;
  const size_t max_spad_rows = sp_addr_range / 2;
  const size_t max_acc_rows = acc_addr_range / 2;

  for (size_t tile_I = 4; tile_I <= 16; ++tile_I) {
    for (size_t tile_J = 1; tile_J <= 16; ++tile_J) {
      for (size_t tile_K = 4; tile_K <= 16; ++tile_K) {
        if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K) <= max_spad_rows &&
            tiled_matmul_total_acc_rows(tile_I, tile_J) <= max_acc_rows) {

          gemmini_flush(custom0, 0);
          gemmini_flush(custom1, 0);
          gemmini_flush(custom2, 0);
          gemmini_flush(custom3, 0);

          memset(&j0, 0, sizeof(shared_multi_matmul_job_t));
          j0.tile_id = 1;
          j0.gemmini_list = gemmini_configuration;
          j0.sp_addr_start_stack = 0;
          j0.sp_addr_end_stack = 0;
          j0.acc_addr_start_stack = 0;
          j0.sp_addr_range = TOTAL_SPAD_ROWS;
          j0.acc_addr_range = TOTAL_ACC_ROWS;
          j0.dim_I = MAT_DIM_I;
          j0.dim_J = MAT_DIM_J;
          j0.dim_K = MAT_DIM_K;
          j0.A = (elem_t *)full_A;
          j0.B = (elem_t *)full_B;
          j0.D = NO_BIAS ? NULL : &full_D[0][0];
          j0.C = (elem_t *)full_C;
          j0.stride_A = MAT_DIM_K;
          j0.stride_B = MAT_DIM_J;
          j0.stride_D = MAT_DIM_J;
          j0.stride_C = MAT_DIM_J;
          j0.A_scale_factor = MVIN_SCALE_IDENTITY;
          j0.B_scale_factor = MVIN_SCALE_IDENTITY;
          j0.D_scale_factor = MVIN_SCALE_IDENTITY;
          j0.act = NO_ACTIVATION;
          j0.scale = ACC_SCALE_IDENTITY;
          j0.bert_scale = 0;
          j0.repeating_bias = REPEATING_BIAS;
          j0.a_transpose = false;
          j0.b_transpose = false;
          j0.full_C = false;
          j0.low_D = !FULL_BIAS_WIDTH;
          j0.weightA = 1;
          j0.dataflow = WEIGHT_STATIONARY;

          uint64_t matmul_start = read_cycles();

          shared_multi_choose_tiling_factors_static(&j0, tile_I, tile_J, tile_K);
          shared_multi_tiled_matmul_job_init(&j0);
          while (!j0.done)
          {
            shared_multi_tiled_matmul_job_step(&j0);
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

            return 1;
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

            return 1;
          }
#endif
          printf("Output matrix came out as expected\n");
#endif

          printf("tile_I: %d, tile_J: %d, tile_K: %d Matmul cycle: %d\n", tile_I, tile_J, tile_K, matmul_end - matmul_start);
          total_cycles = total_cycles + (matmul_end - matmul_start);
        }
      }
    }
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

  printf("Total Matmul cycle: %d\n", total_cycles);

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

  return 0;
}

#ifndef BAREMETAL
static void *matmul_thread(void *arg) {
  (void)arg;
  int cpu_id = sched_getcpu();
  printf("entered matmul thread - cpu_id: %d\n", cpu_id);
  int rc = run_matmul();
  return (void *)(intptr_t)rc;
}
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }

    printf("create threading\n");
    pthread_t thread;
    pthread_attr_t attr;
    cpu_set_t cpuset;

    pthread_attr_init(&attr);
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
    pthread_create(&thread, &attr, matmul_thread, NULL);

    void *thread_ret = NULL;
    pthread_join(thread, &thread_ret);
    pthread_attr_destroy(&attr);

    return (int)(intptr_t)thread_ret;
#else
    return run_matmul();
#endif
}
