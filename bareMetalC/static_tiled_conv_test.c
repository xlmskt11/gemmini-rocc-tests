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

#define PROFILE false
#define profile_data_num 5000

#define MULTI true
#define gemmini_configuration 15

#define FAST true
#define NO_BIAS false
#define CHECK true

#define BATCH_SIZE 1
#define IN_ROW_DIM 28
#define IN_COL_DIM 28
#define IN_CHANNELS 64
#define OUT_CHANNELS 128
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 1

#define OUT_ROW_DIM ((IN_ROW_DIM + 2 * PADDING - KERNEL_DIM) / STRIDE + 1)
#define OUT_COL_DIM ((IN_COL_DIM + 2 * PADDING - KERNEL_DIM) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM)

#define TILE_BATCHES 1
#define TILE_POROWS 8
#define TILE_POCOLS 16
#define TILE_POCHS 64
#define TILE_KROWS 3
#define TILE_KCOLS 3
#define TILE_KCHS 64

#define q_type(p) (p >> 62)
#define start(p) ((p >> 31) & ((1 << 31) - 1))
#define end(p) (p & ((1 << 31) - 1))

static void print_gemmini_use(unsigned mask) {
  int idx[4];
  int n = 0;

  for (int g = 0; g < 4; ++g) {
    if (mask & (1u << g))
      idx[n++] = g;
  }

  printf("Use %d Gemmini", n);
  if (n > 0) {
    printf(" ");
    for (int i = 0; i < n; ++i)
      printf("%d%s", idx[i], (i == n - 1) ? "" : ", ");
  }
  printf(" with DIM: %d\n", DIM);
}

static bool vec_is_equal(const elem_t *a, const elem_t *b, size_t len) {
  for (size_t i = 0; i < len; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

static void init_random(elem_t *buf, size_t len) {
  for (elem_t *ptr = buf; ptr < buf + len; ptr++) {
#if FAST
    *ptr = 1;
#else
    *ptr = (rand() % 5) - 2;
#endif
  }
}

static void init_random_acc(acc_t *buf, size_t len) {
  for (acc_t *ptr = buf; ptr < buf + len; ptr++) {
#if FAST
    *ptr = 1;
#else
    *ptr = (rand() % 5) - 2;
#endif
  }
}

static void init_zeros_acc(acc_t *buf, size_t len) {
  for (acc_t *ptr = buf; ptr < buf + len; ptr++)
    *ptr = 0;
}

#if CHECK && FAST
static elem_t expected_fast_output(int orow, int ocol) {
  acc_t result = NO_BIAS ? 0 : 1;

  for (int krow = 0; krow < KERNEL_DIM; krow++) {
    const int irow = orow * STRIDE + krow - PADDING;
    if (irow < 0 || irow >= IN_ROW_DIM)
      continue;

    for (int kcol = 0; kcol < KERNEL_DIM; kcol++) {
      const int icol = ocol * STRIDE + kcol - PADDING;
      if (icol < 0 || icol >= IN_COL_DIM)
        continue;

      result += IN_CHANNELS;
    }
  }

  return scale_and_sat(result, NO_ACTIVATION, ACC_SCALE_IDENTITY, 0);
}
#endif

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

  printf("Input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM, IN_COL_DIM);
  printf("Input channels: %u\n", IN_CHANNELS);
  printf("Batch size: %u\n", BATCH_SIZE);
  printf("Kernel dimensions: %u by %u\n", KERNEL_DIM, KERNEL_DIM);
  printf("Stride: %u\n", STRIDE);
  printf("Padding: %u\n", PADDING);
  printf("Output channels: %u\n", OUT_CHANNELS);
  printf("Output dimensions (rows by columns): %u by %u\n", OUT_ROW_DIM, OUT_COL_DIM);

  printf("Flush All Gemmini TLB of stale virtual addresses\n");
#if MULTI
  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
#endif
  gemmini_flush(custom3, 0);

  static elem_t input[BATCH_SIZE][IN_ROW_DIM][IN_COL_DIM][IN_CHANNELS] row_align(MAX_BLOCK_LEN);
  static elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);
  static acc_t bias[OUT_CHANNELS] row_align_acc(MAX_BLOCK_LEN_ACC);
  static elem_t output_mat[N_PATCHES][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);

#if CHECK && !FAST
  static elem_t output_cpu[BATCH_SIZE][OUT_ROW_DIM][OUT_COL_DIM][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);
#endif

  printf("Initialize input, weights, and bias\n");
  init_random(&input[0][0][0][0], sizeof(input) / sizeof(elem_t));
  init_random(&weights_mat[0][0], sizeof(weights_mat) / sizeof(elem_t));

  if (NO_BIAS)
    init_zeros_acc(&bias[0], sizeof(bias) / sizeof(acc_t));
  else
    init_random_acc(&bias[0], sizeof(bias) / sizeof(acc_t));

#if CHECK && !FAST
  printf("Run CPU conv for reference\n");
  uint64_t cpu_start = read_cycles();
  conv_cpu(BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, IN_CHANNELS,
           OUT_CHANNELS, OUT_ROW_DIM, OUT_COL_DIM,
           STRIDE, 1, 1, PADDING, KERNEL_DIM,
           false, false, false, false, false,
           (elem_t *)input,
           (elem_t *)weights_mat,
           NO_BIAS ? NULL : (acc_t *)bias,
           (elem_t *)output_cpu,
           NO_ACTIVATION, ACC_SCALE_IDENTITY,
           1, 0, 0);
  uint64_t cpu_end = read_cycles();
  printf("CPU conv took %llu cycles\n", cpu_end - cpu_start);
#endif

  uint64_t conv_start = 0;
  uint64_t conv_end = 0;

#if MULTI
  const int pool_size = 1;
  const int pool_stride = 1;
  const int pool_padding = 0;
  const bool downsample = STRIDE == 2 && KERNEL_DIM == 1 && PADDING == 0 &&
                          IN_ROW_DIM % 2 == 0 && IN_COL_DIM % 2 == 0;

  const int spad_rows = tiled_conv_total_spad_rows(false,
                                                   STRIDE, 1, 1, downsample, false, false,
                                                   TILE_BATCHES, TILE_POROWS, TILE_POCOLS, TILE_POCHS,
                                                   TILE_KROWS, TILE_KCOLS, TILE_KCHS,
                                                   pool_size, pool_stride);
  const int acc_rows = tiled_conv_total_spad_rows(true,
                                                  STRIDE, 1, 1, downsample, false, false,
                                                  TILE_BATCHES, TILE_POROWS, TILE_POCOLS, TILE_POCHS,
                                                  TILE_KROWS, TILE_KCOLS, TILE_KCHS,
                                                  pool_size, pool_stride);

  printf("Static tiles: batches=%d, porows=%d, pocols=%d, pochs=%d, krows=%d, kcols=%d, kchs=%d\n",
         TILE_BATCHES, TILE_POROWS, TILE_POCOLS, TILE_POCHS,
         TILE_KROWS, TILE_KCOLS, TILE_KCHS);
  printf("Reserved rows: spad=%d, acc=%d\n", spad_rows, acc_rows);

  if (spad_rows > TOTAL_SPAD_ROWS / 2 || acc_rows > TOTAL_ACC_ROWS / 2) {
    printf("Static tile configuration exceeds Gemmini capacity\n");
    exit(1);
  }

  shared_multi_conv_job_t j0;
  memset(&j0, 0, sizeof(shared_multi_conv_job_t));

  j0.tile_id = 1;
  j0.gemmini_list = gemmini_configuration;
  j0.sp_addr_start_stack = 0;
  j0.sp_addr_end_stack = 0;
  j0.acc_addr_start_stack = 0;
  j0.sp_addr_range = TOTAL_SPAD_ROWS;
  j0.acc_addr_range = TOTAL_ACC_ROWS;
  j0.batch_size = BATCH_SIZE;
  j0.in_row_dim = IN_ROW_DIM;
  j0.in_col_dim = IN_COL_DIM;
  j0.in_channels = IN_CHANNELS;
  j0.out_channels = OUT_CHANNELS;
  j0.out_row_dim = OUT_ROW_DIM;
  j0.out_col_dim = OUT_COL_DIM;
  j0.stride = STRIDE;
  j0.input_dilation = 1;
  j0.kernel_dilation = 1;
  j0.padding = PADDING;
  j0.kernel_dim = KERNEL_DIM;
  j0.wrot180 = false;
  j0.trans_output_1203 = false;
  j0.trans_input_3120 = false;
  j0.trans_weight_1203 = false;
  j0.trans_weight_0132 = false;
  j0.input = (elem_t *)input;
  j0.weights = (elem_t *)weights_mat;
  j0.bias = NO_BIAS ? NULL : (acc_t *)bias;
  j0.output = (elem_t *)output_mat;
  j0.act = NO_ACTIVATION;
  j0.scale = ACC_SCALE_IDENTITY;
  j0.pool_size = pool_size;
  j0.pool_stride = pool_stride;
  j0.pool_padding = pool_padding;
  j0.tiled_conv_type = WS;

  printf("Run Gemmini static tiled conv\n");
  conv_start = read_cycles();
  // uint64_t tiling_calc_start = read_cycles();
  shared_multi_choose_conv_tiling_factors_static(&j0,
                                                 TILE_BATCHES, TILE_POROWS, TILE_POCOLS, TILE_POCHS,
                                                 TILE_KROWS, TILE_KCOLS, TILE_KCHS);
  // uint64_t tiling_calc_end = read_cycles();
  // printf("tiling_calc: %llu cycles\n", tiling_calc_end - tiling_calc_start);
  // uint64_t gemmini_initialization_start = read_cycles();
  shared_multi_tiled_conv_job_init(&j0);
  // uint64_t gemmini_initialization_end = read_cycles();
  // printf("gemmini initialization: %llu cycles\n", gemmini_initialization_end - gemmini_initialization_start);
  while (!j0.done)
    shared_multi_tiled_conv_job_step(&j0);
  gemmini_fence();
  conv_end = read_cycles();
#else
  printf("Run Gemmini conv\n");
  conv_start = read_cycles();
  tiled_conv_auto(custom3,
                  BATCH_SIZE, IN_ROW_DIM, IN_COL_DIM, IN_CHANNELS,
                  OUT_CHANNELS, OUT_ROW_DIM, OUT_COL_DIM,
                  STRIDE, 1, 1, PADDING, KERNEL_DIM,
                  false, false, false, false, false,
                  (elem_t *)input,
                  (elem_t *)weights_mat,
                  NO_BIAS ? NULL : (acc_t *)bias,
                  (elem_t *)output_mat,
                  NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,
                  WS);
  gemmini_fence();
  conv_end = read_cycles();
#endif

  printf("Total Conv cycle: %llu\n", conv_end - conv_start);

#if CHECK
#if FAST
  bool success = true;

  for (int batch = 0; batch < BATCH_SIZE && success; batch++) {
    for (int orow = 0; orow < OUT_ROW_DIM && success; orow++) {
      for (int ocol = 0; ocol < OUT_COL_DIM && success; ocol++) {
        const elem_t expected = expected_fast_output(orow, ocol);
        const int patch = batch * OUT_ROW_DIM * OUT_COL_DIM + orow * OUT_COL_DIM + ocol;

        for (int och = 0; och < OUT_CHANNELS; och++) {
          const elem_t actual = output_mat[patch][och];
          if (actual != expected) {
            printf("Mismatch at batch=%d, orow=%d, ocol=%d, och=%d: expected=%d actual=%d\n",
                   batch, orow, ocol, och, expected, actual);
            success = false;
            break;
          }
        }
      }
    }
  }
#else
  const bool success = vec_is_equal(&output_cpu[0][0][0][0],
                                    &output_mat[0][0],
                                    sizeof(output_cpu) / sizeof(elem_t));
  if (!success) {
    printf("Incorrect output matrix\n");
  }
#endif

  if (!success)
    exit(1);

  printf("Output matrix came out as expected\n");
#endif

#if PROFILE
  for (int i = 0; i < total_gemmini_num; i++) {
    for (int j = 0; j < profile_data_num; j++) {
      if (P[i][j] == 0)
        break;
      printf("%d, %d, %d, %d\n", i, q_type(P[i][j]), start(P[i][j]), end(P[i][j]));
    }
  }
#endif

  return 0;
}
