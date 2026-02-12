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

#define gemmini_configuration_m0 1
#define gemmini_configuration_m1 14
#define profile_data_num 30

#define NO_BIAS false

#define FAST true
#define CHECK true
#define FENCE true
#define PROFILE false
#define MULTI true

#define q_type(p) (p >> 62)
#define start(p) ((p >> 31) & ((1 << 31) - 1))
#define end(p) (p & ((1 << 31) - 1))

#define BATCH_SIZE_m0 3
#define IN_ROW_DIM_m0 25
#define IN_COL_DIM_m0 25
#define IN_CHANNELS_m0 3
#define OUT_CHANNELS_m0 28
#define KERNEL_DIM_m0 3
#define PADDING_m0 1
#define STRIDE_m0 2

#define BATCH_SIZE_m1 5
#define IN_ROW_DIM_m1 16
#define IN_COL_DIM_m1 16
#define IN_CHANNELS_m1 9
#define OUT_CHANNELS_m1 88
#define KERNEL_DIM_m1 3
#define PADDING_m1 1
#define STRIDE_m1 2

#define OUT_ROW_DIM_m0 ((IN_ROW_DIM_m0 + 2 * PADDING_m0 - KERNEL_DIM_m0) / STRIDE_m0 + 1)
#define OUT_COL_DIM_m0 ((IN_COL_DIM_m0 + 2 * PADDING_m0 - KERNEL_DIM_m0) / STRIDE_m0 + 1)
#define PATCH_SIZE_m0 (KERNEL_DIM_m0 * KERNEL_DIM_m0 * IN_CHANNELS_m0)
#define N_PATCHES_m0 (BATCH_SIZE_m0 * OUT_ROW_DIM_m0 * OUT_COL_DIM_m0)

#define OUT_ROW_DIM_m1 ((IN_ROW_DIM_m1 + 2 * PADDING_m1 - KERNEL_DIM_m1) / STRIDE_m1 + 1)
#define OUT_COL_DIM_m1 ((IN_COL_DIM_m1 + 2 * PADDING_m1 - KERNEL_DIM_m1) / STRIDE_m1 + 1)
#define PATCH_SIZE_m1 (KERNEL_DIM_m1 * KERNEL_DIM_m1 * IN_CHANNELS_m1)
#define N_PATCHES_m1 (BATCH_SIZE_m1 * OUT_ROW_DIM_m1 * OUT_COL_DIM_m1)

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

void conv(int batch_size, int in_channels,
          int in_row_dim, int in_col_dim,
          int out_channels, int kernel_dim,
          int out_row_dim, int out_col_dim,
          int stride, int padding,
          elem_t input[batch_size][in_row_dim][in_col_dim][in_channels],
          elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
          acc_t bias[out_channels],
          elem_t output[batch_size][out_row_dim][out_col_dim][out_channels])
{

#ifdef GEMMINI_ASSERTIONS
  if (out_row_dim != (in_row_dim + 2 * padding - kernel_dim) / stride + 1)
  {
    printf("conv out_row_dim is not correct\n");
    exit(1);
  }

  if (out_col_dim != (in_col_dim + 2 * padding - kernel_dim) / stride + 1)
  {
    printf("conv out_col_dim is not correct\n");
    exit(1);
  }
#endif

  for (int b = 0; b < batch_size; b++)
  {
    for (int orow = 0; orow < out_row_dim; orow++)
    {
      for (int ocol = 0; ocol < out_col_dim; ocol++)
      {
        for (int och = 0; och < out_channels; och++)
        {
          acc_t result = bias[och];

          for (int krow = 0; krow < kernel_dim; krow++)
          {
            for (int kcol = 0; kcol < kernel_dim; kcol++)
            {
              for (int kch = 0; kch < in_channels; kch++)
              {
                int irow = orow * stride + krow - padding;
                int icol = ocol * stride + kcol - padding;

                elem_t pixel = irow < 0 || irow >= in_row_dim ||
                                       icol < 0 || icol >= in_col_dim
                                   ? 0
                                   : input[b][irow][icol][kch];

                result +=
                    weights[och][krow][kcol][kch] *
                    pixel;
              }
            }
          }

          // Clip result
          result = result > elem_t_max ? elem_t_max : (result < elem_t_min ? elem_t_min : result);

          output[b][orow][ocol][och] = result;
        }
      }
    }
  }
}

void flatten_weights(int out_channels, int kernel_dim, int in_channels,
                     int patch_size,
                     elem_t weights[out_channels][kernel_dim][kernel_dim][in_channels],
                     elem_t weights_mat[patch_size][out_channels])
{

  assert(patch_size == kernel_dim * kernel_dim * in_channels);

  for (int outc = 0; outc < out_channels; outc++)
  {
    for (int krow = 0; krow < kernel_dim; krow++)
    {
      for (int kcol = 0; kcol < kernel_dim; kcol++)
      {
        for (int inc = 0; inc < in_channels; inc++)
        {
          int wmatrow = krow * kernel_dim * in_channels +
                        kcol * in_channels +
                        inc;

          weights_mat[wmatrow][outc] =
              weights[outc][krow][kcol][inc];
        }
      }
    }
  }
}

bool vec_is_equal(elem_t *a, elem_t *b, int len)
{
  for (int i = 0; i < len; i++)
    if (a[i] != b[i])
      return false;
  return true;
}

void init_random(elem_t *buf, int len)
{
  for (elem_t *ptr = buf; ptr < buf + len; ptr++)
  {
#if FAST
    *ptr = 1;
#else
    *ptr = (rand() % 5) - 2;
#endif
  }
}

void init_random_acc(acc_t *buf, int len)
{
  for (acc_t *ptr = buf; ptr < buf + len; ptr++)
  {
#if FAST
    *ptr = 1;
#else
    *ptr = (rand() % 5) - 2;
#endif
  }
}

void init_zeros_acc(acc_t *buf, int len)
{
  for (acc_t *ptr = buf; ptr < buf + len; ptr++)
  {
    *ptr = 0;
  }
}

static bool check_fast_conv_output(elem_t *output_mat, int rows, int cols, int kernel_dim, int in_channels)
{
  const int expected0 = (kernel_dim - 1) * (kernel_dim - 1) * in_channels + 1;
  const int expected1 = kernel_dim * (kernel_dim - 1) * in_channels + 1;
  const int expected2 = kernel_dim * kernel_dim * in_channels + 1;

  for (int r = 0; r < rows; r++)
  {
    for (int c = 0; c < cols; c++)
    {
      elem_t v = output_mat[r * cols + c];
      if (v != expected0 && v != expected1 && v != expected2)
      {
        return false;
      }
    }
  }

  return true;
}

static void print_output_matrix_debug(const char *label, elem_t *output_mat, int rows, int cols)
{
  printf("%s:\n", label);
  for (int r = 0; r < rows; r++)
  {
    printf("[");
    for (int c = 0; c < cols; c++)
    {
      printf("%d,", output_mat[r * cols + c]);
    }
    printf("\b],\n");
  }
  printf("\b\n\n");
}

static void print_conv_tensor_debug(const char *label, elem_t *tensor, int batch, int rows, int cols, int channels)
{
  printf("%s:\n", label);
  for (int b = 0; b < batch; b++)
  {
    printf("[");
    for (int r = 0; r < rows; r++)
    {
      printf("[");
      for (int c = 0; c < cols; c++)
      {
        printf("[");
        for (int ch = 0; ch < channels; ch++)
        {
          printf("%d,", tensor[((b * rows + r) * cols + c) * channels + ch]);
        }
        printf("\b],");
      }
      printf("\b],\n");
    }
    printf("\b],");
  }
  printf("\b\n\n");
}

static void print_bias_debug(const char *label, acc_t *bias, int len)
{
  printf("%s:\n", label);
  for (int i = 0; i < len; i++)
  {
    printf("%d,", bias[i]);
  }
  printf("\b\n\n");
}

static void print_weights_debug(const char *label, elem_t *weights, int out_channels, int kernel_dim, int in_channels)
{
  printf("%s:\n", label);
  for (int och = 0; och < out_channels; och++)
  {
    printf("[");
    for (int krow = 0; krow < kernel_dim; krow++)
    {
      printf("[");
      for (int kcol = 0; kcol < kernel_dim; kcol++)
      {
        printf("[");
        for (int ich = 0; ich < in_channels; ich++)
        {
          int idx = (((och * kernel_dim) + krow) * kernel_dim + kcol) * in_channels + ich;
          printf("%d,", weights[idx]);
        }
        printf("\b],");
      }
      printf("\b],\n");
    }
    printf("\b],");
  }
  printf("\b\n\n");
}

static void print_weights_mat_debug(const char *label, elem_t *weights_mat, int rows, int cols)
{
  printf("%s:\n", label);
  for (int r = 0; r < rows; r++)
  {
    printf("[");
    for (int c = 0; c < cols; c++)
    {
      printf("%d,", weights_mat[r * cols + c]);
    }
    printf("\b],\n");
  }
  printf("\b\n\n");
}

static void print_input_debug(const char *label, elem_t *input, int batch, int rows, int cols, int channels)
{
  printf("%s:\n", label);
  for (int b = 0; b < batch; b++)
  {
    printf("[");
    for (int r = 0; r < rows; r++)
    {
      printf("[");
      for (int c = 0; c < cols; c++)
      {
        printf("[");
        for (int ch = 0; ch < channels; ch++)
        {
          int idx = (((b * rows) + r) * cols + c) * channels + ch;
          printf("%d,", input[idx]);
        }
        printf("\b],");
      }
      printf("\b],\n");
    }
    printf("\b],");
  }
  printf("\b\n\n");
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
  printf("Model m0 input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM_m0, IN_COL_DIM_m0);
  printf("Model m0 output dimensions (rows by columns): %u by %u\n\n", OUT_ROW_DIM_m0, OUT_COL_DIM_m0);

  print_gemmini_use(gemmini_configuration_m1);
  printf("Model m1 input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM_m1, IN_COL_DIM_m1);
  printf("Model m1 output dimensions (rows by columns): %u by %u\n\n", OUT_ROW_DIM_m1, OUT_COL_DIM_m1);

  printf("Flush All Gemmini TLB of stale virtual addresses\n");
#if MULTI
  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
#endif
  gemmini_flush(custom3, 0);

  static elem_t input_m0[BATCH_SIZE_m0][IN_ROW_DIM_m0][IN_COL_DIM_m0][IN_CHANNELS_m0] row_align(MAX_BLOCK_LEN);
  static elem_t weights_m0[OUT_CHANNELS_m0][KERNEL_DIM_m0][KERNEL_DIM_m0][IN_CHANNELS_m0] row_align(MAX_BLOCK_LEN);
  static acc_t bias_m0[OUT_CHANNELS_m0] row_align_acc(MAX_BLOCK_LEN_ACC);
  static elem_t weights_mat_m0[PATCH_SIZE_m0][OUT_CHANNELS_m0] row_align(MAX_BLOCK_LEN);
  static elem_t output_mat_m0[N_PATCHES_m0][OUT_CHANNELS_m0] row_align(MAX_BLOCK_LEN);

  static elem_t input_m1[BATCH_SIZE_m1][IN_ROW_DIM_m1][IN_COL_DIM_m1][IN_CHANNELS_m1] row_align(MAX_BLOCK_LEN);
  static elem_t weights_m1[OUT_CHANNELS_m1][KERNEL_DIM_m1][KERNEL_DIM_m1][IN_CHANNELS_m1] row_align(MAX_BLOCK_LEN);
  static acc_t bias_m1[OUT_CHANNELS_m1] row_align_acc(MAX_BLOCK_LEN_ACC);
  static elem_t weights_mat_m1[PATCH_SIZE_m1][OUT_CHANNELS_m1] row_align(MAX_BLOCK_LEN);
  static elem_t output_mat_m1[N_PATCHES_m1][OUT_CHANNELS_m1] row_align(MAX_BLOCK_LEN);

#if !FAST && CHECK
  static elem_t output_m0[BATCH_SIZE_m0][OUT_ROW_DIM_m0][OUT_COL_DIM_m0][OUT_CHANNELS_m0] row_align(MAX_BLOCK_LEN);
  static elem_t output_m1[BATCH_SIZE_m1][OUT_ROW_DIM_m1][OUT_COL_DIM_m1][OUT_CHANNELS_m1] row_align(MAX_BLOCK_LEN);
#endif

  printf("Randomize inputs...\n");
  init_random(&input_m0[0][0][0][0], sizeof(input_m0) / sizeof(elem_t));
  init_random(&input_m1[0][0][0][0], sizeof(input_m1) / sizeof(elem_t));

  printf("Randomize weights...\n");
  init_random(&weights_m0[0][0][0][0], sizeof(weights_m0) / sizeof(elem_t));
  init_random(&weights_m1[0][0][0][0], sizeof(weights_m1) / sizeof(elem_t));

  printf("Randomize bias...\n");
  if (NO_BIAS)
  {
    init_zeros_acc(&bias_m0[0], sizeof(bias_m0) / sizeof(acc_t));
    init_zeros_acc(&bias_m1[0], sizeof(bias_m1) / sizeof(acc_t));
  }
  else
  {
    init_random_acc(&bias_m0[0], sizeof(bias_m0) / sizeof(acc_t));
    init_random_acc(&bias_m1[0], sizeof(bias_m1) / sizeof(acc_t));
  }

#if !FAST && CHECK
  printf("CPU conv m0...\n");
  uint64_t start_cpu_m0 = read_cycles();
  conv(BATCH_SIZE_m0, IN_CHANNELS_m0,
       IN_ROW_DIM_m0, IN_COL_DIM_m0,
       OUT_CHANNELS_m0, KERNEL_DIM_m0,
       OUT_ROW_DIM_m0, OUT_COL_DIM_m0,
       STRIDE_m0, PADDING_m0,
       input_m0,
       weights_m0,
       bias_m0,
       output_m0);
  uint64_t end_cpu_m0 = read_cycles();
  printf("CPU conv m0 took %llu cycles\n", end_cpu_m0 - start_cpu_m0);

  printf("CPU conv m1...\n");
  uint64_t start_cpu_m1 = read_cycles();
  conv(BATCH_SIZE_m1, IN_CHANNELS_m1,
       IN_ROW_DIM_m1, IN_COL_DIM_m1,
       OUT_CHANNELS_m1, KERNEL_DIM_m1,
       OUT_ROW_DIM_m1, OUT_COL_DIM_m1,
       STRIDE_m1, PADDING_m1,
       input_m1,
       weights_m1,
       bias_m1,
       output_m1);
  uint64_t end_cpu_m1 = read_cycles();
  printf("CPU conv m1 took %llu cycles\n", end_cpu_m1 - start_cpu_m1);
#endif

  printf("Flatten weights...\n");
  flatten_weights(OUT_CHANNELS_m0, KERNEL_DIM_m0, IN_CHANNELS_m0,
                  PATCH_SIZE_m0,
                  weights_m0,
                  weights_mat_m0);
  flatten_weights(OUT_CHANNELS_m1, KERNEL_DIM_m1, IN_CHANNELS_m1,
                  PATCH_SIZE_m1,
                  weights_m1,
                  weights_mat_m1);

  int tile_id_m0 = 1;
  int tile_id_m1 = 2;
  printf("Gemmini conv...\n");
  uint64_t start_gemmini = read_cycles();
  // shared_multi_tiled_conv_auto_test(gemmini_configuration_m0, tile_id_m0,
  //                                   0, 0,
  //                                   BANK_NUM * BANK_ROWS / 4, ACC_ROWS / 4,
  //                                   BATCH_SIZE_m0, IN_ROW_DIM_m0, IN_COL_DIM_m0, IN_CHANNELS_m0,
  //                                   OUT_CHANNELS_m0, OUT_ROW_DIM_m0, OUT_COL_DIM_m0,
  //                                   STRIDE_m0, 1, 1, PADDING_m0, KERNEL_DIM_m0,
  //                                   false, false, false, false, false,

  //                                   (elem_t *)input_m0,
  //                                   (elem_t *)weights_mat_m0,
  //                                   NO_BIAS ? NULL : (acc_t *)bias_m0,
  //                                   (elem_t *)output_mat_m0,

  //                                   NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,

  //                                   WS);
  // shared_multi_tiled_conv_auto_test(gemmini_configuration_m1, tile_id_m1,
  //                                   BANK_NUM * BANK_ROWS / 4, ACC_ROWS / 4,
  //                                   BANK_NUM * BANK_ROWS * 3 / 4, ACC_ROWS * 3 / 4,
  //                                   BATCH_SIZE_m1, IN_ROW_DIM_m1, IN_COL_DIM_m1, IN_CHANNELS_m1,
  //                                   OUT_CHANNELS_m1, OUT_ROW_DIM_m1, OUT_COL_DIM_m1,
  //                                   STRIDE_m1, 1, 1, PADDING_m1, KERNEL_DIM_m1,
  //                                   false, false, false, false, false,

  //                                   (elem_t *)input_m1,
  //                                   (elem_t *)weights_mat_m1,
  //                                   NO_BIAS ? NULL : (acc_t *)bias_m1,
  //                                   (elem_t *)output_mat_m1,

  //                                   NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,

  //                                   WS);

#if MULTI
  shared_multi_conv_job_t j0, j1;

  size_t spad_start_addr = 0;
  size_t acc_start_addr = 0;
  j0.tile_id = tile_id_m0;
  j0.gemmini_list = gemmini_configuration_m0;
  // j0.sp_addr_start = spad_start_addr;
  // j0.acc_addr_start = acc_start_addr;
  j0.sp_addr_range = TOTAL_SPAD_ROWS / 4;
  j0.acc_addr_range = TOTAL_ACC_ROWS / 4;
  j0.batch_size = BATCH_SIZE_m0;
  j0.in_row_dim = IN_ROW_DIM_m0;
  j0.in_col_dim = IN_COL_DIM_m0;
  j0.in_channels = IN_CHANNELS_m0;
  j0.out_channels = OUT_CHANNELS_m0;
  j0.out_row_dim = OUT_ROW_DIM_m0;
  j0.out_col_dim = OUT_COL_DIM_m0;
  j0.stride = STRIDE_m0;
  j0.input_dilation = 1;
  j0.kernel_dilation = 1;
  j0.padding = PADDING_m0;
  j0.kernel_dim = KERNEL_DIM_m0;
  j0.wrot180 = false;
  j0.trans_output_1203 = false;
  j0.trans_input_3120 = false;
  j0.trans_weight_1203 = false;
  j0.trans_weight_0132 = false;
  j0.input = (elem_t *)input_m0;
  j0.weights = (elem_t *)weights_mat_m0;
  j0.bias = NO_BIAS ? NULL : (acc_t *)bias_m0;
  j0.output = (elem_t *)output_mat_m0;
  j0.act = NO_ACTIVATION;
  j0.scale = ACC_SCALE_IDENTITY;
  j0.pool_size = 0;
  j0.pool_stride = 0;
  j0.pool_padding = 0;
  j0.tiled_conv_type = WS;

  shared_multi_choose_conv_tiling_factors(&j0);

  shared_multi_tiled_conv_job_init(&j0);

  spad_start_addr += j0.sp_addr_range;
  acc_start_addr += j0.acc_addr_range;

  j1.tile_id = tile_id_m1;
  j1.gemmini_list = gemmini_configuration_m1;
  // j1.sp_addr_start = spad_start_addr;
  // j1.acc_addr_start = acc_start_addr;
  j1.sp_addr_range = TOTAL_SPAD_ROWS - spad_start_addr;
  j1.acc_addr_range = TOTAL_ACC_ROWS - acc_start_addr;
  j1.batch_size = BATCH_SIZE_m1;
  j1.in_row_dim = IN_ROW_DIM_m1;
  j1.in_col_dim = IN_COL_DIM_m1;
  j1.in_channels = IN_CHANNELS_m1;
  j1.out_channels = OUT_CHANNELS_m1;
  j1.out_row_dim = OUT_ROW_DIM_m1;
  j1.out_col_dim = OUT_COL_DIM_m1;
  j1.stride = STRIDE_m1;
  j1.input_dilation = 1;
  j1.kernel_dilation = 1;
  j1.padding = PADDING_m1;
  j1.kernel_dim = KERNEL_DIM_m1;
  j1.wrot180 = false;
  j1.trans_output_1203 = false;
  j1.trans_input_3120 = false;
  j1.trans_weight_1203 = false;
  j1.trans_weight_0132 = false;
  j1.input = (elem_t *)input_m1;
  j1.weights = (elem_t *)weights_mat_m1;
  j1.bias = NO_BIAS ? NULL : (acc_t *)bias_m1;
  j1.output = (elem_t *)output_mat_m1;
  j1.act = NO_ACTIVATION;
  j1.scale = ACC_SCALE_IDENTITY;
  j1.pool_size = 0;
  j1.pool_stride = 0;
  j1.pool_padding = 0;
  j1.tiled_conv_type = WS;

  shared_multi_choose_conv_tiling_factors(&j1);

  shared_multi_tiled_conv_job_init(&j1);

  spad_start_addr += j1.sp_addr_range;
  acc_start_addr += j1.acc_addr_range;

  while (!j0.done || !j1.done)
  {
    if (!j0.done)
      shared_multi_tiled_conv_job_step(&j0);
    if (!j1.done)
      shared_multi_tiled_conv_job_step(&j1);
  }

#else
  tiled_conv_auto(custom3,
                  BATCH_SIZE_m0, IN_ROW_DIM_m0, IN_COL_DIM_m0, IN_CHANNELS_m0,
                  OUT_CHANNELS_m0, OUT_ROW_DIM_m0, OUT_COL_DIM_m0,
                  STRIDE_m0, 1, 1, PADDING_m0, KERNEL_DIM_m0,
                  false, false, false, false, false,

                  (elem_t *)input_m0,
                  (elem_t *)weights_mat_m0,
                  NO_BIAS ? NULL : (acc_t *)bias_m0,
                  (elem_t *)output_mat_m0,

                  NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,

                  WS);
  tiled_conv_auto(custom3,
                  BATCH_SIZE_m1, IN_ROW_DIM_m1, IN_COL_DIM_m1, IN_CHANNELS_m1,
                  OUT_CHANNELS_m1, OUT_ROW_DIM_m1, OUT_COL_DIM_m1,
                  STRIDE_m1, 1, 1, PADDING_m1, KERNEL_DIM_m1,
                  false, false, false, false, false,

                  (elem_t *)input_m1,
                  (elem_t *)weights_mat_m1,
                  NO_BIAS ? NULL : (acc_t *)bias_m1,
                  (elem_t *)output_mat_m1,

                  NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0,

                  WS);
#endif

#if FENCE
  gemmini_fence();
#endif

  uint64_t end_gemmini = read_cycles();

#if MULTI
  printf("total spad_rows reserved: %d\n", spad_start_addr);
  printf("total acc_rows reserved: %d\n\n", acc_start_addr);

  printf("scratchpad row utilization: %d%%\n", (spad_start_addr * 100) / TOTAL_SPAD_ROWS);
  printf("accumulator row utilization: %d%%\n\n", (acc_start_addr * 100) / TOTAL_ACC_ROWS);
#endif

  printf("Gemmini convs took %llu cycles\n", end_gemmini - start_gemmini);

#if CHECK
  printf("Check \"Out\" matrices\n");

#if FAST
  bool success_m0 = check_fast_conv_output(&output_mat_m0[0][0], N_PATCHES_m0, OUT_CHANNELS_m0, KERNEL_DIM_m0, IN_CHANNELS_m0);
  bool success_m1 = check_fast_conv_output(&output_mat_m1[0][0], N_PATCHES_m1, OUT_CHANNELS_m1, KERNEL_DIM_m1, IN_CHANNELS_m1);
#else
  assert(sizeof(output_mat_m0) == sizeof(output_m0));
  assert(sizeof(output_mat_m1) == sizeof(output_m1));
  bool success_m0 = vec_is_equal(&output_m0[0][0][0][0], &output_mat_m0[0][0], sizeof(output_m0) / sizeof(elem_t));
  bool success_m1 = vec_is_equal(&output_m1[0][0][0][0], &output_mat_m1[0][0], sizeof(output_m1) / sizeof(elem_t));
#endif

  if (!success_m0 || !success_m1)
  {
    if (!success_m0 && !success_m1)
    {
      printf("Incorrect output for both models!\n");
    }
    
    if (!success_m0)
    {
      printf("Incorrect output for model m0!\n");
      print_output_matrix_debug("output_mat_m0", &output_mat_m0[0][0], N_PATCHES_m0, OUT_CHANNELS_m0);
#if !FAST
      print_conv_tensor_debug("output_m0", &output_m0[0][0][0][0], BATCH_SIZE_m0, OUT_ROW_DIM_m0, OUT_COL_DIM_m0, OUT_CHANNELS_m0);
#endif
      print_bias_debug("bias_m0", &bias_m0[0], OUT_CHANNELS_m0);
      print_weights_debug("weights_m0", &weights_m0[0][0][0][0], OUT_CHANNELS_m0, KERNEL_DIM_m0, IN_CHANNELS_m0);
      print_weights_mat_debug("weights_mat_m0", &weights_mat_m0[0][0], PATCH_SIZE_m0, OUT_CHANNELS_m0);
      print_input_debug("input_m0", &input_m0[0][0][0][0], BATCH_SIZE_m0, IN_ROW_DIM_m0, IN_COL_DIM_m0, IN_CHANNELS_m0);
    }

    if (!success_m1)
    {
      printf("Incorrect output for model m1!\n");
      print_output_matrix_debug("output_mat_m1", &output_mat_m1[0][0], N_PATCHES_m1, OUT_CHANNELS_m1);
#if !FAST
      print_conv_tensor_debug("output_m1", &output_m1[0][0][0][0], BATCH_SIZE_m1, OUT_ROW_DIM_m1, OUT_COL_DIM_m1, OUT_CHANNELS_m1);
#endif
      print_bias_debug("bias_m1", &bias_m1[0], OUT_CHANNELS_m1);
      print_weights_debug("weights_m1", &weights_m1[0][0][0][0], OUT_CHANNELS_m1, KERNEL_DIM_m1, IN_CHANNELS_m1);
      print_weights_mat_debug("weights_mat_m1", &weights_mat_m1[0][0], PATCH_SIZE_m1, OUT_CHANNELS_m1);
      print_input_debug("input_m1", &input_m1[0][0][0][0], BATCH_SIZE_m1, IN_ROW_DIM_m1, IN_COL_DIM_m1, IN_CHANNELS_m1);
    }

    return 1;
  }

  printf("Output matrices came out as expected\n");
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

  return 0;
}
