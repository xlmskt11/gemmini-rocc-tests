#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils_all.h"

#define PROFILE false
#define profile_data_num 30

#define MULTI true
#define gemmini_configuration 15

#define FAST true
#define NO_BIAS false
#define CHECK false

#define q_type(p) (p >> 62)
#define start(p) ((p >> 31) & ((1 << 31) - 1))
#define end(p) (p & ((1 << 31) - 1))

#ifndef BAREMETAL

#define BATCH_SIZE 4
#define IN_ROW_DIM 224
#define IN_COL_DIM 224
#define IN_CHANNELS 3
#define OUT_CHANNELS 32
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 2

#else

#if FAST

#define IN_ROW_DIM 28
#define IN_COL_DIM 28
#define IN_CHANNELS 64
#define OUT_CHANNELS 128

#else

#define IN_ROW_DIM 28
#define IN_COL_DIM 28
#define IN_CHANNELS 64
#define OUT_CHANNELS 128

#endif

#define BATCH_SIZE 1
#define KERNEL_DIM 3
#define PADDING 1
#define STRIDE 1

#endif

#define OUT_ROW_DIM ((IN_ROW_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define OUT_COL_DIM ((IN_COL_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM)

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
        elem_t output[batch_size][out_row_dim][out_col_dim][out_channels]) {

#ifdef GEMMINI_ASSERTIONS
    if (out_row_dim != (in_row_dim + 2 * padding - kernel_dim) / stride + 1) {
        printf("conv out_row_dim is not correct\n");
        exit(1);
    }

    if (out_col_dim != (in_col_dim + 2 * padding - kernel_dim) / stride + 1) {
        printf("conv out_col_dim is not correct\n");
        exit(1);
    }
#endif

    for (int b = 0; b < batch_size; b++) {
        for (int orow = 0; orow < out_row_dim; orow++) {
            for (int ocol = 0; ocol < out_col_dim; ocol++) {
                for (int och = 0; och < out_channels; och++) {
                    acc_t result = bias[och];

                    for (int krow = 0; krow < kernel_dim; krow++) {
                        for (int kcol = 0; kcol < kernel_dim; kcol++) {
                            for (int kch = 0; kch < in_channels; kch++) {
                                int irow = orow * stride + krow - padding;
                                int icol = ocol * stride + kcol - padding;

                                elem_t pixel = irow < 0 || irow >= in_row_dim ||
                                    icol < 0 || icol >= in_col_dim ?
                                    0 : input[b][irow][icol][kch];

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
        elem_t weights_mat[patch_size][out_channels]) {

    assert(patch_size == kernel_dim * kernel_dim * in_channels);

    for (int outc = 0; outc < out_channels; outc++) {
        for (int krow = 0; krow < kernel_dim; krow++) {
            for (int kcol = 0; kcol < kernel_dim; kcol++) {
                for (int inc = 0; inc < in_channels; inc++) {
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

bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            return false;
    return true;
}

void init_random(elem_t * buf, int len) {
    elem_t i = 0;
    for (elem_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
#if FAST
      *ptr = 1;
#else
      *ptr = (rand() % 5) - 2;
#endif
    }
}

void init_random_acc(acc_t * buf, int len) {
    elem_t i = 0;
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
#if FAST
      *ptr = 1;
#else
      *ptr = (rand() % 5) - 2;
#endif
    }
}

void init_zeros_acc(acc_t * buf, int len) {
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        *ptr = 0;
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

#if MULTI
    gemmini_flush(custom0, 0);
    gemmini_flush(custom1, 0);
    gemmini_flush(custom2, 0);
#endif
    gemmini_flush(custom3, 0);

    // assert((in_dim + 2*padding - kernel_dim) % stride == 0);

    printf("Input dimensions (rows by columns): %u by %u\n", IN_ROW_DIM, IN_COL_DIM);
    printf("Input channels: %u\n", IN_CHANNELS);
    printf("Batch size: %u\n", BATCH_SIZE);
    printf("Kernel dimensions: %u by %u\n", KERNEL_DIM, KERNEL_DIM);
    printf("Stride: %u\n", STRIDE);
    printf("Padding: %u\n", PADDING);
    printf("Output channels: %u\n", OUT_CHANNELS);
    printf("Output dimensions (rows by columns): %u by %u\n\n", OUT_ROW_DIM, OUT_COL_DIM);

    static elem_t input[BATCH_SIZE][IN_ROW_DIM][IN_COL_DIM][IN_CHANNELS] row_align(MAX_BLOCK_LEN);
    static acc_t bias[OUT_CHANNELS] row_align_acc(MAX_BLOCK_LEN_ACC);

    printf("Randomize inputs...\n");
    init_random(&input[0][0][0][0], sizeof(input) / sizeof(elem_t));

#if CHECK && !FAST
    printf("Randomize weights...\n");
    static elem_t weights[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS] row_align(MAX_BLOCK_LEN);
    init_random(&weights[0][0][0][0], sizeof(weights) / sizeof(elem_t));
#endif

    printf("Randomize bias...\n");
    if (NO_BIAS)
        init_zeros_acc(&bias[0], sizeof(bias) / sizeof(acc_t));
    else
        init_random_acc(&bias[0], sizeof(bias) / sizeof(acc_t));

#if CHECK && !FAST
    static elem_t output[BATCH_SIZE][OUT_ROW_DIM][OUT_COL_DIM][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);

    printf("CPU conv...\n");
    uint64_t start_cpu = read_cycles();

    conv(BATCH_SIZE, IN_CHANNELS,
            IN_ROW_DIM, IN_COL_DIM,
            OUT_CHANNELS, KERNEL_DIM,
            OUT_ROW_DIM, OUT_COL_DIM,
            STRIDE, PADDING,
            input,
            weights,
            bias,
            output);

    uint64_t end_cpu = read_cycles();
    printf("CPU conv took %llu cycles\n", end_cpu - start_cpu);
#endif

    static elem_t weights_mat[PATCH_SIZE][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);
    static elem_t output_mat[N_PATCHES][OUT_CHANNELS] row_align(MAX_BLOCK_LEN);

#if CHECK && !FAST
    printf("Flatten weights...\n");
    flatten_weights(OUT_CHANNELS, KERNEL_DIM, IN_CHANNELS,
                    PATCH_SIZE,
                    weights,
                    weights_mat);
#else
    printf("Init flattened weights directly...\n");
    init_random(&weights_mat[0][0], sizeof(weights_mat) / sizeof(elem_t));
#endif

    shared_multi_conv_job_t j0;

#if MULTI
    int total_cycles = 0;
    printf("Do Gemmini tiled conv process\n");

    const size_t sp_addr_range = BANK_NUM * BANK_ROWS;
    const size_t acc_addr_range = ACC_ROWS;
    const size_t max_spad_rows = sp_addr_range / 2;
    const size_t max_acc_rows = acc_addr_range / 2;

    const int pool_size = 1;
    const int pool_stride = 1;
    const int pool_padding = 0;
    const bool downsample = STRIDE == 2 && KERNEL_DIM == 1 && PADDING == 0 &&
                            IN_ROW_DIM % 2 == 0 && IN_COL_DIM % 2 == 0;

    for (int porows = 1; porows <= OUT_ROW_DIM; porows += 1)
    {
        for (int krows = 1; krows <= KERNEL_DIM; krows += 1)
        {
          for (int kcols = 1; kcols <= KERNEL_DIM; kcols += 1)
          {
            for (int pocols = 16; pocols <= OUT_COL_DIM; pocols += 16)
            {
                for (int pochs = 64; pochs <= OUT_CHANNELS; pochs += 16)
                {
                    for (int kchs = 64; kchs <= IN_CHANNELS; kchs += 16)
                    {
                        const int batches = 1;

                        const int spad_rows = tiled_conv_total_spad_rows(false,
                                                                        STRIDE, 1, 1, downsample, false, false,
                                                                        batches, porows, pocols, pochs, krows, kcols, kchs,
                                                                        pool_size, pool_stride);
                        
                        const int acc_rows = tiled_conv_total_spad_rows(true,
                                                                        STRIDE, 1, 1, downsample, false, false,
                                                                        batches, porows, pocols, pochs, krows, kcols, kchs,
                                                                        pool_size, pool_stride);

                        if (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
                            printf("spad_rows: %d, acc_rows: %d exceed max limit\n", spad_rows, acc_rows);
                            continue;
                        }

                        gemmini_flush(custom0, 0);
                        gemmini_flush(custom1, 0);
                        gemmini_flush(custom2, 0);
                        gemmini_flush(custom3, 0);

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
                        j0.pool_size = 1;
                        j0.pool_stride = 1;
                        j0.pool_padding = 0;
                        j0.tiled_conv_type = WS;

                        uint64_t start_gemmini = read_cycles();

                        shared_multi_choose_conv_tiling_factors_static(&j0, batches, porows, pocols, pochs, krows, kcols, kchs);
                        shared_multi_tiled_conv_job_init(&j0);
                        while (!j0.done)
                        {
                        shared_multi_tiled_conv_job_step(&j0);
                        }

                        gemmini_fence();
                        uint64_t end_gemmini = read_cycles();

                        printf("batches: %d, porows: %d, pocols: %d, pochs: %d, krows: %d, kcols: %d, kchs: %d, total_spad_rows: %d, total_acc_rows: %d, Conv cycle: %llu\n",
                            batches, porows, pocols, pochs, krows, kcols, kchs, spad_rows, acc_rows, end_gemmini - start_gemmini);
                        total_cycles += (end_gemmini - start_gemmini);
                    }
                    }
                }
            }
        }
    }

    printf("Total Conv cycle: %d\n", total_cycles);
#else
    printf("Gemmini conv...\n");
    uint64_t start_gemmini = read_cycles();
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
    uint64_t end_gemmini = read_cycles();
    printf("Gemmini conv took %llu cycles\n", end_gemmini - start_gemmini);
#endif

#if !FAST && CHECK
    assert(sizeof(output_mat) == sizeof(output));
#endif

#if CHECK
#if FAST
    bool success = true;
    for (int orow = 0; orow < BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM; orow++) {
      for (int ocol = 0; ocol < OUT_CHANNELS; ocol++) {
	elem_t v = output_mat[orow][ocol];
    if (v != (KERNEL_DIM - 1) * (KERNEL_DIM - 1) * IN_CHANNELS + 1 && v != KERNEL_DIM * (KERNEL_DIM - 1) * IN_CHANNELS + 1 && v != KERNEL_DIM * KERNEL_DIM * IN_CHANNELS + 1)
    {
        success = false;
        break;
    }
      }
    }
#else
    bool success = vec_is_equal(&output[0][0][0][0], &output_mat[0][0], sizeof(output) / sizeof(elem_t));
#endif

    if (!success) {
        // return 1;
        printf("Incorrect output!\n");
        
        printf("output_mat:\n");
        for (int orow = 0; orow < BATCH_SIZE * OUT_ROW_DIM * OUT_COL_DIM; orow++)
        {
            printf("[");
            for (int ocol = 0; ocol < OUT_CHANNELS; ocol++)
            {
                printf("%d,", output_mat[orow][ocol]);
            }
            printf("\b],\n");
        }
        printf("\b\n\n");

#if !FAST
        printf("output:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++)
        {
            printf("[");
            for (int orow = 0; orow < OUT_ROW_DIM; orow++)
            {
                printf("[");
                for (int ocol = 0; ocol < OUT_COL_DIM; ocol++)
                {
                    printf("[");
                    for (int och = 0; och < OUT_CHANNELS; och++)
                    {
                        printf("%d,", output[batch][orow][ocol][och]);
                    }
                    printf("\b],");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");
#endif

        printf("bias:\n");
        for (int och = 0; och < OUT_CHANNELS; och++) {
            printf("%d,", bias[och]);
        }
        printf("\b\n\n");

        printf("weights:\n");
        for (int och = 0; och < OUT_CHANNELS; och++) {
            printf("[");
            for (int wrow = 0; wrow < KERNEL_DIM; wrow++) {
                printf("[");
                for (int wcol = 0; wcol < KERNEL_DIM; wcol++) {
                    printf("[");
                    for (int ich = 0; ich < IN_CHANNELS; ich++) {
                        printf("%d,", weights[och][wrow][wcol][ich]);
                    }
                    printf("\b],");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");

        printf("weights_mat:\n");
        for (int wrow = 0; wrow < KERNEL_DIM * KERNEL_DIM * IN_CHANNELS; wrow++) {
            printf("[");
            for (int wcol = 0; wcol < OUT_CHANNELS; wcol++) {
                printf("%d,", weights_mat[wrow][wcol]);
            }
            printf("\b],\n");
        }
        printf("\b\n\n");

        printf("input:\n");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("[");
            for (int irow = 0; irow < IN_ROW_DIM; irow++) {
                printf("[");
                for (int icol = 0; icol < IN_COL_DIM; icol++) {
                    printf("[");
                    for (int ich = 0; ich < IN_CHANNELS; ich++) {
                        printf("%d,", input[batch][irow][icol][ich]);
                    }
                    printf("\b],");
                }
                printf("\b],\n");
            }
            printf("\b],");
        }
        printf("\b\n\n");

        return 1;
    }
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
