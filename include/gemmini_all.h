// See LICENSE for license details.

#ifndef SRC_MAIN_C_GEMMINI_H
#define SRC_MAIN_C_GEMMINI_H

#undef abs

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>

#include "include/gemmini_params.h"

#define GEMMINI_ASSERTIONS

// Accelerator interface
#include "rocc-software/src/xcustom.h"

// Counter Definition
#include "include/gemmini_counter.h"

// made
// #define PRINT_TILE true
#define TOTAL_SPAD_ROWS (BANK_NUM * BANK_ROWS)
#define TOTAL_ACC_ROWS (ACC_ROWS)
#define total_gemmini_num 4
#define nSharers 4
#define group_w 4
#define custom0 0
#define custom1 1
#define custom2 2
#define custom3 3

#define k_CONFIG 0
#define k_MVIN2 1
#define k_MVIN 2
#define k_MVOUT 3
#define k_COMPUTE_PRELOADED 4
#define k_COMPUTE_ACCUMULATE 5
#define k_PRELOAD 6
#define k_FLUSH 7

#define k_LOOP_WS 8
#define k_LOOP_WS_CONFIG_BOUNDS 9
#define k_LOOP_WS_CONFIG_ADDRS_AB 10
#define k_LOOP_WS_CONFIG_ADDRS_DC 11
#define k_LOOP_WS_CONFIG_STRIDES_AB 12
#define k_LOOP_WS_CONFIG_STRIDES_DC 13

// made
#define SET_PROFILER_PADDR 23
#define k_LOOP_WS_CONFIG_MV_BOUNDS_1 24
#define k_LOOP_WS_CONFIG_SPADDR 25
#define k_LOOP_CONV_WS_CONFIG_MV_BOUNDS_1 26
#define k_LOOP_CONV_WS_CONFIG_SPADDR 27

#define k_MVIN3 14

#define k_COUNTER 126

#define k_LOOP_CONV_WS 15
#define k_LOOP_CONV_WS_CONFIG_1 16
#define k_LOOP_CONV_WS_CONFIG_2 17
#define k_LOOP_CONV_WS_CONFIG_3 18
#define k_LOOP_CONV_WS_CONFIG_4 19
#define k_LOOP_CONV_WS_CONFIG_5 20
#define k_LOOP_CONV_WS_CONFIG_6 21

#define CONFIG_EX 0
#define CONFIG_LD 1
#define CONFIG_ST 2
#define CONFIG_BERT 3

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4

#ifdef ELEM_T_IS_FLOAT
elem_t elem_t_bits_to_elem_t(elem_t_bits x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.b = x;
    return un.f;
}

elem_t_bits elem_t_to_elem_t_bits(elem_t x) {
    union {
        elem_t_bits b;
        elem_t f;
    } un;

    un.f = x;
    return un.b;
}

acc_t acc_t_bits_to_acc_t(acc_t_bits x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.b = x;
    return un.f;
}

acc_t_bits acc_t_to_acc_t_bits(acc_t x) {
    union {
        acc_t_bits b;
        acc_t f;
    } un;

    un.f = x;
    return un.b;
}

bool elem_t_isnan(elem_t x) {
    elem_t_bits bits = elem_t_to_elem_t_bits(x);
    uint64_t exp = (bits >> (ELEM_T_SIG_BITS-1)) & (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ELEM_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ELEM_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}

bool acc_t_isnan(acc_t x) {
    acc_t_bits bits = acc_t_to_acc_t_bits(x);
    uint64_t exp = (bits >> (ACC_T_SIG_BITS-1)) & (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    uint64_t sig = bits & (((uint64_t)1 << ACC_T_SIG_BITS) - 1);
    bool is_nan_or_inf = exp == (((uint64_t)1 << ACC_T_EXP_BITS) - 1);
    bool is_not_inf = sig != 0;
    return is_nan_or_inf && is_not_inf;
}
#endif

#ifdef HAS_MVIN_SCALE
static scale_t scale_t_bits_to_scale_t(scale_t_bits x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.b = x;
    return un.f;
}

static scale_t_bits scale_t_to_scale_t_bits(scale_t x) {
    union {
        scale_t_bits b;
        scale_t f;
    } un;

    un.f = x;
    return un.b;
}
#else
#define scale_t_to_scale_t_bits(x) 0
#endif

#ifdef HAS_MVIN_ACC_SCALE
static scale_acc_t scale_acc_t_bits_to_scale_acc_t(scale_acc_t_bits x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.b = x;
    return un.f;
}

static scale_acc_t_bits scale_acc_t_to_scale_acc_t_bits(scale_acc_t x) {
    union {
        scale_acc_t_bits b;
        scale_acc_t f;
    } un;

    un.f = x;
    return un.b;
}
#endif

static acc_scale_t acc_scale_t_bits_to_acc_scale_t(acc_scale_t_bits x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.b = x;
    return un.f;
}

static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x) {
    union {
        acc_scale_t_bits b;
        acc_scale_t f;
    } un;

    un.f = x;
    return un.b;
}

#define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct) \
  ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

// mvin and mvout
#define gemmini_extended_mvin(custom_num, dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN)

#define gemmini_extended_mvin2(custom_num, dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN2)

#define gemmini_extended_mvin3(custom_num, dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (spad_addr), k_MVIN3)

#define gemmini_block_mvin(custom_num, dram_addr, spad_addr, len) \
  gemmini_extended_mvin(custom_num, dram_addr, spad_addr, (len) * DIM, DIM)

#define gemmini_mvin(custom_num, dram_addr, spad_addr) \
  gemmini_extended_mvin(custom_num, dram_addr, spad_addr, DIM, DIM)

#define gemmini_extended_mvout(custom_num, dram_addr, spad_addr, cols, rows) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, dram_addr, ((uint64_t)(rows) << (ADDR_LEN + 16)) | ((uint64_t)(cols) << ADDR_LEN) | (uint64_t)(spad_addr), k_MVOUT)

#define gemmini_mvout(custom_num, dram_addr, spad_addr) \
  gemmini_extended_mvout(custom_num, dram_addr, spad_addr, DIM, DIM)

// compute
#define gemmini_extended_compute_preloaded(custom_num, A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_PRELOADED)

#define gemmini_extended_compute_accumulated(custom_num, A, BD, A_cols, A_rows, BD_cols, BD_rows) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(A_rows) << (ADDR_LEN + 16)) | ((uint64_t)(A_cols) << ADDR_LEN) | (uint64_t)(A), ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), k_COMPUTE_ACCUMULATE)

#define gemmini_compute_preloaded(custom_num, A, BD) \
  gemmini_extended_compute_preloaded(custom_num, A, BD, DIM, DIM, DIM, DIM)

#define gemmini_compute_accumulated(custom_num, A, BD) \
  gemmini_extended_compute_accumulated(custom_num, A, BD, DIM, DIM, DIM, DIM)

// preload
#define gemmini_extended_preload(custom_num, BD, C, BD_cols, BD_rows, C_cols, C_rows) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(BD_rows) << (ADDR_LEN + 16)) | ((uint64_t)(BD_cols) << ADDR_LEN) | (uint64_t)(BD), ((uint64_t)(C_rows) << (ADDR_LEN + 16)) | ((uint64_t)(C_cols) << ADDR_LEN) | (uint64_t)(C), k_PRELOAD)

#define gemmini_preload(custom_num, BD, C) \
  gemmini_extended_preload(custom_num, BD, C, DIM, DIM, DIM, DIM)

#define gemmini_preload_zeros(custom_num, C) \
  gemmini_preload(custom_num, GARBAGE_ADDR, C)

// config
#define gemmini_extended3_config_ex(custom_num, dataflow, sys_act, sys_shift, sys_acc_scale, C_stride, A_stride, A_transpose, B_transpose, set_only_strides) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)sys_acc_scale) << 32) | ((uint64_t)(A_stride) << 16) | (B_transpose << 9) | (A_transpose << 8) | ((set_only_strides) << 7) | ((sys_act) << 3) | ((dataflow) << 2) | CONFIG_EX, ((uint64_t)(C_stride) << 48) | (sys_shift), k_CONFIG); \

#define gemmini_extended2_config_ex(custom_num, dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended3_config_ex(custom_num, dataflow, sys_act, sys_shift, ACC_SCALE_IDENTITY, 1, A_stride, A_transpose, B_transpose, false)

#define gemmini_extended_config_ex(custom_num, dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose) \
  gemmini_extended2_config_ex(custom_num, dataflow, sys_act, sys_shift, A_stride, A_transpose, B_transpose)

#define gemmini_config_ex(custom_num, dataflow, sys_act, sys_shift) \
    gemmini_extended_config_ex(custom_num, dataflow, sys_act, sys_shift, 1, 0, 0)

// Note: The "pixel_repeats" parameter below is still experimental, andthere is
// a high chance that it will be removed in future releases.
#define gemmini_extended5_config_ld(custom_num, stride, scale, shrunk, block_mvin_stride, pixel_repeats, id) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(scale_t_to_scale_t_bits(scale)) << 32) | ((uint64_t)(block_mvin_stride) << 16) | ((uint64_t)(pixel_repeats) << 8) | ((id) << 3) | ((shrunk) << 2) | CONFIG_LD, stride, k_CONFIG)

#define gemmini_extended4_config_ld(custom_num, stride, scale, shrunk, block_mvin_stride, id) \
  gemmini_extended5_config_ld(custom_num, stride, scale, shrunk, block_mvin_stride, 1, id) \

#define gemmini_extended3_config_ld(custom_num, stride, scale, shrunk, id) \
  gemmini_extended4_config_ld(custom_num, stride, scale, shrunk, DIM, id)

#define gemmini_extended2_config_ld(custom_num, stride, scale, shrunk) \
  gemmini_extended3_config_ld(custom_num, stride, scale, shrunk, 0)

#define gemmini_extended_config_ld(custom_num, stride, scale) \
  gemmini_extended2_config_ld(custom_num, stride, scale, false)

#define gemmini_config_ld(custom_num, stride) \
  gemmini_extended_config_ld(custom_num, stride, MVIN_SCALE_IDENTITY)

#define gemmini_extended2_config_st(custom_num, stride, acc_act, acc_scale, pool_stride, pool_size, pool_out_dim, porows, pocols, orows, ocols, upad, lpad) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(ocols) << 56) | ((uint64_t)(orows) << 48) | ((uint64_t)(pocols) << 40) | ((uint64_t)(porows) << 32) | ((uint64_t)(pool_out_dim) << 24) | ((uint64_t)(lpad) << 10) | ((uint64_t)(upad) << 8) | ((uint64_t)(pool_size) << 6) | ((uint64_t)(pool_stride) << 4) | ((uint64_t)(acc_act) << 2) | CONFIG_ST, ((uint64_t)acc_scale_t_to_acc_scale_t_bits((acc_scale_t)acc_scale) << 32) | ((uint32_t)stride), k_CONFIG)

#define gemmini_extended_config_st(custom_num, stride, acc_act, acc_scale) \
    gemmini_extended2_config_st(custom_num, stride, acc_act, acc_scale, 0, 0, 0, 0, 0, 0, 0, 0, 0)

#define gemmini_config_st(custom_num, stride) \
    gemmini_extended_config_st(custom_num, stride, NO_ACTIVATION, ACC_SCALE_IDENTITY)

#define gemmini_config_norm(custom_num, q_const, q_const_type, set_stats_id_only, act_msb, stat_id, igelu_qb, igelu_qc) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, (((uint64_t) ((uint32_t) q_const)) << 32) | ((q_const_type & 1) << 18) | ((set_stats_id_only & 1) << 17) | ((act_msb & 1) << 16) | ((uint64_t)stat_id << 8) | CONFIG_BERT, ((uint64_t)((uint32_t)(igelu_qc)) << 32) | ((uint64_t)((uint32_t)(igelu_qb))), k_CONFIG)

// made
#define gemmini_profiler(custom_num, P) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, P, 0, SET_PROFILER_PADDR)

// flush
#define gemmini_flush(custom_num, skip) \
  ROCC_INSTRUCTION_RS1_RS2(custom_num, skip, 0, k_FLUSH)

// fence
#define gemmini_fence() asm volatile("fence")

// Counter access
#define gemmini_counter_access(custom_num, rd, config_reg) \
  { \
    uint32_t _placeholder; \
    ROCC_INSTRUCTION(custom_num, rd, config_reg, _placeholder, k_COUNTER) \
  }

// Read counter
static uint32_t counter_read(int custom_num, size_t index) {
  uint32_t config_reg = (index & 0x7) << 4;
  uint32_t res;
  switch (custom_num)
  {
  case 0:
    gemmini_counter_access(custom0, res, config_reg);
    break;
  case 1:
    gemmini_counter_access(custom1, res, config_reg);
    break;
  case 2:
    gemmini_counter_access(custom2, res, config_reg);
    break;
  case 3:
    gemmini_counter_access(custom3, res, config_reg);
    break;
  }
  return res;
}

// Configure counter to take a new signal
static void counter_configure(int custom_num, size_t index, size_t counter_code) {
  int non_incremental = counter_code > INCREMENTAL_COUNTERS;
  if (non_incremental) {
    counter_code -= INCREMENTAL_COUNTERS;
  }

  uint32_t config_reg = (index & 0x7) << 4 | 0x8 | (counter_code & 0x3f) << 12 | non_incremental << 31;
  uint32_t placeholder;
  switch (custom_num)
  {
  case 0:
    gemmini_counter_access(custom0, placeholder, config_reg);
    break;
  case 1:
    gemmini_counter_access(custom1, placeholder, config_reg);
    break;
  case 2:
    gemmini_counter_access(custom2, placeholder, config_reg);
    break;
  case 3:
    gemmini_counter_access(custom3, placeholder, config_reg);
    break;
  }
}

// Take a snapshot
static void counter_snapshot_take(int custom_num) {
  uint32_t config_reg = 0x4;
  uint32_t placeholder;

  switch (custom_num)
  {
  case 0:
    gemmini_counter_access(custom0, placeholder, config_reg);
    break;
  case 1:
    gemmini_counter_access(custom1, placeholder, config_reg);
    break;
  case 2:
    gemmini_counter_access(custom2, placeholder, config_reg);
    break;
  case 3:
    gemmini_counter_access(custom3, placeholder, config_reg);
    break;
  }
}

// Counter snapshot reset
static void counter_snapshot_reset(int custom_num) {
  uint32_t config_reg = 0x2;
  uint32_t placeholder;
  gemmini_counter_access(custom_num, placeholder, config_reg);
}

// Counter module reset
static void counter_reset(int custom_num) {
  uint32_t config_reg = 0x1;
  uint32_t placeholder;
  switch (custom_num)
  {
  case 0:
    gemmini_counter_access(custom0, placeholder, config_reg);
    break;
  case 1:
    gemmini_counter_access(custom1, placeholder, config_reg);
    break;
  case 2:
    gemmini_counter_access(custom2, placeholder, config_reg);
    break;
  case 3:
    gemmini_counter_access(custom3, placeholder, config_reg);
    break;
  }
}

// weight-stationary matmul loop
#define gemmini_loop_ws(custom_num, I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(0) << 36) | ((uint64_t)(0) << 32) | (uint64_t)(0) << 16 | (uint64_t)(0), ((uint64_t)(K) << 32) | ((uint64_t)(pad_K) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_MV_BOUNDS_1) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

// made
#define shared_gemmini_loop_ws(custom_num, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_I, mv_K, mv_pad_K, laddrI_offset, laddrK_offset, I, J, K, pad_I, pad_J, pad_K, A, B, D, C, A_stride, B_stride, D_stride, C_stride, A_transpose, B_transpose, full_C, low_D, ex_accumulate, act) \
  { \
      ROCC_INSTRUCTION_RS1_RS2(custom_num, acc_addr_start, ((uint64_t)(sp_addr_end) << 16) | (uint64_t)(sp_addr_start), k_LOOP_WS_CONFIG_SPADDR) \
      ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(group_list) << 48) | ((uint64_t)(group_id) << 32) | (uint64_t)(laddrK_offset) << 16 | (uint64_t)(laddrI_offset), ((uint64_t)(mv_K) << 32) | ((uint64_t)(mv_pad_K) << 16) | (uint64_t)(ex_I), k_LOOP_WS_CONFIG_MV_BOUNDS_1) \
      ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(I), k_LOOP_WS_CONFIG_BOUNDS) \
      ROCC_INSTRUCTION_RS1_RS2(custom_num, A, B, k_LOOP_WS_CONFIG_ADDRS_AB) \
      ROCC_INSTRUCTION_RS1_RS2(custom_num, D, C, k_LOOP_WS_CONFIG_ADDRS_DC) \
      ROCC_INSTRUCTION_RS1_RS2(custom_num, A_stride, B_stride, k_LOOP_WS_CONFIG_STRIDES_AB) \
      ROCC_INSTRUCTION_RS1_RS2(custom_num, D_stride, C_stride, k_LOOP_WS_CONFIG_STRIDES_DC) \
      ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (ex_accumulate), ((B_transpose) << 1) | (A_transpose), k_LOOP_WS) \
  }

// weight-stationary conv loop
#define gemmini_loop_conv_ws(custom_num, batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, activation, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(0) << 36) | ((uint64_t)(0) << 32) | (uint64_t)(0) << 16 | (uint64_t)(0), ((uint64_t)(kchs) << 16) | (uint64_t)(pochs), k_LOOP_CONV_WS_CONFIG_MV_BOUNDS_1) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(out_channels) << 48) | ((uint64_t)(in_channels) << 32) | ((uint64_t)(in_dim) << 16) | (uint64_t)(batch_size), \
      ((uint64_t)(padding) << 48) | ((uint64_t)(stride) << 32) | ((uint64_t)(pool_out_dim) << 16) | (uint64_t)(out_dim), k_LOOP_CONV_WS_CONFIG_1) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(kernel_dim) << 48) | ((uint64_t)(pool_size) << 32) | ((uint64_t)(pool_stride) << 16) | (uint64_t)(pool_padding), \
      ((uint64_t)(batches) << 48) | ((uint64_t)(porows) << 32) | ((uint64_t)(pocols) << 16) | (uint64_t)(pochs), k_LOOP_CONV_WS_CONFIG_2) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(krows) << 48) | ((uint64_t)(kcols) << 32) | ((uint64_t)(kchs) << 16) | (uint64_t)(lpad), \
      ((uint64_t)(rpad) << 48) | ((uint64_t)(upad) << 32) | ((uint64_t)(dpad) << 16) | (uint64_t)(plpad), k_LOOP_CONV_WS_CONFIG_3) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(orows) << 48) | ((uint64_t)(prpad) << 32) | ((uint64_t)(pupad) << 16) | (uint64_t)(pdpad), \
      ((uint64_t)(kernel_dilation) << 16) | (uint64_t)(ocols), k_LOOP_CONV_WS_CONFIG_4) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, weights, \
      output, k_LOOP_CONV_WS_CONFIG_5) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, bias, \
      input, k_LOOP_CONV_WS_CONFIG_6) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(max_pixels_per_row) << 8) | ((dw) << 6) | ((trans_input_3120) << 5) | ((trans_weight_0132) << 4) | ((trans_weight_1203) << 3) | ((trans_output_1203) << 2) | ((wrot180) << 1) | (no_bias), \
      ((activation) << 3)| ((input_dilated) << 2) | ((downsample) << 1) | (no_pool), \
      k_LOOP_CONV_WS) \
  }

// made
#define shared_gemmini_loop_conv_ws(custom_num, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_ochs, mv_kchs, laddrochs_offset, laddrkchs_offset, batch_size, in_dim, in_channels, out_channels, out_dim, pool_out_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, activation, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw) \
  { \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, acc_addr_start, ((uint64_t)(sp_addr_end) << 16) | sp_addr_start, k_LOOP_CONV_WS_CONFIG_SPADDR) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(group_list) << 48) | ((uint64_t)(group_id) << 32) | (uint64_t)(laddrkchs_offset) << 16 | (uint64_t)(laddrochs_offset), ((uint64_t)(mv_kchs) << 16) | (uint64_t)(ex_ochs), k_LOOP_CONV_WS_CONFIG_MV_BOUNDS_1) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(out_channels) << 48) | ((uint64_t)(in_channels) << 32) | ((uint64_t)(in_dim) << 16) | (uint64_t)(batch_size), \
      ((uint64_t)(padding) << 48) | ((uint64_t)(stride) << 32) | ((uint64_t)(pool_out_dim) << 16) | (uint64_t)(out_dim), k_LOOP_CONV_WS_CONFIG_1) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(kernel_dim) << 48) | ((uint64_t)(pool_size) << 32) | ((uint64_t)(pool_stride) << 16) | (uint64_t)(pool_padding), \
      ((uint64_t)(batches) << 48) | ((uint64_t)(porows) << 32) | ((uint64_t)(pocols) << 16) | (uint64_t)(pochs), k_LOOP_CONV_WS_CONFIG_2) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(krows) << 48) | ((uint64_t)(kcols) << 32) | ((uint64_t)(kchs) << 16) | (uint64_t)(lpad), \
      ((uint64_t)(rpad) << 48) | ((uint64_t)(upad) << 32) | ((uint64_t)(dpad) << 16) | (uint64_t)(plpad), k_LOOP_CONV_WS_CONFIG_3) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(orows) << 48) | ((uint64_t)(prpad) << 32) | ((uint64_t)(pupad) << 16) | (uint64_t)(pdpad), \
      ((uint64_t)(kernel_dilation) << 16) | (uint64_t)(ocols), k_LOOP_CONV_WS_CONFIG_4) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, weights, \
      output, k_LOOP_CONV_WS_CONFIG_5) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, bias, \
      input, k_LOOP_CONV_WS_CONFIG_6) \
    ROCC_INSTRUCTION_RS1_RS2(custom_num, ((uint64_t)(max_pixels_per_row) << 8) | ((dw) << 6) | ((trans_input_3120) << 5) | ((trans_weight_0132) << 4) | ((trans_weight_1203) << 3) | ((trans_output_1203) << 2) | ((wrot180) << 1) | (no_bias), \
      ((activation) << 3)| ((input_dilated) << 2) | ((downsample) << 1) | (no_pool), \
      k_LOOP_CONV_WS) \
  }

// Tiling functions
static void sp_tiled_matmul_os(int custom_num, const elem_t * A, const elem_t * B, const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act) {

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN-2)) | (full_C << (ADDR_LEN-3));

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks = J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC;

  // Move-in D
  if (D != NULL && !no_bias) {
    const size_t D_stride = repeating_bias ? 0 : D_row_stride * sizeof(acc_t);
    switch(custom_num) {
      case 0:
        gemmini_extended_config_ld(custom0, D_stride, D_scale_factor);
        break;
      case 1:
        gemmini_extended_config_ld(custom1, D_stride, D_scale_factor);
        break;
      case 2:
        gemmini_extended_config_ld(custom2, D_stride, D_scale_factor);
        break;
      case 3:
        gemmini_extended_config_ld(custom3, D_stride, D_scale_factor);
        break;
    }

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const acc_t * const D_dram_addr = (acc_t *)D + (bias_row * D_row_stride + j)*DIM;

        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;

        const size_t blocks = j + D_blocks <= J ? D_blocks : J-j;

        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        const size_t rows = DIM - (i == I-1 ? pad_I : 0);
        switch(custom_num) {
          case 0:
            gemmini_extended_mvin(custom0, D_dram_addr, D_sp_addr_acc, cols, rows);
            break;
          case 1:
            gemmini_extended_mvin(custom1, D_dram_addr, D_sp_addr_acc, cols, rows);
            break;
          case 2:
            gemmini_extended_mvin(custom2, D_dram_addr, D_sp_addr_acc, cols, rows);
            break;
          case 3:
            gemmini_extended_mvin(custom3, D_dram_addr, D_sp_addr_acc, cols, rows);
            break;
        }
      }
    }
  }

  // Move-in B
  switch(custom_num) {
    case 0:
      gemmini_extended_config_ld(custom0, B_row_stride * sizeof(elem_t), B_scale_factor);
      break;
    case 1:
      gemmini_extended_config_ld(custom1, B_row_stride * sizeof(elem_t), B_scale_factor);
      break;
    case 2:
      gemmini_extended_config_ld(custom2, B_row_stride * sizeof(elem_t), B_scale_factor);
      break;
    case 3:
      gemmini_extended_config_ld(custom3, B_row_stride * sizeof(elem_t), B_scale_factor);
      break;
  }
  for (size_t j = 0; j < J; j += B_blocks) {
    for (size_t k = 0; k < K; k++) {
      const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
      const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;
      const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
      const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
      const size_t rows = DIM - (k == K-1 ? pad_K : 0);
      switch(custom_num) {
        case 0:
          gemmini_extended_mvin(custom0, B_dram_addr, B_sp_addr, cols, rows);
          break;
        case 1:
          gemmini_extended_mvin(custom1, B_dram_addr, B_sp_addr, cols, rows);
          break;
        case 2:
          gemmini_extended_mvin(custom2, B_dram_addr, B_sp_addr, cols, rows);
          break;
        case 3:
          gemmini_extended_mvin(custom3, B_dram_addr, B_sp_addr, cols, rows);
          break;
      }
    }
  }

  // Move-in A
  switch(custom_num) {
    case 0:
      gemmini_extended_config_ld(custom0, A_row_stride * sizeof(elem_t), A_scale_factor);
      break;
    case 1:
      gemmini_extended_config_ld(custom1, A_row_stride * sizeof(elem_t), A_scale_factor);
      break;
    case 2:
      gemmini_extended_config_ld(custom2, A_row_stride * sizeof(elem_t), A_scale_factor);
      break;
    case 3:
      gemmini_extended_config_ld(custom3, A_row_stride * sizeof(elem_t), A_scale_factor);
      break;
  }
  for (size_t i = 0; i < I; i++) {
    for (size_t k = 0; k < K; k += A_blocks) {
      const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
      const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
      const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
      const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      switch(custom_num) {
        case 0:
          gemmini_extended_mvin(custom0, A_dram_addr, A_sp_addr, cols, rows);
          break;
        case 1:
          gemmini_extended_mvin(custom1, A_dram_addr, A_sp_addr, cols, rows);
          break;
        case 2:
          gemmini_extended_mvin(custom2, A_dram_addr, A_sp_addr, cols, rows);
          break;
        case 3:
          gemmini_extended_mvin(custom3, A_dram_addr, A_sp_addr, cols, rows);
          break;
      }
    }
  }

  for (size_t i = 0; i < I; i++) {
    for (size_t j = 0; j < J; j++) {
      const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

      for (size_t k = 0; k < K; k++) {

        const uint32_t A_sp_addr = A_sp_addr_start + (i*K + k)*DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k*J + j)*DIM;

        uint32_t out_sp_addr = k == K-1 ? C_sp_addr : GARBAGE_ADDR;

        // If we're not using a bias, then we want to overwrite what's in the
        // accumulator, rather than writing over it
        int no_bias_new_matrix = no_bias && D != NULL && k == K-1;
        if (no_bias_new_matrix) {
          out_sp_addr &= ~(1 << (ADDR_LEN-2));
        }

        const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
        const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
        const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        switch(custom_num) {
          case 0:
            gemmini_extended_preload(custom0, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
            break;
          case 1:
            gemmini_extended_preload(custom1, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
            break;
          case 2:
            gemmini_extended_preload(custom2, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
            break;
          case 3:
            gemmini_extended_preload(custom3, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
            break;
        }

        if (k == 0) { // First iteration
          switch(custom_num) {
            case 0:
              gemmini_extended_compute_preloaded(custom0, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
              break;
            case 1:
              gemmini_extended_compute_preloaded(custom1, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
              break;
            case 2:
              gemmini_extended_compute_preloaded(custom2, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
              break;
            case 3:
              gemmini_extended_compute_preloaded(custom3, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
              break;
          }
        } else { // All other iterations
          switch(custom_num) {
            case 0:
              gemmini_extended_compute_accumulated(custom0, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
              break;
            case 1:
              gemmini_extended_compute_accumulated(custom1, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
              break;
            case 2:
              gemmini_extended_compute_accumulated(custom2, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
              break;
            case 3:
              gemmini_extended_compute_accumulated(custom3, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
              break;
          }
        }
      }
    }
  }

  // Move-out C
  if (C != NULL) {
    const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

    for (size_t i = 0; i < I; i++) {
      for (size_t j = 0; j < J; j++) {
        void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + j)*DIM*sizeof_C;
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;

        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);
        
        switch(custom_num) {
          case 0:
            gemmini_extended_mvout(custom0, C_dram_addr, C_sp_addr, C_cols, C_rows);
            break;
          case 1:
            gemmini_extended_mvout(custom1, C_dram_addr, C_sp_addr, C_cols, C_rows);
            break;
          case 2:
            gemmini_extended_mvout(custom2, C_dram_addr, C_sp_addr, C_cols, C_rows);
            break;
          case 3:
            gemmini_extended_mvout(custom3, C_dram_addr, C_sp_addr, C_cols, C_rows);
            break;
        }
      }
    }
  }
}


// made
static void multi_sp_tiled_matmul_os(int custom_num, const size_t added_gemmini_num,
                                     size_t i0, const size_t I0, const size_t divided_dim_I, const size_t added_divided_dim_I,
                                     const elem_t *A, const elem_t *B, const void *D, void *C,
                                     scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                     size_t I, size_t added_I, size_t J, size_t K, size_t pad_I, size_t added_pad_I, size_t pad_J, size_t pad_K,
                                     size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
                                     bool a_transpose, bool b_transpose,
                                     bool full_C, bool low_D,
                                     bool no_bias, bool repeating_bias,
                                     int act)
{

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN - 1);
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) | (full_C << (ADDR_LEN - 3));

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks = J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC;

  // Move-in D
  if (D != NULL && !no_bias)
  {
    const size_t D_stride = repeating_bias ? 0 : D_row_stride * sizeof(acc_t);
    switch (custom_num)
    {
    case 0:
      gemmini_extended_config_ld(custom0, D_stride, D_scale_factor);
      break;
    case 1:
      gemmini_extended_config_ld(custom1, D_stride, D_scale_factor);
      break;
    case 2:
      gemmini_extended_config_ld(custom2, D_stride, D_scale_factor);
      break;
    case 3:
      gemmini_extended_config_ld(custom3, D_stride, D_scale_factor);
      break;
    }

    for (size_t i = 0; i < I; i++)
    {
      for (size_t j = 0; j < J; j += D_blocks)
      {
        const size_t bias_row = repeating_bias ? 0 : i;
        const acc_t *const D_dram_addr = (acc_t *)D + (bias_row * D_row_stride + j) * DIM;

        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i * J + j) * DIM;

        const size_t blocks = j + D_blocks <= J ? D_blocks : J - j;

        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        const size_t rows = DIM - (i == I - 1 ? pad_I : 0);
        switch (custom_num)
        {
        case 0:
          gemmini_extended_mvin(custom0, D_dram_addr, D_sp_addr_acc, cols, rows);
          break;
        case 1:
          gemmini_extended_mvin(custom1, D_dram_addr, D_sp_addr_acc, cols, rows);
          break;
        case 2:
          gemmini_extended_mvin(custom2, D_dram_addr, D_sp_addr_acc, cols, rows);
          break;
        case 3:
          gemmini_extended_mvin(custom3, D_dram_addr, D_sp_addr_acc, cols, rows);
          break;
        }
      }
    }
  }

  // Move-in B
  switch (custom_num)
  {
  case 0:
    gemmini_extended_config_ld(custom0, B_row_stride * sizeof(elem_t), B_scale_factor);
    break;
  case 1:
    gemmini_extended_config_ld(custom1, B_row_stride * sizeof(elem_t), B_scale_factor);
    break;
  case 2:
    gemmini_extended_config_ld(custom2, B_row_stride * sizeof(elem_t), B_scale_factor);
    break;
  case 3:
    gemmini_extended_config_ld(custom3, B_row_stride * sizeof(elem_t), B_scale_factor);
    break;
  }
  for (size_t j = 0; j < J; j += B_blocks)
  {
    for (size_t k = 0; k < K; k++)
    {
      const elem_t *const B_dram_addr = B + (k * B_row_stride + j) * DIM;
      const uint32_t B_sp_addr = B_sp_addr_start + (k * J + j) * DIM;
      const size_t blocks = j + B_blocks <= J ? B_blocks : J - j;
      const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
      const size_t rows = DIM - (k == K - 1 ? pad_K : 0);
      switch (custom_num)
      {
      case 0:
        gemmini_extended_mvin(custom0, B_dram_addr, B_sp_addr, cols, rows);
        break;
      case 1:
        gemmini_extended_mvin(custom1, B_dram_addr, B_sp_addr, cols, rows);
        break;
      case 2:
        gemmini_extended_mvin(custom2, B_dram_addr, B_sp_addr, cols, rows);
        break;
      case 3:
        gemmini_extended_mvin(custom3, B_dram_addr, B_sp_addr, cols, rows);
        break;
      }
    }
  }

  // Move-in A
  switch (custom_num)
  {
  case 0:
    gemmini_extended_config_ld(custom0, A_row_stride * sizeof(elem_t), A_scale_factor);
    break;
  case 1:
    gemmini_extended_config_ld(custom1, A_row_stride * sizeof(elem_t), A_scale_factor);
    break;
  case 2:
    gemmini_extended_config_ld(custom2, A_row_stride * sizeof(elem_t), A_scale_factor);
    break;
  case 3:
    gemmini_extended_config_ld(custom3, A_row_stride * sizeof(elem_t), A_scale_factor);
    break;
  }
  for (size_t i = 0; i < I; i++)
  {
    for (size_t k = 0; k < K; k += A_blocks)
    {
      const elem_t *const A_dram_addr = A + (i * A_row_stride + k) * DIM;
      const uint32_t A_sp_addr = A_sp_addr_start + (i * K + k) * DIM;
      const size_t blocks = k + A_blocks <= K ? A_blocks : K - k;
      const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
      const size_t rows = DIM - (i == I - 1 ? pad_I : 0);
      switch (custom_num)
      {
      case 0:
        gemmini_extended_mvin(custom0, A_dram_addr, A_sp_addr, cols, rows);
        break;
      case 1:
        gemmini_extended_mvin(custom1, A_dram_addr, A_sp_addr, cols, rows);
        break;
      case 2:
        gemmini_extended_mvin(custom2, A_dram_addr, A_sp_addr, cols, rows);
        break;
      case 3:
        gemmini_extended_mvin(custom3, A_dram_addr, A_sp_addr, cols, rows);
        break;
      }
    }
  }

  for (size_t i = 0; i < I; i++)
  {
    for (size_t j = 0; j < J; j++)
    {
      const uint32_t C_sp_addr = C_sp_addr_start + (i * J + j) * DIM;

      for (size_t k = 0; k < K; k++)
      {

        const uint32_t A_sp_addr = A_sp_addr_start + (i * K + k) * DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k * J + j) * DIM;

        uint32_t out_sp_addr = k == K - 1 ? C_sp_addr : GARBAGE_ADDR;

        // If we're not using a bias, then we want to overwrite what's in the
        // accumulator, rather than writing over it
        int no_bias_new_matrix = no_bias && D != NULL && k == K - 1;
        if (no_bias_new_matrix)
        {
          out_sp_addr &= ~(1 << (ADDR_LEN - 2));
        }

        const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
        const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
        const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        switch (custom_num)
        {
        case 0:
          gemmini_extended_preload(custom0, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
          break;
        case 1:
          gemmini_extended_preload(custom1, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
          break;
        case 2:
          gemmini_extended_preload(custom2, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
          break;
        case 3:
          gemmini_extended_preload(custom3, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
          break;
        }

        if (k == 0)
        { // First iteration
          switch (custom_num)
          {
          case 0:
            gemmini_extended_compute_preloaded(custom0, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 1:
            gemmini_extended_compute_preloaded(custom1, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 2:
            gemmini_extended_compute_preloaded(custom2, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 3:
            gemmini_extended_compute_preloaded(custom3, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          }
        }
        else
        { // All other iterations
          switch (custom_num)
          {
          case 0:
            gemmini_extended_compute_accumulated(custom0, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 1:
            gemmini_extended_compute_accumulated(custom1, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 2:
            gemmini_extended_compute_accumulated(custom2, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 3:
            gemmini_extended_compute_accumulated(custom3, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          }
        }
      }
    }
  }

  // Move-out C
  if (C != NULL)
  {
    const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

    for (size_t i = 0; i < I; i++)
    {
      for (size_t j = 0; j < J; j++)
      {
        void *const C_dram_addr = (int8_t *)C + (i * C_row_stride + j) * DIM * sizeof_C;
        const uint32_t C_sp_addr = C_sp_addr_start + (i * J + j) * DIM;

        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        switch (custom_num)
        {
        case 0:
          gemmini_extended_mvout(custom0, C_dram_addr, C_sp_addr, C_cols, C_rows);
          break;
        case 1:
          gemmini_extended_mvout(custom1, C_dram_addr, C_sp_addr, C_cols, C_rows);
          break;
        case 2:
          gemmini_extended_mvout(custom2, C_dram_addr, C_sp_addr, C_cols, C_rows);
          break;
        case 3:
          gemmini_extended_mvout(custom3, C_dram_addr, C_sp_addr, C_cols, C_rows);
          break;
        }
      }
    }
  }
}

// made
static void shared_multi_sp_tiled_matmul_os(int custom_num, int group_list, int group_id,
                                            size_t sp_addr_start, size_t acc_addr_start,
                                            const elem_t *A, const elem_t *B, const void *D, void *C,
                                            scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                            size_t ex_I, size_t mv_K, size_t mv_pad_K, size_t laddrI_offset, size_t laddrK_offset,
                                            size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
                                            size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
                                            bool a_transpose, bool b_transpose,
                                            bool full_C, bool low_D,
                                            bool no_bias, bool repeating_bias,
                                            int act)
{

  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN - 1);
  const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) | (full_C << (ADDR_LEN - 3));

  const int A_blocks = K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN;
  const int B_blocks = J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN;
  const int D_blocks = J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC;

  // Move-in D
  if (D != NULL && !no_bias)
  {
    const size_t D_stride = repeating_bias ? 0 : D_row_stride * sizeof(acc_t);
    switch (custom_num)
    {
    case 0:
      gemmini_extended_config_ld(custom0, D_stride, D_scale_factor);
      break;
    case 1:
      gemmini_extended_config_ld(custom1, D_stride, D_scale_factor);
      break;
    case 2:
      gemmini_extended_config_ld(custom2, D_stride, D_scale_factor);
      break;
    case 3:
      gemmini_extended_config_ld(custom3, D_stride, D_scale_factor);
      break;
    }

    for (size_t i = 0; i < I; i++)
    {
      for (size_t j = 0; j < J; j += D_blocks)
      {
        const size_t bias_row = repeating_bias ? 0 : i;
        const acc_t *const D_dram_addr = (acc_t *)D + (bias_row * D_row_stride + j) * DIM;

        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i * J + j) * DIM;

        const size_t blocks = j + D_blocks <= J ? D_blocks : J - j;

        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        const size_t rows = DIM - (i == I - 1 ? pad_I : 0);
        switch (custom_num)
        {
        case 0:
          gemmini_extended_mvin(custom0, D_dram_addr, D_sp_addr_acc, cols, rows);
          break;
        case 1:
          gemmini_extended_mvin(custom1, D_dram_addr, D_sp_addr_acc, cols, rows);
          break;
        case 2:
          gemmini_extended_mvin(custom2, D_dram_addr, D_sp_addr_acc, cols, rows);
          break;
        case 3:
          gemmini_extended_mvin(custom3, D_dram_addr, D_sp_addr_acc, cols, rows);
          break;
        }
      }
    }
  }

  // Move-in B
  switch (custom_num)
  {
  case 0:
    gemmini_extended_config_ld(custom0, B_row_stride * sizeof(elem_t), B_scale_factor);
    break;
  case 1:
    gemmini_extended_config_ld(custom1, B_row_stride * sizeof(elem_t), B_scale_factor);
    break;
  case 2:
    gemmini_extended_config_ld(custom2, B_row_stride * sizeof(elem_t), B_scale_factor);
    break;
  case 3:
    gemmini_extended_config_ld(custom3, B_row_stride * sizeof(elem_t), B_scale_factor);
    break;
  }
  for (size_t j = 0; j < J; j += B_blocks)
  {
    for (size_t k = 0; k < K; k++)
    {
      const elem_t *const B_dram_addr = B + (k * B_row_stride + j) * DIM;
      const uint32_t B_sp_addr = B_sp_addr_start + (k * J + j) * DIM;
      const size_t blocks = j + B_blocks <= J ? B_blocks : J - j;
      const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
      const size_t rows = DIM - (k == K - 1 ? pad_K : 0);
      switch (custom_num)
      {
      case 0:
        gemmini_extended_mvin(custom0, B_dram_addr, B_sp_addr, cols, rows);
        break;
      case 1:
        gemmini_extended_mvin(custom1, B_dram_addr, B_sp_addr, cols, rows);
        break;
      case 2:
        gemmini_extended_mvin(custom2, B_dram_addr, B_sp_addr, cols, rows);
        break;
      case 3:
        gemmini_extended_mvin(custom3, B_dram_addr, B_sp_addr, cols, rows);
        break;
      }
    }
  }

  // Move-in A
  switch (custom_num)
  {
  case 0:
    gemmini_extended_config_ld(custom0, A_row_stride * sizeof(elem_t), A_scale_factor);
    break;
  case 1:
    gemmini_extended_config_ld(custom1, A_row_stride * sizeof(elem_t), A_scale_factor);
    break;
  case 2:
    gemmini_extended_config_ld(custom2, A_row_stride * sizeof(elem_t), A_scale_factor);
    break;
  case 3:
    gemmini_extended_config_ld(custom3, A_row_stride * sizeof(elem_t), A_scale_factor);
    break;
  }
  for (size_t i = 0; i < I; i++)
  {
    for (size_t k = 0; k < K; k += A_blocks)
    {
      const elem_t *const A_dram_addr = A + (i * A_row_stride + k) * DIM;
      const uint32_t A_sp_addr = A_sp_addr_start + (i * K + k) * DIM;
      const size_t blocks = k + A_blocks <= K ? A_blocks : K - k;
      const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
      const size_t rows = DIM - (i == I - 1 ? pad_I : 0);
      switch (custom_num)
      {
      case 0:
        gemmini_extended_mvin(custom0, A_dram_addr, A_sp_addr, cols, rows);
        break;
      case 1:
        gemmini_extended_mvin(custom1, A_dram_addr, A_sp_addr, cols, rows);
        break;
      case 2:
        gemmini_extended_mvin(custom2, A_dram_addr, A_sp_addr, cols, rows);
        break;
      case 3:
        gemmini_extended_mvin(custom3, A_dram_addr, A_sp_addr, cols, rows);
        break;
      }
    }
  }

  for (size_t i = 0; i < I; i++)
  {
    for (size_t j = 0; j < J; j++)
    {
      const uint32_t C_sp_addr = C_sp_addr_start + (i * J + j) * DIM;

      for (size_t k = 0; k < K; k++)
      {

        const uint32_t A_sp_addr = A_sp_addr_start + (i * K + k) * DIM;
        const uint32_t B_sp_addr = B_sp_addr_start + (k * J + j) * DIM;

        uint32_t out_sp_addr = k == K - 1 ? C_sp_addr : GARBAGE_ADDR;

        // If we're not using a bias, then we want to overwrite what's in the
        // accumulator, rather than writing over it
        int no_bias_new_matrix = no_bias && D != NULL && k == K - 1;
        if (no_bias_new_matrix)
        {
          out_sp_addr &= ~(1 << (ADDR_LEN - 2));
        }

        const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
        const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
        const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        switch (custom_num)
        {
        case 0:
          gemmini_extended_preload(custom0, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
          break;
        case 1:
          gemmini_extended_preload(custom1, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
          break;
        case 2:
          gemmini_extended_preload(custom2, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
          break;
        case 3:
          gemmini_extended_preload(custom3, GARBAGE_ADDR, out_sp_addr, DIM, DIM, C_cols, C_rows);
          break;
        }

        if (k == 0)
        { // First iteration
          switch (custom_num)
          {
          case 0:
            gemmini_extended_compute_preloaded(custom0, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 1:
            gemmini_extended_compute_preloaded(custom1, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 2:
            gemmini_extended_compute_preloaded(custom2, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 3:
            gemmini_extended_compute_preloaded(custom3, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          }
        }
        else
        { // All other iterations
          switch (custom_num)
          {
          case 0:
            gemmini_extended_compute_accumulated(custom0, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 1:
            gemmini_extended_compute_accumulated(custom1, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 2:
            gemmini_extended_compute_accumulated(custom2, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          case 3:
            gemmini_extended_compute_accumulated(custom3, A_sp_addr, B_sp_addr, A_cols, A_rows, B_cols, B_rows);
            break;
          }
        }
      }
    }
  }

  // Move-out C
  if (C != NULL)
  {
    const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

    for (size_t i = 0; i < I; i++)
    {
      for (size_t j = 0; j < J; j++)
      {
        void *const C_dram_addr = (int8_t *)C + (i * C_row_stride + j) * DIM * sizeof_C;
        const uint32_t C_sp_addr = C_sp_addr_start + (i * J + j) * DIM;

        const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
        const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);

        switch (custom_num)
        {
        case 0:
          gemmini_extended_mvout(custom0, C_dram_addr, C_sp_addr, C_cols, C_rows);
          break;
        case 1:
          gemmini_extended_mvout(custom1, C_dram_addr, C_sp_addr, C_cols, C_rows);
          break;
        case 2:
          gemmini_extended_mvout(custom2, C_dram_addr, C_sp_addr, C_cols, C_rows);
          break;
        case 3:
          gemmini_extended_mvout(custom3, C_dram_addr, C_sp_addr, C_cols, C_rows);
          break;
        }
      }
    }
  }
}

static void sp_tiled_matmul_ws(int custom_num, const elem_t * A, const elem_t * B,
        const void * D, void * C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
        size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        bool no_bias, bool repeating_bias,
        int act) {
/*
  const uint32_t A_sp_addr_start = 0;
  const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
  const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
  const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));
  const int A_blocks = a_transpose ? (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN) :
    (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN);
  const int B_blocks = b_transpose ? (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
    (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
  const int C_blocks = full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
  // Move-in D
  if (D != NULL && !no_bias) {
    for (size_t i = 0; i < I; i++) {
      const size_t rows = DIM - (i == I-1 ? pad_I : 0);
      for (size_t j = 0; j < J; j += D_blocks) {
        const size_t bias_row = repeating_bias ? 0 : i;
        const void * const D_dram_addr = (int8_t *)D + (bias_row * D_row_stride + j)*DIM*sizeof_D;
        const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
        size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
        const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
        gemmini_extended_mvin3(D_dram_addr, D_sp_addr_acc, cols, rows);
      }
    }
  }
  for (size_t k = 0; k < K; k++) {
    for (size_t j = 0; j < J; j++) {
      for (size_t i = 0; i < I; i++) {
        const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
          (A_sp_addr_start + (i*K + k)*DIM);
        const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
          (B_sp_addr_start + (k*J + j)*DIM);
        const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;
        // Mvin A
        if (a_transpose) {
          if (j == 0 && i % A_blocks == 0) {
            const elem_t * const A_dram_addr = A + (k*A_row_stride + i)*DIM;
            const size_t blocks = i + A_blocks <= I ? A_blocks : I-i;
            const size_t cols = blocks * DIM - (i + blocks >= I ? pad_I : 0);
            const size_t rows = DIM - (k == K-1 ? pad_K : 0);
            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
          }
        } else {
          if (j == 0 && k % A_blocks == 0) {
            const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
            const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
            const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
            const size_t rows = DIM - (i == I-1 ? pad_I : 0);
            gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
          }
        }
        // Mvin B
        if (b_transpose) {
          if (i == 0 && k % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (j*B_row_stride + k)*DIM;
            const size_t blocks = k + B_blocks <= K ? B_blocks : K-k;
            const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
            const size_t rows = DIM - (j == J-1 ? pad_J : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
          }
        } else {
          if (i == 0 && j % B_blocks == 0) {
            const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
            const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
            const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
            const size_t rows = DIM - (k == K-1 ? pad_K : 0);
            gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
          }
        }
        // Compute
        {
          uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
          uint32_t out_sp_addr = C_sp_addr;
          // If we're not using a bias, then we want to overwrite what's in the
          // accumulator, rather than writing over it
          int no_bias_new_matrix = no_bias && D != NULL && k == 0;
          if (no_bias_new_matrix) {
            out_sp_addr &= ~(1 << (ADDR_LEN-2));
          }
          const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
          const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
          const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
          const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
          const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);
          gemmini_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
          if (i == 0) { // First iteration
            gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
          } else { // All other iterations
            gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
          }
        }
        if (C != NULL && k == K-1) {
          // Move-out C (if not normalizing)
          if (((act != LAYERNORM) && (act != SOFTMAX)) && (j == J-1 || j % C_blocks == C_blocks-1)) {
            const size_t rounded_j = (j / C_blocks) * C_blocks;
            const uint32_t rounded_C_sp_addr = C_sp_addr_start + (i*J + rounded_j)*DIM;
            void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + rounded_j)*DIM*sizeof_C;
            const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
            const size_t cols = blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
            const size_t rows = DIM - (i == I - 1 ? pad_I : 0);
            gemmini_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
          }
          // Move-out C (if normalizing)
          if (act == LAYERNORM && j == J - 1) {
            uint32_t norm_cmds[][2] = {{1,2},{3,4},{0,0}};
            const int norm_cmds_size = sizeof(norm_cmds) / sizeof(norm_cmds[0]);
            const size_t rows = DIM - (i == I-1 ? pad_I : 0);
            for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
              const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                NORM_STAT_IDS : rows - row;
              for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                  gemmini_config_norm(0, 0, 0, 0, stat_id, 0, 0);
                  const size_t r = row + stat_id;
                  for (size_t jj = 0; jj < J; jj += C_blocks) {
                    uint32_t norm_C_sp_addr = C_sp_addr_start + (i*J + jj)*DIM + r;
                    if (jj + C_blocks >= J) {
                      norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                    } else {
                      norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                    }
                    void * const C_dram_addr = (int8_t*)C +
                      (i*C_row_stride + jj) * DIM * sizeof_C +
                      r * C_row_stride * sizeof_C;
                    const size_t blocks = jj + C_blocks <= J ? C_blocks : J-jj;
                    const size_t cols = blocks * DIM - (jj + blocks >= J ? pad_J : 0);
                    gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                  }
                }
              }
            }
          } else if (act == SOFTMAX && j == J - 1) {
            uint32_t norm_cmds[][2] = {{5,5},{6,7},{0,0}};
            const int norm_cmds_size = sizeof(norm_cmds) / sizeof(norm_cmds[0]);
            const size_t rows = DIM - (i == I-1 ? pad_I : 0);
            for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
              const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                NORM_STAT_IDS : rows - row;
              for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                  // set stat id only
                  gemmini_config_norm(0, 0, 1, 0, stat_id, 0, 0);
                  const size_t r = row + stat_id;
                  for (size_t jj = 0; jj < J; jj += C_blocks) {
                    uint32_t norm_C_sp_addr = C_sp_addr_start + (i*J + jj)*DIM + r;
                    if (jj + C_blocks >= J) {
                      norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                    } else {
                      norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                    }
                    void * const C_dram_addr = (int8_t*)C +
                      (i*C_row_stride + jj) * DIM * sizeof_C +
                      r * C_row_stride * sizeof_C;
                    const size_t blocks = jj + C_blocks <= J ? C_blocks : J-jj;
                    const size_t cols = blocks * DIM - (jj + blocks >= J ? pad_J : 0);
                    gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
*/

  // Combined loop
  switch(custom_num) {
    case 0:
      gemmini_loop_ws(custom0, I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, !no_bias || D == NULL,
        act);
      break;
    case 1:
      gemmini_loop_ws(custom1, I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, !no_bias || D == NULL,
        act);
      break;
    case 2:
      gemmini_loop_ws(custom2, I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, !no_bias || D == NULL,
        act);
      break;
    case 3:
      gemmini_loop_ws(custom3, I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
        A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
        a_transpose, b_transpose,
        full_C, low_D, !no_bias || D == NULL,
        act);
      break;
  }
}

// made
static void multi_sp_tiled_matmul_ws(int gemmini_num, const size_t added_gemmini_num,
                                     size_t i0, const size_t I0, const size_t divided_dim_I, const size_t added_divided_dim_I,
                                     const elem_t *A, const elem_t *B,
                                     const void *D, void *C,
                                     scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                     size_t I, size_t added_I, size_t J, size_t K, size_t pad_I, size_t added_pad_I, size_t pad_J, size_t pad_K,
                                     size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
                                     bool a_transpose, bool b_transpose,
                                     bool full_C, bool low_D,
                                     bool no_bias, bool repeating_bias,
                                     int act)
{
  /*
    const uint32_t A_sp_addr_start = 0;
    const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
    const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));
    const int A_blocks = a_transpose ? (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN) :
      (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN);
    const int B_blocks = b_transpose ? (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN) :
      (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
    const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
      (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
    const int C_blocks = full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
    const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
    const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
    // Move-in D
    if (D != NULL && !no_bias) {
      for (size_t i = 0; i < I; i++) {
        const size_t rows = DIM - (i == I-1 ? pad_I : 0);
        for (size_t j = 0; j < J; j += D_blocks) {
          const size_t bias_row = repeating_bias ? 0 : i;
          const void * const D_dram_addr = (int8_t *)D + (bias_row * D_row_stride + j)*DIM*sizeof_D;
          const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
          size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
          const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
          gemmini_extended_mvin3(D_dram_addr, D_sp_addr_acc, cols, rows);
        }
      }
    }
    for (size_t k = 0; k < K; k++) {
      for (size_t j = 0; j < J; j++) {
        for (size_t i = 0; i < I; i++) {
          const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
            (A_sp_addr_start + (i*K + k)*DIM);
          const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
            (B_sp_addr_start + (k*J + j)*DIM);
          const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;
          // Mvin A
          if (a_transpose) {
            if (j == 0 && i % A_blocks == 0) {
              const elem_t * const A_dram_addr = A + (k*A_row_stride + i)*DIM;
              const size_t blocks = i + A_blocks <= I ? A_blocks : I-i;
              const size_t cols = blocks * DIM - (i + blocks >= I ? pad_I : 0);
              const size_t rows = DIM - (k == K-1 ? pad_K : 0);
              gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
            }
          } else {
            if (j == 0 && k % A_blocks == 0) {
              const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
              const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
              const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
              const size_t rows = DIM - (i == I-1 ? pad_I : 0);
              gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
            }
          }
          // Mvin B
          if (b_transpose) {
            if (i == 0 && k % B_blocks == 0) {
              const elem_t * const B_dram_addr = B + (j*B_row_stride + k)*DIM;
              const size_t blocks = k + B_blocks <= K ? B_blocks : K-k;
              const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
              const size_t rows = DIM - (j == J-1 ? pad_J : 0);
              gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
            }
          } else {
            if (i == 0 && j % B_blocks == 0) {
              const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
              const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
              const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
              const size_t rows = DIM - (k == K-1 ? pad_K : 0);
              gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
            }
          }
          // Compute
          {
            uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
            uint32_t out_sp_addr = C_sp_addr;
            // If we're not using a bias, then we want to overwrite what's in the
            // accumulator, rather than writing over it
            int no_bias_new_matrix = no_bias && D != NULL && k == 0;
            if (no_bias_new_matrix) {
              out_sp_addr &= ~(1 << (ADDR_LEN-2));
            }
            const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
            const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
            const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
            const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
            const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
            const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);
            gemmini_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
            if (i == 0) { // First iteration
              gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            } else { // All other iterations
              gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            }
          }
          if (C != NULL && k == K-1) {
            // Move-out C (if not normalizing)
            if (((act != LAYERNORM) && (act != SOFTMAX)) && (j == J-1 || j % C_blocks == C_blocks-1)) {
              const size_t rounded_j = (j / C_blocks) * C_blocks;
              const uint32_t rounded_C_sp_addr = C_sp_addr_start + (i*J + rounded_j)*DIM;
              void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + rounded_j)*DIM*sizeof_C;
              const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
              const size_t cols = blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
              const size_t rows = DIM - (i == I - 1 ? pad_I : 0);
              gemmini_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
            }
            // Move-out C (if normalizing)
            if (act == LAYERNORM && j == J - 1) {
              uint32_t norm_cmds[][2] = {{1,2},{3,4},{0,0}};
              const int norm_cmds_size = sizeof(norm_cmds) / sizeof(norm_cmds[0]);
              const size_t rows = DIM - (i == I-1 ? pad_I : 0);
              for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
                const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                  NORM_STAT_IDS : rows - row;
                for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                  for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                    gemmini_config_norm(0, 0, 0, 0, stat_id, 0, 0);
                    const size_t r = row + stat_id;
                    for (size_t jj = 0; jj < J; jj += C_blocks) {
                      uint32_t norm_C_sp_addr = C_sp_addr_start + (i*J + jj)*DIM + r;
                      if (jj + C_blocks >= J) {
                        norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                      } else {
                        norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                      }
                      void * const C_dram_addr = (int8_t*)C +
                        (i*C_row_stride + jj) * DIM * sizeof_C +
                        r * C_row_stride * sizeof_C;
                      const size_t blocks = jj + C_blocks <= J ? C_blocks : J-jj;
                      const size_t cols = blocks * DIM - (jj + blocks >= J ? pad_J : 0);
                      gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                    }
                  }
                }
              }
            } else if (act == SOFTMAX && j == J - 1) {
              uint32_t norm_cmds[][2] = {{5,5},{6,7},{0,0}};
              const int norm_cmds_size = sizeof(norm_cmds) / sizeof(norm_cmds[0]);
              const size_t rows = DIM - (i == I-1 ? pad_I : 0);
              for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
                const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                  NORM_STAT_IDS : rows - row;
                for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                  for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                    // set stat id only
                    gemmini_config_norm(0, 0, 1, 0, stat_id, 0, 0);
                    const size_t r = row + stat_id;
                    for (size_t jj = 0; jj < J; jj += C_blocks) {
                      uint32_t norm_C_sp_addr = C_sp_addr_start + (i*J + jj)*DIM + r;
                      if (jj + C_blocks >= J) {
                        norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                      } else {
                        norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                      }
                      void * const C_dram_addr = (int8_t*)C +
                        (i*C_row_stride + jj) * DIM * sizeof_C +
                        r * C_row_stride * sizeof_C;
                      const size_t blocks = jj + C_blocks <= J ? C_blocks : J-jj;
                      const size_t cols = blocks * DIM - (jj + blocks >= J ? pad_J : 0);
                      gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  */

  // Combined loop
  size_t custom0_I = (added_gemmini_num > 0) ? added_I : I;
  size_t custom0_pad_I = (added_gemmini_num > 0) ? added_pad_I : pad_I;
  size_t custom1_I = (added_gemmini_num > 1) ? added_I : I;
  size_t custom1_pad_I = (added_gemmini_num > 1) ? added_pad_I : pad_I;
  size_t custom2_I = (added_gemmini_num > 2) ? added_I : I;
  size_t custom2_pad_I = (added_gemmini_num > 2) ? added_pad_I : pad_I;
  size_t custom3_I = (added_gemmini_num > 3) ? added_I : I;
  size_t custom3_pad_I = (added_gemmini_num > 3) ? added_pad_I : pad_I;
  const elem_t *custom0_A_local = A;
  const elem_t *custom1_A_local = custom0_A_local + A_row_stride * ((added_gemmini_num > 0) ? added_divided_dim_I : divided_dim_I);
  const elem_t *custom2_A_local = custom1_A_local + A_row_stride * ((added_gemmini_num > 1) ? added_divided_dim_I : divided_dim_I);
  const elem_t *custom3_A_local = custom2_A_local + A_row_stride * ((added_gemmini_num > 2) ? added_divided_dim_I : divided_dim_I);
  void *custom0_C_local = C;
  void *custom1_C_local = C;
  void *custom2_C_local = C;
  void *custom3_C_local = C;
  if (C != NULL) {
    custom1_C_local = (void *)((int8_t *)custom0_C_local + C_row_stride * ((added_gemmini_num > 0) ? added_divided_dim_I : divided_dim_I));
    custom2_C_local = (void *)((int8_t *)custom1_C_local + C_row_stride * ((added_gemmini_num > 1) ? added_divided_dim_I : divided_dim_I));
    custom3_C_local = (void *)((int8_t *)custom2_C_local + C_row_stride * ((added_gemmini_num > 2) ? added_divided_dim_I : divided_dim_I));
  }

  switch (gemmini_num)
  {
    case 1:
      if (i0 < I0) {
        ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom0_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom0_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom0, custom0_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom0, no_bias ? NULL : D, custom0_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom0, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom0, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
      }
      break;
    case 2:
      if (i0 < I0)
      {
        ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom0_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom0_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom1_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom1_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom0, custom0_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom1, custom1_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom0, no_bias ? NULL : D, custom0_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom1, no_bias ? NULL : D, custom1_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom0, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom1, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom0, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom1, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
        ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
      }
      else if (added_gemmini_num > 0) {
        ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom0_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom0_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom0, custom0_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom0, no_bias ? NULL : D, custom0_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom0, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom0, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
      }
      break;
    case 3:
      if (i0 < I0)
      {
        ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom0_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom0_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom1_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom1_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom2, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom2_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom2_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom0, custom0_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom1, custom1_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom2, custom2_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom0, no_bias ? NULL : D, custom0_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom1, no_bias ? NULL : D, custom1_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom2, no_bias ? NULL : D, custom2_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom0, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom1, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom2, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom0, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom1, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom2, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
        ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
        ROCC_INSTRUCTION_RS1_RS2(custom2, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
      }
      else
      {
        switch (added_gemmini_num) {
          case 1:
            ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom0_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom0_I), k_LOOP_WS_CONFIG_BOUNDS);
            ROCC_INSTRUCTION_RS1_RS2(custom0, custom0_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
            ROCC_INSTRUCTION_RS1_RS2(custom0, no_bias ? NULL : D, custom0_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
            ROCC_INSTRUCTION_RS1_RS2(custom0, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
            ROCC_INSTRUCTION_RS1_RS2(custom0, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
            ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
            break;
          case 2:
            ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom0_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom0_I), k_LOOP_WS_CONFIG_BOUNDS);
            ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom1_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom1_I), k_LOOP_WS_CONFIG_BOUNDS);
            ROCC_INSTRUCTION_RS1_RS2(custom0, custom0_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
            ROCC_INSTRUCTION_RS1_RS2(custom1, custom1_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
            ROCC_INSTRUCTION_RS1_RS2(custom0, no_bias ? NULL : D, custom0_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
            ROCC_INSTRUCTION_RS1_RS2(custom1, no_bias ? NULL : D, custom1_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
            ROCC_INSTRUCTION_RS1_RS2(custom0, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
            ROCC_INSTRUCTION_RS1_RS2(custom1, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
            ROCC_INSTRUCTION_RS1_RS2(custom0, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
            ROCC_INSTRUCTION_RS1_RS2(custom1, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
            ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
            ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
            break;
        }
      }
      break;
    case 4:
      if (i0 < I0)
      {
        ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom0_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom0_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom1_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom1_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom2, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom2_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom2_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom3, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom3_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom3_I), k_LOOP_WS_CONFIG_BOUNDS);
        ROCC_INSTRUCTION_RS1_RS2(custom0, custom0_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom1, custom1_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom2, custom2_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom3, custom3_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom0, no_bias ? NULL : D, custom0_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom1, no_bias ? NULL : D, custom1_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom2, no_bias ? NULL : D, custom2_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom3, no_bias ? NULL : D, custom3_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom0, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom1, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom2, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom3, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
        ROCC_INSTRUCTION_RS1_RS2(custom0, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom1, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom2, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom3, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
        ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
        ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
        ROCC_INSTRUCTION_RS1_RS2(custom2, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
        ROCC_INSTRUCTION_RS1_RS2(custom3, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
      }
      else
      {
        switch (added_gemmini_num)
        {
        case 1:
          ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom0_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom0_I), k_LOOP_WS_CONFIG_BOUNDS);
          ROCC_INSTRUCTION_RS1_RS2(custom0, custom0_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom0, no_bias ? NULL : D, custom0_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom0, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom0, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
          break;
        case 2:
          ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom0_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom0_I), k_LOOP_WS_CONFIG_BOUNDS);
          ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom1_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom1_I), k_LOOP_WS_CONFIG_BOUNDS);
          ROCC_INSTRUCTION_RS1_RS2(custom0, custom0_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom1, custom1_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom0, no_bias ? NULL : D, custom0_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom1, no_bias ? NULL : D, custom1_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom0, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom1, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom0, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom1, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
          ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
          break;
        case 3:
          ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom0_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom0_I), k_LOOP_WS_CONFIG_BOUNDS);
          ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom1_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom1_I), k_LOOP_WS_CONFIG_BOUNDS);
          ROCC_INSTRUCTION_RS1_RS2(custom2, ((uint64_t)(pad_K) << 32) | ((uint64_t)(pad_J) << 16) | (uint64_t)(custom2_pad_I), ((uint64_t)(K) << 32) | ((uint64_t)(J) << 16) | (uint64_t)(custom2_I), k_LOOP_WS_CONFIG_BOUNDS);
          ROCC_INSTRUCTION_RS1_RS2(custom0, custom0_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom1, custom1_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom2, custom2_A_local, B, k_LOOP_WS_CONFIG_ADDRS_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom0, no_bias ? NULL : D, custom0_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom1, no_bias ? NULL : D, custom1_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom2, no_bias ? NULL : D, custom2_C_local, k_LOOP_WS_CONFIG_ADDRS_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom0, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom1, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom2, A_row_stride, B_row_stride, k_LOOP_WS_CONFIG_STRIDES_AB);
          ROCC_INSTRUCTION_RS1_RS2(custom0, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom1, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom2, repeating_bias ? 0 : D_row_stride, C_row_stride, k_LOOP_WS_CONFIG_STRIDES_DC);
          ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
          ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
          ROCC_INSTRUCTION_RS1_RS2(custom2, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || D == NULL), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
          break;
        }
      }
      break;
  }
}

// made
static void shared_multi_sp_tiled_matmul_ws(int custom_num, int group_list, int group_id,
                                            size_t sp_addr_start, size_t sp_addr_end, size_t acc_addr_start,
                                            const elem_t *A, const elem_t *B, const void *D, void *C,
                                            scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                            size_t ex_I, size_t mv_K, size_t mv_pad_K, size_t laddrI_offset, size_t laddrK_offset,
                                            size_t I, size_t J, size_t K, size_t pad_I, size_t pad_J, size_t pad_K,
                                            size_t A_row_stride, size_t B_row_stride, size_t D_row_stride, size_t C_row_stride,
                                            bool a_transpose, bool b_transpose,
                                            bool full_C, bool low_D,
                                            bool no_bias, bool repeating_bias,
                                            int act)
{
  /*
    const uint32_t A_sp_addr_start = 0;
    const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - K * J * DIM;
    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
    const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2) | (full_C << (ADDR_LEN-3));
    const int A_blocks = a_transpose ? (I <= MAX_BLOCK_LEN ? I : MAX_BLOCK_LEN) :
      (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN);
    const int B_blocks = b_transpose ? (K <= MAX_BLOCK_LEN ? K : MAX_BLOCK_LEN) :
      (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
    const int D_blocks = low_D ? (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN) :
      (J <= MAX_BLOCK_LEN_ACC ? J : MAX_BLOCK_LEN_ACC);
    const int C_blocks = full_C ? 1 : (J <= MAX_BLOCK_LEN ? J : MAX_BLOCK_LEN);
    const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
    const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);
    // Move-in D
    if (D != NULL && !no_bias) {
      for (size_t i = 0; i < I; i++) {
        const size_t rows = DIM - (i == I-1 ? pad_I : 0);
        for (size_t j = 0; j < J; j += D_blocks) {
          const size_t bias_row = repeating_bias ? 0 : i;
          const void * const D_dram_addr = (int8_t *)D + (bias_row * D_row_stride + j)*DIM*sizeof_D;
          const uint32_t D_sp_addr_acc = D_sp_addr_start + (i*J + j)*DIM;
          size_t blocks = j + D_blocks <= J ? D_blocks : J-j;
          const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
          gemmini_extended_mvin3(D_dram_addr, D_sp_addr_acc, cols, rows);
        }
      }
    }
    for (size_t k = 0; k < K; k++) {
      for (size_t j = 0; j < J; j++) {
        for (size_t i = 0; i < I; i++) {
          const uint32_t A_sp_addr = a_transpose ? (A_sp_addr_start + (k*I + i)*DIM) :
            (A_sp_addr_start + (i*K + k)*DIM);
          const uint32_t B_sp_addr = b_transpose ? (B_sp_addr_start + (j*K + k)*DIM) :
            (B_sp_addr_start + (k*J + j)*DIM);
          const uint32_t C_sp_addr = C_sp_addr_start + (i*J + j)*DIM;
          // Mvin A
          if (a_transpose) {
            if (j == 0 && i % A_blocks == 0) {
              const elem_t * const A_dram_addr = A + (k*A_row_stride + i)*DIM;
              const size_t blocks = i + A_blocks <= I ? A_blocks : I-i;
              const size_t cols = blocks * DIM - (i + blocks >= I ? pad_I : 0);
              const size_t rows = DIM - (k == K-1 ? pad_K : 0);
              gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
            }
          } else {
            if (j == 0 && k % A_blocks == 0) {
              const elem_t * const A_dram_addr = A + (i*A_row_stride + k)*DIM;
              const size_t blocks = k + A_blocks <= K ? A_blocks : K-k;
              const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
              const size_t rows = DIM - (i == I-1 ? pad_I : 0);
              gemmini_extended_mvin(A_dram_addr, A_sp_addr, cols, rows);
            }
          }
          // Mvin B
          if (b_transpose) {
            if (i == 0 && k % B_blocks == 0) {
              const elem_t * const B_dram_addr = B + (j*B_row_stride + k)*DIM;
              const size_t blocks = k + B_blocks <= K ? B_blocks : K-k;
              const size_t cols = blocks * DIM - (k + blocks >= K ? pad_K : 0);
              const size_t rows = DIM - (j == J-1 ? pad_J : 0);
              gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
            }
          } else {
            if (i == 0 && j % B_blocks == 0) {
              const elem_t * const B_dram_addr = B + (k*B_row_stride + j)*DIM;
              const size_t blocks = j + B_blocks <= J ? B_blocks : J-j;
              const size_t cols = blocks * DIM - (j + blocks >= J ? pad_J : 0);
              const size_t rows = DIM - (k == K-1 ? pad_K : 0);
              gemmini_extended_mvin2(B_dram_addr, B_sp_addr, cols, rows);
            }
          }
          // Compute
          {
            uint32_t pre_sp_addr = i == 0 ? B_sp_addr : GARBAGE_ADDR;
            uint32_t out_sp_addr = C_sp_addr;
            // If we're not using a bias, then we want to overwrite what's in the
            // accumulator, rather than writing over it
            int no_bias_new_matrix = no_bias && D != NULL && k == 0;
            if (no_bias_new_matrix) {
              out_sp_addr &= ~(1 << (ADDR_LEN-2));
            }
            const size_t A_cols = DIM - (k == K - 1 ? pad_K : 0);
            const size_t A_rows = DIM - (i == I - 1 ? pad_I : 0);
            const size_t B_cols = DIM - (j == J - 1 ? pad_J : 0);
            const size_t B_rows = DIM - (k == K - 1 ? pad_K : 0);
            const size_t C_cols = DIM - (j == J - 1 ? pad_J : 0);
            const size_t C_rows = DIM - (i == I - 1 ? pad_I : 0);
            gemmini_extended_preload(pre_sp_addr, out_sp_addr, B_cols, B_rows, C_cols, C_rows);
            if (i == 0) { // First iteration
              gemmini_extended_compute_preloaded(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            } else { // All other iterations
              gemmini_extended_compute_accumulated(A_sp_addr, GARBAGE_ADDR, A_cols, A_rows, DIM, DIM);
            }
          }
          if (C != NULL && k == K-1) {
            // Move-out C (if not normalizing)
            if (((act != LAYERNORM) && (act != SOFTMAX)) && (j == J-1 || j % C_blocks == C_blocks-1)) {
              const size_t rounded_j = (j / C_blocks) * C_blocks;
              const uint32_t rounded_C_sp_addr = C_sp_addr_start + (i*J + rounded_j)*DIM;
              void * const C_dram_addr = (int8_t*)C + (i*C_row_stride + rounded_j)*DIM*sizeof_C;
              const size_t blocks = rounded_j + C_blocks <= J ? C_blocks : J-rounded_j;
              const size_t cols = blocks * DIM - (rounded_j + blocks >= J ? pad_J : 0);
              const size_t rows = DIM - (i == I - 1 ? pad_I : 0);
              gemmini_extended_mvout(C_dram_addr, rounded_C_sp_addr, cols, rows);
            }
            // Move-out C (if normalizing)
            if (act == LAYERNORM && j == J - 1) {
              uint32_t norm_cmds[][2] = {{1,2},{3,4},{0,0}};
              const int norm_cmds_size = sizeof(norm_cmds) / sizeof(norm_cmds[0]);
              const size_t rows = DIM - (i == I-1 ? pad_I : 0);
              for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
                const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                  NORM_STAT_IDS : rows - row;
                for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                  for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                    gemmini_config_norm(0, 0, 0, 0, stat_id, 0, 0);
                    const size_t r = row + stat_id;
                    for (size_t jj = 0; jj < J; jj += C_blocks) {
                      uint32_t norm_C_sp_addr = C_sp_addr_start + (i*J + jj)*DIM + r;
                      if (jj + C_blocks >= J) {
                        norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                      } else {
                        norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                      }
                      void * const C_dram_addr = (int8_t*)C +
                        (i*C_row_stride + jj) * DIM * sizeof_C +
                        r * C_row_stride * sizeof_C;
                      const size_t blocks = jj + C_blocks <= J ? C_blocks : J-jj;
                      const size_t cols = blocks * DIM - (jj + blocks >= J ? pad_J : 0);
                      gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                    }
                  }
                }
              }
            } else if (act == SOFTMAX && j == J - 1) {
              uint32_t norm_cmds[][2] = {{5,5},{6,7},{0,0}};
              const int norm_cmds_size = sizeof(norm_cmds) / sizeof(norm_cmds[0]);
              const size_t rows = DIM - (i == I-1 ? pad_I : 0);
              for (size_t row = 0; row < rows; row += NORM_STAT_IDS) {
                const size_t stat_ids = rows - row > NORM_STAT_IDS ?
                  NORM_STAT_IDS : rows - row;
                for (int cmd = 0; cmd < norm_cmds_size; cmd++) {
                  for (size_t stat_id = 0; stat_id < stat_ids; stat_id++) {
                    // set stat id only
                    gemmini_config_norm(0, 0, 1, 0, stat_id, 0, 0);
                    const size_t r = row + stat_id;
                    for (size_t jj = 0; jj < J; jj += C_blocks) {
                      uint32_t norm_C_sp_addr = C_sp_addr_start + (i*J + jj)*DIM + r;
                      if (jj + C_blocks >= J) {
                        norm_C_sp_addr |= (norm_cmds[cmd][1] << 26); // Final mean/inv-std-dev calculation
                      } else {
                        norm_C_sp_addr |= (norm_cmds[cmd][0] << 26); // Accumulate sum/variance
                      }
                      void * const C_dram_addr = (int8_t*)C +
                        (i*C_row_stride + jj) * DIM * sizeof_C +
                        r * C_row_stride * sizeof_C;
                      const size_t blocks = jj + C_blocks <= J ? C_blocks : J-jj;
                      const size_t cols = blocks * DIM - (jj + blocks >= J ? pad_J : 0);
                      gemmini_extended_mvout(C_dram_addr, norm_C_sp_addr, cols, 1);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  */

  // Combined loop
  switch (custom_num)
  {
  case 0:
    shared_gemmini_loop_ws(custom0, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_I, mv_K, mv_pad_K, laddrI_offset, laddrK_offset,
                           I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
                           A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
                           a_transpose, b_transpose,
                           full_C, low_D, !no_bias || D == NULL,
                           act);
    break;
  case 1:
    shared_gemmini_loop_ws(custom1, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_I, mv_K, mv_pad_K, laddrI_offset, laddrK_offset,
                           I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
                           A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
                           a_transpose, b_transpose,
                           full_C, low_D, !no_bias || D == NULL,
                           act);
    break;
  case 2:
    shared_gemmini_loop_ws(custom2, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_I, mv_K, mv_pad_K, laddrI_offset, laddrK_offset,
                           I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
                           A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
                           a_transpose, b_transpose,
                           full_C, low_D, !no_bias || D == NULL,
                           act);
    break;
  case 3:
    shared_gemmini_loop_ws(custom3, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_I, mv_K, mv_pad_K, laddrI_offset, laddrK_offset,
                           I, J, K, pad_I, pad_J, pad_K, A, B, no_bias ? NULL : D, C,
                           A_row_stride, B_row_stride, repeating_bias ? 0 : D_row_stride, C_row_stride,
                           a_transpose, b_transpose,
                           full_C, low_D, !no_bias || D == NULL,
                           act);
    break;
  }
}

static void tiled_matmul_outer(int custom_num, size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        size_t tile_I, size_t tile_J, size_t tile_K,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool a_transpose, bool b_transpose,
        bool full_C, bool low_D,
        uint8_t weightA,
        int dataflow) {

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const size_t I0 = dim_I_padded / (tile_I*DIM) + (dim_I_padded % (tile_I*DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J*DIM) + (dim_J_padded % (tile_J*DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K*DIM) + (dim_K_padded % (tile_K*DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I*DIM) == 0 ? tile_I : (dim_I_padded/DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J*DIM) == 0 ? tile_J : (dim_J_padded/DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K*DIM) == 0 ? tile_K : (dim_K_padded/DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias) {
    D = (void*) 1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t) ;
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  switch(custom_num) {
    case 0:
      gemmini_extended_config_ex(custom0, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
      gemmini_extended_config_st(custom0, stride_C * sizeof_C, act & 3, scale);
      gemmini_extended3_config_ld(custom0, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
      gemmini_extended3_config_ld(custom0, stride_B * sizeof(elem_t), B_scale_factor, false, 1)
      gemmini_extended3_config_ld(custom0, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
      break;
    case 1:
      gemmini_extended_config_ex(custom1, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
      gemmini_extended_config_st(custom1, stride_C * sizeof_C, act & 3, scale);
      gemmini_extended3_config_ld(custom1, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
      gemmini_extended3_config_ld(custom1, stride_B * sizeof(elem_t), B_scale_factor, false, 1)
      gemmini_extended3_config_ld(custom1, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
      break;
    case 2:
      gemmini_extended_config_ex(custom2, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
      gemmini_extended_config_st(custom2, stride_C * sizeof_C, act & 3, scale);
      gemmini_extended3_config_ld(custom2, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
      gemmini_extended3_config_ld(custom2, stride_B * sizeof(elem_t), B_scale_factor, false, 1)
      gemmini_extended3_config_ld(custom2, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
      break;
    case 3:
      gemmini_extended_config_ex(custom3, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
      gemmini_extended_config_st(custom3, stride_C * sizeof_C, act & 3, scale);
      gemmini_extended3_config_ld(custom3, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
      gemmini_extended3_config_ld(custom3, stride_B * sizeof(elem_t), B_scale_factor, false, 1)
      gemmini_extended3_config_ld(custom3, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
      break;
  }

  if (act == IGELU) {
    const acc_scale_t sqrt_2 = 1.41421356237;
    const acc_scale_t S = bert_scale;
    const acc_scale_t S_erf = (-0.2888 * ((S*S) / 2));

    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / S_erf;
    switch(custom_num) {
      case 0:
        gemmini_config_norm(custom0, 0, 0, 0, 0, 0, qb, qc);
        break;
      case 1:
        gemmini_config_norm(custom1, 0, 0, 0, 0, 0, qb, qc);
        break;
      case 2:
        gemmini_config_norm(custom2, 0, 0, 0, 0, 0, qb, qc);
        break;
      case 3:
        gemmini_config_norm(custom3, 0, 0, 0, 0, 0, qb, qc);
        break;
    }
  }

  if (act == SOFTMAX) {
    const scale_t a = 0.3585;
    const scale_t b = 1.353;
    const scale_t c = 0.344;

    const acc_t qln2 = (int) (0.693147 / bert_scale);
    const acc_t qln2_inv = 65536 / qln2;
    const acc_t qb = b / bert_scale;
    const acc_t qc = c / (a*bert_scale*bert_scale);
    
    switch(custom_num) {
      case 0:
        gemmini_config_norm(custom0, qln2, 0, 0, 1, 0, qb, qc);
        gemmini_config_norm(custom0, qln2_inv, 1, 0, 1, 0, qb, qc);
        break;
      case 1:
        gemmini_config_norm(custom1, qln2, 0, 0, 1, 0, qb, qc);
        gemmini_config_norm(custom1, qln2_inv, 1, 0, 1, 0, qb, qc);
        break;
      case 2:
        gemmini_config_norm(custom2, qln2, 0, 0, 1, 0, qb, qc);
        gemmini_config_norm(custom2, qln2_inv, 1, 0, 1, 0, qb, qc);
        break;
      case 3:
        gemmini_config_norm(custom3, qln2, 0, 0, 1, 0, qb, qc);
        gemmini_config_norm(custom3, qln2_inv, 1, 0, 1, 0, qb, qc);
        break;
    }
  }

  void (*inner)(int, const elem_t *, const elem_t *, const void *, void *,
        scale_t, scale_t, scale_acc_t,
        size_t, size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t,
        bool, bool,
        bool, bool,
        bool, bool,
        int);

  if (dataflow == OUTPUT_STATIONARY) {
    inner = &sp_tiled_matmul_os;
  } else /* if (dataflow == WEIGHT_STATIONARY) */ {
    inner = &sp_tiled_matmul_ws;
  }

  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++) {

        const void * pre;
        if (k0 != 0) {
          pre = NULL;
        } else {
          size_t bias_row = repeating_bias ? 0 : i0*tile_I*DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t*)D + (bias_row * stride_D + j0 * tile_J * DIM)*sizeof_D;
        }

        void * out = k0 == K0-1 ? (int8_t*)C + (i0*tile_I*DIM*stride_C + j0*tile_J*DIM)*sizeof_C : NULL;

        const size_t I = i0 < I0-1 ? tile_I : last_I;
        const size_t J = j0 < J0-1 ? tile_J : last_J;
        const size_t K = k0 < K0-1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0-1 ? padding_I : 0;
        const size_t pad_J = j0 == J0-1 ? padding_J : 0;
        const size_t pad_K = k0 == K0-1 ? padding_K : 0;

        const elem_t * a = a_transpose ? (A + k0*tile_K*DIM*stride_A + i0*tile_I*DIM)
          : (A + i0*tile_I*DIM*stride_A + k0*tile_K*DIM);

        const elem_t * b = b_transpose ? (B + j0*tile_J*DIM*stride_B + k0*tile_K*DIM)
          : (B + k0*tile_K*DIM*stride_B + j0*tile_J*DIM);

        (*inner)(custom_num, a, b, pre, out,
            A_scale_factor, B_scale_factor, D_scale_factor,
            I, J, K,
            pad_I, pad_J, pad_K,
            stride_A, stride_B, stride_D, stride_C,
            a_transpose, b_transpose,
            full_C, low_D,
            no_bias, repeating_bias,
            act);
      }

  //gemmini_fence();
}

// made
static void multi_tiled_matmul_outer1(int gemmini_num,
                               size_t dim_I, size_t dim_J, size_t dim_K,
                               const elem_t *A, const elem_t *B,
                               const void *D, void *C,
                               size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                               scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                               size_t tile_I, size_t tile_J, size_t tile_K,
                               int act, acc_scale_t scale, acc_scale_t bert_scale,
                               bool repeating_bias,
                               bool a_transpose, bool b_transpose,
                               bool full_C, bool low_D,
                               uint8_t weightA,
                               int dataflow)
{

  const size_t divided_dim_I = dim_I / gemmini_num;
  const size_t added_divided_dim_I = divided_dim_I + 1;
  const size_t added_gemmini_num = dim_I % gemmini_num;

  const size_t dim_I_padded = (divided_dim_I / DIM + (divided_dim_I % DIM != 0)) * DIM;
  const size_t added_dim_I_padded = (added_divided_dim_I / DIM + (added_divided_dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const size_t I0 = dim_I_padded / (tile_I * DIM) + (dim_I_padded % (tile_I * DIM) != 0);
  const size_t added_I0 = added_dim_I_padded / (tile_I * DIM) + (added_dim_I_padded % (tile_I * DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J * DIM) + (dim_J_padded % (tile_J * DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K * DIM) + (dim_K_padded % (tile_K * DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I * DIM) == 0 ? tile_I : (dim_I_padded / DIM) % tile_I;
  const size_t added_last_I = added_dim_I_padded % (tile_I * DIM) == 0 ? tile_I : (added_dim_I_padded / DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J * DIM) == 0 ? tile_J : (dim_J_padded / DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K * DIM) == 0 ? tile_K : (dim_K_padded / DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - divided_dim_I;
  const size_t added_padding_I = added_dim_I_padded - added_divided_dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias)
  {
    D = (void *)1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  //gemmini_fence();

  switch (gemmini_num)
  {
  case 4:
    gemmini_extended_config_ex(custom3, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
    gemmini_extended_config_st(custom3, stride_C * sizeof_C, act & 3, scale);
    gemmini_extended3_config_ld(custom3, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
    gemmini_extended3_config_ld(custom3, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
    gemmini_extended3_config_ld(custom3, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
  case 3:
    gemmini_extended_config_ex(custom2, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
    gemmini_extended_config_st(custom2, stride_C * sizeof_C, act & 3, scale);
    gemmini_extended3_config_ld(custom2, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
    gemmini_extended3_config_ld(custom2, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
    gemmini_extended3_config_ld(custom2, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
  case 2:
    gemmini_extended_config_ex(custom1, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
    gemmini_extended_config_st(custom1, stride_C * sizeof_C, act & 3, scale);
    gemmini_extended3_config_ld(custom1, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
    gemmini_extended3_config_ld(custom1, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
    gemmini_extended3_config_ld(custom1, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
  case 1:
    gemmini_extended_config_ex(custom0, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
    gemmini_extended_config_st(custom0, stride_C * sizeof_C, act & 3, scale);
    gemmini_extended3_config_ld(custom0, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
    gemmini_extended3_config_ld(custom0, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
    gemmini_extended3_config_ld(custom0, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
    break;
  }

  if (act == IGELU)
  {
    const acc_scale_t sqrt_2 = 1.41421356237;
    const acc_scale_t S = bert_scale;
    const acc_scale_t S_erf = (-0.2888 * ((S * S) / 2));

    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / S_erf;
    switch (gemmini_num)
    {
    case 4:
      gemmini_config_norm(custom3, 0, 0, 0, 0, 0, qb, qc);
    case 3:
      gemmini_config_norm(custom2, 0, 0, 0, 0, 0, qb, qc);
    case 2:
      gemmini_config_norm(custom1, 0, 0, 0, 0, 0, qb, qc);
    case 1:
      gemmini_config_norm(custom0, 0, 0, 0, 0, 0, qb, qc);
      break;
    }
  }

  if (act == SOFTMAX)
  {
    const scale_t a = 0.3585;
    const scale_t b = 1.353;
    const scale_t c = 0.344;

    const acc_t qln2 = (int)(0.693147 / bert_scale);
    const acc_t qln2_inv = 65536 / qln2;
    const acc_t qb = b / bert_scale;
    const acc_t qc = c / (a * bert_scale * bert_scale);

    switch (gemmini_num)
    {
    case 4:
      gemmini_config_norm(custom3, qln2, 0, 0, 1, 0, qb, qc);
      gemmini_config_norm(custom3, qln2_inv, 1, 0, 1, 0, qb, qc);
    case 3:
      gemmini_config_norm(custom2, qln2, 0, 0, 1, 0, qb, qc);
      gemmini_config_norm(custom2, qln2_inv, 1, 0, 1, 0, qb, qc);
    case 2:
      gemmini_config_norm(custom1, qln2, 0, 0, 1, 0, qb, qc);
      gemmini_config_norm(custom1, qln2_inv, 1, 0, 1, 0, qb, qc);
    case 1:
      gemmini_config_norm(custom0, qln2, 0, 0, 1, 0, qb, qc);
      gemmini_config_norm(custom0, qln2_inv, 1, 0, 1, 0, qb, qc);
      break;
    }
  }

  void (*inner)(int, const size_t, size_t, const size_t, const size_t, const size_t, const elem_t *, const elem_t *, const void *, void *,
                scale_t, scale_t, scale_acc_t,
                size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
                size_t, size_t, size_t, size_t,
                bool, bool,
                bool, bool,
                bool, bool,
                int);

  if (dataflow == OUTPUT_STATIONARY)
  {
    inner = &multi_sp_tiled_matmul_os;
  }
  else /* if (dataflow == WEIGHT_STATIONARY) */
  {
    inner = &multi_sp_tiled_matmul_ws;
  }

  for (size_t i0 = 0; i0 < added_I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++)
      {

        const void *pre;
        if (k0 != 0)
        {
          pre = NULL;
        }
        else
        {
          size_t bias_row = repeating_bias ? 0 : i0 * tile_I * DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t *)D + (bias_row * stride_D + j0 * tile_J * DIM) * sizeof_D;
        }

        void *out = k0 == K0 - 1 ? (int8_t *)C + (i0 * tile_I * DIM * stride_C + j0 * tile_J * DIM) * sizeof_C : NULL;

        const size_t I = i0 < I0 - 1 ? tile_I : last_I;
        const size_t added_I = i0 < added_I0 - 1 ? tile_I : added_last_I;
        const size_t J = j0 < J0 - 1 ? tile_J : last_J;
        const size_t K = k0 < K0 - 1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0 - 1 ? padding_I : 0;
        const size_t added_pad_I = i0 == added_I0 - 1 ? added_padding_I : 0;
        const size_t pad_J = j0 == J0 - 1 ? padding_J : 0;
        const size_t pad_K = k0 == K0 - 1 ? padding_K : 0;

        const elem_t *a = a_transpose ? (A + k0 * tile_K * DIM * stride_A + i0 * tile_I * DIM)
                                      : (A + i0 * tile_I * DIM * stride_A + k0 * tile_K * DIM);

        const elem_t *b = b_transpose ? (B + j0 * tile_J * DIM * stride_B + k0 * tile_K * DIM)
                                      : (B + k0 * tile_K * DIM * stride_B + j0 * tile_J * DIM);

        (*inner)(gemmini_num, added_gemmini_num,
                 i0, I0, divided_dim_I, added_divided_dim_I,
                 a, b, pre, out,
                 A_scale_factor, B_scale_factor, D_scale_factor,
                 I, added_I, J, K,
                 pad_I, added_pad_I, pad_J, pad_K,
                 stride_A, stride_B, stride_D, stride_C,
                 a_transpose, b_transpose,
                 full_C, low_D,
                 no_bias, repeating_bias,
                 act);
      }

  // gemmini_fence();
}

// made
static void shared_multi_tiled_matmul_outer(int gemmini_list, int tile_id,
                                            size_t sp_addr_start, size_t acc_addr_start,
                                            size_t dim_I, size_t dim_J, size_t dim_K,
                                            const elem_t *A, const elem_t *B,
                                            const void *D, void *C,
                                            size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                            scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                            size_t tile_I, size_t tile_J, size_t tile_K,
                                            int act, acc_scale_t scale, acc_scale_t bert_scale,
                                            bool repeating_bias,
                                            bool a_transpose, bool b_transpose,
                                            bool full_C, bool low_D,
                                            uint8_t weightA,
                                            int dataflow)
{
  int gemmini_num = 0;

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const size_t I0 = dim_I_padded / (tile_I * DIM) + (dim_I_padded % (tile_I * DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J * DIM) + (dim_J_padded % (tile_J * DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K * DIM) + (dim_K_padded % (tile_K * DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I * DIM) == 0 ? tile_I : (dim_I_padded / DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J * DIM) == 0 ? tile_J : (dim_J_padded / DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K * DIM) == 0 ? tile_K : (dim_K_padded / DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias)
  {
    D = (void *)1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  // gemmini_fence();

  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((gemmini_list >> i) & 1)
    {
      gemmini_num++;
      switch (i)
      {
      case 3:
        gemmini_extended_config_ex(custom3, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
        gemmini_extended_config_st(custom3, stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(custom3, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom3, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom3, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
        break;
      case 2:
        gemmini_extended_config_ex(custom2, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
        gemmini_extended_config_st(custom2, stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(custom2, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom2, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom2, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
        break;
      case 1:
        gemmini_extended_config_ex(custom1, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
        gemmini_extended_config_st(custom1, stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(custom1, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom1, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom1, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
        break;
      case 0:
        gemmini_extended_config_ex(custom0, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
        gemmini_extended_config_st(custom0, stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(custom0, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom0, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom0, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
        break;
      }
    }
  }

      if (act == IGELU)
      {
        const acc_scale_t sqrt_2 = 1.41421356237;
        const acc_scale_t S = bert_scale;
        const acc_scale_t S_erf = (-0.2888 * ((S * S) / 2));

        const acc_t qb = -1.769 / (S / sqrt_2);
        const acc_t qc = 1.0 / S_erf;
        
        for (int i = 0; i < total_gemmini_num; i++)
        {
          if ((gemmini_list >> i) & 1)
          {
            switch (i)
            {
            case 3:
              gemmini_config_norm(custom3, 0, 0, 0, 0, 0, qb, qc);
              break;
            case 2:
              gemmini_config_norm(custom2, 0, 0, 0, 0, 0, qb, qc);
              break;
            case 1:
              gemmini_config_norm(custom1, 0, 0, 0, 0, 0, qb, qc);
              break;
            case 0:
              gemmini_config_norm(custom0, 0, 0, 0, 0, 0, qb, qc);
              break;
            }
          }
        }
      }

          if (act == SOFTMAX)
          {
            const scale_t a = 0.3585;
            const scale_t b = 1.353;
            const scale_t c = 0.344;

            const acc_t qln2 = (int)(0.693147 / bert_scale);
            const acc_t qln2_inv = 65536 / qln2;
            const acc_t qb = b / bert_scale;
            const acc_t qc = c / (a * bert_scale * bert_scale);

            for (int i = 0; i < total_gemmini_num; i++)
            {
              if ((gemmini_list >> i) & 1)
              {
                switch (i)
                {
                case 3:
                  gemmini_config_norm(custom3, qln2, 0, 0, 1, 0, qb, qc);
                  gemmini_config_norm(custom3, qln2_inv, 1, 0, 1, 0, qb, qc);
                  break;
                case 2:
                  gemmini_config_norm(custom2, qln2, 0, 0, 1, 0, qb, qc);
                  gemmini_config_norm(custom2, qln2_inv, 1, 0, 1, 0, qb, qc);
                  break;
                case 1:
                  gemmini_config_norm(custom1, qln2, 0, 0, 1, 0, qb, qc);
                  gemmini_config_norm(custom1, qln2_inv, 1, 0, 1, 0, qb, qc);
                  break;
                case 0:
                  gemmini_config_norm(custom0, qln2, 0, 0, 1, 0, qb, qc);
                  gemmini_config_norm(custom0, qln2_inv, 1, 0, 1, 0, qb, qc);
                  break;
                }
              }
            }
          }

  void (*inner)(int, int, int,
                size_t, size_t,
                const elem_t *, const elem_t *, const void *, void *,
                scale_t, scale_t, scale_acc_t,
                size_t, size_t, size_t, size_t, size_t,
                size_t, size_t, size_t, size_t, size_t, size_t,
                size_t, size_t, size_t, size_t,
                bool, bool,
                bool, bool,
                bool, bool,
                int);

  if (dataflow == OUTPUT_STATIONARY)
  {
    inner = &shared_multi_sp_tiled_matmul_os;
  }
  else /* if (dataflow == WEIGHT_STATIONARY) */
  {
    inner = &shared_multi_sp_tiled_matmul_ws;
  }

  int inner_call_counter = 0;
  int lastK_toggle = 0;
  // printf("tile_I = %d, tile_J = %d, tile_K = %d\n", tile_I, tile_J, tile_K);
  // printf("I0 = %d, J0 = %d, K0 = %d\n", I0, J0, K0);
  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++)
      {
        const void *pre;
        if (k0 != 0)
        {
          pre = NULL;
        }
        else
        {
          size_t bias_row = repeating_bias ? 0 : i0 * tile_I * DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t *)D + (bias_row * stride_D + j0 * tile_J * DIM) * sizeof_D;
        }

        void *out = k0 == K0 - 1 ? (int8_t *)C + (i0 * tile_I * DIM * stride_C + j0 * tile_J * DIM) * sizeof_C : NULL;

        const size_t I = i0 < I0 - 1 ? tile_I : last_I;
        const size_t J = j0 < J0 - 1 ? tile_J : last_J;
        const size_t K = k0 < K0 - 1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0 - 1 ? padding_I : 0;
        const size_t pad_J = j0 == J0 - 1 ? padding_J : 0;
        const size_t pad_K = k0 == K0 - 1 ? padding_K : 0;

        const elem_t *a = a_transpose ? (A + k0 * tile_K * DIM * stride_A + i0 * tile_I * DIM)
                                      : (A + i0 * tile_I * DIM * stride_A + k0 * tile_K * DIM);

        const elem_t *b = b_transpose ? (B + j0 * tile_J * DIM * stride_B + k0 * tile_K * DIM)
                                      : (B + k0 * tile_K * DIM * stride_B + j0 * tile_J * DIM);

        int using_gemmini_num = gemmini_num;
        if (I < gemmini_num && K < gemmini_num)
        {
          using_gemmini_num = (I < K) ? K : I;
        }

        size_t group_list = gemmini_list;
        int shift_num = 0;
        for (int i = 0; i < total_gemmini_num; i++)
        {
          size_t shifted_list = (gemmini_list >> i);
          if (shift_num == using_gemmini_num)
          {
            group_list = group_list & ~(shifted_list << i);
            break;
          }
          if (shifted_list & 1)
          {
            shift_num++;
          }
        }

        // for ldA, ldD, stC, ex
        const size_t I_div = I / using_gemmini_num;
        const size_t I_div_added = I_div + 1;
        const size_t I_added_gemmini_num = I % using_gemmini_num;

        // for ldB
        const size_t K_div = K / using_gemmini_num;
        const size_t K_div_added = K_div + 1;
        const size_t K_added_gemmini_num = K % using_gemmini_num;

        // for ld, st dram address
        const elem_t *a_local = a;
        const elem_t *b_local = b;
        const int8_t *c_local = (int8_t *)out;
        const int8_t *d_local = (int8_t *)pre;

        // for local addr
        size_t laddrI_offset = 0;
        size_t laddrK_offset = 0;

        const int t = (inner_call_counter & 1);
        const size_t local_sp_addr_start = (t == 0) ? sp_addr_start : sp_addr_start + BANK_NUM * BANK_ROWS / 2;
        // const size_t sp_addr_end = (t == 0) ? sp_addr_start + BANK_NUM * BANK_ROWS / 2 : sp_addr_start + BANK_NUM * BANK_ROWS - 1;
        const size_t local_acc_addr_start = lastK_toggle ? acc_addr_start : acc_addr_start + ACC_ROWS / 2;

        int activated_gemmini_num = 0;
        for (int i = 0; i < total_gemmini_num; i++)
        {
          if ((group_list >> i) & 1)
          {
            size_t this_I = (activated_gemmini_num < I_added_gemmini_num) ? I_div_added : I_div;
            int ex_gemmini_num = (using_gemmini_num < I) ? using_gemmini_num : I;
            size_t this_pad_I = (activated_gemmini_num == ex_gemmini_num - 1) ? pad_I : 0;

            size_t this_K = (activated_gemmini_num < K_added_gemmini_num) ? K_div_added : K_div;
            int ldB_gemmini_num = (using_gemmini_num < K) ? using_gemmini_num : K;
            size_t this_pad_K = (activated_gemmini_num == ldB_gemmini_num - 1) ? pad_K : 0;

            (*inner)(i, group_list, tile_id << 1 | t,
                     local_sp_addr_start, local_acc_addr_start,
                     a_local, b_local, (k0 != 0) ? NULL : (void *)d_local, (k0 == K0 - 1) ? (void *)c_local : NULL,
                     A_scale_factor, B_scale_factor, D_scale_factor,
                     this_I, this_K, this_pad_K, laddrI_offset, laddrK_offset, // mv_K, mv_pad_K, laddrI_offset, laddrK_offset
                     I, J, K,
                     this_pad_I, pad_J, pad_K,
                     stride_A, stride_B, stride_D, stride_C,
                     a_transpose, b_transpose,
                     full_C, low_D,
                     no_bias, repeating_bias,
                     act);

            laddrI_offset += this_I;
            laddrK_offset += this_K;

            size_t local_stride_A = a_transpose ? 1 : stride_A;
            a_local += local_stride_A * DIM * this_I;
            size_t local_stride_B = b_transpose ? 1 : stride_B;
            b_local += local_stride_B * DIM * this_K;
            c_local += stride_C * DIM * this_I * sizeof_C;
            d_local += stride_D * DIM * this_I * sizeof_D;

            activated_gemmini_num++;
          }
        }
        
        for (int i = 0; i < total_gemmini_num; i++)
        {
          if ((group_list >> i) & 1)
          {
            switch (i)
            {
            case 3:
              // printf("delay");
              ROCC_INSTRUCTION_RS1_RS2(custom3, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
              break;
            case 2:
              ROCC_INSTRUCTION_RS1_RS2(custom2, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
              break;
            case 1:
              ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
              break;
            case 0:
              ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
              break;
            }
          }
        }

        inner_call_counter++;
        if (k0 == K0 - 1)
          lastK_toggle ^= 1;
      }

  // gemmini_fence();
}

// made
static void shared_multi_tiled_matmul_outer_test(int gemmini_list, int tile_id,
                                            size_t sp_addr_start, size_t acc_addr_start,
                                            size_t sp_addr_range, size_t acc_addr_range,
                                            size_t dim_I, size_t dim_J, size_t dim_K,
                                            const elem_t *A, const elem_t *B,
                                            const void *D, void *C,
                                            size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                            scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                            size_t tile_I, size_t tile_J, size_t tile_K,
                                            int act, acc_scale_t scale, acc_scale_t bert_scale,
                                            bool repeating_bias,
                                            bool a_transpose, bool b_transpose,
                                            bool full_C, bool low_D,
                                            uint8_t weightA,
                                            int dataflow)
{
  int gemmini_num = 0;

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const size_t I0 = dim_I_padded / (tile_I * DIM) + (dim_I_padded % (tile_I * DIM) != 0);
  const size_t J0 = dim_J_padded / (tile_J * DIM) + (dim_J_padded % (tile_J * DIM) != 0);
  const size_t K0 = dim_K_padded / (tile_K * DIM) + (dim_K_padded % (tile_K * DIM) != 0);

  // These lines here are supposed to help us deal with when the dimensions of
  // the systolic array aren't divisible by the tiling factors
  const size_t last_I = dim_I_padded % (tile_I * DIM) == 0 ? tile_I : (dim_I_padded / DIM) % tile_I;
  const size_t last_J = dim_J_padded % (tile_J * DIM) == 0 ? tile_J : (dim_J_padded / DIM) % tile_J;
  const size_t last_K = dim_K_padded % (tile_K * DIM) == 0 ? tile_K : (dim_K_padded / DIM) % tile_K;

  // These lines are supposed to figure out how much padding the hardware is
  // supposed to add for the final tile
  const size_t padding_I = dim_I_padded - dim_I;
  const size_t padding_J = dim_J_padded - dim_J;
  const size_t padding_K = dim_K_padded - dim_K;

  const bool no_bias = D == NULL;

  if (no_bias)
  {
    D = (void *)1; // Dummy address which isn't NULL
  }

  const size_t sizeof_D = low_D ? sizeof(elem_t) : sizeof(acc_t);
  const size_t sizeof_C = full_C ? sizeof(acc_t) : sizeof(elem_t);

  // gemmini_fence();

  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((gemmini_list >> i) & 1)
    {
      gemmini_num++;
      switch (i)
      {
      case 3:
        gemmini_extended_config_ex(custom3, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
        gemmini_extended_config_st(custom3, stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(custom3, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom3, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom3, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
        break;
      case 2:
        gemmini_extended_config_ex(custom2, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
        gemmini_extended_config_st(custom2, stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(custom2, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom2, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom2, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
        break;
      case 1:
        gemmini_extended_config_ex(custom1, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
        gemmini_extended_config_st(custom1, stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(custom1, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom1, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom1, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
        break;
      case 0:
        gemmini_extended_config_ex(custom0, dataflow, act & 3, 0, 1, a_transpose, b_transpose);
        gemmini_extended_config_st(custom0, stride_C * sizeof_C, act & 3, scale);
        gemmini_extended3_config_ld(custom0, stride_A * sizeof(elem_t), A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom0, stride_B * sizeof(elem_t), B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom0, repeating_bias ? 0 : (stride_D * sizeof_D), D_scale_factor, low_D, 2);
        break;
      }
    }
  }

  if (act == IGELU)
  {
    const acc_scale_t sqrt_2 = 1.41421356237;
    const acc_scale_t S = bert_scale;
    const acc_scale_t S_erf = (-0.2888 * ((S * S) / 2));

    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / S_erf;

    for (int i = 0; i < total_gemmini_num; i++)
    {
      if ((gemmini_list >> i) & 1)
      {
        switch (i)
        {
        case 3:
          gemmini_config_norm(custom3, 0, 0, 0, 0, 0, qb, qc);
          break;
        case 2:
          gemmini_config_norm(custom2, 0, 0, 0, 0, 0, qb, qc);
          break;
        case 1:
          gemmini_config_norm(custom1, 0, 0, 0, 0, 0, qb, qc);
          break;
        case 0:
          gemmini_config_norm(custom0, 0, 0, 0, 0, 0, qb, qc);
          break;
        }
      }
    }
  }

  if (act == SOFTMAX)
  {
    const scale_t a = 0.3585;
    const scale_t b = 1.353;
    const scale_t c = 0.344;

    const acc_t qln2 = (int)(0.693147 / bert_scale);
    const acc_t qln2_inv = 65536 / qln2;
    const acc_t qb = b / bert_scale;
    const acc_t qc = c / (a * bert_scale * bert_scale);

    for (int i = 0; i < total_gemmini_num; i++)
    {
      if ((gemmini_list >> i) & 1)
      {
        switch (i)
        {
        case 3:
          gemmini_config_norm(custom3, qln2, 0, 0, 1, 0, qb, qc);
          gemmini_config_norm(custom3, qln2_inv, 1, 0, 1, 0, qb, qc);
          break;
        case 2:
          gemmini_config_norm(custom2, qln2, 0, 0, 1, 0, qb, qc);
          gemmini_config_norm(custom2, qln2_inv, 1, 0, 1, 0, qb, qc);
          break;
        case 1:
          gemmini_config_norm(custom1, qln2, 0, 0, 1, 0, qb, qc);
          gemmini_config_norm(custom1, qln2_inv, 1, 0, 1, 0, qb, qc);
          break;
        case 0:
          gemmini_config_norm(custom0, qln2, 0, 0, 1, 0, qb, qc);
          gemmini_config_norm(custom0, qln2_inv, 1, 0, 1, 0, qb, qc);
          break;
        }
      }
    }
  }

  void (*inner)(int, int, int,
                size_t, size_t,
                const elem_t *, const elem_t *, const void *, void *,
                scale_t, scale_t, scale_acc_t,
                size_t, size_t, size_t, size_t, size_t,
                size_t, size_t, size_t, size_t, size_t, size_t,
                size_t, size_t, size_t, size_t,
                bool, bool,
                bool, bool,
                bool, bool,
                int);

  if (dataflow == OUTPUT_STATIONARY)
  {
    inner = &shared_multi_sp_tiled_matmul_os;
  }
  else /* if (dataflow == WEIGHT_STATIONARY) */
  {
    inner = &shared_multi_sp_tiled_matmul_ws;
  }

  int inner_call_counter = 0;
  int lastK_toggle = 0;
  // printf("tile_I = %d, tile_J = %d, tile_K = %d\n", tile_I, tile_J, tile_K);
  // printf("I0 = %d, J0 = %d, K0 = %d\n", I0, J0, K0);
  for (size_t i0 = 0; i0 < I0; i0++)
    for (size_t j0 = 0; j0 < J0; j0++)
      for (size_t k0 = 0; k0 < K0; k0++)
      {
        const void *pre;
        if (k0 != 0)
        {
          pre = NULL;
        }
        else
        {
          size_t bias_row = repeating_bias ? 0 : i0 * tile_I * DIM;
          // pre = &(((acc_t*)D)[bias_row * stride_D + j0 * tile_J * DIM]);
          pre = (int8_t *)D + (bias_row * stride_D + j0 * tile_J * DIM) * sizeof_D;
        }

        void *out = k0 == K0 - 1 ? (int8_t *)C + (i0 * tile_I * DIM * stride_C + j0 * tile_J * DIM) * sizeof_C : NULL;

        const size_t I = i0 < I0 - 1 ? tile_I : last_I;
        const size_t J = j0 < J0 - 1 ? tile_J : last_J;
        const size_t K = k0 < K0 - 1 ? tile_K : last_K;

        const size_t pad_I = i0 == I0 - 1 ? padding_I : 0;
        const size_t pad_J = j0 == J0 - 1 ? padding_J : 0;
        const size_t pad_K = k0 == K0 - 1 ? padding_K : 0;

        const elem_t *a = a_transpose ? (A + k0 * tile_K * DIM * stride_A + i0 * tile_I * DIM)
                                      : (A + i0 * tile_I * DIM * stride_A + k0 * tile_K * DIM);

        const elem_t *b = b_transpose ? (B + j0 * tile_J * DIM * stride_B + k0 * tile_K * DIM)
                                      : (B + k0 * tile_K * DIM * stride_B + j0 * tile_J * DIM);

        int using_gemmini_num = gemmini_num;
        if (I < gemmini_num && K < gemmini_num)
        {
          using_gemmini_num = (I < K) ? K : I;
        }

        size_t group_list = gemmini_list;
        int shift_num = 0;
        for (int i = 0; i < total_gemmini_num; i++)
        {
          size_t shifted_list = (gemmini_list >> i);
          if (shift_num == using_gemmini_num)
          {
            group_list = group_list & ~(shifted_list << i);
            break;
          }
          if (shifted_list & 1)
          {
            shift_num++;
          }
        }

        // for ldA, ldD, stC, ex
        const size_t I_div = I / using_gemmini_num;
        const size_t I_div_added = I_div + 1;
        const size_t I_added_gemmini_num = I % using_gemmini_num;

        // for ldB
        const size_t K_div = K / using_gemmini_num;
        const size_t K_div_added = K_div + 1;
        const size_t K_added_gemmini_num = K % using_gemmini_num;

        // for ld, st dram address
        const elem_t *a_local = a;
        const elem_t *b_local = b;
        const int8_t *c_local = (int8_t *)out;
        const int8_t *d_local = (int8_t *)pre;

        // for local addr
        size_t laddrI_offset = 0;
        size_t laddrK_offset = 0;

        const int t = (inner_call_counter & 1);
        const size_t local_sp_addr_start = (t == 0) ? sp_addr_start : sp_addr_start + sp_addr_range / 2;
        // const size_t sp_addr_end = (t == 0) ? sp_addr_start + BANK_NUM * BANK_ROWS / 2 : sp_addr_start + BANK_NUM * BANK_ROWS - 1;
        const size_t local_acc_addr_start = lastK_toggle ? acc_addr_start : acc_addr_start + acc_addr_range / 2;

        int activated_gemmini_num = 0;
        for (int i = 0; i < total_gemmini_num; i++)
        {
          if ((group_list >> i) & 1)
          {
            size_t this_I = (activated_gemmini_num < I_added_gemmini_num) ? I_div_added : I_div;
            int ex_gemmini_num = (using_gemmini_num < I) ? using_gemmini_num : I;
            size_t this_pad_I = (activated_gemmini_num == ex_gemmini_num - 1) ? pad_I : 0;

            size_t this_K = (activated_gemmini_num < K_added_gemmini_num) ? K_div_added : K_div;
            int ldB_gemmini_num = (using_gemmini_num < K) ? using_gemmini_num : K;
            size_t this_pad_K = (activated_gemmini_num == ldB_gemmini_num - 1) ? pad_K : 0;

            (*inner)(i, group_list, tile_id << 1 | t,
                     local_sp_addr_start, local_acc_addr_start,
                     a_local, b_local, (k0 != 0) ? NULL : (void *)d_local, (k0 == K0 - 1) ? (void *)c_local : NULL,
                     A_scale_factor, B_scale_factor, D_scale_factor,
                     this_I, this_K, this_pad_K, laddrI_offset, laddrK_offset, // mv_K, mv_pad_K, laddrI_offset, laddrK_offset
                     I, J, K,
                     this_pad_I, pad_J, pad_K,
                     stride_A, stride_B, stride_D, stride_C,
                     a_transpose, b_transpose,
                     full_C, low_D,
                     no_bias, repeating_bias,
                     act);

            laddrI_offset += this_I;
            laddrK_offset += this_K;

            size_t local_stride_A = a_transpose ? 1 : stride_A;
            a_local += local_stride_A * DIM * this_I;
            size_t local_stride_B = b_transpose ? 1 : stride_B;
            b_local += local_stride_B * DIM * this_K;
            c_local += stride_C * DIM * this_I * sizeof_C;
            d_local += stride_D * DIM * this_I * sizeof_D;

            activated_gemmini_num++;
          }
        }

        // for (int i = 0; i < total_gemmini_num; i++)
        // {
        //   if ((group_list >> i) & 1)
        //   {
        //     switch (i)
        //     {
        //     case 3:
        //       // printf("delay");
        //       ROCC_INSTRUCTION_RS1_RS2(custom3, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
        //       break;
        //     case 2:
        //       ROCC_INSTRUCTION_RS1_RS2(custom2, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
        //       break;
        //     case 1:
        //       ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
        //       break;
        //     case 0:
        //       ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
        //       break;
        //     }
        //   }
        // }

        // if (tile_id == 7) {
        //   tile_id = 0;
        // } else {
        //   tile_id++;
        // }

        inner_call_counter++;
        if (k0 == K0 - 1)
          lastK_toggle ^= 1;
      }

  // gemmini_fence();
}

static acc_t int_sqrt(acc_t n)
{
  if (n == 0)
    return 0;

  int bits = 0;
  for (acc_t x = n; x > 0; x /= 2)
    bits++;

  acc_t x_prev = 1 << ((bits + 1) / 2);

  while (1)
  {
    acc_t x_next = (x_prev + n / x_prev) / 2;
    if (x_next >= x_prev)
      return x_prev;
    x_prev = x_next;
  };
}

static elem_t scale_and_sat(acc_t x, int act, acc_scale_t scale, acc_scale_t bert_scale)
{
  // Apply I-GELU if needed
  if (act == IGELU)
  {
    const acc_scale_t sqrt_2 = 1.41421356237;

    const acc_scale_t S = bert_scale;

    const acc_scale_t S_erf = (-0.2888 * (S / sqrt_2) * (S / sqrt_2));
    const acc_t q1 = 1 / S_erf;
    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / (-0.2888 * (S / sqrt_2) * (S / sqrt_2));

    const acc_t q = x;

    const acc_t q_sign = q < 0 ? -1 : 1;
    const acc_t q_clipped = abs(q) > (-qb) ? (-qb) : abs(q);
    const acc_t q_poly = (q_clipped + qb) * (q_clipped + qb) + qc;
    const acc_t q_erf = q_sign * q_poly;

    x = q * (q_erf + q1);
  }

  // Scale value down and round it
  x = ACC_SCALE(x, scale);
  // Clip result
  x = x > elem_t_max ? elem_t_max : (x < elem_t_min ? elem_t_min : x);
  // Apply activation function
  if (act == RELU)
  {
    x = x < 0 ? 0 : x;
  }
  return x;
}

#ifdef HAS_MVIN_SCALE
#define GEMMINI_SCALE(x, scale) MVIN_SCALE((x), (scale))
#else
#define GEMMINI_SCALE(x, scale) (x)
#endif

#ifdef HAS_MVIN_ACC_SCALE
#define GEMMINI_ACC_SCALE(x, scale) MVIN_SCALE_ACC((x), (scale))
#else
#define GEMMINI_ACC_SCALE(x, scale) (x)
#endif

static void matmul_cpu(bool transA, bool transB, size_t DIM_I, size_t DIM_J, size_t DIM_K,
        const elem_t* A, const elem_t* B, const acc_t * D,
        elem_t* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale, bool repeating_bias) {

  const int no_bias = D == NULL;
  if (act != LAYERNORM && act != SOFTMAX && !transA && !transB && DIM_I % 4 == 0 && DIM_J % 4 == 0) {
    for (size_t i = 0; i < DIM_I; i += 4) {
      for (size_t j = 0; j < DIM_J; j += 4) {

        acc_t result[4][4]; // = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

        for (size_t ii = 0; ii < 4; ii++)
          for (size_t jj = 0; jj < 4; jj++) {
            const size_t bias_row = repeating_bias ? 0 : i + ii;
            result[ii][jj] = no_bias ? 0 :
              GEMMINI_ACC_SCALE(*(D + bias_row*stride_D + j + jj), D_scale_factor);
          }

        for (size_t k = 0; k < DIM_K; k++) {
          result[0][0] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[0][1] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[0][2] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[0][3] +=
                GEMMINI_SCALE(*(A + i*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[1][0] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[1][1] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[1][2] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[1][3] +=
                GEMMINI_SCALE(*(A + (i+1)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[2][0] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[2][1] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[2][2] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[2][3] +=
                GEMMINI_SCALE(*(A + (i+2)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
          result[3][0] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j), B_scale_factor);
          result[3][1] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+1), B_scale_factor);
          result[3][2] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+2), B_scale_factor);
          result[3][3] +=
                GEMMINI_SCALE(*(A + (i+3)*stride_A + k), A_scale_factor) *
                GEMMINI_SCALE(*(B + k*stride_B + j+3), B_scale_factor);
        }

        *(C + i*stride_C + j) =
             scale_and_sat(result[0][0], act, scale, bert_scale);
        *(C + i*stride_C + j+1) =
             scale_and_sat(result[0][1], act, scale, bert_scale);
        *(C + i*stride_C + j+2) =
             scale_and_sat(result[0][2], act, scale, bert_scale);
        *(C + i*stride_C + j+3) =
             scale_and_sat(result[0][3], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j) =
             scale_and_sat(result[1][0], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+1) =
             scale_and_sat(result[1][1], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+2) =
             scale_and_sat(result[1][2], act, scale, bert_scale);
        *(C + (i+1)*stride_C + j+3) =
             scale_and_sat(result[1][3], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j) =
             scale_and_sat(result[2][0], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+1) =
             scale_and_sat(result[2][1], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+2) =
             scale_and_sat(result[2][2], act, scale, bert_scale);
        *(C + (i+2)*stride_C + j+3) =
             scale_and_sat(result[2][3], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j) =
             scale_and_sat(result[3][0], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+1) =
             scale_and_sat(result[3][1], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+2) =
             scale_and_sat(result[3][2], act, scale, bert_scale);
        *(C + (i+3)*stride_C + j+3) =
             scale_and_sat(result[3][3], act, scale, bert_scale);
      }
    }
  } else {
    size_t A_dim_strides[2] = {!transA ? stride_A : 1, !transA ? 1 : stride_A}; // i, j stride
    size_t B_dim_strides[2] = {!transB ? 1 : stride_B, !transB ? stride_B : 1}; // j, k stride

    // We also create a buffer that we can use for layernorms and softmaxes
    static acc_t c_buffer[1024];
    const size_t c_buffer_sz = sizeof(c_buffer)/sizeof(c_buffer[0]);
    if ((act == LAYERNORM || act == SOFTMAX) && DIM_J > c_buffer_sz) {
      printf("Matmul is too large to normalize\n");
      exit(1);
    }

    for (size_t i = 0; i < DIM_I; i++) {
      for (size_t j = 0; j < DIM_J; j++) {
        elem_t* c = C + (i * stride_C) + j;

        const size_t bias_row = repeating_bias ? 0 : i;
        acc_t sum = no_bias ? 0 : GEMMINI_ACC_SCALE(*(D + bias_row * stride_D + j), D_scale_factor);

        for (size_t k = 0; k < DIM_K; k++) {
          const elem_t* a = A + i * A_dim_strides[0] + k * A_dim_strides[1];
          const elem_t* b = B + j * B_dim_strides[0] + k * B_dim_strides[1];
          sum += (GEMMINI_SCALE(*a, A_scale_factor) * GEMMINI_SCALE(*b, B_scale_factor));
        }

        if (act == LAYERNORM || act == SOFTMAX)
          c_buffer[j] = sum;
        else
          *c = scale_and_sat(sum, act, scale, bert_scale);
      }

      if (act == LAYERNORM) {
        acc_t sum = 0;
        for (size_t j = 0; j < DIM_J; j++)
          sum += c_buffer[j];
        acc_t mean = sum / (acc_t)DIM_J;

        acc_t total_err_sq = 0;
        for (size_t j = 0; j < DIM_J; j++)
          total_err_sq += (c_buffer[j] - mean)*(c_buffer[j] - mean);
        acc_t variance = total_err_sq / (acc_t)DIM_J;

        acc_t stddev = int_sqrt(variance);
        if (variance == 0) stddev = 1;

        for (size_t j = 0; j < DIM_J; j++) {
          c_buffer[j] -= mean;
          c_buffer[j] /= stddev;

          elem_t* c = C + (i * stride_C) + j;
          *c = scale_and_sat(c_buffer[j], act, scale, bert_scale);
        }
      } else if (act == SOFTMAX) {
        const scale_t a = 0.3585;
        const scale_t b = 1.353;
        const scale_t c = 0.344;

        // is SCALE supposed to be input scale?
        const acc_t qln2 = (acc_t) (0.693147 / bert_scale);
        const acc_t qln2_inv = 65536 / qln2;
        const acc_t qb = b / bert_scale;
        const acc_t qc = c / (a*bert_scale*bert_scale);

        // pass 1: get max_q
        acc_t max_q = -2147483648;
        for (size_t j = 0; j < DIM_J; j++) {
          if (c_buffer[j] > max_q) max_q = c_buffer[j];
        }

        // pass 2: calculate iexp(q_tilde) and sum(q_tilde)
        acc_t sum_exp = 0;
        for (size_t j = 0; j < DIM_J; j++) {
          acc_t q = c_buffer[j] - max_q;
          acc_t z = (acc_t) (-q * qln2_inv) >> 16;
          acc_t qp = q + z * qln2;
          acc_t q_exp = (qp + qb)*(qp + qb) + qc;
          c_buffer[j] = q_exp >> z;
          sum_exp += c_buffer[j];
        }

        // pass 3: divide by sum
        scale_t factor = (127.f) / (float) sum_exp; // what corresponds to 1 in output?
        for (size_t j = 0; j < DIM_J; j++) {
          elem_t* c = C + (i * stride_C) + j;
          *c = scale_and_sat(c_buffer[j], act, factor, bert_scale);
        }
      }
    }
  }
}

#undef GEMMINI_SCALE

// General matmul which can be run with different dataflows, or on the CPU
enum tiled_matmul_type_t {OS, WS, CPU}; // TODO rename this so it's name also applies to convs

// This function runs a tiled matrix mulctiplication, with hardcoded tiling
// factors
static void tiled_matmul (int custom_num, size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void* C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        size_t tile_I, size_t tile_J, size_t tile_K,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0) {
    printf("tile_I is non-positive\n");
    exit(1);
  } else if (tile_J <= 0) {
    printf("tile_J is non-positive\n");
    exit(1);
  } else if (tile_K <= 0) {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded) {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  } else if (tile_J * DIM > dim_J_padded) {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  } else if (tile_K * DIM > dim_K_padded) {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t total_spad_size = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
  const size_t total_acc_size = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) +   // Rows to store A
      (tile_K * tile_J * DIM);    // Rows to store B

  if (total_spad_rows > total_spad_size) {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM;      // Rows to store C

  if (total_acc_rows > total_acc_size) {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535) {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }

  char matmul_type_str[][4] = {"OS", "WS", "CPU"};

  // Check if transpose options are correct
  if (((tiled_matmul_type == OS) && (transpose_A || transpose_B)) ||
    (tiled_matmul_type == WS && transpose_A && transpose_B)) {
    printf("Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\n", matmul_type_str[tiled_matmul_type], transpose_A, transpose_B);
    exit(1);
  }

  // Check if full_C options are correct
  if ((tiled_matmul_type == CPU && (full_C || low_D)) ||
      (tiled_matmul_type == OS && low_D)) {
    printf("Not implemented: %s matmul, full_C=%d, low_D=%d\n", matmul_type_str[tiled_matmul_type], full_C, low_D);
  }

  if (act == LAYERNORM || act == SOFTMAX) {
    if (tiled_matmul_type == OS) {
      printf("Not implemented: %s matmul, act=%d\n", matmul_type_str[tiled_matmul_type], act);
    }
    if (tile_J * DIM < dim_J) {
      printf("When doing layernorm or softmax, the full J dimension of the matrix must fit in the accumulator\n");
    }
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU
  if (tiled_matmul_type == OS || tiled_matmul_type == WS) {
    tiled_matmul_outer(custom_num, dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        tile_I, tile_J, tile_K,
        act, scale, bert_scale, repeating_bias,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        (int)tiled_matmul_type);
  } else /*if (tiled_matmul_type == CPU)*/ {
    matmul_cpu(transpose_A, transpose_B, dim_I, dim_J, dim_K,
            A, B, (const acc_t*) D, (elem_t*)C,
            stride_A, stride_B, stride_D, stride_C,
            A_scale_factor, B_scale_factor, D_scale_factor,
            act, scale, bert_scale, repeating_bias);
  }
}

// made
static void multi_tiled_matmul1(int gemmini_num,
                         size_t dim_I, size_t dim_J, size_t dim_K,
                         const elem_t *A, const elem_t *B,
                         const void *D, void *C,
                         size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                         scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                         int act, acc_scale_t scale, acc_scale_t bert_scale,
                         bool repeating_bias,
                         size_t tile_I, size_t tile_J, size_t tile_K,
                         bool transpose_A, bool transpose_B,
                         bool full_C, bool low_D,
                         uint8_t weightA,
                         enum tiled_matmul_type_t tiled_matmul_type)
{

#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0)
  {
    printf("tile_I is non-positive\n");
    exit(1);
  }
  else if (tile_J <= 0)
  {
    printf("tile_J is non-positive\n");
    exit(1);
  }
  else if (tile_K <= 0)
  {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t divided_dim_I = dim_I / gemmini_num + 1;

  const size_t dim_I_padded = (divided_dim_I / DIM + (divided_dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded)
  {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  }
  else if (tile_J * DIM > dim_J_padded)
  {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  }
  else if (tile_K * DIM > dim_K_padded)
  {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t total_spad_size = double_buffered ? BANK_NUM * BANK_ROWS / 2 : BANK_NUM * BANK_ROWS;
  const size_t total_acc_size = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) + // Rows to store A
      (tile_K * tile_J * DIM);  // Rows to store B

  if (total_spad_rows > total_spad_size)
  {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM; // Rows to store C

  if (total_acc_rows > total_acc_size)
  {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535)
  {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }

  char matmul_type_str[][4] = {"OS", "WS", "CPU"};

  // Check if transpose options are correct
  if (((tiled_matmul_type == OS) && (transpose_A || transpose_B)) ||
      (tiled_matmul_type == WS && transpose_A && transpose_B))
  {
    printf("Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\n", matmul_type_str[tiled_matmul_type], transpose_A, transpose_B);
    exit(1);
  }

  // Check if full_C options are correct
  if ((tiled_matmul_type == CPU && (full_C || low_D)) ||
      (tiled_matmul_type == OS && low_D))
  {
    printf("Not implemented: %s matmul, full_C=%d, low_D=%d\n", matmul_type_str[tiled_matmul_type], full_C, low_D);
  }

  if (act == LAYERNORM || act == SOFTMAX)
  {
    if (tiled_matmul_type == OS)
    {
      printf("Not implemented: %s matmul, act=%d\n", matmul_type_str[tiled_matmul_type], act);
    }
    if (tile_J * DIM < dim_J)
    {
      printf("When doing layernorm or softmax, the full J dimension of the matrix must fit in the accumulator\n");
    }
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU
  if (tiled_matmul_type == OS || tiled_matmul_type == WS)
  {
    multi_tiled_matmul_outer1(gemmini_num,
                       dim_I, dim_J, dim_K,
                       A, B, D, C,
                       stride_A, stride_B, stride_D, stride_C,
                       A_scale_factor, B_scale_factor, D_scale_factor,
                       tile_I, tile_J, tile_K,
                       act, scale, bert_scale, repeating_bias,
                       transpose_A, transpose_B,
                       full_C, low_D,
                       weightA,
                       (int)tiled_matmul_type);
  }
  else /*if (tiled_matmul_type == CPU)*/
  {
    matmul_cpu(transpose_A, transpose_B, dim_I, dim_J, dim_K,
               A, B, (const acc_t *)D, (elem_t *)C,
               stride_A, stride_B, stride_D, stride_C,
               A_scale_factor, B_scale_factor, D_scale_factor,
               act, scale, bert_scale, repeating_bias);
  }
}

// made
static void shared_multi_tiled_matmul(int gemmini_list, int tile_id,
                                      size_t sp_addr_start, size_t acc_addr_start,
                                      size_t dim_I, size_t dim_J, size_t dim_K,
                                      const elem_t *A, const elem_t *B,
                                      const void *D, void *C,
                                      size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                      scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                      int act, acc_scale_t scale, acc_scale_t bert_scale,
                                      bool repeating_bias,
                                      size_t tile_I, size_t tile_J, size_t tile_K,
                                      bool transpose_A, bool transpose_B,
                                      bool full_C, bool low_D,
                                      uint8_t weightA,
                                      enum tiled_matmul_type_t tiled_matmul_type)
{

#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0)
  {
    printf("tile_I is non-positive\n");
    exit(1);
  }
  else if (tile_J <= 0)
  {
    printf("tile_J is non-positive\n");
    exit(1);
  }
  else if (tile_K <= 0)
  {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded)
  {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  }
  else if (tile_J * DIM > dim_J_padded)
  {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  }
  else if (tile_K * DIM > dim_K_padded)
  {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t total_spad_size = double_buffered ? BANK_NUM * BANK_ROWS / 2 : BANK_NUM * BANK_ROWS;
  const size_t total_acc_size = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) + // Rows to store A
      (tile_K * tile_J * DIM);  // Rows to store B

  if (total_spad_rows > total_spad_size)
  {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM; // Rows to store C

  if (total_acc_rows > total_acc_size)
  {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535)
  {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }

  char matmul_type_str[][4] = {"OS", "WS", "CPU"};

  // Check if transpose options are correct
  if (((tiled_matmul_type == OS) && (transpose_A || transpose_B)) ||
      (tiled_matmul_type == WS && transpose_A && transpose_B))
  {
    printf("Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\n", matmul_type_str[tiled_matmul_type], transpose_A, transpose_B);
    exit(1);
  }

  // Check if full_C options are correct
  if ((tiled_matmul_type == CPU && (full_C || low_D)) ||
      (tiled_matmul_type == OS && low_D))
  {
    printf("Not implemented: %s matmul, full_C=%d, low_D=%d\n", matmul_type_str[tiled_matmul_type], full_C, low_D);
  }

  if (act == LAYERNORM || act == SOFTMAX)
  {
    if (tiled_matmul_type == OS)
    {
      printf("Not implemented: %s matmul, act=%d\n", matmul_type_str[tiled_matmul_type], act);
    }
    if (tile_J * DIM < dim_J)
    {
      printf("When doing layernorm or softmax, the full J dimension of the matrix must fit in the accumulator\n");
    }
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU
  if (tiled_matmul_type == OS || tiled_matmul_type == WS)
  {
    shared_multi_tiled_matmul_outer(gemmini_list, tile_id,
                                    sp_addr_start, acc_addr_start,
                                    dim_I, dim_J, dim_K,
                                    A, B, D, C,
                                    stride_A, stride_B, stride_D, stride_C,
                                    A_scale_factor, B_scale_factor, D_scale_factor,
                                    tile_I, tile_J, tile_K,
                                    act, scale, bert_scale, repeating_bias,
                                    transpose_A, transpose_B,
                                    full_C, low_D,
                                    weightA,
                                    (int)tiled_matmul_type);
  }
  else /*if (tiled_matmul_type == CPU)*/
  {
    matmul_cpu(transpose_A, transpose_B, dim_I, dim_J, dim_K,
               A, B, (const acc_t *)D, (elem_t *)C,
               stride_A, stride_B, stride_D, stride_C,
               A_scale_factor, B_scale_factor, D_scale_factor,
               act, scale, bert_scale, repeating_bias);
  }
}

// made
static void shared_multi_tiled_matmul_test(int gemmini_list, int tile_id,
                                      size_t sp_addr_start, size_t acc_addr_start,
                                      size_t sp_addr_range, size_t acc_addr_range,
                                      size_t dim_I, size_t dim_J, size_t dim_K,
                                      const elem_t *A, const elem_t *B,
                                      const void *D, void *C,
                                      size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                      scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                      int act, acc_scale_t scale, acc_scale_t bert_scale,
                                      bool repeating_bias,
                                      size_t tile_I, size_t tile_J, size_t tile_K,
                                      bool transpose_A, bool transpose_B,
                                      bool full_C, bool low_D,
                                      uint8_t weightA,
                                      enum tiled_matmul_type_t tiled_matmul_type)
{

#ifdef GEMMINI_ASSERTIONS
  // Make sure that the tiling factors make sense
  if (tile_I <= 0)
  {
    printf("tile_I is non-positive\n");
    exit(1);
  }
  else if (tile_J <= 0)
  {
    printf("tile_J is non-positive\n");
    exit(1);
  }
  else if (tile_K <= 0)
  {
    printf("tile_K is non-positive\n");
    exit(1);
  }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  if (tile_I * DIM > dim_I_padded)
  {
    printf("tile_I is too large (tile_I * DIM > dim_I_padded)\n");
    exit(1);
  }
  else if (tile_J * DIM > dim_J_padded)
  {
    printf("tile_J is too large (tile_J * DIM > dim_J_padded)\n");
    exit(1);
  }
  else if (tile_K * DIM > dim_K_padded)
  {
    printf("tile_K is too large (tile_K * DIM > dim_K_padded)\n");
    exit(1);
  }

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t total_spad_size = double_buffered ? sp_addr_range / 2 : sp_addr_range;
  const size_t total_acc_size = double_buffered ? acc_addr_range / 2 : acc_addr_range;

  const size_t total_spad_rows =
      (tile_I * tile_K * DIM) + // Rows to store A
      (tile_K * tile_J * DIM);  // Rows to store B

  if (total_spad_rows > total_spad_size)
  {
    printf("Not enough space in scratchpad to store A and B matrices\n");
    exit(1);
  }

  const size_t total_acc_rows =
      tile_I * tile_J * DIM; // Rows to store C

  if (total_acc_rows > total_acc_size)
  {
    printf("Not enough space in accumulator to store C\n");
    exit(1);
  }

  if (tile_I > 65535 || tile_J > 65535 || tile_K > 65535)
  {
    printf("I, J, and K tiling factors must be less than 65535, to fit within the bounds of the LOOP_WS function");
    exit(1);
  }

  char matmul_type_str[][4] = {"OS", "WS", "CPU"};

  // Check if transpose options are correct
  if (((tiled_matmul_type == OS) && (transpose_A || transpose_B)) ||
      (tiled_matmul_type == WS && transpose_A && transpose_B))
  {
    printf("Not implemented: %s matmul, a_transpose=%d, b_transpose=%d\n", matmul_type_str[tiled_matmul_type], transpose_A, transpose_B);
    exit(1);
  }

  // Check if full_C options are correct
  if ((tiled_matmul_type == CPU && (full_C || low_D)) ||
      (tiled_matmul_type == OS && low_D))
  {
    printf("Not implemented: %s matmul, full_C=%d, low_D=%d\n", matmul_type_str[tiled_matmul_type], full_C, low_D);
  }

  if (act == LAYERNORM || act == SOFTMAX)
  {
    if (tiled_matmul_type == OS)
    {
      printf("Not implemented: %s matmul, act=%d\n", matmul_type_str[tiled_matmul_type], act);
    }
    if (tile_J * DIM < dim_J)
    {
      printf("When doing layernorm or softmax, the full J dimension of the matrix must fit in the accumulator\n");
    }
  }
#endif

  // Run a tiled matrix multiplication on either Gemmini or the CPU
  if (tiled_matmul_type == OS || tiled_matmul_type == WS)
  {
    shared_multi_tiled_matmul_outer_test(gemmini_list, tile_id,
                                    sp_addr_start, acc_addr_start,
                                    sp_addr_range, acc_addr_range,
                                    dim_I, dim_J, dim_K,
                                    A, B, D, C,
                                    stride_A, stride_B, stride_D, stride_C,
                                    A_scale_factor, B_scale_factor, D_scale_factor,
                                    tile_I, tile_J, tile_K,
                                    act, scale, bert_scale, repeating_bias,
                                    transpose_A, transpose_B,
                                    full_C, low_D,
                                    weightA,
                                    (int)tiled_matmul_type);
  }
  else /*if (tiled_matmul_type == CPU)*/
  {
    matmul_cpu(transpose_A, transpose_B, dim_I, dim_J, dim_K,
               A, B, (const acc_t *)D, (elem_t *)C,
               stride_A, stride_B, stride_D, stride_C,
               A_scale_factor, B_scale_factor, D_scale_factor,
               act, scale, bert_scale, repeating_bias);
  }
}

static size_t tiled_matmul_total_spad_rows(size_t I, size_t J, size_t K) {
  return (I * K + K * J) * DIM;
}

// made
static size_t tiled_matmul_A_spad_rows(size_t I, size_t J, size_t K)
{
  return (I * K) * DIM;
}

// made
static size_t tiled_matmul_B_spad_rows(size_t I, size_t J, size_t K)
{
  return (K * J) * DIM;
}

static size_t tiled_matmul_total_acc_rows(size_t I, size_t J) {
  return (I * J) * DIM;
}

// This function runs a tiled matrix multiplication, with automatically
// calculated tiling factors
static void tiled_matmul_auto(int custom_num, size_t dim_I, size_t dim_J, size_t dim_K,
        const elem_t* A, const elem_t* B,
        const void * D, void * C,
        size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
        scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
        int act, acc_scale_t scale, acc_scale_t bert_scale,
        bool repeating_bias,
        bool transpose_A, bool transpose_B,
        bool full_C, bool low_D,
        uint8_t weightA,
        enum tiled_matmul_type_t tiled_matmul_type) {

#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

    // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

    const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
    const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
    const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

    const bool double_buffered = tiled_matmul_type == WS;

    const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 :
      BANK_NUM * BANK_ROWS;
    const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

    size_t tile_I, tile_J, tile_K;

    if (act == LAYERNORM || act == SOFTMAX) {
       tile_I = 1;
       tile_J = dim_J_padded/DIM;
       tile_K = 1;
    } else if (double_buffered) {
       tile_I = dim_I_padded/DIM < db_max_tile_i_j ? dim_I_padded/DIM : db_max_tile_i_j;
       tile_J = dim_J_padded/DIM < db_max_tile_i_j ? dim_J_padded/DIM : db_max_tile_i_j;
       tile_K = dim_K_padded/DIM < db_max_tile_k ? dim_K_padded/DIM : db_max_tile_k;
    } else {
       tile_I = dim_I_padded/DIM < max_tile_i_j ? dim_I_padded/DIM : max_tile_i_j;
       tile_J = dim_J_padded/DIM < max_tile_i_j ? dim_J_padded/DIM : max_tile_i_j;
       tile_K = dim_K_padded/DIM < max_tile_k ? dim_K_padded/DIM : max_tile_k;
    }

    // Fill scratchpad as much as possible
    while (true) {
      bool increased = false;

      if (tiled_matmul_total_spad_rows(tile_I, tile_J+1, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I, tile_J+1) <= max_acc_rows &&
          (tile_J+1) * DIM <= dim_J_padded) {
        tile_J++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tile_I+1, tile_J, tile_K) <= max_spad_rows &&
          tiled_matmul_total_acc_rows(tile_I+1, tile_J) <= max_acc_rows &&
          (tile_I+1) * DIM <= dim_I_padded) {
        tile_I++;
        increased = true;
      }

      if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K+1) <= max_spad_rows &&
          (tile_K+1) * DIM <= dim_K_padded) {
        tile_K++;
        increased = true;
      }

      if (!increased)
        break;
    }

#ifdef PRINT_TILE
#if PRINT_TILE
    const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
    const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

    printf("tile_I: %d\n", tile_I);
    printf("tile_J: %d\n", tile_J);
    printf("tile_K: %d\n\n", tile_K);

    printf("spad_rows: %d\n", spad_rows);
    printf("acc_rows: %d\n\n", acc_rows);

    printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
    printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

    // exit(EXIT_SUCCESS);
#endif
#endif

    tiled_matmul(custom_num, dim_I, dim_J, dim_K,
        A, B, D, C,
        stride_A, stride_B, stride_D, stride_C,
        A_scale_factor, B_scale_factor, D_scale_factor,
        act, scale, bert_scale, repeating_bias,
        tile_I, tile_J, tile_K,
        transpose_A, transpose_B,
        full_C, low_D,
        weightA,
        tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

// made
static void multi_tiled_matmul_auto1(int gemmini_num,
                                          size_t dim_I, size_t dim_J, size_t dim_K,
                                          const elem_t *A, const elem_t *B,
                                          const void *D, void *C,
                                          size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                          scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                          int act, acc_scale_t scale, acc_scale_t bert_scale,
                                          bool repeating_bias,
                                          bool transpose_A, bool transpose_B,
                                          bool full_C, bool low_D,
                                          uint8_t weightA,
                                          enum tiled_matmul_type_t tiled_matmul_type)
{

#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

  // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

  if (gemmini_num == 0) {
    printf("We need at least one gemmini!\n");
    exit(1);
  }
  else if (gemmini_num > 4) {
    printf("We can have maximum four gemminis!\n");
    exit(1);
  }

  const size_t divided_dim_I = dim_I / gemmini_num + 1;

  const size_t dim_I_padded = (divided_dim_I / DIM + (divided_dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 : BANK_NUM * BANK_ROWS;
  const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

  size_t tile_I, tile_J, tile_K;

  if (act == LAYERNORM || act == SOFTMAX)
  {
    tile_I = 1;
    tile_J = dim_J_padded / DIM;
    tile_K = 1;
  }
  else if (double_buffered)
  {
    tile_I = dim_I_padded / DIM < db_max_tile_i_j ? dim_I_padded / DIM : db_max_tile_i_j;
    tile_J = dim_J_padded / DIM < db_max_tile_i_j ? dim_J_padded / DIM : db_max_tile_i_j;
    tile_K = dim_K_padded / DIM < db_max_tile_k ? dim_K_padded / DIM : db_max_tile_k;
  }
  else
  {
    tile_I = dim_I_padded / DIM < max_tile_i_j ? dim_I_padded / DIM : max_tile_i_j;
    tile_J = dim_J_padded / DIM < max_tile_i_j ? dim_J_padded / DIM : max_tile_i_j;
    tile_K = dim_K_padded / DIM < max_tile_k ? dim_K_padded / DIM : max_tile_k;
  }

  // Fill scratchpad as much as possible
  while (true)
  {
    bool increased = false;

    if (tiled_matmul_total_spad_rows(tile_I, tile_J + 1, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I, tile_J + 1) <= max_acc_rows &&
        (tile_J + 1) * DIM <= dim_J_padded)
    {
      tile_J++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tile_I + 1, tile_J, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I + 1, tile_J) <= max_acc_rows &&
        (tile_I + 1) * DIM <= dim_I_padded)
    {
      tile_I++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K + 1) <= max_spad_rows &&
        (tile_K + 1) * DIM <= dim_K_padded)
    {
      tile_K++;
      increased = true;
    }

    if (!increased)
      break;
  }

#ifdef PRINT_TILE
#if PRINT_TILE
  const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
  const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

  printf("tile_I: %d\n", tile_I);
  printf("tile_J: %d\n", tile_J);
  printf("tile_K: %d\n\n", tile_K);

  printf("spad_rows: %d\n", spad_rows);
  printf("acc_rows: %d\n\n", acc_rows);

  printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

  exit(EXIT_SUCCESS);
#endif
#endif

  multi_tiled_matmul1(gemmini_num,
               dim_I, dim_J, dim_K,
               A, B, D, C,
               stride_A, stride_B, stride_D, stride_C,
               A_scale_factor, B_scale_factor, D_scale_factor,
               act, scale, bert_scale, repeating_bias,
               tile_I, tile_J, tile_K,
               transpose_A, transpose_B,
               full_C, low_D,
               weightA,
               tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

// made
static void shared_multi_tiled_matmul_auto(int gemmini_list, int tile_id,
                                           size_t sp_addr_start, size_t acc_addr_start,
                                           size_t dim_I, size_t dim_J, size_t dim_K,
                                           const elem_t *A, const elem_t *B,
                                           const void *D, void *C,
                                           size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                           scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                           int act, acc_scale_t scale, acc_scale_t bert_scale,
                                           bool repeating_bias,
                                           bool transpose_A, bool transpose_B,
                                           bool full_C, bool low_D,
                                           uint8_t weightA,
                                           enum tiled_matmul_type_t tiled_matmul_type)
{

#define partition_rows (BANK_NUM * BANK_ROWS / 2)
#define mats_in_partition (partition_rows / DIM)
#define mats_in_acc (ACC_ROWS / DIM)
#define max_tile_i_j ((size_t)sqrt(mats_in_acc))
#define max_tile_k (mats_in_partition / max_tile_i_j)

  // "db_" means "double-buffered"
#define db_partition_rows ((BANK_NUM * BANK_ROWS / 2) / 2)
#define db_mats_in_partition (db_partition_rows / DIM)
#define db_mats_in_acc ((ACC_ROWS / 2) / DIM)
#define db_max_tile_i_j ((size_t)sqrt(db_mats_in_acc))
#define db_max_tile_k (db_mats_in_partition / db_max_tile_i_j)

  // if (gemmini_num == 0)
  // {
  //   printf("We need at least one gemmini!\n");
  //   exit(1);
  // }
  // else if (gemmini_num > 4)
  // {
  //   printf("We can have maximum four gemminis!\n");
  //   exit(1);
  // }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t max_spad_rows = double_buffered ? BANK_NUM * BANK_ROWS / 2 : BANK_NUM * BANK_ROWS;
  const size_t max_acc_rows = double_buffered ? ACC_ROWS / 2 : ACC_ROWS;

  size_t tile_I, tile_J, tile_K;

  if (act == LAYERNORM || act == SOFTMAX)
  {
    tile_I = 1;
    tile_J = dim_J_padded / DIM;
    tile_K = 1;
  }
  else if (double_buffered)
  {
    tile_I = dim_I_padded / DIM < db_max_tile_i_j ? dim_I_padded / DIM : db_max_tile_i_j;
    tile_J = dim_J_padded / DIM < db_max_tile_i_j ? dim_J_padded / DIM : db_max_tile_i_j;
    tile_K = dim_K_padded / DIM < db_max_tile_k ? dim_K_padded / DIM : db_max_tile_k;
  }
  else
  {
    tile_I = dim_I_padded / DIM < max_tile_i_j ? dim_I_padded / DIM : max_tile_i_j;
    tile_J = dim_J_padded / DIM < max_tile_i_j ? dim_J_padded / DIM : max_tile_i_j;
    tile_K = dim_K_padded / DIM < max_tile_k ? dim_K_padded / DIM : max_tile_k;
  }

  // Fill scratchpad as much as possible
  while (true)
  {
    bool increased = false;

    if (tiled_matmul_total_spad_rows(tile_I, tile_J + 1, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I, tile_J + 1) <= max_acc_rows &&
        (tile_J + 1) * DIM <= dim_J_padded)
    {
      tile_J++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tile_I + 1, tile_J, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I + 1, tile_J) <= max_acc_rows &&
        (tile_I + 1) * DIM <= dim_I_padded)
    {
      tile_I++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K + 1) <= max_spad_rows &&
        (tile_K + 1) * DIM <= dim_K_padded)
    {
      tile_K++;
      increased = true;
    }

    if (!increased)
      break;
  }

#ifdef PRINT_TILE
#if PRINT_TILE
  const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
  const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

  printf("tile_I: %d\n", tile_I);
  printf("tile_J: %d\n", tile_J);
  printf("tile_K: %d\n\n", tile_K);

  printf("spad_rows: %d\n", spad_rows);
  printf("acc_rows: %d\n\n", acc_rows);

  printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

  exit(EXIT_SUCCESS);
#endif
#endif

  shared_multi_tiled_matmul(gemmini_list, tile_id,
                            sp_addr_start, acc_addr_start,
                            dim_I, dim_J, dim_K,
                            A, B, D, C,
                            stride_A, stride_B, stride_D, stride_C,
                            A_scale_factor, B_scale_factor, D_scale_factor,
                            act, scale, bert_scale, repeating_bias,
                            tile_I, tile_J, tile_K,
                            transpose_A, transpose_B,
                            full_C, low_D,
                            weightA,
                            tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

// made
static void shared_multi_tiled_matmul_auto_test_2(int gemmini_list, int tile_id,
                                           size_t sp_addr_start, size_t acc_addr_start,
                                           size_t sp_addr_range, size_t acc_addr_range,
                                           size_t dim_I, size_t dim_J, size_t dim_K,
                                           const elem_t *A, const elem_t *B,
                                           const void *D, void *C,
                                           size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                           scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                           int act, acc_scale_t scale, acc_scale_t bert_scale,
                                           bool repeating_bias,
                                           bool transpose_A, bool transpose_B,
                                           bool full_C, bool low_D,
                                           uint8_t weightA,
                                           enum tiled_matmul_type_t tiled_matmul_type)
{
  if (sp_addr_start % 2 != 0 || acc_addr_start % 2 != 0 || 
      sp_addr_range % 2 != 0 || acc_addr_range % 2 != 0) {
    printf("sp_addr_start and acc_addr_start must be even numbers\n");
    exit(1);
  }

  int gemmini_num = 0;
  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((gemmini_list >> i) & 1)
      gemmini_num++;
  }

  size_t partition_rows = (sp_addr_range / 2);
  size_t mats_in_partition = (partition_rows / DIM);
  size_t mats_in_acc = (acc_addr_range / DIM);
  size_t max_tile_i_j = ((size_t)sqrt(mats_in_acc));
  size_t max_tile_k = (mats_in_partition / max_tile_i_j);

  // "db_" means "double-buffered"
  size_t db_partition_rows_1 = ((sp_addr_range / 2) / 2);
  size_t db_mats_in_partition_1 = (db_partition_rows_1 / DIM);
  size_t db_mats_in_acc_1 = ((acc_addr_range / 2) / DIM);
  size_t db_max_tile_i_j_1 = ((size_t)sqrt(db_mats_in_acc_1));
  size_t db_max_tile_k_1 = (db_mats_in_partition_1 / db_max_tile_i_j_1);

  // if (gemmini_num == 0)
  // {
  //   printf("We need at least one gemmini!\n");
  //   exit(1);
  // }
  // else if (gemmini_num > 4)
  // {
  //   printf("We can have maximum four gemminis!\n");
  //   exit(1);
  // }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t max_spad_rows = double_buffered ? sp_addr_range / 2 : sp_addr_range;
  const size_t max_acc_rows = double_buffered ? acc_addr_range / 2 : acc_addr_range;

  size_t tile_I, tile_J, tile_K;

  if (act == LAYERNORM || act == SOFTMAX)
  {
    tile_I = 1;
    tile_J = dim_J_padded / DIM;
    tile_K = 1;
  }
  else if (double_buffered)
  {
    tile_I = dim_I_padded / DIM < db_max_tile_i_j_1 ? dim_I_padded / DIM : db_max_tile_i_j_1;
    tile_J = dim_J_padded / DIM < db_max_tile_i_j_1 ? dim_J_padded / DIM : db_max_tile_i_j_1;
    tile_K = dim_K_padded / DIM < db_max_tile_k_1 ? dim_K_padded / DIM : db_max_tile_k_1;
  }
  else
  {
    tile_I = dim_I_padded / DIM < max_tile_i_j ? dim_I_padded / DIM : max_tile_i_j;
    tile_J = dim_J_padded / DIM < max_tile_i_j ? dim_J_padded / DIM : max_tile_i_j;
    tile_K = dim_K_padded / DIM < max_tile_k ? dim_K_padded / DIM : max_tile_k;
  }

  tile_K = (tile_K / gemmini_num) * gemmini_num;
  tile_I = (tile_I / gemmini_num) * gemmini_num;
  
  while (true)
  {
    bool increasing = false;

    bool tile_I_can_increase = tile_I < gemmini_num && (tile_I + 1) * DIM <= dim_I_padded;
    bool tile_K_can_increase = tile_K < gemmini_num && (tile_K + 1) * DIM <= dim_K_padded;
    bool tile_I_can_decrease = tile_I > gemmini_num;
    bool tile_K_can_decrease = tile_K > gemmini_num;
    bool tile_J_can_decrease = tile_J > 1;

    if (tile_I_can_increase &&
        tiled_matmul_total_spad_rows(tile_I + 1, tile_J, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I + 1, tile_J) <= max_acc_rows)
    {
      tile_I = tile_I + 1;
      increasing = true;
    }

    if (tile_K_can_increase &&
        tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K + 1) <= max_spad_rows)
    {
      tile_K = tile_K + 1;
      increasing = true;
    }

    if (tile_J > 1 && (tile_I_can_increase || tile_K_can_increase))
    {
      tile_J = tile_J - 1;
      increasing = true;
    }

    if (!increasing)
      break;
  }

  // Fill scratchpad as much as possible
  while (true)
  {
    bool increased = false;

    if (tiled_matmul_total_spad_rows(tile_I, tile_J + 1, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I, tile_J + 1) <= max_acc_rows &&
        (tile_J + 1) * DIM <= dim_J_padded)
    {
      tile_J++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tile_I + gemmini_num, tile_J, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I + gemmini_num, tile_J) <= max_acc_rows &&
        (tile_I + gemmini_num) * DIM <= dim_I_padded)
    {
      tile_I = tile_I + gemmini_num;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K + gemmini_num) <= max_spad_rows &&
        (tile_K + gemmini_num) * DIM <= dim_K_padded)
    {
      tile_K = tile_K + gemmini_num;
      increased = true;
    }

    if (!increased)
      break;
  }

  // tile_I = 3;
  // tile_J = 4;
  // tile_K = 9;
  // printf("tile_I: %d, tile_J: %d, tile_K: %d\n", tile_I, tile_J, tile_K);

#ifdef PRINT_TILE
#if PRINT_TILE
  const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
  const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

  printf("tile_I: %d\n", tile_I);
  printf("tile_J: %d\n", tile_J);
  printf("tile_K: %d\n\n", tile_K);

  printf("spad_rows: %d\n", spad_rows);
  printf("acc_rows: %d\n\n", acc_rows);

  printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

  exit(EXIT_SUCCESS);
#endif
#endif

  shared_multi_tiled_matmul_test(gemmini_list, tile_id,
                            sp_addr_start, acc_addr_start,
                            sp_addr_range, acc_addr_range,
                            dim_I, dim_J, dim_K,
                            A, B, D, C,
                            stride_A, stride_B, stride_D, stride_C,
                            A_scale_factor, B_scale_factor, D_scale_factor,
                            act, scale, bert_scale, repeating_bias,
                            tile_I, tile_J, tile_K,
                            transpose_A, transpose_B,
                            full_C, low_D,
                            weightA,
                            tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

// made
static void shared_multi_tiled_matmul_auto_test(int gemmini_list, int tile_id,
                                                size_t sp_addr_start, size_t acc_addr_start,
                                                size_t sp_addr_range, size_t acc_addr_range,
                                                size_t dim_I, size_t dim_J, size_t dim_K,
                                                const elem_t *A, const elem_t *B,
                                                const void *D, void *C,
                                                size_t stride_A, size_t stride_B, size_t stride_D, size_t stride_C,
                                                scale_t A_scale_factor, scale_t B_scale_factor, scale_acc_t D_scale_factor,
                                                int act, acc_scale_t scale, acc_scale_t bert_scale,
                                                bool repeating_bias,
                                                bool transpose_A, bool transpose_B,
                                                bool full_C, bool low_D,
                                                uint8_t weightA,
                                                enum tiled_matmul_type_t tiled_matmul_type)
{
  if (sp_addr_start % 2 != 0 || acc_addr_start % 2 != 0 ||
      sp_addr_range % 2 != 0 || acc_addr_range % 2 != 0)
  {
    printf("sp_addr_start and acc_addr_start must be even numbers\n");
    exit(1);
  }

  size_t partition_rows = (sp_addr_range / 2);
  size_t mats_in_partition = (partition_rows / DIM);
  size_t mats_in_acc = (acc_addr_range / DIM);
  size_t max_tile_i_j = ((size_t)sqrt(mats_in_acc));
  size_t max_tile_k = (mats_in_partition / max_tile_i_j);

  // "db_" means "double-buffered"
  size_t db_partition_rows_1 = ((sp_addr_range / 2) / 2);
  size_t db_mats_in_partition_1 = (db_partition_rows_1 / DIM);
  size_t db_mats_in_acc_1 = ((acc_addr_range / 2) / DIM);
  size_t db_max_tile_i_j_1 = ((size_t)sqrt(db_mats_in_acc_1));
  size_t db_max_tile_k_1 = (db_mats_in_partition_1 / db_max_tile_i_j_1);

  // if (gemmini_num == 0)
  // {
  //   printf("We need at least one gemmini!\n");
  //   exit(1);
  // }
  // else if (gemmini_num > 4)
  // {
  //   printf("We can have maximum four gemminis!\n");
  //   exit(1);
  // }

  const size_t dim_I_padded = (dim_I / DIM + (dim_I % DIM != 0)) * DIM;
  const size_t dim_J_padded = (dim_J / DIM + (dim_J % DIM != 0)) * DIM;
  const size_t dim_K_padded = (dim_K / DIM + (dim_K % DIM != 0)) * DIM;

  const bool double_buffered = tiled_matmul_type == WS;

  const size_t max_spad_rows = double_buffered ? sp_addr_range / 2 : sp_addr_range;
  const size_t max_acc_rows = double_buffered ? acc_addr_range / 2 : acc_addr_range;

  size_t tile_I, tile_J, tile_K;

  if (act == LAYERNORM || act == SOFTMAX)
  {
    tile_I = 1;
    tile_J = dim_J_padded / DIM;
    tile_K = 1;
  }
  else if (double_buffered)
  {
    tile_I = dim_I_padded / DIM < db_max_tile_i_j_1 ? dim_I_padded / DIM : db_max_tile_i_j_1;
    tile_J = dim_J_padded / DIM < db_max_tile_i_j_1 ? dim_J_padded / DIM : db_max_tile_i_j_1;
    tile_K = dim_K_padded / DIM < db_max_tile_k_1 ? dim_K_padded / DIM : db_max_tile_k_1;
  }
  else
  {
    tile_I = dim_I_padded / DIM < max_tile_i_j ? dim_I_padded / DIM : max_tile_i_j;
    tile_J = dim_J_padded / DIM < max_tile_i_j ? dim_J_padded / DIM : max_tile_i_j;
    tile_K = dim_K_padded / DIM < max_tile_k ? dim_K_padded / DIM : max_tile_k;
  }

  // Fill scratchpad as much as possible
  while (true)
  {
    bool increased = false;

    if (tiled_matmul_total_spad_rows(tile_I, tile_J + 1, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I, tile_J + 1) <= max_acc_rows &&
        (tile_J + 1) * DIM <= dim_J_padded)
    {
      tile_J++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tile_I + 1, tile_J, tile_K) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tile_I + 1, tile_J) <= max_acc_rows &&
        (tile_I + 1) * DIM <= dim_I_padded)
    {
      tile_I++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K + 1) <= max_spad_rows &&
        (tile_K + 1) * DIM <= dim_K_padded)
    {
      tile_K++;
      increased = true;
    }

    if (!increased)
      break;
  }

  // tile_I = 3;
  // tile_J = 4;
  // tile_K = 9;
  // printf("tile_I: %d, tile_J: %d, tile_K: %d\n", tile_I, tile_J, tile_K);

#ifdef PRINT_TILE
#if PRINT_TILE
  const int spad_rows = tiled_matmul_total_spad_rows(tile_I, tile_J, tile_K);
  const int acc_rows = tiled_matmul_total_acc_rows(tile_I, tile_J);

  printf("tile_I: %d\n", tile_I);
  printf("tile_J: %d\n", tile_J);
  printf("tile_K: %d\n\n", tile_K);

  printf("spad_rows: %d\n", spad_rows);
  printf("acc_rows: %d\n\n", acc_rows);

  printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

  exit(EXIT_SUCCESS);
#endif
#endif

  shared_multi_tiled_matmul_test(gemmini_list, tile_id,
                                 sp_addr_start, acc_addr_start,
                                 sp_addr_range, acc_addr_range,
                                 dim_I, dim_J, dim_K,
                                 A, B, D, C,
                                 stride_A, stride_B, stride_D, stride_C,
                                 A_scale_factor, B_scale_factor, D_scale_factor,
                                 act, scale, bert_scale, repeating_bias,
                                 tile_I, tile_J, tile_K,
                                 transpose_A, transpose_B,
                                 full_C, low_D,
                                 weightA,
                                 tiled_matmul_type);

#undef partition_rows
#undef mats_in_partition
#undef mats_in_acc
#undef max_tile_i_j
#undef max_tile_k
}

static void sp_tiled_conv(
        int custom_num, 
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int pool_out_row_dim, int pool_out_col_dim,

        int stride, int padding, int kernel_dim, int kernel_dilation,

        int pool_size, int pool_stride, int pool_padding,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        int lpad, int rpad, int upad, int dpad,
        int plpad, int prpad, int pupad, int pdpad,

        const elem_t * input,
        const elem_t * weights,
        elem_t * output,
        const acc_t * bias,

        int act, acc_scale_t scale,

        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        bool no_bias, bool no_pool, bool downsample, bool input_dilated,
        bool dw) {

  // When dw convs are true, we assume that kchs and ochs are 1
  if (dw) { kchs = 1; pochs = 1; }

  const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
  const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
  const int ochs = pochs;

  // Calculate image dimensions
  // Note: "irows" and "icols" includes padding
  const int dilated_krows = krows + (kernel_dilation - 1)*(krows - 1);
  const int dilated_kcols = kcols + (kernel_dilation - 1)*(kcols - 1);
  int irows = orows * stride + dilated_krows - 1;
  int icols = ocols * stride + dilated_kcols - 1;
  int irows_unpadded = irows - upad - dpad;
  int icols_unpadded = icols - lpad - rpad;
  const int ichs = kchs;

#define UNDILATED(x) ((input_dilated) ? (((x)+1)/2) : (x))

  if (input_dilated) {
    irows_unpadded = (irows_unpadded+1)/2;
    icols_unpadded = (icols_unpadded+1)/2;

    irows = irows_unpadded + UNDILATED(upad) + UNDILATED(dpad);
    icols = icols_unpadded + UNDILATED(lpad) + UNDILATED(rpad);
  }

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
  const bool transposed = trans_output_1203 || trans_input_3120 ||
      trans_weight_1203 || trans_weight_0132;
  int max_pixels_per_row = transposed || wrot180 || downsample ||
      input_dilated || kernel_dilation > 1 ||
      ichs > DIM ? 1 : DIM/ichs;
  if (max_pixels_per_row > kcols) max_pixels_per_row = kcols;
#else
  const int max_pixels_per_row = 1;
#endif

  // Calculate spad address offsets
  // const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
  // const int in_channels_per_bank = kchs / DIM + (kchs % DIM != 0);
  // const int B_rows = trans_weight_0132 ?
  //   in_channels_per_bank * kcols * krows * ochs :
  //   out_channels_per_bank * kcols * krows * kchs;

  // static uint32_t D_sp_addr_row = 0;
  // static uint32_t C_sp_addr_row = 0;

  // const uint32_t A_sp_addr_start = 0;
  // const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - B_rows;
  // const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
  // const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;

  // if (bias != 0) {
  //   D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  // }

  // if (output != 0) {
  //   C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  // }

  if (in_row_dim == in_col_dim && out_row_dim == out_col_dim && pool_out_row_dim == pool_out_col_dim) {
    switch (custom_num) {
      case 0:
        gemmini_loop_conv_ws(custom0, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
        break;
      case 1:
        gemmini_loop_conv_ws(custom1, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
        break;
      case 2:
        gemmini_loop_conv_ws(custom2, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
        break;
      case 3:
        gemmini_loop_conv_ws(custom3, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
        break;
    }
    
    return;
  }

//   if (!no_pool) {
//     printf("Pooling with rectangular convolutions is currently not supported.\n");
//     exit(1);
//   }

//   // Only rectangular convolutions will use the following C code

//   // mvin bias
//   if (bias != NULL) {
//     // TODO we probably don't need quite this many nested loops for this part

//     const int max_ochs_per_mvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs :
//         MAX_BLOCK_LEN_ACC * DIM;

//     gemmini_extended4_config_ld(custom_num, 0, MVIN_SCALE_IDENTITY, false, batches * orows * ocols, 2);

//     for (int b = 0; b < batches; b++)
//       for (int orow = 0; orow < orows; orow++)
//         for (int ocol = 0; ocol < ocols; ocol += DIM) {
//           const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

//           for (int och = 0; och < ochs; och += max_ochs_per_mvin) {
//             const int J = ochs - och > max_ochs_per_mvin ? max_ochs_per_mvin : ochs - och;

//             const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

//             const acc_t * bias_dram_addr = no_bias ? NULL : bias + och;

//             gemmini_extended_mvin3(custom_num, bias_dram_addr,
//                     D_sp_addr,
//                     J, I);
//           }
//         }
//   }

//   // mvin input
//   {
//     int max_chs_per_mvin = ichs < MAX_BLOCK_LEN * DIM ? ichs :
//       MAX_BLOCK_LEN * DIM;
//     if (trans_input_3120) {
//       max_chs_per_mvin = batches < MAX_BLOCK_LEN * DIM ? batches :
//         MAX_BLOCK_LEN * DIM;
//     }

//     const int dram_stride = trans_input_3120 ?
//       batch_size * sizeof(elem_t) :
//       in_channels * sizeof(elem_t);

//     const int spad_stride = trans_input_3120 ?
//       ichs * (irows >> downsample) * (icols >> downsample) :
//       batches * (irows >> downsample) * (icols >> downsample);

//     gemmini_extended5_config_ld(custom_num, dram_stride << downsample, MVIN_SCALE_IDENTITY, false, spad_stride, max_pixels_per_row, 0);

//     const int b_it = trans_input_3120 ? max_chs_per_mvin : 1;
//     const int ich_it = trans_input_3120 ? 1 : max_chs_per_mvin;

//     for (int b = 0; b < batches; b += b_it)
//       for (int irow = -UNDILATED(upad); irow < irows_unpadded + UNDILATED(dpad); irow += 1 + downsample) {
//         const int irow_padded = irow + UNDILATED(upad);

//         for (int icol = -UNDILATED(lpad); icol < icols_unpadded + UNDILATED(rpad);) {
//           // TODO There might be some unnecessary mvins here at the edge of the image

//           int I = icols_unpadded - icol > (DIM << downsample) ?
//             (DIM << downsample) : icols_unpadded - icol;

//           if (icol < 0) {
//             I = -icol > DIM ? DIM : -icol;
//           } else if (icol >= icols_unpadded) {
//             I = icols_unpadded + UNDILATED(rpad) - icol > DIM ? DIM : icols_unpadded + UNDILATED(rpad) - icol;
//           }

//           const int icol_padded = icol + UNDILATED(lpad);

//           for (int ich = 0; ich < ichs; ich += ich_it) {
//             int K = ichs - ich > max_chs_per_mvin ?
//               max_chs_per_mvin : ichs - ich;
//             if (trans_input_3120) {
//               K = batches - b > max_chs_per_mvin ?
//                 max_chs_per_mvin : batches - b;
//             }

// #define DS(x) ((x) >> (downsample))

//             uint32_t A_sp_addr = A_sp_addr_start + (ich / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
//             if (trans_input_3120) {
//               A_sp_addr = A_sp_addr_start + (b / DIM) * ichs * DS(irows) * DS(icols) + ich * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
//             }

//             const bool is_zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;

//             const elem_t * in = input + (b*in_row_dim*in_col_dim + irow*in_col_dim + icol) * in_channels + ich;
//             if (is_zeros) {
//               in = NULL;
//             } else if (trans_input_3120) {
//               in = input + (ich*in_row_dim*in_col_dim + irow*in_col_dim + icol) * batch_size + b;
//             }

//             gemmini_extended_mvin(custom_num, in,
//                 A_sp_addr,
//                 K, I >> downsample);
//           }

//           icol += I;
//         }
//       }
//   }

//   // mvin weights
//   {
//     int max_chs_per_mvin = ochs < MAX_BLOCK_LEN * DIM ? ochs :
//         MAX_BLOCK_LEN * DIM;
//     if (trans_weight_0132) {
//       max_chs_per_mvin = kchs < MAX_BLOCK_LEN * DIM ? kchs :
//           MAX_BLOCK_LEN * DIM;
//     }

//     size_t dram_stride = out_channels * sizeof(elem_t);
//     if (dw) {
//       dram_stride = sizeof(elem_t);
//     } else if (trans_weight_1203) {
//       dram_stride = kernel_dim * kernel_dim * out_channels * sizeof(elem_t);
//     } else if (trans_weight_0132) {
//       dram_stride = in_channels * sizeof(elem_t);
//     }

//     const size_t spad_block_stride = trans_weight_0132 ?
//       krows * kcols * ochs : krows * kcols * kchs;

//     gemmini_extended4_config_ld(custom_num, dram_stride, MVIN_SCALE_IDENTITY, false, spad_block_stride, 1);

//     const size_t och_it = trans_weight_0132 ? DIM : max_chs_per_mvin;
//     const size_t kch_it = trans_weight_0132 ? max_chs_per_mvin : DIM;

//     for (int och = 0; och < ochs; och += och_it) {
//       for (int krow = 0; krow < krows; krow++)
//         for (int kcol = 0; kcol < kcols; kcol++)
//           for (int kch = 0; kch < kchs; kch += kch_it) {
//             int K = kchs - kch > DIM ? DIM : kchs - kch;
//             int J = ochs - och > max_chs_per_mvin ? max_chs_per_mvin : ochs - och;
//             if (trans_weight_0132) {
//               K = ochs - och > DIM ? DIM : ochs - och;
//               J = kchs - kch > max_chs_per_mvin ? max_chs_per_mvin : kchs - kch;
//             }

//             uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow * kcols * kchs + kcol * kchs + kch;
//             if (trans_weight_0132) {
//               B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow * kcols * ochs + kcol * ochs + och;
//             }

//             const elem_t * w = weights + (krow*kernel_dim*in_channels + kcol*in_channels + kch) * out_channels + och;
//             if (dw) {
//               w = weights + krow * kernel_dim + kcol;
//             } else if (trans_weight_1203) {
//               w = weights + (kch * kernel_dim * kernel_dim + krow * kernel_dim + kcol) * out_channels + och;
//             } else if (trans_weight_0132) {
//               w = weights + (krow * kernel_dim * out_channels + kcol * out_channels + och) * in_channels + kch;
//             }

//             gemmini_extended_mvin2(custom_num, w, B_sp_addr, J, K);
//           }
//     }
//   }

//   // Compute
//   {
//     const int b_it = trans_input_3120 ? DIM : 1;
//     const int ocol_it = trans_input_3120 ? 1 : (DIM << input_dilated);

//     if (trans_input_3120) {
//       gemmini_extended3_config_ex(custom_num, 0, 0, 0, 0, orows * ocols, irows * icols, 0, 0, true);
//     }

//     for (int och = 0; och < ochs; och += DIM) {
//       for (int krow = 0; krow < krows; krow++) {
//         for (int kcol = 0; kcol < kcols; kcol += max_pixels_per_row) {
//           for (int kch = 0; kch < kchs; kch += DIM) {
//             bool new_weights = true;

//             for (int b = 0; b < batches; b += b_it) {
//               for (int orow = 0; orow < orows; orow++) {
//                 // Skip some kernel rows due to input-dilation
//                 if (input_dilated && ((krow * kernel_dilation + orow * stride - upad) % 2 != 0)) {
//                   continue;
//                 }

//                 for (int ocol = 0; ocol < ocols;) {
//                   // Skip some cols dimensions due to input-dilation
//                   if (input_dilated && ((kcol + ocol * stride - lpad) % 2 != 0)) {
//                     ocol++;
//                     continue;
//                   }

//                   int irow = orow * stride + krow * kernel_dilation;
//                   int icol = ocol * stride + kcol * kernel_dilation;

//                   if (input_dilated) {
//                     irow = (irow + 1) / 2;
//                     icol = (icol + 1) / 2;
//                   }

//                   const int pixels = kcols - kcol > max_pixels_per_row ?
//                     max_pixels_per_row : kcols - kcol;

//                   const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

//                   // Over here, construct a new matrix
//                   //
//                   // Let us assume that we only ever operate on
//                   // one pixel in one row.
//                   // Thus, krows == kcols == 1
//                   //
//                   // Then, for every set of I, J, and K values
//                   //     - I = ocols
//                   //     - J = ochs
//                   //     - K = kchs

//                   int I = UNDILATED(ocols - ocol > (DIM << input_dilated) ? (DIM << input_dilated) : ocols - ocol);
//                   const int J = ochs - och > DIM ? DIM : ochs - och;
//                   const int K = pixels * (kchs - kch > DIM ? DIM : kchs - kch);

//                   if (trans_input_3120) {
//                     I = batches - b > DIM ? DIM : batches - b;
//                   }

//                   uint32_t A_sp_addr = A_sp_addr_start + (kch / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
//                   if (trans_input_3120) {
//                     A_sp_addr = A_sp_addr_start + (b / DIM) * kchs * DS(irows) * DS(icols) + kch * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
//                   }

//                   const int krow_ = wrot180 ? krows - krow - 1 : krow;
//                   const int kcol_ = wrot180 ? kcols - kcol - 1 : kcol;

//                   uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow_ * kcols * kchs + kcol_ * kchs + kch;
//                   if (trans_weight_0132) {
//                     B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow_ * kcols * ochs + kcol_ * ochs + och;
//                   }

//                   const uint32_t pre_sp_addr = new_weights ?
//                     B_sp_addr : GARBAGE_ADDR;

//                   // perform matmul
//                   gemmini_extended_preload(custom_num, pre_sp_addr, C_sp_addr, J, K, J, I);

//                   if (new_weights) {
//                     gemmini_extended_compute_preloaded(custom_num, A_sp_addr, GARBAGE_ADDR, K, I, J, I);
//                   } else {
//                     gemmini_extended_compute_accumulated(custom_num, A_sp_addr, GARBAGE_ADDR, K, I, J, I);
//                   }

//                   ocol += ocol_it;
//                   new_weights = false;
//                 }
//               }
//             }
//           }
//         }
//       }
//     }
//   }

// #undef DS
// #undef UNDILATED

//   // mvout output
//   if (output != NULL) {
//     if (no_pool) {
//       for (int b = 0; b < batches; b++)
//         for (int orow = 0; orow < orows; orow++)
//           for (int ocol = 0; ocol < ocols; ocol += DIM) {
//             const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

//             for (int och = 0; och < ochs; och += DIM) {
//               const int J = ochs - och > DIM ? DIM : ochs - och;

//               const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

//               elem_t * out = output + (b*out_row_dim*out_col_dim + orow*out_col_dim + ocol) * out_channels + och;
//               if (trans_output_1203) {
//                 out = output + (orow*out_col_dim*batch_size + ocol*batch_size + b) * out_channels + och;
//               }

//               gemmini_extended_mvout(custom_num, out,
//                   C_sp_addr,
//                   J, I);
//             }
//           }
//     } else {
//       printf("Pooling with rectangular convolutions is currently not supported.\n");
//       exit(1);

//       /*
//       gemmini_extended2_config_st(out_channels * sizeof(elem_t), act, scale, pool_stride, pool_size, pool_out_row_dim, porows, pocols, orows, ocols, pupad, plpad);

//       for (int b = 0; b < batches; b++) {
//         for (int poch = 0; poch < pochs; poch += DIM) {
//           const int channels = poch + DIM >= pochs ? pochs - poch : DIM;

//           elem_t * pout = output + (b * pool_out_row_dim * pool_out_col_dim)*out_channels + poch;

//           const uint32_t C_sp_addr = C_sp_addr_start + (poch / DIM) * batches * orows * ocols + b * orows * ocols;

//           gemmini_extended_mvout(pout,
//               C_sp_addr,
//               channels, 0);
//         }
//       }

//       gemmini_extended_config_st(out_channels * sizeof(elem_t), act, scale);
//       */
//     }
//   }
}

// made
static void shared_multi_sp_tiled_conv(
    int custom_num, int group_list, int group_id,
    size_t sp_addr_start, size_t sp_addr_end, size_t acc_addr_start,
    int batch_size, int in_row_dim, int in_col_dim, int in_channels,
    int out_channels, int out_row_dim, int out_col_dim,
    int pool_out_row_dim, int pool_out_col_dim,

    int stride, int padding, int kernel_dim, int kernel_dilation,

    int pool_size, int pool_stride, int pool_padding,

    size_t ex_ochs, size_t mv_kchs, size_t laddrochs_offset, size_t laddrkchs_offset,

    int batches,
    int porows, int pocols, int pochs,
    int krows, int kcols, int kchs,

    int lpad, int rpad, int upad, int dpad,
    int plpad, int prpad, int pupad, int pdpad,

    const elem_t *input,
    const elem_t *weights,
    elem_t *output,
    const acc_t *bias,

    int act, acc_scale_t scale,

    bool wrot180, bool trans_output_1203, bool trans_input_3120,
    bool trans_weight_1203, bool trans_weight_0132,

    bool no_bias, bool no_pool, bool downsample, bool input_dilated,
    bool dw)
{

  // When dw convs are true, we assume that kchs and ochs are 1
  if (dw)
  {
    kchs = 1;
    pochs = 1;
  }

  const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
  const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
  const int ochs = pochs;

  // Calculate image dimensions
  // Note: "irows" and "icols" includes padding
  const int dilated_krows = krows + (kernel_dilation - 1) * (krows - 1);
  const int dilated_kcols = kcols + (kernel_dilation - 1) * (kcols - 1);
  int irows = orows * stride + dilated_krows - 1;
  int icols = ocols * stride + dilated_kcols - 1;
  int irows_unpadded = irows - upad - dpad;
  int icols_unpadded = icols - lpad - rpad;
  const int ichs = kchs;

#define UNDILATED(x) ((input_dilated) ? (((x) + 1) / 2) : (x))

  if (input_dilated)
  {
    irows_unpadded = (irows_unpadded + 1) / 2;
    icols_unpadded = (icols_unpadded + 1) / 2;

    irows = irows_unpadded + UNDILATED(upad) + UNDILATED(dpad);
    icols = icols_unpadded + UNDILATED(lpad) + UNDILATED(rpad);
  }

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
  const bool transposed = trans_output_1203 || trans_input_3120 ||
                          trans_weight_1203 || trans_weight_0132;
  int max_pixels_per_row = transposed || wrot180 || downsample ||
                                   input_dilated || kernel_dilation > 1 ||
                                   ichs > DIM
                               ? 1
                               : DIM / ichs;
  if (max_pixels_per_row > kcols)
    max_pixels_per_row = kcols;
#else
  const int max_pixels_per_row = 1;
#endif

  // Calculate spad address offsets
  // const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
  // const int in_channels_per_bank = kchs / DIM + (kchs % DIM != 0);
  // const int B_rows = trans_weight_0132 ?
  //   in_channels_per_bank * kcols * krows * ochs :
  //   out_channels_per_bank * kcols * krows * kchs;

  // static uint32_t D_sp_addr_row = 0;
  // static uint32_t C_sp_addr_row = 0;

  // const uint32_t A_sp_addr_start = 0;
  // const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - B_rows;
  // const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
  // const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;

  // if (bias != 0) {
  //   D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  // }

  // if (output != 0) {
  //   C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  // }

  if (in_row_dim == in_col_dim && out_row_dim == out_col_dim && pool_out_row_dim == pool_out_col_dim)
  {
    switch (custom_num)
    {
    case 0:
      shared_gemmini_loop_conv_ws(custom0, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_ochs, mv_kchs, laddrochs_offset, laddrkchs_offset, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
      break;
    case 1:
      shared_gemmini_loop_conv_ws(custom1, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_ochs, mv_kchs, laddrochs_offset, laddrkchs_offset, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
      break;
    case 2:
      shared_gemmini_loop_conv_ws(custom2, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_ochs, mv_kchs, laddrochs_offset, laddrkchs_offset, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
      break;
    case 3:
      shared_gemmini_loop_conv_ws(custom3, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_ochs, mv_kchs, laddrochs_offset, laddrkchs_offset, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
      break;
    }

    return;
  }

  //   if (!no_pool) {
  //     printf("Pooling with rectangular convolutions is currently not supported.\n");
  //     exit(1);
  //   }

  //   // Only rectangular convolutions will use the following C code

  //   // mvin bias
  //   if (bias != NULL) {
  //     // TODO we probably don't need quite this many nested loops for this part

  //     const int max_ochs_per_mvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs :
  //         MAX_BLOCK_LEN_ACC * DIM;

  //     gemmini_extended4_config_ld(custom_num, 0, MVIN_SCALE_IDENTITY, false, batches * orows * ocols, 2);

  //     for (int b = 0; b < batches; b++)
  //       for (int orow = 0; orow < orows; orow++)
  //         for (int ocol = 0; ocol < ocols; ocol += DIM) {
  //           const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

  //           for (int och = 0; och < ochs; och += max_ochs_per_mvin) {
  //             const int J = ochs - och > max_ochs_per_mvin ? max_ochs_per_mvin : ochs - och;

  //             const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

  //             const acc_t * bias_dram_addr = no_bias ? NULL : bias + och;

  //             gemmini_extended_mvin3(custom_num, bias_dram_addr,
  //                     D_sp_addr,
  //                     J, I);
  //           }
  //         }
  //   }

  //   // mvin input
  //   {
  //     int max_chs_per_mvin = ichs < MAX_BLOCK_LEN * DIM ? ichs :
  //       MAX_BLOCK_LEN * DIM;
  //     if (trans_input_3120) {
  //       max_chs_per_mvin = batches < MAX_BLOCK_LEN * DIM ? batches :
  //         MAX_BLOCK_LEN * DIM;
  //     }

  //     const int dram_stride = trans_input_3120 ?
  //       batch_size * sizeof(elem_t) :
  //       in_channels * sizeof(elem_t);

  //     const int spad_stride = trans_input_3120 ?
  //       ichs * (irows >> downsample) * (icols >> downsample) :
  //       batches * (irows >> downsample) * (icols >> downsample);

  //     gemmini_extended5_config_ld(custom_num, dram_stride << downsample, MVIN_SCALE_IDENTITY, false, spad_stride, max_pixels_per_row, 0);

  //     const int b_it = trans_input_3120 ? max_chs_per_mvin : 1;
  //     const int ich_it = trans_input_3120 ? 1 : max_chs_per_mvin;

  //     for (int b = 0; b < batches; b += b_it)
  //       for (int irow = -UNDILATED(upad); irow < irows_unpadded + UNDILATED(dpad); irow += 1 + downsample) {
  //         const int irow_padded = irow + UNDILATED(upad);

  //         for (int icol = -UNDILATED(lpad); icol < icols_unpadded + UNDILATED(rpad);) {
  //           // TODO There might be some unnecessary mvins here at the edge of the image

  //           int I = icols_unpadded - icol > (DIM << downsample) ?
  //             (DIM << downsample) : icols_unpadded - icol;

  //           if (icol < 0) {
  //             I = -icol > DIM ? DIM : -icol;
  //           } else if (icol >= icols_unpadded) {
  //             I = icols_unpadded + UNDILATED(rpad) - icol > DIM ? DIM : icols_unpadded + UNDILATED(rpad) - icol;
  //           }

  //           const int icol_padded = icol + UNDILATED(lpad);

  //           for (int ich = 0; ich < ichs; ich += ich_it) {
  //             int K = ichs - ich > max_chs_per_mvin ?
  //               max_chs_per_mvin : ichs - ich;
  //             if (trans_input_3120) {
  //               K = batches - b > max_chs_per_mvin ?
  //                 max_chs_per_mvin : batches - b;
  //             }

  // #define DS(x) ((x) >> (downsample))

  //             uint32_t A_sp_addr = A_sp_addr_start + (ich / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
  //             if (trans_input_3120) {
  //               A_sp_addr = A_sp_addr_start + (b / DIM) * ichs * DS(irows) * DS(icols) + ich * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
  //             }

  //             const bool is_zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;

  //             const elem_t * in = input + (b*in_row_dim*in_col_dim + irow*in_col_dim + icol) * in_channels + ich;
  //             if (is_zeros) {
  //               in = NULL;
  //             } else if (trans_input_3120) {
  //               in = input + (ich*in_row_dim*in_col_dim + irow*in_col_dim + icol) * batch_size + b;
  //             }

  //             gemmini_extended_mvin(custom_num, in,
  //                 A_sp_addr,
  //                 K, I >> downsample);
  //           }

  //           icol += I;
  //         }
  //       }
  //   }

  //   // mvin weights
  //   {
  //     int max_chs_per_mvin = ochs < MAX_BLOCK_LEN * DIM ? ochs :
  //         MAX_BLOCK_LEN * DIM;
  //     if (trans_weight_0132) {
  //       max_chs_per_mvin = kchs < MAX_BLOCK_LEN * DIM ? kchs :
  //           MAX_BLOCK_LEN * DIM;
  //     }

  //     size_t dram_stride = out_channels * sizeof(elem_t);
  //     if (dw) {
  //       dram_stride = sizeof(elem_t);
  //     } else if (trans_weight_1203) {
  //       dram_stride = kernel_dim * kernel_dim * out_channels * sizeof(elem_t);
  //     } else if (trans_weight_0132) {
  //       dram_stride = in_channels * sizeof(elem_t);
  //     }

  //     const size_t spad_block_stride = trans_weight_0132 ?
  //       krows * kcols * ochs : krows * kcols * kchs;

  //     gemmini_extended4_config_ld(custom_num, dram_stride, MVIN_SCALE_IDENTITY, false, spad_block_stride, 1);

  //     const size_t och_it = trans_weight_0132 ? DIM : max_chs_per_mvin;
  //     const size_t kch_it = trans_weight_0132 ? max_chs_per_mvin : DIM;

  //     for (int och = 0; och < ochs; och += och_it) {
  //       for (int krow = 0; krow < krows; krow++)
  //         for (int kcol = 0; kcol < kcols; kcol++)
  //           for (int kch = 0; kch < kchs; kch += kch_it) {
  //             int K = kchs - kch > DIM ? DIM : kchs - kch;
  //             int J = ochs - och > max_chs_per_mvin ? max_chs_per_mvin : ochs - och;
  //             if (trans_weight_0132) {
  //               K = ochs - och > DIM ? DIM : ochs - och;
  //               J = kchs - kch > max_chs_per_mvin ? max_chs_per_mvin : kchs - kch;
  //             }

  //             uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow * kcols * kchs + kcol * kchs + kch;
  //             if (trans_weight_0132) {
  //               B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow * kcols * ochs + kcol * ochs + och;
  //             }

  //             const elem_t * w = weights + (krow*kernel_dim*in_channels + kcol*in_channels + kch) * out_channels + och;
  //             if (dw) {
  //               w = weights + krow * kernel_dim + kcol;
  //             } else if (trans_weight_1203) {
  //               w = weights + (kch * kernel_dim * kernel_dim + krow * kernel_dim + kcol) * out_channels + och;
  //             } else if (trans_weight_0132) {
  //               w = weights + (krow * kernel_dim * out_channels + kcol * out_channels + och) * in_channels + kch;
  //             }

  //             gemmini_extended_mvin2(custom_num, w, B_sp_addr, J, K);
  //           }
  //     }
  //   }

  //   // Compute
  //   {
  //     const int b_it = trans_input_3120 ? DIM : 1;
  //     const int ocol_it = trans_input_3120 ? 1 : (DIM << input_dilated);

  //     if (trans_input_3120) {
  //       gemmini_extended3_config_ex(custom_num, 0, 0, 0, 0, orows * ocols, irows * icols, 0, 0, true);
  //     }

  //     for (int och = 0; och < ochs; och += DIM) {
  //       for (int krow = 0; krow < krows; krow++) {
  //         for (int kcol = 0; kcol < kcols; kcol += max_pixels_per_row) {
  //           for (int kch = 0; kch < kchs; kch += DIM) {
  //             bool new_weights = true;

  //             for (int b = 0; b < batches; b += b_it) {
  //               for (int orow = 0; orow < orows; orow++) {
  //                 // Skip some kernel rows due to input-dilation
  //                 if (input_dilated && ((krow * kernel_dilation + orow * stride - upad) % 2 != 0)) {
  //                   continue;
  //                 }

  //                 for (int ocol = 0; ocol < ocols;) {
  //                   // Skip some cols dimensions due to input-dilation
  //                   if (input_dilated && ((kcol + ocol * stride - lpad) % 2 != 0)) {
  //                     ocol++;
  //                     continue;
  //                   }

  //                   int irow = orow * stride + krow * kernel_dilation;
  //                   int icol = ocol * stride + kcol * kernel_dilation;

  //                   if (input_dilated) {
  //                     irow = (irow + 1) / 2;
  //                     icol = (icol + 1) / 2;
  //                   }

  //                   const int pixels = kcols - kcol > max_pixels_per_row ?
  //                     max_pixels_per_row : kcols - kcol;

  //                   const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

  //                   // Over here, construct a new matrix
  //                   //
  //                   // Let us assume that we only ever operate on
  //                   // one pixel in one row.
  //                   // Thus, krows == kcols == 1
  //                   //
  //                   // Then, for every set of I, J, and K values
  //                   //     - I = ocols
  //                   //     - J = ochs
  //                   //     - K = kchs

  //                   int I = UNDILATED(ocols - ocol > (DIM << input_dilated) ? (DIM << input_dilated) : ocols - ocol);
  //                   const int J = ochs - och > DIM ? DIM : ochs - och;
  //                   const int K = pixels * (kchs - kch > DIM ? DIM : kchs - kch);

  //                   if (trans_input_3120) {
  //                     I = batches - b > DIM ? DIM : batches - b;
  //                   }

  //                   uint32_t A_sp_addr = A_sp_addr_start + (kch / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
  //                   if (trans_input_3120) {
  //                     A_sp_addr = A_sp_addr_start + (b / DIM) * kchs * DS(irows) * DS(icols) + kch * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
  //                   }

  //                   const int krow_ = wrot180 ? krows - krow - 1 : krow;
  //                   const int kcol_ = wrot180 ? kcols - kcol - 1 : kcol;

  //                   uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow_ * kcols * kchs + kcol_ * kchs + kch;
  //                   if (trans_weight_0132) {
  //                     B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow_ * kcols * ochs + kcol_ * ochs + och;
  //                   }

  //                   const uint32_t pre_sp_addr = new_weights ?
  //                     B_sp_addr : GARBAGE_ADDR;

  //                   // perform matmul
  //                   gemmini_extended_preload(custom_num, pre_sp_addr, C_sp_addr, J, K, J, I);

  //                   if (new_weights) {
  //                     gemmini_extended_compute_preloaded(custom_num, A_sp_addr, GARBAGE_ADDR, K, I, J, I);
  //                   } else {
  //                     gemmini_extended_compute_accumulated(custom_num, A_sp_addr, GARBAGE_ADDR, K, I, J, I);
  //                   }

  //                   ocol += ocol_it;
  //                   new_weights = false;
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }

  // #undef DS
  // #undef UNDILATED

  //   // mvout output
  //   if (output != NULL) {
  //     if (no_pool) {
  //       for (int b = 0; b < batches; b++)
  //         for (int orow = 0; orow < orows; orow++)
  //           for (int ocol = 0; ocol < ocols; ocol += DIM) {
  //             const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

  //             for (int och = 0; och < ochs; och += DIM) {
  //               const int J = ochs - och > DIM ? DIM : ochs - och;

  //               const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

  //               elem_t * out = output + (b*out_row_dim*out_col_dim + orow*out_col_dim + ocol) * out_channels + och;
  //               if (trans_output_1203) {
  //                 out = output + (orow*out_col_dim*batch_size + ocol*batch_size + b) * out_channels + och;
  //               }

  //               gemmini_extended_mvout(custom_num, out,
  //                   C_sp_addr,
  //                   J, I);
  //             }
  //           }
  //     } else {
  //       printf("Pooling with rectangular convolutions is currently not supported.\n");
  //       exit(1);

  //       /*
  //       gemmini_extended2_config_st(out_channels * sizeof(elem_t), act, scale, pool_stride, pool_size, pool_out_row_dim, porows, pocols, orows, ocols, pupad, plpad);

  //       for (int b = 0; b < batches; b++) {
  //         for (int poch = 0; poch < pochs; poch += DIM) {
  //           const int channels = poch + DIM >= pochs ? pochs - poch : DIM;

  //           elem_t * pout = output + (b * pool_out_row_dim * pool_out_col_dim)*out_channels + poch;

  //           const uint32_t C_sp_addr = C_sp_addr_start + (poch / DIM) * batches * orows * ocols + b * orows * ocols;

  //           gemmini_extended_mvout(pout,
  //               C_sp_addr,
  //               channels, 0);
  //         }
  //       }

  //       gemmini_extended_config_st(out_channels * sizeof(elem_t), act, scale);
  //       */
  //     }
  //   }
}

// made
static void shared_multi_sp_tiled_conv_test(
    int custom_num, int group_list, int group_id,
    size_t sp_addr_start, size_t sp_addr_end, size_t acc_addr_start,
    int batch_size, int in_row_dim, int in_col_dim, int in_channels,
    int out_channels, int out_row_dim, int out_col_dim,
    int pool_out_row_dim, int pool_out_col_dim,

    int stride, int padding, int kernel_dim, int kernel_dilation,

    int pool_size, int pool_stride, int pool_padding,

    size_t ex_ochs, size_t mv_kchs, size_t laddrochs_offset, size_t laddrkchs_offset,

    int batches,
    int porows, int pocols, int pochs,
    int krows, int kcols, int kchs,

    int lpad, int rpad, int upad, int dpad,
    int plpad, int prpad, int pupad, int pdpad,

    const elem_t *input,
    const elem_t *weights,
    elem_t *output,
    const acc_t *bias,

    int act, acc_scale_t scale,

    bool wrot180, bool trans_output_1203, bool trans_input_3120,
    bool trans_weight_1203, bool trans_weight_0132,

    bool no_bias, bool no_pool, bool downsample, bool input_dilated,
    bool dw)
{

  // When dw convs are true, we assume that kchs and ochs are 1
  if (dw)
  {
    kchs = 1;
    pochs = 1;
  }

  const int orows = porows * pool_stride + pool_size - 1 - pupad - pdpad;
  const int ocols = pocols * pool_stride + pool_size - 1 - plpad - prpad;
  const int ochs = pochs;

  // Calculate image dimensions
  // Note: "irows" and "icols" includes padding
  const int dilated_krows = krows + (kernel_dilation - 1) * (krows - 1);
  const int dilated_kcols = kcols + (kernel_dilation - 1) * (kcols - 1);
  int irows = orows * stride + dilated_krows - 1;
  int icols = ocols * stride + dilated_kcols - 1;
  int irows_unpadded = irows - upad - dpad;
  int icols_unpadded = icols - lpad - rpad;
  const int ichs = kchs;

#define UNDILATED(x) ((input_dilated) ? (((x) + 1) / 2) : (x))

  if (input_dilated)
  {
    irows_unpadded = (irows_unpadded + 1) / 2;
    icols_unpadded = (icols_unpadded + 1) / 2;

    irows = irows_unpadded + UNDILATED(upad) + UNDILATED(dpad);
    icols = icols_unpadded + UNDILATED(lpad) + UNDILATED(rpad);
  }

#ifdef HAS_FIRST_LAYER_OPTIMIZATIONS
  const bool transposed = trans_output_1203 || trans_input_3120 ||
                          trans_weight_1203 || trans_weight_0132;
  int max_pixels_per_row = transposed || wrot180 || downsample ||
                                   input_dilated || kernel_dilation > 1 ||
                                   ichs > DIM
                               ? 1
                               : DIM / ichs;
  if (max_pixels_per_row > kcols)
    max_pixels_per_row = kcols;
#else
  const int max_pixels_per_row = 1;
#endif

  // Calculate spad address offsets
  // const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
  // const int in_channels_per_bank = kchs / DIM + (kchs % DIM != 0);
  // const int B_rows = trans_weight_0132 ?
  //   in_channels_per_bank * kcols * krows * ochs :
  //   out_channels_per_bank * kcols * krows * kchs;

  // static uint32_t D_sp_addr_row = 0;
  // static uint32_t C_sp_addr_row = 0;

  // const uint32_t A_sp_addr_start = 0;
  // const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - B_rows;
  // const uint32_t D_sp_addr_start = (1 << (ADDR_LEN - 1)) + D_sp_addr_row;
  // const uint32_t C_sp_addr_start = (3 << (ADDR_LEN - 2)) + C_sp_addr_row;

  // if (bias != 0) {
  //   D_sp_addr_row = (D_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  // }

  // if (output != 0) {
  //   C_sp_addr_row = (C_sp_addr_row + ACC_ROWS / 2) % ACC_ROWS;
  // }

  if (in_row_dim == in_col_dim && out_row_dim == out_col_dim && pool_out_row_dim == pool_out_col_dim)
  {
    switch (custom_num)
    {
    case 0:
      shared_gemmini_loop_conv_ws(custom0, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_ochs, mv_kchs, laddrochs_offset, laddrkchs_offset, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
      break;
    case 1:
      shared_gemmini_loop_conv_ws(custom1, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_ochs, mv_kchs, laddrochs_offset, laddrkchs_offset, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
      break;
    case 2:
      shared_gemmini_loop_conv_ws(custom2, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_ochs, mv_kchs, laddrochs_offset, laddrkchs_offset, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
      break;
    case 3:
      shared_gemmini_loop_conv_ws(custom3, group_list, group_id, sp_addr_start, sp_addr_end, acc_addr_start, ex_ochs, mv_kchs, laddrochs_offset, laddrkchs_offset, batch_size, in_row_dim, in_channels, out_channels, out_row_dim, pool_out_row_dim, stride, padding, kernel_dim, kernel_dilation, pool_size, pool_stride, pool_padding, batches, porows, pocols, pochs, krows, kcols, kchs, lpad, rpad, upad, dpad, plpad, prpad, pupad, pdpad, orows, ocols, weights, output, bias, input, no_bias, no_pool, downsample, wrot180, input_dilated, act, trans_output_1203, trans_weight_1203, trans_weight_0132, trans_input_3120, max_pixels_per_row, dw);
      break;
    }

    return;
  }

  //   if (!no_pool) {
  //     printf("Pooling with rectangular convolutions is currently not supported.\n");
  //     exit(1);
  //   }

  //   // Only rectangular convolutions will use the following C code

  //   // mvin bias
  //   if (bias != NULL) {
  //     // TODO we probably don't need quite this many nested loops for this part

  //     const int max_ochs_per_mvin = ochs < MAX_BLOCK_LEN_ACC * DIM ? ochs :
  //         MAX_BLOCK_LEN_ACC * DIM;

  //     gemmini_extended4_config_ld(custom_num, 0, MVIN_SCALE_IDENTITY, false, batches * orows * ocols, 2);

  //     for (int b = 0; b < batches; b++)
  //       for (int orow = 0; orow < orows; orow++)
  //         for (int ocol = 0; ocol < ocols; ocol += DIM) {
  //           const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

  //           for (int och = 0; och < ochs; och += max_ochs_per_mvin) {
  //             const int J = ochs - och > max_ochs_per_mvin ? max_ochs_per_mvin : ochs - och;

  //             const uint32_t D_sp_addr = D_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

  //             const acc_t * bias_dram_addr = no_bias ? NULL : bias + och;

  //             gemmini_extended_mvin3(custom_num, bias_dram_addr,
  //                     D_sp_addr,
  //                     J, I);
  //           }
  //         }
  //   }

  //   // mvin input
  //   {
  //     int max_chs_per_mvin = ichs < MAX_BLOCK_LEN * DIM ? ichs :
  //       MAX_BLOCK_LEN * DIM;
  //     if (trans_input_3120) {
  //       max_chs_per_mvin = batches < MAX_BLOCK_LEN * DIM ? batches :
  //         MAX_BLOCK_LEN * DIM;
  //     }

  //     const int dram_stride = trans_input_3120 ?
  //       batch_size * sizeof(elem_t) :
  //       in_channels * sizeof(elem_t);

  //     const int spad_stride = trans_input_3120 ?
  //       ichs * (irows >> downsample) * (icols >> downsample) :
  //       batches * (irows >> downsample) * (icols >> downsample);

  //     gemmini_extended5_config_ld(custom_num, dram_stride << downsample, MVIN_SCALE_IDENTITY, false, spad_stride, max_pixels_per_row, 0);

  //     const int b_it = trans_input_3120 ? max_chs_per_mvin : 1;
  //     const int ich_it = trans_input_3120 ? 1 : max_chs_per_mvin;

  //     for (int b = 0; b < batches; b += b_it)
  //       for (int irow = -UNDILATED(upad); irow < irows_unpadded + UNDILATED(dpad); irow += 1 + downsample) {
  //         const int irow_padded = irow + UNDILATED(upad);

  //         for (int icol = -UNDILATED(lpad); icol < icols_unpadded + UNDILATED(rpad);) {
  //           // TODO There might be some unnecessary mvins here at the edge of the image

  //           int I = icols_unpadded - icol > (DIM << downsample) ?
  //             (DIM << downsample) : icols_unpadded - icol;

  //           if (icol < 0) {
  //             I = -icol > DIM ? DIM : -icol;
  //           } else if (icol >= icols_unpadded) {
  //             I = icols_unpadded + UNDILATED(rpad) - icol > DIM ? DIM : icols_unpadded + UNDILATED(rpad) - icol;
  //           }

  //           const int icol_padded = icol + UNDILATED(lpad);

  //           for (int ich = 0; ich < ichs; ich += ich_it) {
  //             int K = ichs - ich > max_chs_per_mvin ?
  //               max_chs_per_mvin : ichs - ich;
  //             if (trans_input_3120) {
  //               K = batches - b > max_chs_per_mvin ?
  //                 max_chs_per_mvin : batches - b;
  //             }

  // #define DS(x) ((x) >> (downsample))

  //             uint32_t A_sp_addr = A_sp_addr_start + (ich / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
  //             if (trans_input_3120) {
  //               A_sp_addr = A_sp_addr_start + (b / DIM) * ichs * DS(irows) * DS(icols) + ich * DS(irows) * DS(icols) + DS(irow_padded) * DS(icols) + DS(icol_padded);
  //             }

  //             const bool is_zeros = irow < 0 || irow >= irows_unpadded || icol < 0 || icol >= icols_unpadded;

  //             const elem_t * in = input + (b*in_row_dim*in_col_dim + irow*in_col_dim + icol) * in_channels + ich;
  //             if (is_zeros) {
  //               in = NULL;
  //             } else if (trans_input_3120) {
  //               in = input + (ich*in_row_dim*in_col_dim + irow*in_col_dim + icol) * batch_size + b;
  //             }

  //             gemmini_extended_mvin(custom_num, in,
  //                 A_sp_addr,
  //                 K, I >> downsample);
  //           }

  //           icol += I;
  //         }
  //       }
  //   }

  //   // mvin weights
  //   {
  //     int max_chs_per_mvin = ochs < MAX_BLOCK_LEN * DIM ? ochs :
  //         MAX_BLOCK_LEN * DIM;
  //     if (trans_weight_0132) {
  //       max_chs_per_mvin = kchs < MAX_BLOCK_LEN * DIM ? kchs :
  //           MAX_BLOCK_LEN * DIM;
  //     }

  //     size_t dram_stride = out_channels * sizeof(elem_t);
  //     if (dw) {
  //       dram_stride = sizeof(elem_t);
  //     } else if (trans_weight_1203) {
  //       dram_stride = kernel_dim * kernel_dim * out_channels * sizeof(elem_t);
  //     } else if (trans_weight_0132) {
  //       dram_stride = in_channels * sizeof(elem_t);
  //     }

  //     const size_t spad_block_stride = trans_weight_0132 ?
  //       krows * kcols * ochs : krows * kcols * kchs;

  //     gemmini_extended4_config_ld(custom_num, dram_stride, MVIN_SCALE_IDENTITY, false, spad_block_stride, 1);

  //     const size_t och_it = trans_weight_0132 ? DIM : max_chs_per_mvin;
  //     const size_t kch_it = trans_weight_0132 ? max_chs_per_mvin : DIM;

  //     for (int och = 0; och < ochs; och += och_it) {
  //       for (int krow = 0; krow < krows; krow++)
  //         for (int kcol = 0; kcol < kcols; kcol++)
  //           for (int kch = 0; kch < kchs; kch += kch_it) {
  //             int K = kchs - kch > DIM ? DIM : kchs - kch;
  //             int J = ochs - och > max_chs_per_mvin ? max_chs_per_mvin : ochs - och;
  //             if (trans_weight_0132) {
  //               K = ochs - och > DIM ? DIM : ochs - och;
  //               J = kchs - kch > max_chs_per_mvin ? max_chs_per_mvin : kchs - kch;
  //             }

  //             uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow * kcols * kchs + kcol * kchs + kch;
  //             if (trans_weight_0132) {
  //               B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow * kcols * ochs + kcol * ochs + och;
  //             }

  //             const elem_t * w = weights + (krow*kernel_dim*in_channels + kcol*in_channels + kch) * out_channels + och;
  //             if (dw) {
  //               w = weights + krow * kernel_dim + kcol;
  //             } else if (trans_weight_1203) {
  //               w = weights + (kch * kernel_dim * kernel_dim + krow * kernel_dim + kcol) * out_channels + och;
  //             } else if (trans_weight_0132) {
  //               w = weights + (krow * kernel_dim * out_channels + kcol * out_channels + och) * in_channels + kch;
  //             }

  //             gemmini_extended_mvin2(custom_num, w, B_sp_addr, J, K);
  //           }
  //     }
  //   }

  //   // Compute
  //   {
  //     const int b_it = trans_input_3120 ? DIM : 1;
  //     const int ocol_it = trans_input_3120 ? 1 : (DIM << input_dilated);

  //     if (trans_input_3120) {
  //       gemmini_extended3_config_ex(custom_num, 0, 0, 0, 0, orows * ocols, irows * icols, 0, 0, true);
  //     }

  //     for (int och = 0; och < ochs; och += DIM) {
  //       for (int krow = 0; krow < krows; krow++) {
  //         for (int kcol = 0; kcol < kcols; kcol += max_pixels_per_row) {
  //           for (int kch = 0; kch < kchs; kch += DIM) {
  //             bool new_weights = true;

  //             for (int b = 0; b < batches; b += b_it) {
  //               for (int orow = 0; orow < orows; orow++) {
  //                 // Skip some kernel rows due to input-dilation
  //                 if (input_dilated && ((krow * kernel_dilation + orow * stride - upad) % 2 != 0)) {
  //                   continue;
  //                 }

  //                 for (int ocol = 0; ocol < ocols;) {
  //                   // Skip some cols dimensions due to input-dilation
  //                   if (input_dilated && ((kcol + ocol * stride - lpad) % 2 != 0)) {
  //                     ocol++;
  //                     continue;
  //                   }

  //                   int irow = orow * stride + krow * kernel_dilation;
  //                   int icol = ocol * stride + kcol * kernel_dilation;

  //                   if (input_dilated) {
  //                     irow = (irow + 1) / 2;
  //                     icol = (icol + 1) / 2;
  //                   }

  //                   const int pixels = kcols - kcol > max_pixels_per_row ?
  //                     max_pixels_per_row : kcols - kcol;

  //                   const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

  //                   // Over here, construct a new matrix
  //                   //
  //                   // Let us assume that we only ever operate on
  //                   // one pixel in one row.
  //                   // Thus, krows == kcols == 1
  //                   //
  //                   // Then, for every set of I, J, and K values
  //                   //     - I = ocols
  //                   //     - J = ochs
  //                   //     - K = kchs

  //                   int I = UNDILATED(ocols - ocol > (DIM << input_dilated) ? (DIM << input_dilated) : ocols - ocol);
  //                   const int J = ochs - och > DIM ? DIM : ochs - och;
  //                   const int K = pixels * (kchs - kch > DIM ? DIM : kchs - kch);

  //                   if (trans_input_3120) {
  //                     I = batches - b > DIM ? DIM : batches - b;
  //                   }

  //                   uint32_t A_sp_addr = A_sp_addr_start + (kch / DIM) * batches * DS(irows) * DS(icols) + b * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
  //                   if (trans_input_3120) {
  //                     A_sp_addr = A_sp_addr_start + (b / DIM) * kchs * DS(irows) * DS(icols) + kch * DS(irows) * DS(icols) + DS(irow) * DS(icols) + DS(icol);
  //                   }

  //                   const int krow_ = wrot180 ? krows - krow - 1 : krow;
  //                   const int kcol_ = wrot180 ? kcols - kcol - 1 : kcol;

  //                   uint32_t B_sp_addr = B_sp_addr_start + (och / DIM) * krows * kcols * kchs + krow_ * kcols * kchs + kcol_ * kchs + kch;
  //                   if (trans_weight_0132) {
  //                     B_sp_addr = B_sp_addr_start + (kch / DIM) * krows * kcols * ochs + krow_ * kcols * ochs + kcol_ * ochs + och;
  //                   }

  //                   const uint32_t pre_sp_addr = new_weights ?
  //                     B_sp_addr : GARBAGE_ADDR;

  //                   // perform matmul
  //                   gemmini_extended_preload(custom_num, pre_sp_addr, C_sp_addr, J, K, J, I);

  //                   if (new_weights) {
  //                     gemmini_extended_compute_preloaded(custom_num, A_sp_addr, GARBAGE_ADDR, K, I, J, I);
  //                   } else {
  //                     gemmini_extended_compute_accumulated(custom_num, A_sp_addr, GARBAGE_ADDR, K, I, J, I);
  //                   }

  //                   ocol += ocol_it;
  //                   new_weights = false;
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }

  // #undef DS
  // #undef UNDILATED

  //   // mvout output
  //   if (output != NULL) {
  //     if (no_pool) {
  //       for (int b = 0; b < batches; b++)
  //         for (int orow = 0; orow < orows; orow++)
  //           for (int ocol = 0; ocol < ocols; ocol += DIM) {
  //             const int I = ocols - ocol > DIM ? DIM : ocols - ocol;

  //             for (int och = 0; och < ochs; och += DIM) {
  //               const int J = ochs - och > DIM ? DIM : ochs - och;

  //               const uint32_t C_sp_addr = C_sp_addr_start + (och / DIM) * batches * orows * ocols + b * orows * ocols + orow * ocols + ocol;

  //               elem_t * out = output + (b*out_row_dim*out_col_dim + orow*out_col_dim + ocol) * out_channels + och;
  //               if (trans_output_1203) {
  //                 out = output + (orow*out_col_dim*batch_size + ocol*batch_size + b) * out_channels + och;
  //               }

  //               gemmini_extended_mvout(custom_num, out,
  //                   C_sp_addr,
  //                   J, I);
  //             }
  //           }
  //     } else {
  //       printf("Pooling with rectangular convolutions is currently not supported.\n");
  //       exit(1);

  //       /*
  //       gemmini_extended2_config_st(out_channels * sizeof(elem_t), act, scale, pool_stride, pool_size, pool_out_row_dim, porows, pocols, orows, ocols, pupad, plpad);

  //       for (int b = 0; b < batches; b++) {
  //         for (int poch = 0; poch < pochs; poch += DIM) {
  //           const int channels = poch + DIM >= pochs ? pochs - poch : DIM;

  //           elem_t * pout = output + (b * pool_out_row_dim * pool_out_col_dim)*out_channels + poch;

  //           const uint32_t C_sp_addr = C_sp_addr_start + (poch / DIM) * batches * orows * ocols + b * orows * ocols;

  //           gemmini_extended_mvout(pout,
  //               C_sp_addr,
  //               channels, 0);
  //         }
  //       }

  //       gemmini_extended_config_st(out_channels * sizeof(elem_t), act, scale);
  //       */
  //     }
  //   }
}

static int tiled_conv_total_spad_rows_dw(bool acc, bool weight,
        int stride,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int irows = orows * stride + krows - 1; // - 2 * padding;
    const int icols = ocols * stride + kcols - 1; // - 2 * padding;
    const int ichs = kchs;

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);

    const int A_rows = in_channels_per_bank * batches * irows * icols;
    const int B_rows = out_channels_per_bank * kcols * krows * kchs;
    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    if (acc)
        return C_rows;
    else if(weight)
        return B_rows;
    else
        return A_rows;
}


static int tiled_conv_total_spad_rows(bool acc,
        int stride,
        int input_dilation,
        int kernel_dilation,
        bool downsample,
        bool trans_weight_0132,
        bool trans_input_3120,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int krows_dilated = krows + (kernel_dilation - 1)*(krows - 1);
    const int kcols_dilated = kcols + (kernel_dilation - 1)*(kcols - 1);

    int irows = orows * stride + krows_dilated - 1; // - 2 * padding;
    int icols = ocols * stride + kcols_dilated - 1; // - 2 * padding;
    const int ichs = kchs;

    irows = irows / input_dilation + (irows % input_dilation != 0);
    icols = icols / input_dilation + (icols % input_dilation != 0);

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
    const int batches_per_bank = batches / DIM + (batches % DIM != 0);

    const int A_rows = trans_input_3120 ?
        (batches_per_bank * ichs * (irows >> downsample) * (icols >> downsample)) :
        (in_channels_per_bank * batches * (irows >> downsample) * (icols >> downsample));

    const int B_rows = trans_weight_0132 ?
      in_channels_per_bank * kcols * krows * ochs :
      out_channels_per_bank * kcols * krows * kchs;

    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    return acc ? C_rows : A_rows + B_rows;
}

// made
static int tiled_conv_total_spad_rows_test(bool acc, bool A, bool B,
        int stride,
        int input_dilation,
        int kernel_dilation,
        bool downsample,
        bool trans_weight_0132,
        bool trans_input_3120,
        int batches,
        int porows, int pocols, int ochs,
        int krows, int kcols, int kchs,
        int pool_size, int pool_stride) {

    const int orows = porows * pool_stride + pool_size - 1;
    const int ocols = pocols * pool_stride + pool_size - 1;

    const int krows_dilated = krows + (kernel_dilation - 1)*(krows - 1);
    const int kcols_dilated = kcols + (kernel_dilation - 1)*(kcols - 1);

    int irows = orows * stride + krows_dilated - 1; // - 2 * padding;
    int icols = ocols * stride + kcols_dilated - 1; // - 2 * padding;
    const int ichs = kchs;

    irows = irows / input_dilation + (irows % input_dilation != 0);
    icols = icols / input_dilation + (icols % input_dilation != 0);

    const int in_channels_per_bank = ichs / DIM + (ichs % DIM != 0);
    const int out_channels_per_bank = ochs / DIM + (ochs % DIM != 0);
    const int batches_per_bank = batches / DIM + (batches % DIM != 0);

    const int A_rows = trans_input_3120 ?
        (batches_per_bank * ichs * (irows >> downsample) * (icols >> downsample)) :
        (in_channels_per_bank * batches * (irows >> downsample) * (icols >> downsample));

    const int B_rows = trans_weight_0132 ?
      in_channels_per_bank * kcols * krows * ochs :
      out_channels_per_bank * kcols * krows * kchs;

    const int C_rows = out_channels_per_bank * batches * orows * ocols;

    return acc ? C_rows : A ? A_rows : B ? B_rows : A_rows + B_rows;
}


static void conv_cpu_without_pool(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale) {

  bool no_bias = bias == NULL;

  for (int b = 0; b < batch_size; b++) {
    for (int orow = 0; orow < out_row_dim; orow++) {
      for (int ocol = 0; ocol < out_col_dim; ocol++) {
        for (int och = 0; och < out_channels; och++) {

          acc_t opixel = no_bias ? 0 : bias[och];

          for (int krow = 0; krow < kernel_dim; krow++) {
            if ((orow * stride + krow * kernel_dilation - padding) % input_dilation != 0)
              continue;

            const int irow = (orow * stride + krow * kernel_dilation - padding) / input_dilation;

            for (int kcol = 0; kcol < kernel_dim; kcol++) {
              if ((ocol * stride + kcol * kernel_dilation - padding) % input_dilation != 0)
                continue;

              const int icol = (ocol * stride + kcol * kernel_dilation - padding) / input_dilation;

              for (int kch = 0; kch < in_channels; kch++) {
                const elem_t *in = input + (b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * in_channels + kch;
                if (trans_input_3120) {
                  // NHWC to CHWN
                  in = input + (kch * in_row_dim * in_col_dim + irow * in_col_dim + icol) * batch_size + b;
                }

                elem_t ipixel = irow < 0 || irow >= in_row_dim || icol < 0 || icol >= in_col_dim ?
                    0 : *in;

                const int krow_ = wrot180 ? kernel_dim - krow - 1 : krow;
                const int kcol_ = wrot180 ? kernel_dim - kcol - 1 : kcol;

                elem_t weight = *(weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + och);
                if (trans_weight_1203) {
                  // HWIO to WIHO
                  weight = *(weights + (kch * kernel_dim * kernel_dim  + krow_ * kernel_dim + kcol_) * out_channels + och);
                } else if (trans_weight_0132) {
                  // HWIO to HWOI
                  weight = *(weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + och) * in_channels + kch);
                }

                opixel += weight * ipixel;
              }
            }
          }

          elem_t *out = output + (b * out_row_dim * out_col_dim + orow * out_col_dim + ocol) * out_channels + och;
          if (trans_output_1203) {
            // NHWC to HWNC
            out = output + (orow * out_col_dim * batch_size + ocol * batch_size + b) * out_channels + och;
          }

          *out = scale_and_sat(opixel, act, scale, 0);
        }
      }
    }
  }
}


static void conv_dw_cpu_without_pool(
        int batch_size, int in_row_dim, int in_col_dim,
        int channels, int out_row_dim, int out_col_dim,
        int stride, int padding, int kernel_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale) {

  bool no_bias = bias == NULL;

  for (int b = 0; b < batch_size; b++) {
    for (int orow = 0; orow < out_row_dim; orow++) {
      for (int ocol = 0; ocol < out_col_dim; ocol++) {
        for (int ch = 0; ch < channels; ch++) {
          acc_t opixel = no_bias ? 0 : bias[ch];

          for (int krow = 0; krow < kernel_dim; krow++) {
            const int irow = orow * stride + krow - padding;

            for (int kcol = 0; kcol < kernel_dim; kcol++) {
              const int icol = ocol * stride + kcol - padding;

              const elem_t * in = input + (b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * channels + ch;

              const elem_t ipixel = irow < 0 || irow >= in_row_dim || icol < 0 || icol >= in_col_dim ?
                  0 : *in;

              const elem_t weight = *(weights + (ch * kernel_dim + krow) * kernel_dim  + kcol);

              opixel += weight * ipixel;
            }
          }

          elem_t *out = output + (b * out_row_dim * out_col_dim + orow * out_col_dim + ocol) * channels + ch;

          *out = scale_and_sat(opixel, act, scale, 0);
        }
      }
    }
  }
}


static void conv_cpu(
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding) {

  const bool no_pool = pool_stride == 0;
  if (no_pool) {
    conv_cpu_without_pool(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale);
    return;
  }

  const bool no_bias = bias == NULL;
  const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

  for (int b = 0; b < batch_size; b++) {
    for (int porow = 0; porow < pool_out_row_dim; porow++) {
      for (int pocol = 0; pocol < pool_out_col_dim; pocol++) {
        for (int poch = 0; poch < out_channels; poch++) {

          elem_t running_max = 0;
          bool running_max_initialized = false;

          for (int pwrow = 0; pwrow < pool_size; pwrow++) {
            const int orow = porow * pool_stride + pwrow - pool_padding;

            for (int pwcol = 0; pwcol < pool_size; pwcol++) {
              const int ocol = pocol * pool_stride + pwcol - pool_padding;

              if (orow < 0 || orow >= out_row_dim || ocol < 0 || ocol >= out_col_dim) {
                if (!running_max_initialized || running_max < 0) {
                  running_max = 0;
                  running_max_initialized = true;
                }
              } else {

                acc_t opixel = no_bias ? 0 : bias[poch];

                for (int krow = 0; krow < kernel_dim; krow++) {
                  if ((orow * stride + krow * kernel_dilation - padding) % input_dilation != 0)
                    continue;

                  const int irow = (orow * stride + krow * kernel_dilation - padding) / input_dilation;

                  for (int kcol = 0; kcol < kernel_dim; kcol++) {
                    if ((ocol * stride + kcol * kernel_dilation - padding) % input_dilation != 0)
                      continue;

                    const int icol = (ocol * stride + kcol * kernel_dilation - padding) / input_dilation;

                    for (int kch = 0; kch < in_channels; kch++) {
                      const elem_t * in = input + (b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * in_channels + kch;
                      if (trans_input_3120) {
                        // NHWC to CHWN
                        in = input + (kch * in_row_dim * in_col_dim + irow * in_col_dim + icol) * batch_size + b;
                      }

                      elem_t ipixel = irow < 0 || irow >= in_row_dim || icol < 0 || icol >= in_col_dim ?
                          0 : *in;

                      const int krow_ = wrot180 ? kernel_dim - krow - 1 : krow;
                      const int kcol_ = wrot180 ? kernel_dim - kcol - 1 : kcol;

                      elem_t weight = *(weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + poch);
                      if (trans_weight_1203) {
                        // HWIO to WIHO
                        weight = *(weights + (kch * kernel_dim * kernel_dim  + krow_ * kernel_dim + kcol_) * out_channels + poch);
                      } else if (trans_weight_0132) {
                        // HWIO to HWOI
                        weight = *(weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + poch) * in_channels + kch);
                      }

                      opixel += weight * ipixel;
                    }
                  }
                }

                opixel = scale_and_sat(opixel, act, scale, 0);
                if (!running_max_initialized || opixel > running_max) {
                  running_max = opixel;
                  running_max_initialized = true;
                }
              }

              if (pwrow == pool_size - 1 && pwcol == pool_size - 1) {
                elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * out_channels + poch;
                if (trans_output_1203) {
                  // NHWC to HWNC
                  out = output + (porow * pool_out_col_dim * batch_size + pocol * batch_size + b) * out_channels + poch;
                }

                *out = running_max;
              }
            }
          }
        }
      }
    }
  }
}


static void conv_dw_cpu(
        int batch_size, int in_row_dim, int in_col_dim,
        int channels, int out_row_dim, int out_col_dim,
        int stride, int padding, int kernel_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding) {

  const bool no_pool = pool_stride == 0;
  if (no_pool) {
    conv_dw_cpu_without_pool(
        batch_size, in_row_dim, in_col_dim,
        channels, out_row_dim, out_col_dim,
        stride, padding, kernel_dim,
        input, weights, bias, output,
        act, scale);
    return;
  }

  const bool no_bias = bias == NULL;
  const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

  for (int b = 0; b < batch_size; b++) {
    for (int porow = 0; porow < pool_out_row_dim; porow++) {
      for (int pocol = 0; pocol < pool_out_col_dim; pocol++) {
        for (int ch = 0; ch < channels; ch++) {

          elem_t running_max = 0;
          bool running_max_initialized = false;

          for (int pwrow = 0; pwrow < pool_size; pwrow++) {
            const int orow = porow * pool_stride + pwrow - pool_padding;

            for (int pwcol = 0; pwcol < pool_size; pwcol++) {
              const int ocol = pocol * pool_stride + pwcol - pool_padding;

              if (orow < 0 || orow >= out_row_dim || ocol < 0 || ocol >= out_col_dim) {
                if (!running_max_initialized || running_max < 0) {
                  running_max = 0;
                  running_max_initialized = true;
                }
              } else {

                acc_t opixel = no_bias ? 0 : bias[ch];

                for (int krow = 0; krow < kernel_dim; krow++) {
                  const int irow = orow * stride + krow - padding;

                  for (int kcol = 0; kcol < kernel_dim; kcol++) {
                    const int icol = ocol * stride + kcol - padding;

                    const elem_t * in = input + (b * in_row_dim * in_col_dim + irow * in_col_dim + icol) * channels + ch;

                    elem_t ipixel = irow < 0 || irow >= in_row_dim || icol < 0 || icol >= in_col_dim ?
                        0 : *in;

                    const elem_t weight = *(weights + (ch * kernel_dim + krow) * kernel_dim  + kcol);

                    opixel += weight * ipixel;
                  }
                }

                opixel = scale_and_sat(opixel, act, scale, 0);
                if (!running_max_initialized || opixel > running_max) {
                  running_max = opixel;
                  running_max_initialized = true;
                }
              }

              if (pwrow == pool_size - 1 && pwcol == pool_size - 1) {
                elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * channels + ch;

                *out = running_max;
              }
            }
          }
        }
      }
    }
  }
}


static void tiled_conv(
        int custom_num, 
        int batch_size,
        int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        int batches,
        int porows, int pocols, int pochs,
        int krows, int kcols, int kchs,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type) {

#ifdef GEMMINI_ASSERTIONS
  if (trans_weight_1203 && trans_weight_0132) {
    printf("Only one weight transformation can be applied at a time\n");
    exit(1);
  }
#endif

    if (tiled_conv_type == CPU) {
      if (pool_size == 1 && pool_stride == 1 && pool_padding == 0) {
        pool_stride = 0;
      }

      // assume in_dim_rows = in_dim_cols
      // and out_dim_rows = out_dim_cols for now
      conv_cpu(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale,
        pool_size, pool_stride, pool_padding);
      return;
    } else if (tiled_conv_type == OS) {
      printf("Gemmini convs do not currently support OS\n");
      exit(1);
    }

    // TODO move everything below this into a tiled_conv_outer function to match the tiled_matmul function

    bool no_bias = false;
    if (bias == NULL) {
        bias = (acc_t*)1;
        no_bias = true;
    }

    bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const bool downsample = stride == 2 && kernel_dim == 1 && in_row_dim % 2 == 0 && in_col_dim % 2 == 0
      && padding == 0 && no_pool && input_dilation == 1 && !trans_input_3120;

    const int input_dilated = input_dilation == 2;

#ifdef GEMMINI_ASSERTIONS
    {
        // const int orows = porows * pool_stride + pool_size - 1;
        // const int ocols = pocols * pool_stride + pool_size - 1;

        // Check that data will fit in scratchpad
        const int spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
        const int acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

        if (spad_rows > BANK_NUM * BANK_ROWS / 2) {
            printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
            exit(1);
        }
        if (acc_rows > ACC_ROWS / 2) {
            printf("not enough accumulator space to store outputs\n");
            exit(1);
        }
        if (kernel_dim <= padding) {
            printf("kernel_dim must be larger than padding\n");
            exit(1);
        }
        if (input_dilation > 2) {
            printf("input_dilation > 2 is only supported on CPU\n");
            exit(1);
        }
        if (input_dilation > 1 && stride > 1) {
            printf("input input_dilation is only supported when stride == 1\n");
            exit(1);
        }
        if (trans_output_1203 && !no_pool) {
            printf("Output can only be transposed when pooling is disabled\n");
            exit(1);
        }
        if (trans_input_3120 && trans_weight_0132) {
            printf("Cannot transpose innermost dimensions of both inputs and weights on WS.\n");
            exit(1);
        }
    }
#endif

    const size_t st_dram_stride = trans_output_1203 ?
        batch_size * out_channels * sizeof(elem_t) :
        out_channels * sizeof(elem_t);

    switch (custom_num) {
      case 0:
        gemmini_extended_config_st(custom0, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom0, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      case 1:
        gemmini_extended_config_st(custom1, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom1, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      case 2:
        gemmini_extended_config_st(custom2, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom2, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      case 3:
        gemmini_extended_config_st(custom3, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom3, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
    }

    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int dilated_in_row_dim = in_row_dim + (input_dilation - 1) * (in_row_dim- 1);
    const int dilated_in_col_dim = in_col_dim + (input_dilation - 1) * (in_col_dim- 1);

    for (int b = 0; b < batch_size; b += batches) {
        for (int porow = 0; porow < pool_out_row_dim; porow += porows) {
            const int orow = porow * pool_stride - pool_padding;

            for (int pocol = 0; pocol < pool_out_col_dim; pocol += pocols) {
                const int ocol = pocol * pool_stride - pool_padding;

                for (int poch = 0; poch < out_channels; poch += pochs) {
                    for (int krow = 0; krow < kernel_dim; krow += krows) {
                        const int orow_floored = orow < 0 ? 0 : orow;
                        int irow = orow_floored * stride + krow * kernel_dilation - padding;

                        for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                            const int ocol_floored = ocol < 0 ? 0 : ocol;
                            int icol = ocol_floored * stride + kcol * kernel_dilation - padding;

                            for (int kch = 0; kch < in_channels; kch += kchs) {
                                elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * out_channels + poch;
                                if (trans_output_1203) {
                                    out = output + (porow * pool_out_col_dim * batch_size + pocol * batch_size + b) * out_channels + poch;
                                }

                                if (krow + krows < kernel_dim ||
                                        kcol + kcols < kernel_dim ||
                                        kch + kchs < in_channels) {
                                    out = NULL;
                                }

                                const acc_t * bias_ = bias + poch;
                                if (krow > 0 ||
                                        kcol > 0 ||
                                        kch > 0) {
                                    bias_ = NULL;
                                }

                                const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                                const int porows_ = pool_out_row_dim - porow > porows ? porows : pool_out_row_dim - porow;
                                const int pocols_ = pool_out_col_dim - pocol > pocols ? pocols : pool_out_col_dim - pocol;
                                const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
                                const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                                const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                                const int kchs_ = in_channels - kch > kchs ? kchs : in_channels - kch;

                                const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                                const int orows_ = porows_ * pool_stride + pool_size - 1;

                                const int plpad = ocol < 0 ? -ocol : 0;
                                const int prpad = ocol + ocols_ > out_col_dim ? ocol + ocols_ - out_col_dim : 0;
                                const int pupad = orow < 0 ? -orow : 0;
                                const int pdpad = orow + orows_ > out_row_dim ? orow + orows_ - out_row_dim : 0;

                                const int dilated_krows_ = krows_ + (kernel_dilation - 1)*(krows_ - 1);
                                const int dilated_kcols_ = kcols_ + (kernel_dilation - 1)*(kcols_ - 1);

                                const int icols_ = (ocols_ - plpad - prpad) * stride + dilated_kcols_ - 1;
                                const int irows_ = (orows_ - pupad - pdpad) * stride + dilated_krows_ - 1;

                                int lpad = icol < 0 ? -icol : 0;
                                int rpad = icol + icols_ > dilated_in_col_dim ? icol + icols_ - dilated_in_col_dim : 0;
                                int upad = irow < 0 ? -irow : 0;
                                int dpad = irow + irows_ > dilated_in_row_dim ? irow + irows_ - dilated_in_row_dim : 0;

                                if (input_dilated) {
                                  lpad += lpad == 0 && icol % 2 != 0;
                                  rpad += rpad == 0 && (icol + icols_) % 2 != 1;
                                  upad += upad == 0 && irow % 2 != 0;
                                  dpad += dpad == 0 && (irow + irows_) % 2 != 1;
                                }

                                int krow_ = krow;
                                int kcol_ = kcol;
                                if (wrot180) {
                                  krow_ = kernel_dim - krow - krows_;
                                  kcol_ = kernel_dim - kcol - kcols_;
                                }

                                const elem_t * weights_slice = weights + (krow_*kernel_dim*in_channels + kcol_*in_channels + kch) * out_channels + poch;
                                if (trans_weight_1203) {
                                  weights_slice = weights + (kch*kernel_dim*kernel_dim + krow_*kernel_dim+kcol_) * out_channels + poch;
                                } else if (trans_weight_0132) {
                                  weights_slice = weights + (krow_*kernel_dim*out_channels + kcol_*out_channels + poch) * in_channels + kch;
                                }

                                const elem_t * in = input + (b *in_row_dim * in_col_dim + ((irow+upad)>>input_dilated) * in_col_dim + ((icol+lpad)>>input_dilated)) * in_channels + kch;
                                if (trans_input_3120) {
                                  in = input + (kch * in_row_dim * in_col_dim + ((irow+upad)>>input_dilated) * in_col_dim + ((icol+lpad)>>input_dilated)) * batch_size + b;
                                }

                                sp_tiled_conv(
                                    custom_num, 
                                    batch_size, in_row_dim, in_col_dim, in_channels,
                                    out_channels, out_row_dim, out_col_dim,
                                    pool_out_row_dim, pool_out_col_dim,

                                    stride, padding, kernel_dim, kernel_dilation,

                                    pool_size, pool_stride, pool_padding,

                                    batches_,
                                    porows_, pocols_, pochs_,
                                    krows_, kcols_, kchs_,

                                    lpad, rpad, upad, dpad,
                                    plpad, prpad, pupad, pdpad,

                                    in,
                                    weights_slice,
                                    out,
                                    bias_,

                                    act, scale,

                                    wrot180, trans_output_1203, trans_input_3120,
                                    trans_weight_1203, trans_weight_0132,

                                    no_bias, no_pool, downsample, input_dilated,
                                    false);

                            }
                        }
                    }
                }
            }
        }
    }
}

// made
static void shared_multi_tiled_conv(
    int gemmini_list, int tile_id,
    size_t sp_addr_start, size_t acc_addr_start,
    int batch_size,
    int in_row_dim, int in_col_dim, int in_channels,
    int out_channels, int out_row_dim, int out_col_dim,
    int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
    bool wrot180, bool trans_output_1203, bool trans_input_3120,
    bool trans_weight_1203, bool trans_weight_0132,

    int batches,
    int porows, int pocols, int pochs,
    int krows, int kcols, int kchs,

    const elem_t *input,
    const elem_t *weights,
    const acc_t *bias,
    elem_t *output,

    int act, acc_scale_t scale,
    int pool_size, int pool_stride, int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type)
{
  int gemmini_num = 0;

#ifdef GEMMINI_ASSERTIONS
  if (trans_weight_1203 && trans_weight_0132)
  {
    printf("Only one weight transformation can be applied at a time\n");
    exit(1);
  }
#endif

  if (tiled_conv_type == CPU)
  {
    if (pool_size == 1 && pool_stride == 1 && pool_padding == 0)
    {
      pool_stride = 0;
    }

    // assume in_dim_rows = in_dim_cols
    // and out_dim_rows = out_dim_cols for now
    conv_cpu(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale,
        pool_size, pool_stride, pool_padding);
    return;
  }
  else if (tiled_conv_type == OS)
  {
    printf("Gemmini convs do not currently support OS\n");
    exit(1);
  }

  // TODO move everything below this into a tiled_conv_outer function to match the tiled_matmul function

  bool no_bias = false;
  if (bias == NULL)
  {
    bias = (acc_t *)1;
    no_bias = true;
  }

  bool no_pool = pool_stride == 0;
  if (no_pool)
  {
    pool_size = 1;
    pool_stride = 1;
    pool_padding = 0;
  }

  const bool downsample = stride == 2 && kernel_dim == 1 && in_row_dim % 2 == 0 && in_col_dim % 2 == 0 && padding == 0 && no_pool && input_dilation == 1 && !trans_input_3120;

  const int input_dilated = input_dilation == 2;

#ifdef GEMMINI_ASSERTIONS
  {
    // const int orows = porows * pool_stride + pool_size - 1;
    // const int ocols = pocols * pool_stride + pool_size - 1;

    // Check that data will fit in scratchpad
    const int spad_rows = tiled_conv_total_spad_rows(false,
                                                     stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                                     batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
    const int acc_rows = tiled_conv_total_spad_rows(true,
                                                    stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                                    batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

    if (spad_rows > BANK_NUM * BANK_ROWS / 2)
    {
      printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
      exit(1);
    }
    if (acc_rows > ACC_ROWS / 2)
    {
      printf("not enough accumulator space to store outputs\n");
      exit(1);
    }
    if (kernel_dim <= padding)
    {
      printf("kernel_dim must be larger than padding\n");
      exit(1);
    }
    if (input_dilation > 2)
    {
      printf("input_dilation > 2 is only supported on CPU\n");
      exit(1);
    }
    if (input_dilation > 1 && stride > 1)
    {
      printf("input input_dilation is only supported when stride == 1\n");
      exit(1);
    }
    if (trans_output_1203 && !no_pool)
    {
      printf("Output can only be transposed when pooling is disabled\n");
      exit(1);
    }
    if (trans_input_3120 && trans_weight_0132)
    {
      printf("Cannot transpose innermost dimensions of both inputs and weights on WS.\n");
      exit(1);
    }
  }
#endif

  const size_t st_dram_stride = trans_output_1203 ? batch_size * out_channels * sizeof(elem_t) : out_channels * sizeof(elem_t);

  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((gemmini_list >> i) & 1)
    {
      gemmini_num++;
      switch (i)
      {
      case 0:
        gemmini_extended_config_st(custom0, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom0, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      case 1:
        gemmini_extended_config_st(custom1, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom1, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      case 2:
        gemmini_extended_config_st(custom2, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom2, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      case 3:
        gemmini_extended_config_st(custom3, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom3, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      }
    }
  }

  const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int dilated_in_row_dim = in_row_dim + (input_dilation - 1) * (in_row_dim - 1);
  const int dilated_in_col_dim = in_col_dim + (input_dilation - 1) * (in_col_dim - 1);

  int inner_call_counter = 0;
  int lastK_toggle = 0;
  for (int b = 0; b < batch_size; b += batches)
  {
    for (int porow = 0; porow < pool_out_row_dim; porow += porows)
    {
      const int orow = porow * pool_stride - pool_padding;

      for (int pocol = 0; pocol < pool_out_col_dim; pocol += pocols)
      {
        const int ocol = pocol * pool_stride - pool_padding;

        for (int poch = 0; poch < out_channels; poch += pochs)
        {
          for (int krow = 0; krow < kernel_dim; krow += krows)
          {
            const int orow_floored = orow < 0 ? 0 : orow;
            int irow = orow_floored * stride + krow * kernel_dilation - padding;

            for (int kcol = 0; kcol < kernel_dim; kcol += kcols)
            {
              const int ocol_floored = ocol < 0 ? 0 : ocol;
              int icol = ocol_floored * stride + kcol * kernel_dilation - padding;

              for (int kch = 0; kch < in_channels; kch += kchs)
              {
                elem_t *out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * out_channels + poch;
                if (trans_output_1203)
                {
                  out = output + (porow * pool_out_col_dim * batch_size + pocol * batch_size + b) * out_channels + poch;
                }

                if (krow + krows < kernel_dim ||
                    kcol + kcols < kernel_dim ||
                    kch + kchs < in_channels)
                {
                  out = NULL;
                }

                const acc_t *bias_ = bias + poch;
                if (krow > 0 ||
                    kcol > 0 ||
                    kch > 0)
                {
                  bias_ = NULL;
                }

                const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                const int porows_ = pool_out_row_dim - porow > porows ? porows : pool_out_row_dim - porow;
                const int pocols_ = pool_out_col_dim - pocol > pocols ? pocols : pool_out_col_dim - pocol;
                const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
                const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                const int kchs_ = in_channels - kch > kchs ? kchs : in_channels - kch;

                const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                const int orows_ = porows_ * pool_stride + pool_size - 1;

                const int plpad = ocol < 0 ? -ocol : 0;
                const int prpad = ocol + ocols_ > out_col_dim ? ocol + ocols_ - out_col_dim : 0;
                const int pupad = orow < 0 ? -orow : 0;
                const int pdpad = orow + orows_ > out_row_dim ? orow + orows_ - out_row_dim : 0;

                const int dilated_krows_ = krows_ + (kernel_dilation - 1) * (krows_ - 1);
                const int dilated_kcols_ = kcols_ + (kernel_dilation - 1) * (kcols_ - 1);

                const int icols_ = (ocols_ - plpad - prpad) * stride + dilated_kcols_ - 1;
                const int irows_ = (orows_ - pupad - pdpad) * stride + dilated_krows_ - 1;

                int lpad = icol < 0 ? -icol : 0;
                int rpad = icol + icols_ > dilated_in_col_dim ? icol + icols_ - dilated_in_col_dim : 0;
                int upad = irow < 0 ? -irow : 0;
                int dpad = irow + irows_ > dilated_in_row_dim ? irow + irows_ - dilated_in_row_dim : 0;

                if (input_dilated)
                {
                  lpad += lpad == 0 && icol % 2 != 0;
                  rpad += rpad == 0 && (icol + icols_) % 2 != 1;
                  upad += upad == 0 && irow % 2 != 0;
                  dpad += dpad == 0 && (irow + irows_) % 2 != 1;
                }

                int krow_ = krow;
                int kcol_ = kcol;
                if (wrot180)
                {
                  krow_ = kernel_dim - krow - krows_;
                  kcol_ = kernel_dim - kcol - kcols_;
                }

                const elem_t *weights_slice = weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + poch;
                if (trans_weight_1203)
                {
                  weights_slice = weights + (kch * kernel_dim * kernel_dim + krow_ * kernel_dim + kcol_) * out_channels + poch;
                }
                else if (trans_weight_0132)
                {
                  weights_slice = weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + poch) * in_channels + kch;
                }

                const elem_t *in = input + (b * in_row_dim * in_col_dim + ((irow + upad) >> input_dilated) * in_col_dim + ((icol + lpad) >> input_dilated)) * in_channels + kch;
                if (trans_input_3120)
                {
                  in = input + (kch * in_row_dim * in_col_dim + ((irow + upad) >> input_dilated) * in_col_dim + ((icol + lpad) >> input_dilated)) * batch_size + b;
                }

                const int kchs_Remain = kchs_ % DIM;
                const int kchs_DIM = kchs_ / DIM + (kchs_Remain != 0);

                const int pochs_Remain = pochs_ % DIM;
                const int pochs_DIM = pochs_ / DIM + (pochs_Remain != 0);

                int using_gemmini_num = gemmini_num;
                if (kchs_DIM < gemmini_num && pochs_DIM < gemmini_num)
                {
                  using_gemmini_num = (kchs_DIM < pochs_DIM) ? pochs_DIM : kchs_DIM;
                }

                int group_list = gemmini_list;
                int shift_num = 0;
                for (int i = 0; i < total_gemmini_num; i++)
                {
                  size_t shifted_list = (gemmini_list >> i);
                  if (shift_num == using_gemmini_num)
                  {
                    group_list = group_list & ~(shifted_list << i);
                    break;
                  }
                  if (shifted_list & 1)
                  {
                    shift_num++;
                  }
                }

                // for LdInput
                const size_t kchs_DIM_div = kchs_DIM / using_gemmini_num;
                const size_t kchs_DIM_div_added = kchs_DIM_div + 1;
                const size_t kchs_added_gemmini_num = kchs_DIM % using_gemmini_num;
                
                // for LdWeights, LdBias, Execute, St
                const size_t ochs_DIM_div = pochs_DIM / using_gemmini_num;
                const size_t ochs_DIM_div_added = ochs_DIM_div + 1;
                const size_t ochs_added_gemmini_num = pochs_DIM % using_gemmini_num;

                // for ld, st dram address
                const elem_t *in_local = in;
                const elem_t *weights_slice_local = weights_slice;
                const elem_t *out_local = out;
                const acc_t *bias_local = bias_;

                // for local addr
                size_t laddrochs_offset = 0;
                size_t laddrkchs_offset = 0;

                const int t = (inner_call_counter & 1);
                const size_t local_sp_addr_start = (t == 0) ? sp_addr_start : sp_addr_start + BANK_NUM * BANK_ROWS / 2;
                const size_t local_sp_addr_end = (t == 0) ? sp_addr_start + BANK_NUM * BANK_ROWS / 2 : sp_addr_start + BANK_NUM * BANK_ROWS - 1;
                const size_t local_acc_addr_start = lastK_toggle ? acc_addr_start : acc_addr_start + ACC_ROWS / 2;

                int activated_gemmini_num = 0;
                for (int i = 0; i< total_gemmini_num; i++)
                {
                  if ((group_list >> i) & 1)
                  {
                    size_t this_ochs = (activated_gemmini_num < ochs_added_gemmini_num) ? ochs_DIM_div_added * DIM : ochs_DIM_div * DIM;
                    int last_ochs_gemmini = (ochs_DIM_div == 0) ? ochs_added_gemmini_num - 1 : using_gemmini_num - 1;
                    if (pochs_Remain != 0 && activated_gemmini_num == last_ochs_gemmini){
                      this_ochs = this_ochs - DIM + pochs_Remain;
                    }

                    size_t this_kchs = (activated_gemmini_num < kchs_added_gemmini_num) ? kchs_DIM_div_added * DIM : kchs_DIM_div * DIM;
                    int last_kchs_gemmini = (kchs_DIM_div == 0) ? kchs_added_gemmini_num - 1 : using_gemmini_num - 1;
                    if (kchs_Remain != 0 && activated_gemmini_num == last_kchs_gemmini){
                      this_kchs = this_kchs - DIM + kchs_Remain;
                    }

                    shared_multi_sp_tiled_conv(
                        i, group_list, tile_id << 1 | t,
                        local_sp_addr_start, local_sp_addr_end, local_acc_addr_start,
                        batch_size, in_row_dim, in_col_dim, in_channels,
                        out_channels, out_row_dim, out_col_dim,
                        pool_out_row_dim, pool_out_col_dim,

                        stride, padding, kernel_dim, kernel_dilation,

                        pool_size, pool_stride, pool_padding,

                        this_ochs, this_kchs, laddrochs_offset, laddrkchs_offset,

                        batches_,
                        porows_, pocols_, pochs_,
                        krows_, kcols_, kchs_,

                        lpad, rpad, upad, dpad,
                        plpad, prpad, pupad, pdpad,

                        in_local,
                        weights_slice_local,
                        out_local,
                        bias_local,

                        act, scale,

                        wrot180, trans_output_1203, trans_input_3120,
                        trans_weight_1203, trans_weight_0132,

                        no_bias, no_pool, downsample, input_dilated,
                        false);

                    laddrochs_offset += this_ochs;
                    laddrkchs_offset += this_kchs;

                    if (trans_input_3120) {
                      in_local += this_kchs * in_row_dim * in_col_dim * batch_size;
                    } else {
                      in_local += this_kchs;
                    }

                    if (trans_weight_0132) {
                      weights_slice_local += this_ochs * in_channels;
                    } else {
                      weights_slice_local += this_ochs;
                    }
                   
                    out_local += this_ochs;
                    if (krow + krows < kernel_dim ||
                        kcol + kcols < kernel_dim ||
                        kch + kchs < in_channels)
                    {
                      out_local = NULL;
                    }

                    bias_local += this_ochs;
                    if (krow > 0 ||
                        kcol > 0 ||
                        kch > 0)
                    {
                      bias_local = NULL;
                    }

                    activated_gemmini_num++;
                  }
                }

                // for (int i = 0; i < total_gemmini_num; i++)
                // {
                //   if ((group_list >> i) & 1)
                //   {
                //     switch (i)
                //     {
                //     case 3:
                //       // printf("delay");
                //       ROCC_INSTRUCTION_RS1_RS2(custom3, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
                //       break;
                //     case 2:
                //       ROCC_INSTRUCTION_RS1_RS2(custom2, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
                //       break;
                //     case 1:
                //       ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
                //       break;
                //     case 0:
                //       ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)), ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
                //       break;
                //     }
                //   }
                // }

                inner_call_counter++;
                if (krow + krows >= kernel_dim &&
                    kcol + kcols >= kernel_dim &&
                    kch + kchs >= in_channels)
                  lastK_toggle ^= 1;
              }
            }
          }
        }
      }
    }
  }
}

// made
static void shared_multi_tiled_conv_test(
    int gemmini_list, int tile_id,
    size_t sp_addr_start, size_t acc_addr_start,
    size_t sp_addr_range, size_t acc_addr_range,
    int batch_size,
    int in_row_dim, int in_col_dim, int in_channels,
    int out_channels, int out_row_dim, int out_col_dim,
    int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
    bool wrot180, bool trans_output_1203, bool trans_input_3120,
    bool trans_weight_1203, bool trans_weight_0132,

    int batches,
    int porows, int pocols, int pochs,
    int krows, int kcols, int kchs,

    const elem_t *input,
    const elem_t *weights,
    const acc_t *bias,
    elem_t *output,

    int act, acc_scale_t scale,
    int pool_size, int pool_stride, int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type)
{
  int gemmini_num = 0;

#ifdef GEMMINI_ASSERTIONS
  if (trans_weight_1203 && trans_weight_0132)
  {
    printf("Only one weight transformation can be applied at a time\n");
    exit(1);
  }
#endif

  if (tiled_conv_type == CPU)
  {
    if (pool_size == 1 && pool_stride == 1 && pool_padding == 0)
    {
      pool_stride = 0;
    }

    // assume in_dim_rows = in_dim_cols
    // and out_dim_rows = out_dim_cols for now
    conv_cpu(
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,
        input, weights, bias, output,
        act, scale,
        pool_size, pool_stride, pool_padding);
    return;
  }
  else if (tiled_conv_type == OS)
  {
    printf("Gemmini convs do not currently support OS\n");
    exit(1);
  }

  // TODO move everything below this into a tiled_conv_outer function to match the tiled_matmul function

  bool no_bias = false;
  if (bias == NULL)
  {
    bias = (acc_t *)1;
    no_bias = true;
  }

  bool no_pool = pool_stride == 0;
  if (no_pool)
  {
    pool_size = 1;
    pool_stride = 1;
    pool_padding = 0;
  }

  const bool downsample = stride == 2 && kernel_dim == 1 && in_row_dim % 2 == 0 && in_col_dim % 2 == 0 && padding == 0 && no_pool && input_dilation == 1 && !trans_input_3120;

  const int input_dilated = input_dilation == 2;

#ifdef GEMMINI_ASSERTIONS
  {
    // const int orows = porows * pool_stride + pool_size - 1;
    // const int ocols = pocols * pool_stride + pool_size - 1;

    // Check that data will fit in scratchpad
    const int spad_rows = tiled_conv_total_spad_rows(false,
                                                     stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                                     batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);
    const int acc_rows = tiled_conv_total_spad_rows(true,
                                                    stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                                    batches, porows, pocols, pochs, krows, kcols, kchs, pool_size, pool_stride);

    if (spad_rows > sp_addr_range / 2)
    {
      printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
      exit(1);
    }
    if (acc_rows > acc_addr_range / 2)
    {
      printf("not enough accumulator space to store outputs\n");
      exit(1);
    }
    if (kernel_dim <= padding)
    {
      printf("kernel_dim must be larger than padding\n");
      exit(1);
    }
    if (input_dilation > 2)
    {
      printf("input_dilation > 2 is only supported on CPU\n");
      exit(1);
    }
    if (input_dilation > 1 && stride > 1)
    {
      printf("input input_dilation is only supported when stride == 1\n");
      exit(1);
    }
    if (trans_output_1203 && !no_pool)
    {
      printf("Output can only be transposed when pooling is disabled\n");
      exit(1);
    }
    if (trans_input_3120 && trans_weight_0132)
    {
      printf("Cannot transpose innermost dimensions of both inputs and weights on WS.\n");
      exit(1);
    }
  }
#endif

  const size_t st_dram_stride = trans_output_1203 ? batch_size * out_channels * sizeof(elem_t) : out_channels * sizeof(elem_t);

  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((gemmini_list >> i) & 1)
    {
      gemmini_num++;
      switch (i)
      {
      case 0:
        gemmini_extended_config_st(custom0, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom0, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      case 1:
        gemmini_extended_config_st(custom1, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom1, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      case 2:
        gemmini_extended_config_st(custom2, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom2, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      case 3:
        gemmini_extended_config_st(custom3, st_dram_stride, act, scale);
        gemmini_extended3_config_ex(custom3, WEIGHT_STATIONARY, 0, 0, 0, input_dilation, stride >> downsample, trans_input_3120, trans_weight_0132, false);
        break;
      }
    }
  }

  const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int dilated_in_row_dim = in_row_dim + (input_dilation - 1) * (in_row_dim - 1);
  const int dilated_in_col_dim = in_col_dim + (input_dilation - 1) * (in_col_dim - 1);

  int inner_call_counter = 0;
  int lastK_toggle = 0;
  for (int b = 0; b < batch_size; b += batches)
  {
    for (int porow = 0; porow < pool_out_row_dim; porow += porows)
    {
      const int orow = porow * pool_stride - pool_padding;

      for (int pocol = 0; pocol < pool_out_col_dim; pocol += pocols)
      {
        const int ocol = pocol * pool_stride - pool_padding;

        for (int poch = 0; poch < out_channels; poch += pochs)
        {
          for (int krow = 0; krow < kernel_dim; krow += krows)
          {
            const int orow_floored = orow < 0 ? 0 : orow;
            int irow = orow_floored * stride + krow * kernel_dilation - padding;

            for (int kcol = 0; kcol < kernel_dim; kcol += kcols)
            {
              const int ocol_floored = ocol < 0 ? 0 : ocol;
              int icol = ocol_floored * stride + kcol * kernel_dilation - padding;

              for (int kch = 0; kch < in_channels; kch += kchs)
              {
                elem_t *out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * out_channels + poch;
                if (trans_output_1203)
                {
                  out = output + (porow * pool_out_col_dim * batch_size + pocol * batch_size + b) * out_channels + poch;
                }

                if (krow + krows < kernel_dim ||
                    kcol + kcols < kernel_dim ||
                    kch + kchs < in_channels)
                {
                  out = NULL;
                }

                const acc_t *bias_ = bias + poch;
                if (krow > 0 ||
                    kcol > 0 ||
                    kch > 0)
                {
                  bias_ = NULL;
                }

                const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                const int porows_ = pool_out_row_dim - porow > porows ? porows : pool_out_row_dim - porow;
                const int pocols_ = pool_out_col_dim - pocol > pocols ? pocols : pool_out_col_dim - pocol;
                const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
                const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
                const int kchs_ = in_channels - kch > kchs ? kchs : in_channels - kch;

                const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                const int orows_ = porows_ * pool_stride + pool_size - 1;

                const int plpad = ocol < 0 ? -ocol : 0;
                const int prpad = ocol + ocols_ > out_col_dim ? ocol + ocols_ - out_col_dim : 0;
                const int pupad = orow < 0 ? -orow : 0;
                const int pdpad = orow + orows_ > out_row_dim ? orow + orows_ - out_row_dim : 0;

                const int dilated_krows_ = krows_ + (kernel_dilation - 1) * (krows_ - 1);
                const int dilated_kcols_ = kcols_ + (kernel_dilation - 1) * (kcols_ - 1);

                const int icols_ = (ocols_ - plpad - prpad) * stride + dilated_kcols_ - 1;
                const int irows_ = (orows_ - pupad - pdpad) * stride + dilated_krows_ - 1;

                int lpad = icol < 0 ? -icol : 0;
                int rpad = icol + icols_ > dilated_in_col_dim ? icol + icols_ - dilated_in_col_dim : 0;
                int upad = irow < 0 ? -irow : 0;
                int dpad = irow + irows_ > dilated_in_row_dim ? irow + irows_ - dilated_in_row_dim : 0;

                if (input_dilated)
                {
                  lpad += lpad == 0 && icol % 2 != 0;
                  rpad += rpad == 0 && (icol + icols_) % 2 != 1;
                  upad += upad == 0 && irow % 2 != 0;
                  dpad += dpad == 0 && (irow + irows_) % 2 != 1;
                }

                int krow_ = krow;
                int kcol_ = kcol;
                if (wrot180)
                {
                  krow_ = kernel_dim - krow - krows_;
                  kcol_ = kernel_dim - kcol - kcols_;
                }

                const elem_t *weights_slice = weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + poch;
                if (trans_weight_1203)
                {
                  weights_slice = weights + (kch * kernel_dim * kernel_dim + krow_ * kernel_dim + kcol_) * out_channels + poch;
                }
                else if (trans_weight_0132)
                {
                  weights_slice = weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + poch) * in_channels + kch;
                }

                const elem_t *in = input + (b * in_row_dim * in_col_dim + ((irow + upad) >> input_dilated) * in_col_dim + ((icol + lpad) >> input_dilated)) * in_channels + kch;
                if (trans_input_3120)
                {
                  in = input + (kch * in_row_dim * in_col_dim + ((irow + upad) >> input_dilated) * in_col_dim + ((icol + lpad) >> input_dilated)) * batch_size + b;
                }

                const int kchs_Remain = kchs_ % DIM;
                const int kchs_DIM = kchs_ / DIM + (kchs_Remain != 0);

                const int pochs_Remain = pochs_ % DIM;
                const int pochs_DIM = pochs_ / DIM + (pochs_Remain != 0);

                int using_gemmini_num = gemmini_num;
                if (kchs_DIM < gemmini_num && pochs_DIM < gemmini_num)
                {
                  using_gemmini_num = (kchs_DIM < pochs_DIM) ? pochs_DIM : kchs_DIM;
                }

                int group_list = gemmini_list;
                int shift_num = 0;
                for (int i = 0; i < total_gemmini_num; i++)
                {
                  size_t shifted_list = (gemmini_list >> i);
                  if (shift_num == using_gemmini_num)
                  {
                    group_list = group_list & ~(shifted_list << i);
                    break;
                  }
                  if (shifted_list & 1)
                  {
                    shift_num++;
                  }
                }

                // for LdInput
                const size_t kchs_DIM_div = kchs_DIM / using_gemmini_num;
                const size_t kchs_DIM_div_added = kchs_DIM_div + 1;
                const size_t kchs_added_gemmini_num = kchs_DIM % using_gemmini_num;

                // for LdWeights, LdBias, Execute, St
                const size_t ochs_DIM_div = pochs_DIM / using_gemmini_num;
                const size_t ochs_DIM_div_added = ochs_DIM_div + 1;
                const size_t ochs_added_gemmini_num = pochs_DIM % using_gemmini_num;

                // for ld, st dram address
                const elem_t *in_local = in;
                const elem_t *weights_slice_local = weights_slice;
                const elem_t *out_local = out;
                const acc_t *bias_local = bias_;

                // for local addr
                size_t laddrochs_offset = 0;
                size_t laddrkchs_offset = 0;

                const int t = (inner_call_counter & 1);
                const size_t local_sp_addr_start = (t == 0) ? sp_addr_start : sp_addr_start + sp_addr_range / 2;
                const size_t local_sp_addr_end = (t == 0) ? sp_addr_start + BANK_NUM * BANK_ROWS / 2 : sp_addr_start + BANK_NUM * BANK_ROWS - 1;
                const size_t local_acc_addr_start = lastK_toggle ? acc_addr_start : acc_addr_start + acc_addr_range / 2;

                int activated_gemmini_num = 0;
                for (int i = 0; i < total_gemmini_num; i++)
                {
                  if ((group_list >> i) & 1)
                  {
                    size_t this_ochs = (activated_gemmini_num < ochs_added_gemmini_num) ? ochs_DIM_div_added * DIM : ochs_DIM_div * DIM;
                    int last_ochs_gemmini = (ochs_DIM_div == 0) ? ochs_added_gemmini_num - 1 : using_gemmini_num - 1;
                    if (pochs_Remain != 0 && activated_gemmini_num == last_ochs_gemmini)
                    {
                      this_ochs = this_ochs - DIM + pochs_Remain;
                    }

                    size_t this_kchs = (activated_gemmini_num < kchs_added_gemmini_num) ? kchs_DIM_div_added * DIM : kchs_DIM_div * DIM;
                    int last_kchs_gemmini = (kchs_DIM_div == 0) ? kchs_added_gemmini_num - 1 : using_gemmini_num - 1;
                    if (kchs_Remain != 0 && activated_gemmini_num == last_kchs_gemmini)
                    {
                      this_kchs = this_kchs - DIM + kchs_Remain;
                    }

                    shared_multi_sp_tiled_conv_test(
                        i, group_list, tile_id << 1 | t,
                        local_sp_addr_start, local_sp_addr_end, local_acc_addr_start,
                        batch_size, in_row_dim, in_col_dim, in_channels,
                        out_channels, out_row_dim, out_col_dim,
                        pool_out_row_dim, pool_out_col_dim,

                        stride, padding, kernel_dim, kernel_dilation,

                        pool_size, pool_stride, pool_padding,

                        this_ochs, this_kchs, laddrochs_offset, laddrkchs_offset,

                        batches_,
                        porows_, pocols_, pochs_,
                        krows_, kcols_, kchs_,

                        lpad, rpad, upad, dpad,
                        plpad, prpad, pupad, pdpad,

                        in_local,
                        weights_slice_local,
                        out_local,
                        bias_local,

                        act, scale,

                        wrot180, trans_output_1203, trans_input_3120,
                        trans_weight_1203, trans_weight_0132,

                        no_bias, no_pool, downsample, input_dilated,
                        false);

                    laddrochs_offset += this_ochs;
                    laddrkchs_offset += this_kchs;

                    if (trans_input_3120)
                    {
                      in_local += this_kchs * in_row_dim * in_col_dim * batch_size;
                    }
                    else
                    {
                      in_local += this_kchs;
                    }

                    if (trans_weight_0132)
                    {
                      weights_slice_local += this_ochs * in_channels;
                    }
                    else
                    {
                      weights_slice_local += this_ochs;
                    }

                    out_local += this_ochs;
                    if (krow + krows < kernel_dim ||
                        kcol + kcols < kernel_dim ||
                        kch + kchs < in_channels)
                    {
                      out_local = NULL;
                    }

                    bias_local += this_ochs;
                    if (krow > 0 ||
                        kcol > 0 ||
                        kch > 0)
                    {
                      bias_local = NULL;
                    }

                    activated_gemmini_num++;
                  }
                }

                inner_call_counter++;
                if (krow + krows >= kernel_dim &&
                    kcol + kcols >= kernel_dim &&
                    kch + kchs >= in_channels)
                  lastK_toggle ^= 1;
              }
            }
          }
        }
      }
    }
  }
}

static void tiled_conv_dw(
    int custom_num, 
    int batch_size, int in_row_dim, int in_col_dim,
    int channels, int out_row_dim, int out_col_dim,
    int stride, int padding, int kernel_dim,

    int batches,
    int porows, int pocols,
    int krows, int kcols,

    const elem_t * input,
    const elem_t * weights,
    const acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale,
    int pool_size, int pool_stride, int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type) {

    if (tiled_conv_type == CPU) {
      if (pool_size == 1 && pool_stride == 1 && pool_padding == 0) {
        pool_stride = 0;
      }

      conv_dw_cpu(
        batch_size, in_row_dim, in_col_dim,
        channels, out_row_dim, out_col_dim,
        stride, padding, kernel_dim,
        input, weights, bias, output,
        act, scale,
        pool_size, pool_stride, pool_padding);
      return;
    } else if (tiled_conv_type == OS) {
      printf("Gemmini convs do not currently support OS\n");
      exit(1);
    }

    // TODO move everything below this into a tiled_conv_outer function to match the tiled_matmul function

    bool no_bias = false;
    if (bias == NULL) {
        bias = (acc_t*)1;
        no_bias = true;
    }

    bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

#ifdef GEMMINI_ASSERTIONS
    {
        // const int orows = porows * pool_stride + pool_size - 1;
        // const int ocols = pocols * pool_stride + pool_size - 1;

        // Check that data will fit in scratchpad
        const int spad_rows = tiled_conv_total_spad_rows(false,
            stride, 1, 1, false, false, false,
            batches, porows, pocols, 1, krows, kcols, 1, pool_size, pool_stride);
        const int acc_rows = tiled_conv_total_spad_rows(true,
            stride, 1, 1, false, false, false,
            batches, porows, pocols, 1, krows, kcols, 1, pool_size, pool_stride);

        if (spad_rows > BANK_NUM * BANK_ROWS / 2) {
            printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
            exit(1);
        }
        if (acc_rows > ACC_ROWS / 2) {
            printf("not enough accumulator space to store outputs\n");
            exit(1);
        }
        if (kernel_dim <= padding) {
            printf("kernel_dim must be larger than padding\n");
            exit(1);
        }
    }
#endif

    const size_t st_dram_stride = channels * sizeof(elem_t);
    gemmini_extended_config_st(custom_num, st_dram_stride, act, scale);

    gemmini_extended3_config_ex(custom_num, WEIGHT_STATIONARY, 0, 0, 0, 1, stride, false, false, false);

    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

    for (int b = 0; b < batch_size; b += batches) {
        for (int porow = 0; porow < pool_out_row_dim; porow += porows) {
            const int orow = porow * pool_stride - pool_padding;

            for (int pocol = 0; pocol < pool_out_col_dim; pocol += pocols) {
                const int ocol = pocol * pool_stride - pool_padding;

                for (int ch = 0; ch < channels; ch++) {
                    for (int krow = 0; krow < kernel_dim; krow += krows) {
                        const int orow_floored = orow < 0 ? 0 : orow;
                        int irow = orow_floored * stride + krow - padding;

                        for (int kcol = 0; kcol < kernel_dim; kcol += kcols) {
                            const int ocol_floored = ocol < 0 ? 0 : ocol;
                            int icol = ocol_floored * stride + kcol - padding;

                            elem_t * out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * channels + ch;

                            if (krow + krows < kernel_dim ||
                                    kcol + kcols < kernel_dim) {
                                out = NULL;
                            }

                            const acc_t * bias_ = bias + ch;
                            if (krow > 0 ||
                                    kcol > 0) {
                                bias_ = NULL;
                            }

                            const int batches_ = batch_size - b > batches ? batches : batch_size - b;
                            const int porows_ = pool_out_row_dim - porow > porows ? porows : pool_out_row_dim - porow;
                            const int pocols_ = pool_out_col_dim - pocol > pocols ? pocols : pool_out_col_dim - pocol;
                            const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
                            const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;

                            const int ocols_ = pocols_ * pool_stride + pool_size - 1;
                            const int orows_ = porows_ * pool_stride + pool_size - 1;

                            const int plpad = ocol < 0 ? -ocol : 0;
                            const int prpad = ocol + ocols_ > out_col_dim ? ocol + ocols_ - out_col_dim : 0;
                            const int pupad = orow < 0 ? -orow : 0;
                            const int pdpad = orow + orows_ > out_row_dim ? orow + orows_ - out_row_dim : 0;

                            const int icols_ = (ocols_ - plpad - prpad) * stride + kcols_ - 1;
                            const int irows_ = (orows_ - pupad - pdpad) * stride + krows_ - 1;

                            int lpad = icol < 0 ? -icol : 0;
                            int rpad = icol + icols_ > in_col_dim ? icol + icols_ - in_col_dim : 0;
                            int upad = irow < 0 ? -irow : 0;
                            int dpad = irow + irows_ > in_row_dim ? irow + irows_ - in_row_dim : 0;

                            const elem_t * weights_slice = weights + (ch*kernel_dim + krow) * kernel_dim + kcol;

                            const elem_t *in = input + (b * in_row_dim * in_col_dim + (irow+upad) * in_col_dim + (icol+lpad)) * channels + ch;

                            sp_tiled_conv(
                                custom_num, 
                                batch_size, in_row_dim, in_col_dim, channels,
                                channels, out_row_dim, out_col_dim,
                                pool_out_row_dim, pool_out_col_dim,

                                stride, padding, kernel_dim, 1,

                                pool_size, pool_stride, pool_padding,

                                batches_,
                                porows_, pocols_, 1,
                                krows_, kcols_, 1,

                                lpad, rpad, upad, dpad,
                                plpad, prpad, pupad, pdpad,

                                in,
                                weights_slice,
                                out,
                                bias_,

                                act, scale,

                                false, false, false,
                                false, false,

                                no_bias, no_pool, false, false,
                                true);

                        }
                    }
                }
            }
        }
    }
}


static void tiled_conv_auto(
        int custom_num, 
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,
        int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
        bool wrot180, bool trans_output_1203, bool trans_input_3120,
        bool trans_weight_1203, bool trans_weight_0132,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,
        int pool_size, int pool_stride, int pool_padding,

        enum tiled_matmul_type_t tiled_conv_type) {

    const bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

    const bool downsample = stride == 2 && kernel_dim == 1 && padding == 0 && no_pool && in_row_dim % 2 == 0 && in_col_dim % 2 == 0;

    // Tile convolution params

    // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
    int args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};
    const int max_args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};

    const int orows_idx = 1;
    const int ocols_idx = 2;
    const int out_channels_idx = 3;
    const int in_channels_idx = 6;

    // We divide by 2 for the sake of double-buffering
    const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
    const int max_acc_rows = (ACC_ROWS / 2);

    int spad_rows = tiled_conv_total_spad_rows(false,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    int acc_rows = tiled_conv_total_spad_rows(true,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
        int max_val = -1;
        int max_idx = -1;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            // We avoid reducing ocols when possible to keep the spatial array fully utilized
            if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1)
                    && args[i] > max_val) {
                max_val = args[i];
                max_idx = i;
            }
        }

        if (max_idx == out_channels_idx || max_idx == in_channels_idx) {
            // For input and output channels, there's no point in subtracting by just one
            if (args[max_idx] % DIM != 0) {
                args[max_idx] = (args[max_idx] / DIM) * DIM;
            } else {
                args[max_idx] -= DIM;
            }
            args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
        } else {
            args[max_idx]--;
        }

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
        // printf("batches: %d, orows: %d, ocols: %d, ochs: %d, krows: %d, kcols: %d, kchs: %d\n",
        //        args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
    }

    // Check if we can increase ocols
    bool not_increased = false;
    while (!not_increased) {
        not_increased = true;

        int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
        args_candidate[ocols_idx]++;

        if (args_candidate[ocols_idx] > max_args[ocols_idx])
            continue;

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

        if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
            args[ocols_idx] = args_candidate[ocols_idx];
            not_increased = false;
        }
        // printf("batches: %d, orows: %d, ocols: %d, ochs: %d, krows: %d, kcols: %d, kchs: %d\n",
        //        args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
    }

    // Check if there are any parameters that we can currently still increase
    bool nothing_increased = false;
    while (!nothing_increased) {
        nothing_increased = true;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
            args_candidate[i]++;

            if (args_candidate[i] > max_args[i])
                continue;

            spad_rows = tiled_conv_total_spad_rows(false,
                stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
            acc_rows = tiled_conv_total_spad_rows(true,
                stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

            if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
                args[i] = args_candidate[i];
                nothing_increased = false;
            }
            // printf("batches: %d, orows: %d, ocols: %d, ochs: %d, krows: %d, kcols: %d, kchs: %d\n",
            //        args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
        }
    }

    const int batches = args[0];
    const int orows = args[1];
    const int ocols = args[2];
    const int ochs = args[3];
    const int krows = args[4];
    const int kcols = args[5];
    const int kchs = args[6];

    // printf("batches: %d, orows: %d, ocols: %d, ochs: %d, krows: %d, kcols: %d, kchs: %d\n",
    //     batches, orows, ocols, ochs, krows, kcols, kchs);


    spad_rows = tiled_conv_total_spad_rows(false,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
        stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

#ifdef PRINT_TILE
#if PRINT_TILE
    printf("batches = %d\n", batches);
    printf("orows   = %d\n", orows);
    printf("ocols   = %d\n", ocols);
    printf("ochs    = %d\n", ochs);
    printf("krows   = %d\n", krows);
    printf("kcols   = %d\n", kcols);
    printf("kchs    = %d\n\n", kchs);

    printf("total spad_rows reserved: %d\n", spad_rows);
    printf("total acc_rows reserved: %d\n\n", acc_rows);

    printf("scratchpad row utilization: %d%%\n", (spad_rows*100) / max_spad_rows);
    printf("accumulator row utilization: %d%%\n\n", (acc_rows*100) / max_acc_rows);

    printf("inner matmul size: i=%d, j=%d, k=%d\n\n", ocols, ochs, kchs);
#endif
#endif

    tiled_conv(
        custom_num,
        batch_size, in_row_dim, in_col_dim, in_channels,
        out_channels, out_row_dim, out_col_dim,
        stride, input_dilation, kernel_dilation, padding, kernel_dim,
        wrot180, trans_output_1203, trans_input_3120,
        trans_weight_1203, trans_weight_0132,

        batches,
        orows, ocols, ochs,
        krows, kcols, kchs,

        input,
        weights,
        bias,
        output,

        act, scale,
        pool_size, no_pool ? 0 : pool_stride, pool_padding,

        tiled_conv_type);
}

// made
static void shared_multi_tiled_conv_auto(
    int gemmini_list, int tile_id,
    size_t sp_addr_start, size_t acc_addr_start,
    int batch_size, int in_row_dim, int in_col_dim, int in_channels,
    int out_channels, int out_row_dim, int out_col_dim,
    int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
    bool wrot180, bool trans_output_1203, bool trans_input_3120,
    bool trans_weight_1203, bool trans_weight_0132,

    const elem_t *input,
    const elem_t *weights,
    const acc_t *bias,
    elem_t *output,

    int act, acc_scale_t scale,
    int pool_size, int pool_stride, int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type)
{

  const bool no_pool = pool_stride == 0;
  if (no_pool)
  {
    pool_size = 1;
    pool_stride = 1;
    pool_padding = 0;
  }

  const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

  const bool downsample = stride == 2 && kernel_dim == 1 && padding == 0 && no_pool && in_row_dim % 2 == 0 && in_col_dim % 2 == 0;

  // Tile convolution params

  // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
  int args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};
  const int max_args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};

  const int orows_idx = 1;
  const int ocols_idx = 2;
  const int out_channels_idx = 3;
  const int in_channels_idx = 6;

  // We divide by 2 for the sake of double-buffering
  const int max_spad_rows = (BANK_NUM * BANK_ROWS / 2);
  const int max_acc_rows = (ACC_ROWS / 2);

  int spad_rows = tiled_conv_total_spad_rows(false,
                                             stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                             args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  int acc_rows = tiled_conv_total_spad_rows(true,
                                            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

  while (spad_rows > max_spad_rows || acc_rows > max_acc_rows)
  {
    int max_val = -1;
    int max_idx = -1;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    {
      // We avoid reducing ocols when possible to keep the spatial array fully utilized
      if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1) && args[i] > max_val)
      {
        max_val = args[i];
        max_idx = i;
      }
    }

    if (max_idx == out_channels_idx || max_idx == in_channels_idx)
    {
      // For input and output channels, there's no point in subtracting by just one
      if (args[max_idx] % DIM != 0)
      {
        args[max_idx] = (args[max_idx] / DIM) * DIM;
      }
      else
      {
        args[max_idx] -= DIM;
      }
      args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
    }
    else
    {
      args[max_idx]--;
    }

    spad_rows = tiled_conv_total_spad_rows(false,
                                           stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                           args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
                                          stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                          args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  }

  // Check if we can increase ocols
  bool not_increased = false;
  while (!not_increased)
  {
    not_increased = true;

    int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
    args_candidate[ocols_idx]++;

    if (args_candidate[ocols_idx] > max_args[ocols_idx])
      continue;

    spad_rows = tiled_conv_total_spad_rows(false,
                                           stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                           args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
                                          stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                          args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

    if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows)
    {
      args[ocols_idx] = args_candidate[ocols_idx];
      not_increased = false;
    }
  }

  // Check if there are any parameters that we can currently still increase
  bool nothing_increased = false;
  while (!nothing_increased)
  {
    nothing_increased = true;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    {
      int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
      args_candidate[i]++;

      if (args_candidate[i] > max_args[i])
        continue;

      spad_rows = tiled_conv_total_spad_rows(false,
                                             stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                             args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
      acc_rows = tiled_conv_total_spad_rows(true,
                                            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

      if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows)
      {
        args[i] = args_candidate[i];
        nothing_increased = false;
      }
    }
  }

  const int batches = args[0];
  const int orows = args[1];
  const int ocols = args[2];
  const int ochs = args[3];
  const int krows = args[4];
  const int kcols = args[5];
  const int kchs = args[6];

  /*
  spad_rows = tiled_conv_total_spad_rows(false,
      stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  acc_rows = tiled_conv_total_spad_rows(true,
      stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  */

#ifdef PRINT_TILE
#if PRINT_TILE
  printf("batches = %d\n", batches);
  printf("orows   = %d\n", orows);
  printf("ocols   = %d\n", ocols);
  printf("ochs    = %d\n", ochs);
  printf("krows   = %d\n", krows);
  printf("kcols   = %d\n", kcols);
  printf("kchs    = %d\n\n", kchs);

  printf("total spad_rows reserved: %d\n", spad_rows);
  printf("total acc_rows reserved: %d\n\n", acc_rows);

  printf("scratchpad row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("accumulator row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

  printf("inner matmul size: i=%d, j=%d, k=%d\n\n", ocols, ochs, kchs);
#endif
#endif

  shared_multi_tiled_conv(
      gemmini_list, tile_id,
      sp_addr_start, acc_addr_start,
      batch_size, in_row_dim, in_col_dim, in_channels,
      out_channels, out_row_dim, out_col_dim,
      stride, input_dilation, kernel_dilation, padding, kernel_dim,
      wrot180, trans_output_1203, trans_input_3120,
      trans_weight_1203, trans_weight_0132,

      batches,
      orows, ocols, ochs,
      krows, kcols, kchs,

      input,
      weights,
      bias,
      output,

      act, scale,
      pool_size, no_pool ? 0 : pool_stride, pool_padding,

      tiled_conv_type);
}

// made
static void shared_multi_tiled_conv_auto_test(
    int gemmini_list, int tile_id,
    size_t sp_addr_start, size_t acc_addr_start,
    size_t sp_addr_range, size_t acc_addr_range,
    int batch_size, int in_row_dim, int in_col_dim, int in_channels,
    int out_channels, int out_row_dim, int out_col_dim,
    int stride, int input_dilation, int kernel_dilation, int padding, int kernel_dim,
    bool wrot180, bool trans_output_1203, bool trans_input_3120,
    bool trans_weight_1203, bool trans_weight_0132,

    const elem_t *input,
    const elem_t *weights,
    const acc_t *bias,
    elem_t *output,

    int act, acc_scale_t scale,
    int pool_size, int pool_stride, int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type)
{
  int gemmini_num = 0;
  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((gemmini_list >> i) & 1)
    {
      gemmini_num++;
    }
  }
  int minimum_ch = DIM * gemmini_num;

  const bool no_pool = pool_stride == 0;
  if (no_pool)
  {
    pool_size = 1;
    pool_stride = 1;
    pool_padding = 0;
  }

  const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

  const bool downsample = stride == 2 && kernel_dim == 1 && padding == 0 && no_pool && in_row_dim % 2 == 0 && in_col_dim % 2 == 0;

  // Tile convolution params

  // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
  int args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};
  const int max_args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, out_channels, kernel_dim, kernel_dim, in_channels};

  const int orows_idx = 1;
  const int ocols_idx = 2;
  const int out_channels_idx = 3;
  const int in_channels_idx = 6;

  // We divide by 2 for the sake of double-buffering
  const int max_spad_rows = (sp_addr_range / 2);
  const int max_acc_rows = (acc_addr_range / 2);

  int spad_rows = tiled_conv_total_spad_rows(false,
                                             stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                             args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  int acc_rows = tiled_conv_total_spad_rows(true,
                                            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

  while (spad_rows > max_spad_rows || acc_rows > max_acc_rows)
  {
    int max_val = -1;
    int max_idx = -1;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    {
      // We avoid reducing ocols when possible to keep the spatial array fully utilized
      if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1) && args[i] > max_val && !((i == out_channels_idx || i == in_channels_idx) && args[i] <= minimum_ch))
      {
        max_val = args[i];
        max_idx = i;
      }
    }

    if (max_idx == out_channels_idx || max_idx == in_channels_idx)
    {
      // For input and output channels, there's no point in subtracting by just one
      if (args[max_idx] % minimum_ch != 0)
      {
        args[max_idx] = (args[max_idx] / minimum_ch) * minimum_ch;
      }
      else
      {
        args[max_idx] -= minimum_ch;
      }
      args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
    }
    else
    {
      args[max_idx]--;
    }

    spad_rows = tiled_conv_total_spad_rows(false,
                                           stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                           args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
                                          stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                          args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  }

  // Check if we can increase ocols
  bool not_increased = false;
  while (!not_increased)
  {
    not_increased = true;

    int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
    args_candidate[ocols_idx]++;

    if (args_candidate[ocols_idx] > max_args[ocols_idx])
      continue;

    spad_rows = tiled_conv_total_spad_rows(false,
                                           stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                           args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
                                          stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                          args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

    if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows)
    {
      args[ocols_idx] = args_candidate[ocols_idx];
      not_increased = false;
    }
  }

  // Check if there are any parameters that we can currently still increase
  bool nothing_increased = false;
  while (!nothing_increased)
  {
    nothing_increased = true;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    {
      int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
      if (i == out_channels_idx || i == in_channels_idx)
      {
        args_candidate[i] += minimum_ch;
      }
      else
      {
        args_candidate[i]++;
      }

      if (args_candidate[i] > max_args[i])
        continue;

      spad_rows = tiled_conv_total_spad_rows(false,
                                             stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                             args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
      acc_rows = tiled_conv_total_spad_rows(true,
                                            stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
                                            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

      if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows)
      {
        args[i] = args_candidate[i];
        nothing_increased = false;
      }
    }
  }

  const int batches = args[0];
  const int orows = args[1];
  const int ocols = args[2];
  const int ochs = args[3];
  const int krows = args[4];
  const int kcols = args[5];
  const int kchs = args[6];

  /*
  spad_rows = tiled_conv_total_spad_rows(false,
      stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  acc_rows = tiled_conv_total_spad_rows(true,
      stride, input_dilation, kernel_dilation, downsample, trans_weight_0132, trans_input_3120,
      args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  */

#ifdef PRINT_TILE
#if PRINT_TILE
  printf("batches = %d\n", batches);
  printf("orows   = %d\n", orows);
  printf("ocols   = %d\n", ocols);
  printf("ochs    = %d\n", ochs);
  printf("krows   = %d\n", krows);
  printf("kcols   = %d\n", kcols);
  printf("kchs    = %d\n\n", kchs);

  printf("total spad_rows reserved: %d\n", spad_rows);
  printf("total acc_rows reserved: %d\n\n", acc_rows);

  printf("scratchpad row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("accumulator row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

  printf("inner matmul size: i=%d, j=%d, k=%d\n\n", ocols, ochs, kchs);
#endif
#endif

  shared_multi_tiled_conv_test(
      gemmini_list, tile_id,
      sp_addr_start, acc_addr_start,
      sp_addr_range, acc_addr_range,
      batch_size, in_row_dim, in_col_dim, in_channels,
      out_channels, out_row_dim, out_col_dim,
      stride, input_dilation, kernel_dilation, padding, kernel_dim,
      wrot180, trans_output_1203, trans_input_3120,
      trans_weight_1203, trans_weight_0132,

      batches,
      orows, ocols, ochs,
      krows, kcols, kchs,

      input,
      weights,
      bias,
      output,

      act, scale,
      pool_size, no_pool ? 0 : pool_stride, pool_padding,

      tiled_conv_type);
}

// This function is for a convolution with kernel_dim=1, stride==2, padding=0, and no pooling
static void tiled_conv_downsample(
        int custom_num, 
        int batch_size, int in_row_dim, int in_col_dim, int in_channels,
        int out_channels, int out_row_dim, int out_col_dim,

        const elem_t * input,
        const elem_t * weights,
        const acc_t * bias,
        elem_t * output,

        int act, acc_scale_t scale,

        enum tiled_matmul_type_t tiled_conv_type) {

    // Rectangular dimensions for this function are currently not supported
    if (in_row_dim != in_col_dim || out_row_dim != out_col_dim) {
        printf("Rectangular convolutions for tiled_conv_downsample are currently not supported.\n");
        exit(1);
    }

    const int in_dim = in_row_dim;
    const int out_dim = out_row_dim;

    const int stride = 2;

    for (int b = 0; b < batch_size; b++) {
        for (int irow = 0; irow < in_row_dim; irow += stride) {
            const int orow = irow / stride;

            const int I = in_col_dim / stride; // number of columns in row
            const int J = out_channels;
            const int K = in_channels;

            const elem_t * A = input + (b * in_dim + irow) * in_dim * in_channels;
            const elem_t * B = weights;
            const acc_t * D = bias;
            elem_t * C = output + (b * out_dim + orow) * out_dim * out_channels;

            const int A_stride = in_channels * 2;
            const int B_stride = out_channels;
            const int D_stride = out_channels;
            const int C_stride = out_channels;

            tiled_matmul_auto(custom_num, I, J, K, A, B, (void*)D, (void*)C,
                    A_stride, B_stride, D_stride, C_stride,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    MVIN_SCALE_IDENTITY, act, scale, 0,
                    true, false, false, false, false, 0, tiled_conv_type);
        }
    }
}

//for mobilenet's depthwise convs
static void tiled_conv_dw_auto(
    int custom_num, 
    int batch_size, int in_row_dim, int in_col_dim,
    int channels, int out_row_dim, int out_col_dim,
    int stride, int padding, int kernel_dim,

    elem_t * input,
    elem_t * weights,
    acc_t * bias,
    elem_t * output,

    int act, acc_scale_t scale,
    int pool_size, int pool_stride, int pool_padding,

    enum tiled_matmul_type_t tiled_conv_type) {

    const bool no_pool = pool_stride == 0;
    if (no_pool) {
        pool_size = 1;
        pool_stride = 1;
        pool_padding = 0;
    }

    const int pool_out_row_dim = (out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
    const int pool_out_col_dim = (out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

    // Tile convolution params

    // int args[] = {batch_size, porows, pocols, pochs, krows, kcols, kchs};
    int args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, 1, kernel_dim, kernel_dim, 1};
    const int max_args[] = {batch_size, pool_out_row_dim, pool_out_col_dim, 1, kernel_dim, kernel_dim, 1};

    const int orows_idx = 1;
    const int ocols_idx = 2;
    const int out_channels_idx = 3;

    // We divide by 2 for the sake of double-buffering
    const int max_spad_rows = (BANK_NUM*BANK_ROWS / 2);
    const int max_acc_rows = (ACC_ROWS / 2);

    int spad_rows = tiled_conv_total_spad_rows(false,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    int acc_rows = tiled_conv_total_spad_rows(true,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    while (spad_rows > max_spad_rows || acc_rows > max_acc_rows) {
        int max_val = -1;
        int max_idx = -1;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            // We avoid reducing ocols when possible to keep the spatial array fully utilized
            if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1)
                    && args[i] > max_val) {
                max_val = args[i];
                max_idx = i;
            }
        }

        if (max_idx == out_channels_idx) {
            // For input and output channels, there's no point in subtracting by just one
            if (args[max_idx] % DIM != 0) {
                args[max_idx] = (args[max_idx] / DIM) * DIM;
            } else {
                args[max_idx] -= DIM;
            }
            args[max_idx] = args[max_idx] == 0 ? 1 : args[max_idx];
        } else {
            args[max_idx]--;
        }

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, 1, 1, false, false, false,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, 1, 1, false, false, false,
            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    }

    // Check if we can increase ocols
    bool not_increased = false;
    while (!not_increased) {
        not_increased = true;

        int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
        args_candidate[ocols_idx]++;

        if (args_candidate[ocols_idx] > max_args[ocols_idx])
            continue;

        spad_rows = tiled_conv_total_spad_rows(false,
            stride, 1, 1, false, false, false,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
        acc_rows = tiled_conv_total_spad_rows(true,
            stride, 1, 1, false, false, false,
            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

        if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
            args[ocols_idx] = args_candidate[ocols_idx];
            not_increased = false;
        }
    }

    // Check if there are any parameters that we can currently still increase
    bool nothing_increased = false;
    while (!nothing_increased) {
        nothing_increased = true;

        for (size_t i = 0; i < sizeof(args)/sizeof(args[0]); i++) {
            int args_candidate[] = {args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
            args_candidate[i]++;

            if (args_candidate[i] > max_args[i])
                continue;

            spad_rows = tiled_conv_total_spad_rows(false,
                stride, 1, 1, false, false, false,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);
            acc_rows = tiled_conv_total_spad_rows(true,
                stride, 1, 1, false, false, false,
                args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3], args_candidate[4], args_candidate[5], args_candidate[6], pool_size, pool_stride);

            if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows) {
                args[i] = args_candidate[i];
                nothing_increased = false;
            }
        }
    }

    const int batches = args[0];
    const int orows = args[1];
    const int ocols = args[2];
    const int ochs = 1; // args[3];
    const int krows = args[4];
    const int kcols = args[5];
    const int kchs = 1; // args[6];

    /*
    spad_rows = tiled_conv_total_spad_rows(false,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
        stride, 1, 1, false, false, false,
        args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

    printf("batches = %d\n", batches);
    printf("orows   = %d\n", orows);
    printf("ocols   = %d\n", ocols);
    printf("ochs    = %d\n", ochs);
    printf("krows   = %d\n", krows);
    printf("kcols   = %d\n", kcols);
    printf("kchs    = %d\n\n", kchs);

    printf("total spad_rows reserved: %d\n", spad_rows);
    printf("total acc_rows reserved: %d\n\n", acc_rows);

    printf("scratchpad row utilization: %d%%\n", (spad_rows*100) / max_spad_rows);
    printf("accumulator row utilization: %d%%\n\n", (acc_rows*100) / max_acc_rows);

    printf("inner matmul size: i=%d, j=%d, k=%d\n\n", ocols, ochs, kchs);
    */

    tiled_conv_dw(
        custom_num, 
        batch_size, in_row_dim, in_col_dim,
        channels, out_row_dim, out_col_dim,
        stride, padding, kernel_dim,

        batches,
        orows, ocols,
        krows, kcols,

        input,
        weights,
        bias,
        output,

        act, scale,
        pool_size, no_pool ? 0 : pool_stride, pool_padding,

        tiled_conv_type);
}


static void resadd_cpu(const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu) {

	const int minimum = relu ? 0 : elem_t_min;

    for (size_t i = 0; i < I; i++) {
        for (size_t j = 0; j < J; j++) {
            const elem_t * a = A + i * J + j;
            const elem_t * b = B + i * J + j;
            elem_t * c = C + i * J + j;

            acc_t result = MVIN_SCALE(*a, A_scale) + MVIN_SCALE(*b, B_scale);
            result = ACC_SCALE(result, C_scale);
            result = result > elem_t_max ? elem_t_max :
                (result < minimum ? minimum : result);

            *c = result;
        }
    }
}


static void sp_tiled_resadd(int custom_num, const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const elem_t * A, const elem_t * B, elem_t * C,
        size_t A_row_stride, size_t B_row_stride, size_t C_row_stride,
        bool relu) {

    // Use the new mvin2 command to overlap mvin A, mvin B, and mvout C

    size_t blocks = (J/DIM + (J % DIM != 0));
    if (blocks > MAX_BLOCK_LEN) blocks = MAX_BLOCK_LEN;

    const uint32_t D_sp_addr_start = 1 << (ADDR_LEN-1);
    const uint32_t C_sp_addr_start = 3 << (ADDR_LEN-2);

    const size_t rounded_up_J = (J / DIM + (J % DIM != 0)) * DIM;

    // Mvin A
    // printf("Mving A\n");
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const elem_t * const A_dram_addr = A + i * A_row_stride + j;
            const uint32_t A_sp_addr = D_sp_addr_start + i * (rounded_up_J/DIM) + j;

            gemmini_extended_mvin(custom_num, A_dram_addr, A_sp_addr, cols, rows);
        }
    }

    // Mvin B
    // printf("Mving B\n");
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            const elem_t * const B_dram_addr = B + i * B_row_stride + j;
            const uint32_t B_sp_addr = C_sp_addr_start + i * (rounded_up_J/DIM) + j;
            gemmini_extended_mvin2(custom_num, B_dram_addr, B_sp_addr, cols, rows);
        }
    }

    // Mvout C from accumulator
    // printf("Mvout C from accumulator\n");
    for (size_t i = 0; i < I; i += DIM) {
        for (size_t j = 0; j < J; j += blocks * DIM) {
            const size_t cols = j + blocks*DIM <= J ? blocks*DIM : J-j;
            const size_t rows = i + DIM <= I ? DIM : I-i;

            elem_t * const C_dram_addr = C + i * C_row_stride + j;
            const uint32_t C_sp_addr = D_sp_addr_start + i * (rounded_up_J/DIM) + j;
            gemmini_extended_mvout(custom_num, C_dram_addr, C_sp_addr, cols, rows);
        }
    }
}

// Compute MVIN_SCALE(A, A_scale) + MVIN_SCALE(B, B_scale) = C
static void tiled_resadd(int custom_num, const size_t I, const size_t J,
        const size_t tile_I, const size_t tile_J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type) {

    gemmini_extended_config_st(custom_num, J * sizeof(elem_t), relu ? RELU : NO_ACTIVATION, C_scale);
    gemmini_config_ex(custom_num, WS, 0, 0);

    gemmini_extended4_config_ld(custom_num, J * sizeof(elem_t), A_scale, true, DIM, 0);
    gemmini_extended4_config_ld(custom_num, J * sizeof(elem_t), B_scale, true, DIM, 1);

    for (size_t i = 0; i < I; i += tile_I) {
        for (size_t j = 0; j < J; j += tile_J) {
            const size_t I_tile = i + tile_I <= I ? tile_I : I - i;
            const size_t J_tile = j + tile_J <= J ? tile_J : J - j;

            const elem_t * a = A + i * J + j;
            const elem_t * b = B + i * J + j;
            elem_t * c = C + i * J + j;

            sp_tiled_resadd(custom_num, I_tile, J_tile,
                    A_scale, B_scale, a, b, c,
                    J, J, J,
                    relu);
        }
    }

    gemmini_fence();
}

// Compute (A >> A_shift) + B = C
static void tiled_resadd_auto(int custom_num, const size_t I, const size_t J,
        const scale_t A_scale,
        const scale_t B_scale,
        const acc_scale_t C_scale,
        const elem_t * A,
        const elem_t * B,
        elem_t * C,
        bool relu,
        enum tiled_matmul_type_t matadd_type) {

    if (matadd_type == CPU) {
        resadd_cpu(I, J,
            A_scale, B_scale, C_scale, A, B, C,
            relu);
        return;
    }

    size_t tile_I = I, tile_J = J;

    // size_t total_spad_rows = 2 * (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));
    size_t total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));

    // TODO this is a very inefficient way of doing this...
    while (total_acc_rows > ACC_ROWS) {
        if (tile_I >= tile_J)
            tile_I--;
        else
            tile_J--;

        total_acc_rows = (tile_I / DIM + (tile_I % DIM != 0))*DIM * (tile_J / DIM + (tile_J % DIM != 0));
    }

    // printf("tile_I: %llu\n", tile_I);
    // printf("tile_J: %llu\n", tile_J);

    if (matadd_type == WS) {
      tiled_resadd(custom_num, I, J, tile_I, tile_J,
            A_scale, B_scale, C_scale, A, B, C,
            relu, matadd_type);
    } else if(matadd_type == CPU){
	  resadd_cpu(I, J, A_scale, B_scale, C_scale,
		A, B, C, relu);
    }
    else {
      printf("Unsupported type\n");
      exit(1);
    }
}


static void global_average_cpu(const elem_t * input, elem_t * output,
    int batches, int channels, int dim) {
  const int count = dim * dim;

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel++) {
      acc_t sum = 0;
      for (int row = 0; row < dim; row++) {
        for (int col = 0; col < dim; col++) {
          size_t pixel = batch * dim * dim + row * dim + col;

          sum += input[pixel * channels + channel];
        }
      }

#ifdef ELEM_T_IS_FLOAT
      output[batch * channels + channel] = sum / count;
#else
      output[batch * channels + channel] = (sum + count/2) / count;
#endif
    }
  }
}


static void sp_tiled_global_average(int custom_num, const elem_t * input, elem_t * output,
    int batches, int channels, int dim, int channel_tile_size) {
  const uint32_t C_acc_addr_start = ((uint32_t)1 << 31);

  size_t blocks = channel_tile_size/DIM + (channel_tile_size % DIM != 0);
  if (blocks > MAX_BLOCK_LEN) blocks = MAX_BLOCK_LEN;

  for (int channel = 0; channel < channel_tile_size; channel += blocks*DIM) {
    for (int row = 0; row < dim; row++) {
      for (int col = 0; col < dim; col++) {
        const elem_t * in = input +
          (row * dim + col) * channels +
          channel;

        const uint32_t acc_addr_start = C_acc_addr_start |
          ((row != 0 || col != 0) << 30);

        const uint32_t acc_addr = acc_addr_start + channel / DIM;

        const size_t cols = channel + blocks*DIM <= channel_tile_size ?
          blocks*DIM : channel_tile_size - channel;

        const size_t rows = 1;

        gemmini_extended_mvin(custom_num, in, acc_addr, cols, rows);
      }
    }
  }

  for (int channel = 0; channel < channel_tile_size; channel += DIM) {
    elem_t * out = output + channel;

    const uint32_t acc_addr = C_acc_addr_start + channel / DIM;

    const size_t cols = channel + DIM <= channel_tile_size ?
      DIM : channel_tile_size - channel;

    const size_t rows = 1; // TODO we should move out more than just one row here

    gemmini_extended_mvout(custom_num, out, acc_addr, cols, rows);
  }
}


static void tiled_global_average(int custom_num, const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    int channel_tile_size) {

  gemmini_extended4_config_ld(custom_num, DIM*sizeof(elem_t), MVIN_SCALE_IDENTITY, true, 1, 0);
  gemmini_config_ex(custom_num, 0, NO_ACTIVATION, 0);
  gemmini_extended_config_st(custom_num, 0, NO_ACTIVATION, 1.0 / (dim*dim));

  for (int batch = 0; batch < batches; batch++) {
    for (int channel = 0; channel < channels; channel += channel_tile_size) {
      const int tile_size = channel + channel_tile_size <= channels ?
        channel_tile_size : channels - channel;

      sp_tiled_global_average(custom_num, input + batch * dim * dim * channels + channel,
          output + batch * channels + channel,
          batches, channels, dim, tile_size);
    }
  }
}


static void tiled_global_average_auto(int custom_num, const elem_t * input, elem_t * output,
    int batches, int channels, int dim,
    enum tiled_matmul_type_t type) {
  if (type == CPU) {
    return global_average_cpu(input, output, batches, channels, dim);
  }

  int channel_tile_size = channels;

  int acc_rows = channel_tile_size / DIM + (channel_tile_size % DIM != 0);
  while (acc_rows > ACC_ROWS) {
    channel_tile_size--;
    acc_rows = channel_tile_size / DIM + (channel_tile_size % DIM != 0);
  }

  tiled_global_average(custom_num, input, output, batches, channels, dim,
      channel_tile_size);
}

// made
typedef struct
{
  // 입력 인자들
  int gemmini_list;
  int tile_id;
  size_t sp_addr_start_stack, sp_addr_end_stack, acc_addr_start_stack;
  size_t sp_addr_range, acc_addr_range;
  size_t sp_addr_A_stacked, sp_addr_B_stacked, acc_addr_stacked;
  size_t dim_I, dim_J, dim_K;
  const elem_t *A;
  const elem_t *B;
  const void *D;
  void *C;
  size_t stride_A, stride_B, stride_D, stride_C;
  scale_t A_scale_factor, B_scale_factor;
  scale_acc_t D_scale_factor;
  size_t tile_I, tile_J, tile_K;
  int act;
  acc_scale_t scale, bert_scale;
  bool repeating_bias;
  bool a_transpose, b_transpose;
  bool full_C, low_D;
  uint8_t weightA;
  int dataflow; // OUTPUT_STATIONARY or WEIGHT_STATIONARY

  // 파생 값들 (outer_test prologue에서 계산하던 것들)
  size_t dim_I_padded, dim_J_padded, dim_K_padded;
  size_t I0, J0, K0;
  size_t last_I, last_J, last_K;
  size_t padding_I, padding_J, padding_K;
  bool no_bias;
  size_t sizeof_D, sizeof_C;
  int gemmini_num;

  // 루프 진행 상태
  size_t i0, j0, k0;
  int inner_call_counter;
  int lastK_toggle;

  // 이 job이 끝났는지 여부
  bool done;

} shared_multi_matmul_job_t;

static void shared_multi_choose_tiling_factors_static(shared_multi_matmul_job_t *job, size_t tI_static, size_t tJ_static, size_t tK_static)
{

  job->gemmini_num = 0;
  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((job->gemmini_list >> i) & 1)
      job->gemmini_num++;
  }

  // 기존 auto_test 상단에 있던 partition 계산
  size_t partition_rows = (job->sp_addr_range / 2);
  size_t mats_in_partition = (partition_rows / DIM);
  size_t mats_in_acc = (job->acc_addr_range / DIM);
  size_t max_tile_i_j = ((size_t)sqrt(mats_in_acc));
  size_t max_tile_k = (mats_in_partition / max_tile_i_j);

  // double-buffered 기준 partition (db_*)
  size_t db_partition_rows_1 = ((job->sp_addr_range / 2) / 2);
  size_t db_mats_in_partition_1 = (db_partition_rows_1 / DIM);
  size_t db_mats_in_acc_1 = ((job->acc_addr_range / 2) / DIM);
  size_t db_max_tile_i_j_1 = ((size_t)sqrt(db_mats_in_acc_1));
  size_t db_max_tile_k_1 = (db_mats_in_partition_1 / db_max_tile_i_j_1);

  job->dim_I_padded = (job->dim_I / DIM + (job->dim_I % DIM != 0)) * DIM;
  job->dim_J_padded = (job->dim_J / DIM + (job->dim_J % DIM != 0)) * DIM;
  job->dim_K_padded = (job->dim_K / DIM + (job->dim_K % DIM != 0)) * DIM;

  const bool double_buffered = job->dataflow == WEIGHT_STATIONARY;

  const size_t max_spad_rows = double_buffered ? job->sp_addr_range / 2 : job->sp_addr_range;
  const size_t max_acc_rows = double_buffered ? job->acc_addr_range / 2 : job->acc_addr_range;

  size_t tI, tJ, tK;

  if (job->act == LAYERNORM || job->act == SOFTMAX)
  {
    tI = 1;
    tJ = job->dim_J_padded / DIM;
    tK = 1;
  }
  else if (double_buffered)
  {
    size_t max_i = (job->dim_I_padded / DIM);
    size_t max_j = (job->dim_J_padded / DIM);
    size_t max_k = (job->dim_K_padded / DIM);

    tI = max_i < db_max_tile_i_j_1 ? max_i : db_max_tile_i_j_1;
    tJ = max_j < db_max_tile_i_j_1 ? max_j : db_max_tile_i_j_1;
    tK = max_k < db_max_tile_k_1 ? max_k : db_max_tile_k_1;
  }
  else
  {
    size_t max_i = (job->dim_I_padded / DIM);
    size_t max_j = (job->dim_J_padded / DIM);
    size_t max_k = (job->dim_K_padded / DIM);

    tI = max_i < max_tile_i_j ? max_i : max_tile_i_j;
    tJ = max_j < max_tile_i_j ? max_j : max_tile_i_j;
    tK = max_k < max_tile_k ? max_k : max_tile_k;
  }

  if (tI >= job->gemmini_num)
  {
    if (tI % job->gemmini_num != 0)
    {
      tI = (tI / job->gemmini_num) * job->gemmini_num;
    }
  }
  else
  {
    if (job->gemmini_num * DIM <= job->dim_I_padded)
    {
      tI = job->gemmini_num;
    }
  }

  if (tK >= job->gemmini_num)
  {
    if (tK % job->gemmini_num != 0)
    {
      tK = (tK / job->gemmini_num) * job->gemmini_num;
    }
  }
  else
  {
    if (job->gemmini_num * DIM <= job->dim_K_padded)
    {
      tK = job->gemmini_num;
    }
  }

  while (true)
  {
    bool decreased = false;

    if ((tiled_matmul_total_spad_rows(tI, tJ, tK) > max_spad_rows ||
         tiled_matmul_total_acc_rows(tI, tJ) > max_acc_rows) &&
        tJ > 1)
    {
      tJ--;
      decreased = true;
    }

    if (!decreased)
      break;
  }

  // greedy하게 spad/acc를 꽉 채우도록 키우는 부분
  while (true)
  {
    bool increased = false;

    if (tiled_matmul_total_spad_rows(tI, tJ + 1, tK) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tI, tJ + 1) <= max_acc_rows &&
        (tJ + 1) * DIM <= job->dim_J_padded)
    {
      tJ++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tI + job->gemmini_num, tJ, tK) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tI + job->gemmini_num, tJ) <= max_acc_rows &&
        (tI + job->gemmini_num) * DIM <= job->dim_I_padded)
    {
      tI = tI + job->gemmini_num;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tI, tJ, tK + job->gemmini_num) <= max_spad_rows &&
        (tK + job->gemmini_num) * DIM <= job->dim_K_padded)
    {
      tK = tK + job->gemmini_num;
      increased = true;
    }

    if (!increased)
      break;
  }

#ifdef PRINT_TILE
#if PRINT_TILE
  const int spad_rows = tiled_matmul_total_spad_rows(tI, tJ, tK);
  const int acc_rows = tiled_matmul_total_acc_rows(tI, tJ);

  printf("tile_I: %d\n", tI);
  printf("tile_J: %d\n", tJ);
  printf("tile_K: %d\n\n", tK);

  printf("spad_rows: %d\n", spad_rows);
  printf("acc_rows: %d\n\n", acc_rows);

  printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

  // exit(EXIT_SUCCESS);
#endif
#endif

  job->tile_I = tI_static;
  job->tile_J = tJ_static;
  job->tile_K = tK_static;
  job->sp_addr_range = tiled_matmul_total_spad_rows(tI_static, tJ_static, tK_static) * 2;
  job->acc_addr_range = tiled_matmul_total_acc_rows(tI_static, tJ_static) * 2;
  job->sp_addr_A_stacked = tiled_matmul_A_spad_rows(tI_static, tJ_static, tK_static);
  job->sp_addr_B_stacked = tiled_matmul_B_spad_rows(tI_static, tJ_static, tK_static);
  job->acc_addr_stacked = tiled_matmul_total_acc_rows(tI_static, tJ_static);
}

static void shared_multi_choose_tiling_factors(shared_multi_matmul_job_t *job)
{

  job->gemmini_num = 0;
  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((job->gemmini_list >> i) & 1)
      job->gemmini_num++;
  }

  // 기존 auto_test 상단에 있던 partition 계산
  size_t partition_rows = (job->sp_addr_range / 2);
  size_t mats_in_partition = (partition_rows / DIM);
  size_t mats_in_acc = (job->acc_addr_range / DIM);
  size_t max_tile_i_j = ((size_t)sqrt(mats_in_acc));
  size_t max_tile_k = (mats_in_partition / max_tile_i_j);

  // double-buffered 기준 partition (db_*)
  size_t db_partition_rows_1 = ((job->sp_addr_range / 2) / 2);
  size_t db_mats_in_partition_1 = (db_partition_rows_1 / DIM);
  size_t db_mats_in_acc_1 = ((job->acc_addr_range / 2) / DIM);
  size_t db_max_tile_i_j_1 = ((size_t)sqrt(db_mats_in_acc_1));
  size_t db_max_tile_k_1 = (db_mats_in_partition_1 / db_max_tile_i_j_1);

  job->dim_I_padded = (job->dim_I / DIM + (job->dim_I % DIM != 0)) * DIM;
  job->dim_J_padded = (job->dim_J / DIM + (job->dim_J % DIM != 0)) * DIM;
  job->dim_K_padded = (job->dim_K / DIM + (job->dim_K % DIM != 0)) * DIM;

  const bool double_buffered = job->dataflow == WEIGHT_STATIONARY;

  const size_t max_spad_rows = double_buffered ? job->sp_addr_range / 2 : job->sp_addr_range;
  const size_t max_acc_rows = double_buffered ? job->acc_addr_range / 2 : job->acc_addr_range;

  size_t tI, tJ, tK;

  if (job->act == LAYERNORM || job->act == SOFTMAX)
  {
    tI = 1;
    tJ = job->dim_J_padded / DIM;
    tK = 1;
  }
  else if (double_buffered)
  {
    size_t max_i = (job->dim_I_padded / DIM);
    size_t max_j = (job->dim_J_padded / DIM);
    size_t max_k = (job->dim_K_padded / DIM);

    tI = max_i < db_max_tile_i_j_1 ? max_i : db_max_tile_i_j_1;
    tJ = max_j < db_max_tile_i_j_1 ? max_j : db_max_tile_i_j_1;
    tK = max_k < db_max_tile_k_1 ? max_k : db_max_tile_k_1;
  }
  else
  {
    size_t max_i = (job->dim_I_padded / DIM);
    size_t max_j = (job->dim_J_padded / DIM);
    size_t max_k = (job->dim_K_padded / DIM);

    tI = max_i < max_tile_i_j ? max_i : max_tile_i_j;
    tJ = max_j < max_tile_i_j ? max_j : max_tile_i_j;
    tK = max_k < max_tile_k ? max_k : max_tile_k;
  }

  if (tI >= job->gemmini_num)
  {
    if (tI % job->gemmini_num != 0)
    {
      tI = (tI / job->gemmini_num) * job->gemmini_num;
    }
  }
  else
  {
    if (job->gemmini_num * DIM <= job->dim_I_padded)
    {
      tI = job->gemmini_num;
    }
  }

  if (tK >= job->gemmini_num)
  {
    if (tK % job->gemmini_num != 0)
    {
      tK = (tK / job->gemmini_num) * job->gemmini_num;
    }
  }
  else
  {
    if (job->gemmini_num * DIM <= job->dim_K_padded)
    {
      tK = job->gemmini_num;
    }
  }

  while (true)
  {
    bool decreased = false;

    if ((tiled_matmul_total_spad_rows(tI, tJ, tK) > max_spad_rows ||
         tiled_matmul_total_acc_rows(tI, tJ) > max_acc_rows) &&
        tJ > 1)
    {
      tJ--;
      decreased = true;
    }

    if (!decreased)
      break;
  }

  // greedy하게 spad/acc를 꽉 채우도록 키우는 부분
  while (true)
  {
    bool increased = false;

    if (tiled_matmul_total_spad_rows(tI, tJ + 1, tK) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tI, tJ + 1) <= max_acc_rows &&
        (tJ + 1) * DIM <= job->dim_J_padded)
    {
      tJ++;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tI + job->gemmini_num, tJ, tK) <= max_spad_rows &&
        tiled_matmul_total_acc_rows(tI + job->gemmini_num, tJ) <= max_acc_rows &&
        (tI + job->gemmini_num) * DIM <= job->dim_I_padded)
    {
      tI = tI + job->gemmini_num;
      increased = true;
    }

    if (tiled_matmul_total_spad_rows(tI, tJ, tK + job->gemmini_num) <= max_spad_rows &&
        (tK + job->gemmini_num) * DIM <= job->dim_K_padded)
    {
      tK = tK + job->gemmini_num;
      increased = true;
    }

    if (!increased)
      break;
  }

#ifdef PRINT_TILE
#if PRINT_TILE
  const int spad_rows = tiled_matmul_total_spad_rows(tI, tJ, tK);
  const int acc_rows = tiled_matmul_total_acc_rows(tI, tJ);

  printf("tile_I: %d\n", tI);
  printf("tile_J: %d\n", tJ);
  printf("tile_K: %d\n\n", tK);

  printf("spad_rows: %d\n", spad_rows);
  printf("acc_rows: %d\n\n", acc_rows);

  printf("spad_row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("acc_row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

  // exit(EXIT_SUCCESS);
#endif
#endif

  job->tile_I = tI;
  job->tile_J = tJ;
  job->tile_K = tK;
  job->sp_addr_range = tiled_matmul_total_spad_rows(tI, tJ, tK) * 2;
  job->acc_addr_range = tiled_matmul_total_acc_rows(tI, tJ) * 2;
  job->sp_addr_A_stacked = tiled_matmul_A_spad_rows(tI, tJ, tK);
  job->sp_addr_B_stacked = tiled_matmul_B_spad_rows(tI, tJ, tK);
  job->acc_addr_stacked = tiled_matmul_total_acc_rows(tI, tJ);
}

void shared_multi_tiled_matmul_job_init(shared_multi_matmul_job_t *job)
{
  // memset(job, 0, sizeof(*job));

  job->I0 = job->dim_I_padded / (job->tile_I * DIM) + (job->dim_I_padded % (job->tile_I * DIM) != 0);
  job->J0 = job->dim_J_padded / (job->tile_J * DIM) + (job->dim_J_padded % (job->tile_J * DIM) != 0);
  job->K0 = job->dim_K_padded / (job->tile_K * DIM) + (job->dim_K_padded % (job->tile_K * DIM) != 0);

  job->last_I = job->dim_I_padded % (job->tile_I * DIM) == 0 ? job->tile_I : (job->dim_I_padded / DIM) % job->tile_I;
  job->last_J = job->dim_J_padded % (job->tile_J * DIM) == 0 ? job->tile_J : (job->dim_J_padded / DIM) % job->tile_J;
  job->last_K = job->dim_K_padded % (job->tile_K * DIM) == 0 ? job->tile_K : (job->dim_K_padded / DIM) % job->tile_K;

  job->padding_I = job->dim_I_padded - job->dim_I;
  job->padding_J = job->dim_J_padded - job->dim_J;
  job->padding_K = job->dim_K_padded - job->dim_K;

  job->no_bias = (job->D == NULL);
  if (job->no_bias)
  {
    job->D = (void *)1; // dummy
  }

  job->sizeof_D = job->low_D ? sizeof(elem_t) : sizeof(acc_t);
  job->sizeof_C = job->full_C ? sizeof(acc_t) : sizeof(elem_t);

  job->i0 = job->j0 = job->k0 = 0;
  job->inner_call_counter = 0;
  job->lastK_toggle = 1;
  job->done = false;

  // 여기부터는 기존 outer_test에서 했던 Gemmini config 부분을 그대로 가져옴
  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((job->gemmini_list >> i) & 1)
    {
      switch (i)
      {
      case 3:
        gemmini_extended_config_ex(custom3, job->dataflow, job->act & 3, 0, 1, job->a_transpose, job->b_transpose);
        gemmini_extended_config_st(custom3, job->stride_C * job->sizeof_C, job->act & 3, job->scale);
        gemmini_extended3_config_ld(custom3, job->stride_A * sizeof(elem_t), job->A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom3, job->stride_B * sizeof(elem_t), job->B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom3, job->repeating_bias ? 0 : (job->stride_D * job->sizeof_D), job->D_scale_factor, job->low_D, 2);
        break;
      case 2:
        gemmini_extended_config_ex(custom2, job->dataflow, job->act & 3, 0, 1, job->a_transpose, job->b_transpose);
        gemmini_extended_config_st(custom2, job->stride_C * job->sizeof_C, job->act & 3, job->scale);
        gemmini_extended3_config_ld(custom2, job->stride_A * sizeof(elem_t), job->A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom2, job->stride_B * sizeof(elem_t), job->B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom2, job->repeating_bias ? 0 : (job->stride_D * job->sizeof_D), job->D_scale_factor, job->low_D, 2);
        break;
      case 1:
        gemmini_extended_config_ex(custom1, job->dataflow, job->act & 3, 0, 1, job->a_transpose, job->b_transpose);
        gemmini_extended_config_st(custom1, job->stride_C * job->sizeof_C, job->act & 3, job->scale);
        gemmini_extended3_config_ld(custom1, job->stride_A * sizeof(elem_t), job->A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom1, job->stride_B * sizeof(elem_t), job->B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom1, job->repeating_bias ? 0 : (job->stride_D * job->sizeof_D), job->D_scale_factor, job->low_D, 2);
        break;
      case 0:
        gemmini_extended_config_ex(custom0, job->dataflow, job->act & 3, 0, 1, job->a_transpose, job->b_transpose);
        gemmini_extended_config_st(custom0, job->stride_C * job->sizeof_C, job->act & 3, job->scale);
        gemmini_extended3_config_ld(custom0, job->stride_A * sizeof(elem_t), job->A_scale_factor, false, 0);
        gemmini_extended3_config_ld(custom0, job->stride_B * sizeof(elem_t), job->B_scale_factor, false, 1);
        gemmini_extended3_config_ld(custom0, job->repeating_bias ? 0 : (job->stride_D * job->sizeof_D), job->D_scale_factor, job->low_D, 2);
        break;
      }
    }
  }

  // IGELU / SOFTMAX norm 설정도 그대로
  if (job->act == IGELU)
  {
    const acc_scale_t sqrt_2 = 1.41421356237;
    const acc_scale_t S = job->bert_scale;
    const acc_scale_t S_erf = (-0.2888 * ((S * S) / 2));

    const acc_t qb = -1.769 / (S / sqrt_2);
    const acc_t qc = 1.0 / S_erf;

    for (int i = 0; i < total_gemmini_num; i++)
    {
      if ((job->gemmini_list >> i) & 1)
      {
        switch (i)
        {
        case 3:
          gemmini_config_norm(custom3, 0, 0, 0, 0, 0, qb, qc);
          break;
        case 2:
          gemmini_config_norm(custom2, 0, 0, 0, 0, 0, qb, qc);
          break;
        case 1:
          gemmini_config_norm(custom1, 0, 0, 0, 0, 0, qb, qc);
          break;
        case 0:
          gemmini_config_norm(custom0, 0, 0, 0, 0, 0, qb, qc);
          break;
        }
      }
    }
  }

  if (job->act == SOFTMAX)
  {
    const scale_t a = 0.3585;
    const scale_t b = 1.353;
    const scale_t c = 0.344;

    const acc_t qln2 = (int)(0.693147 / job->bert_scale);
    const acc_t qln2_inv = 65536 / qln2;
    const acc_t qb = b / job->bert_scale;
    const acc_t qc = c / (a * job->bert_scale * job->bert_scale);

    for (int i = 0; i < total_gemmini_num; i++)
    {
      if ((job->gemmini_list >> i) & 1)
      {
        switch (i)
        {
        case 3:
          gemmini_config_norm(custom3, qln2, 0, 0, 1, 0, qb, qc);
          gemmini_config_norm(custom3, qln2_inv, 1, 0, 1, 0, qb, qc);
          break;
        case 2:
          gemmini_config_norm(custom2, qln2, 0, 0, 1, 0, qb, qc);
          gemmini_config_norm(custom2, qln2_inv, 1, 0, 1, 0, qb, qc);
          break;
        case 1:
          gemmini_config_norm(custom1, qln2, 0, 0, 1, 0, qb, qc);
          gemmini_config_norm(custom1, qln2_inv, 1, 0, 1, 0, qb, qc);
          break;
        case 0:
          gemmini_config_norm(custom0, qln2, 0, 0, 1, 0, qb, qc);
          gemmini_config_norm(custom0, qln2_inv, 1, 0, 1, 0, qb, qc);
          break;
        }
      }
    }
  }
}

static void shared_multi_tiled_matmul_job_step(shared_multi_matmul_job_t *job)
{
  if (job->done)
    return;

  const size_t i0 = job->i0;
  const size_t j0 = job->j0;
  const size_t k0 = job->k0;

  const bool no_bias = job->no_bias;
  const size_t sizeof_D = job->sizeof_D;
  const size_t sizeof_C = job->sizeof_C;

  void (*inner)(int, int, int,
                size_t, size_t, size_t,
                const elem_t *, const elem_t *, const void *, void *,
                scale_t, scale_t, scale_acc_t,
                size_t, size_t, size_t, size_t, size_t,
                size_t, size_t, size_t, size_t, size_t, size_t,
                size_t, size_t, size_t, size_t,
                bool, bool,
                bool, bool,
                bool, bool,
                int);

  if (job->dataflow == OUTPUT_STATIONARY)
    inner = &shared_multi_sp_tiled_matmul_os;
  else
    inner = &shared_multi_sp_tiled_matmul_ws;

  const size_t I0 = job->I0;
  const size_t J0 = job->J0;
  const size_t K0 = job->K0;

  const size_t tile_I = job->tile_I;
  const size_t tile_J = job->tile_J;
  const size_t tile_K = job->tile_K;

  const size_t last_I = job->last_I;
  const size_t last_J = job->last_J;
  const size_t last_K = job->last_K;

  const size_t padding_I = job->padding_I;
  const size_t padding_J = job->padding_J;
  const size_t padding_K = job->padding_K;

  const size_t dim_I_padded = job->dim_I_padded;
  const size_t dim_J_padded = job->dim_J_padded;
  const size_t dim_K_padded = job->dim_K_padded;

  const elem_t *A = job->A;
  const elem_t *B = job->B;
  const void *D = job->D;
  void *C = job->C;

  const size_t stride_A = job->stride_A;
  const size_t stride_B = job->stride_B;
  const size_t stride_D = job->stride_D;
  const size_t stride_C = job->stride_C;

  const scale_t A_scale_factor = job->A_scale_factor;
  const scale_t B_scale_factor = job->B_scale_factor;
  const scale_acc_t D_scale_factor = job->D_scale_factor;

  const int act = job->act;
  const acc_scale_t scale = job->scale;
  const acc_scale_t bert_scale = job->bert_scale;
  const bool repeating_bias = job->repeating_bias;
  const bool a_transpose = job->a_transpose;
  const bool b_transpose = job->b_transpose;
  const bool full_C = job->full_C;
  const bool low_D = job->low_D;
  const uint8_t weightA = job->weightA;
  const int dataflow = job->dataflow;

  const size_t sp_addr_start_stack = job->sp_addr_start_stack;
  const size_t sp_addr_end_stack = job->sp_addr_end_stack;
  const size_t acc_addr_start_stack = job->acc_addr_start_stack;
  const size_t sp_addr_range = job->sp_addr_range;
  const size_t acc_addr_range = job->acc_addr_range;
  const int tile_id = job->tile_id;
  const int gemmini_list = job->gemmini_list;

  const void *pre;
  if (k0 != 0)
  {
    pre = NULL;
  }
  else
  {
    size_t bias_row = repeating_bias ? 0 : i0 * tile_I * DIM;
    pre = (int8_t *)D + (bias_row * stride_D + j0 * tile_J * DIM) * sizeof_D;
  }

  void *out = (k0 == K0 - 1)
                  ? (int8_t *)C + (i0 * tile_I * DIM * stride_C + j0 * tile_J * DIM) * sizeof_C
                  : NULL;

  const size_t I = i0 < I0 - 1 ? tile_I : last_I;
  const size_t J = j0 < J0 - 1 ? tile_J : last_J;
  const size_t K = k0 < K0 - 1 ? tile_K : last_K;

  const size_t pad_I = i0 == I0 - 1 ? padding_I : 0;
  const size_t pad_J = j0 == J0 - 1 ? padding_J : 0;
  const size_t pad_K = k0 == K0 - 1 ? padding_K : 0;

  const elem_t *a = a_transpose ? (A + k0 * tile_K * DIM * stride_A + i0 * tile_I * DIM)
                                : (A + i0 * tile_I * DIM * stride_A + k0 * tile_K * DIM);

  const elem_t *b = b_transpose ? (B + j0 * tile_J * DIM * stride_B + k0 * tile_K * DIM)
                                : (B + k0 * tile_K * DIM * stride_B + j0 * tile_J * DIM);

  // group_list 계산
  int using_gemmini_num = job->gemmini_num;
  if (I < job->gemmini_num && K < job->gemmini_num)
  {
    using_gemmini_num = (I < K) ? K : I;
  }

  size_t group_list = gemmini_list;
  int shift_num = 0;
  for (int i = 0; i < total_gemmini_num; i++)
  {
    size_t shifted_list = (gemmini_list >> i);
    if (shift_num == using_gemmini_num)
    {
      group_list = group_list & ~(shifted_list << i);
      break;
    }
    if (shifted_list & 1)
    {
      shift_num++;
    }
  }

  const size_t I_div = I / using_gemmini_num;
  const size_t I_div_added = I_div + 1;
  const size_t I_added_gemmini_num = I % using_gemmini_num;

  const size_t K_div = K / using_gemmini_num;
  const size_t K_div_added = K_div + 1;
  const size_t K_added_gemmini_num = K % using_gemmini_num;

  const elem_t *a_local = a;
  const elem_t *b_local = b;
  const int8_t *c_local = (int8_t *)out;
  const int8_t *d_local = (int8_t *)pre;

  size_t laddrI_offset = 0;
  size_t laddrK_offset = 0;

  const int t = (job->inner_call_counter & 1);
  const size_t local_sp_addr_start = (t == 0) ? sp_addr_start_stack : sp_addr_start_stack + BANK_NUM * BANK_ROWS / 2;
  const size_t local_sp_addr_end = (t == 0) ? BANK_NUM * BANK_ROWS / 2 - sp_addr_end_stack : BANK_NUM * BANK_ROWS - sp_addr_end_stack;
  const size_t local_acc_addr_start = job->lastK_toggle ? acc_addr_start_stack : acc_addr_start_stack + ACC_ROWS / 2;

  int activated_gemmini_num = 0;
  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((group_list >> i) & 1)
    {
      size_t this_I = (activated_gemmini_num < I_added_gemmini_num) ? I_div_added : I_div;
      int ex_gemmini_num = (using_gemmini_num < I) ? using_gemmini_num : I;
      size_t this_pad_I = (activated_gemmini_num == ex_gemmini_num - 1) ? pad_I : 0;

      size_t this_K = (activated_gemmini_num < K_added_gemmini_num) ? K_div_added : K_div;
      int ldB_gemmini_num = (using_gemmini_num < K) ? using_gemmini_num : K;
      size_t this_pad_K = (activated_gemmini_num == ldB_gemmini_num - 1) ? pad_K : 0;

      (*inner)(i, group_list, tile_id << 1 | t,
               local_sp_addr_start, local_sp_addr_end, local_acc_addr_start,
               a_local, b_local, (k0 != 0) ? NULL : (void *)d_local, (k0 == K0 - 1) ? (void *)c_local : NULL,
               A_scale_factor, B_scale_factor, D_scale_factor,
               this_I, this_K, this_pad_K, laddrI_offset, laddrK_offset,
               I, J, K,
               this_pad_I, pad_J, pad_K,
               stride_A, stride_B, stride_D, stride_C,
               a_transpose, b_transpose,
               full_C, low_D,
               no_bias, repeating_bias,
               act);

      laddrI_offset += this_I;
      laddrK_offset += this_K;

      size_t local_stride_A = a_transpose ? 1 : stride_A;
      a_local += local_stride_A * DIM * this_I;
      size_t local_stride_B = b_transpose ? 1 : stride_B;
      b_local += local_stride_B * DIM * this_K;
      c_local += stride_C * DIM * this_I * sizeof_C;
      d_local += stride_D * DIM * this_I * sizeof_D;

      activated_gemmini_num++;
    }
  }

  // for (int i = 0; i < total_gemmini_num; i++)
  // {
  //   if ((group_list >> i) & 1)
  //   {
  //     switch (i)
  //     {
  //     case 3:
  //       ROCC_INSTRUCTION_RS1_RS2(custom3, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)),
  //                                ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
  //       break;
  //     case 2:
  //       ROCC_INSTRUCTION_RS1_RS2(custom2, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)),
  //                                ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
  //       break;
  //     case 1:
  //       ROCC_INSTRUCTION_RS1_RS2(custom1, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)),
  //                                ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
  //       break;
  //     case 0:
  //       ROCC_INSTRUCTION_RS1_RS2(custom0, ((uint64_t)(act) << 8) | ((low_D) << 2) | ((full_C) << 1) | (!no_bias || (k0 != 0)),
  //                                ((b_transpose) << 1) | (a_transpose), k_LOOP_WS);
  //       break;
  //     }
  //   }
  // }

  job->inner_call_counter++;
  if (k0 == K0 - 1)
    job->lastK_toggle ^= 1;

  // ---- 여기까지가 원래 한 (i0,j0,k0) 타일에 대해 하던 일 ----
  // 이제 (i0, j0, k0) 인덱스를 다음 타일로 증가시킴

  // k0 증가
  job->k0++;
  if (job->k0 >= K0)
  {
    job->k0 = 0;
    job->j0++;
    if (job->j0 >= J0)
    {
      job->j0 = 0;
      job->i0++;
      if (job->i0 >= I0)
      {
        job->done = true; // 모든 타일 완료
      }
    }
  }
}

typedef struct
{
  // loop indices
  int b, porow, pocol, poch, krow, kcol, kch;

  // tiling 파라미터
  int batch_size, in_row_dim, in_col_dim, in_channels;
  int out_channels, out_row_dim, out_col_dim;
  int stride, input_dilation, kernel_dilation, padding, kernel_dim;
  bool wrot180, trans_output_1203, trans_input_3120;
  bool trans_weight_1203, trans_weight_0132;

  int batches, porows, pocols, pochs;
  int krows, kcols, kchs;

  int pool_size, pool_stride, pool_padding;

  // geometry 관련 precomputed 값들 (pool_out_row_dim, pool_out_col_dim, dilated_in_row_dim, ...)
  int pool_out_row_dim, pool_out_col_dim;
  int dilated_in_row_dim, dilated_in_col_dim;
  int input_dilated;
  bool no_bias, no_pool, downsample;
  enum tiled_matmul_type_t tiled_conv_type;

  // pointer들
  const elem_t *input;
  const elem_t *weights;
  const acc_t *bias;
  elem_t *output;

  // 기타 config
  int act;
  acc_scale_t scale;
  int gemmini_list;
  size_t sp_addr_start_stack, sp_addr_end_stack, acc_addr_start_stack;
  size_t sp_addr_A_stacked, sp_addr_B_stacked, acc_addr_stacked;
  size_t sp_addr_range, acc_addr_range;
  int tile_id;

  // 반복에서 쓰던 변수들
  int gemmini_num;
  int inner_call_counter;
  int lastK_toggle;

  bool done;
} shared_multi_conv_job_t;

static void shared_multi_choose_conv_tiling_factors_static(shared_multi_conv_job_t *job, int batches_static, int porows_static, int pocols_static, int pochs_static, int krows_static, int kcols_static, int kchs_static)
{
  job->gemmini_num = 0;
  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((job->gemmini_list >> i) & 1)
      job->gemmini_num++;
  }
  const int minimum_ch = DIM * job->gemmini_num;

  int pool_size = job->pool_size;
  int pool_stride = job->pool_stride;
  int pool_padding = job->pool_padding;

  const bool no_pool = (pool_stride == 0);
  if (no_pool)
  {
    pool_size = 1;
    pool_stride = 1;
    pool_padding = 0;
  }

  const int pool_out_row_dim = (job->out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (job->out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

  const bool downsample = job->stride == 2 && job->kernel_dim == 1 && job->padding == 0 &&
                          no_pool && job->in_row_dim % 2 == 0 && job->in_col_dim % 2 == 0;

  int args[] = {job->batch_size, pool_out_row_dim, pool_out_col_dim,
                job->out_channels, job->kernel_dim, job->kernel_dim, job->in_channels};
  const int max_args[] = {job->batch_size, pool_out_row_dim, pool_out_col_dim,
                          job->out_channels, job->kernel_dim, job->kernel_dim, job->in_channels};

  const int orows_idx = 1;
  const int ocols_idx = 2;
  const int out_channels_idx = 3;
  const int in_channels_idx = 6;

  const int max_spad_rows = (job->sp_addr_range / 2);
  const int max_acc_rows = (job->acc_addr_range / 2);

  int spad_rows = tiled_conv_total_spad_rows(false,
                                             job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                             args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  int acc_rows = tiled_conv_total_spad_rows(true,
                                            job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

  while (spad_rows > max_spad_rows || acc_rows > max_acc_rows)
  {
    int max_val = -1;
    int max_idx = -1;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    {
      if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1) &&
          args[i] > max_val &&
          !((i == out_channels_idx || i == in_channels_idx) && args[i] <= minimum_ch))
      {
        max_val = args[i];
        max_idx = i;
      }
    }

    if (max_idx == out_channels_idx || max_idx == in_channels_idx)
    {
      if (args[max_idx] % minimum_ch != 0)
      {
        args[max_idx] = (args[max_idx] / minimum_ch) * minimum_ch;
      }
      else
      {
        args[max_idx] -= minimum_ch;
      }
      if (args[max_idx] == 0)
        args[max_idx] = 1;
    }
    else
    {
      args[max_idx]--;
    }

    spad_rows = tiled_conv_total_spad_rows(false,
                                           job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                           args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
                                          job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                          args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  }

  bool not_increased = false;
  while (!not_increased)
  {
    not_increased = true;

    int args_candidate[] = {args[0], args[1], args[2],
                            args[3], args[4], args[5], args[6]};
    args_candidate[ocols_idx]++;

    if (args_candidate[ocols_idx] > max_args[ocols_idx])
      continue;

    spad_rows = tiled_conv_total_spad_rows(false,
                                           job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                           args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3],
                                           args_candidate[4], args_candidate[5], args_candidate[6],
                                           pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
                                          job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                          args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3],
                                          args_candidate[4], args_candidate[5], args_candidate[6],
                                          pool_size, pool_stride);

    if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows)
    {
      args[ocols_idx] = args_candidate[ocols_idx];
      not_increased = false;
    }
  }

  bool nothing_increased = false;
  while (!nothing_increased)
  {
    nothing_increased = true;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    {
      int args_candidate[] = {args[0], args[1], args[2],
                              args[3], args[4], args[5], args[6]};

      if (i == out_channels_idx || i == in_channels_idx)
      {
        args_candidate[i] += minimum_ch;
      }
      else
      {
        args_candidate[i]++;
      }

      if (args_candidate[i] > max_args[i])
        continue;

      spad_rows = tiled_conv_total_spad_rows(false,
                                             job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                             args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3],
                                             args_candidate[4], args_candidate[5], args_candidate[6],
                                             pool_size, pool_stride);
      acc_rows = tiled_conv_total_spad_rows(true,
                                            job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3],
                                            args_candidate[4], args_candidate[5], args_candidate[6],
                                            pool_size, pool_stride);

      if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows)
      {
        args[i] = args_candidate[i];
        nothing_increased = false;
      }
    }
  }

  const int batches = args[0];
  const int porows = args[1];
  const int pocols = args[2];
  const int pochs = args[3];
  const int krows = args[4];
  const int kcols = args[5];
  const int kchs = args[6];

  spad_rows = tiled_conv_total_spad_rows(false,
                                         job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                         batches_static, porows_static, pocols_static, pochs_static, krows_static, kcols_static, kchs_static,
                                         pool_size, pool_stride);
  acc_rows = tiled_conv_total_spad_rows(true,
                                        job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                        batches_static, porows_static, pocols_static, pochs_static, krows_static, kcols_static, kchs_static,
                                        pool_size, pool_stride);

#ifdef PRINT_TILE
#if PRINT_TILE
  printf("batches = %d\n", batches);
  printf("orows   = %d\n", porows);
  printf("ocols   = %d\n", pocols);
  printf("ochs    = %d\n", pochs);
  printf("krows   = %d\n", krows);
  printf("kcols   = %d\n", kcols);
  printf("kchs    = %d\n\n", kchs);

  printf("total spad_rows reserved: %d\n", spad_rows);
  printf("total acc_rows reserved: %d\n\n", acc_rows);

  printf("scratchpad row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("accumulator row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

  printf("inner matmul size: i=%d, j=%d, k=%d\n\n", pocols, pochs, kchs);
#endif
#endif

  job->batches = batches_static;
  job->porows = porows_static;
  job->pocols = pocols_static;
  job->pochs = pochs_static;
  job->krows = krows_static;
  job->kcols = kcols_static;
  job->kchs = kchs_static;

  job->pool_size = pool_size;
  job->pool_stride = no_pool ? 0 : pool_stride;
  job->pool_padding = pool_padding;

  job->sp_addr_range = spad_rows * 2;
  job->acc_addr_range = acc_rows * 2;

  job->sp_addr_A_stacked = tiled_conv_total_spad_rows_test(false, true, false,
                                                           job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                                           batches_static, porows_static, pocols_static, pochs_static, krows_static, kcols_static, kchs_static,
                                                           pool_size, pool_stride);
  job->sp_addr_B_stacked = tiled_conv_total_spad_rows_test(false, false, true,
                                                           job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                                           batches_static, porows_static, pocols_static, pochs_static, krows_static, kcols_static, kchs_static,
                                                           pool_size, pool_stride);
  job->acc_addr_stacked = acc_rows;
}

static void shared_multi_choose_conv_tiling_factors(shared_multi_conv_job_t *job)
{
  job->gemmini_num = 0;
  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((job->gemmini_list >> i) & 1)
      job->gemmini_num++;
  }
  const int minimum_ch = DIM * job->gemmini_num;

  int pool_size = job->pool_size;
  int pool_stride = job->pool_stride;
  int pool_padding = job->pool_padding;

  const bool no_pool = (pool_stride == 0);
  if (no_pool)
  {
    pool_size = 1;
    pool_stride = 1;
    pool_padding = 0;
  }

  const int pool_out_row_dim = (job->out_row_dim + 2 * pool_padding - pool_size) / pool_stride + 1;
  const int pool_out_col_dim = (job->out_col_dim + 2 * pool_padding - pool_size) / pool_stride + 1;

  const bool downsample = job->stride == 2 && job->kernel_dim == 1 && job->padding == 0 &&
                          no_pool && job->in_row_dim % 2 == 0 && job->in_col_dim % 2 == 0;

  int args[] = {job->batch_size, pool_out_row_dim, pool_out_col_dim,
                job->out_channels, job->kernel_dim, job->kernel_dim, job->in_channels};
  const int max_args[] = {job->batch_size, pool_out_row_dim, pool_out_col_dim,
                          job->out_channels, job->kernel_dim, job->kernel_dim, job->in_channels};

  const int orows_idx = 1;
  const int ocols_idx = 2;
  const int out_channels_idx = 3;
  const int in_channels_idx = 6;

  const int max_spad_rows = (job->sp_addr_range / 2);
  const int max_acc_rows = (job->acc_addr_range / 2);

  int spad_rows = tiled_conv_total_spad_rows(false,
                                             job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                             args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  int acc_rows = tiled_conv_total_spad_rows(true,
                                            job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                            args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);

  while (spad_rows > max_spad_rows || acc_rows > max_acc_rows)
  {
    int max_val = -1;
    int max_idx = -1;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    {
      if (!(i == ocols_idx && args[i] <= DIM && args[orows_idx] > 1) &&
          args[i] > max_val &&
          !((i == out_channels_idx || i == in_channels_idx) && args[i] <= minimum_ch))
      {
        max_val = args[i];
        max_idx = i;
      }
    }

    if (max_idx == out_channels_idx || max_idx == in_channels_idx)
    {
      if (args[max_idx] % minimum_ch != 0)
      {
        args[max_idx] = (args[max_idx] / minimum_ch) * minimum_ch;
      }
      else
      {
        args[max_idx] -= minimum_ch;
      }
      if (args[max_idx] == 0)
        args[max_idx] = 1;
    }
    else
    {
      args[max_idx]--;
    }

    spad_rows = tiled_conv_total_spad_rows(false,
                                           job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                           args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
                                          job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                          args[0], args[1], args[2], args[3], args[4], args[5], args[6], pool_size, pool_stride);
  }

  bool not_increased = false;
  while (!not_increased)
  {
    not_increased = true;

    int args_candidate[] = {args[0], args[1], args[2],
                            args[3], args[4], args[5], args[6]};
    args_candidate[ocols_idx]++;

    if (args_candidate[ocols_idx] > max_args[ocols_idx])
      continue;

    spad_rows = tiled_conv_total_spad_rows(false,
                                           job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                           args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3],
                                           args_candidate[4], args_candidate[5], args_candidate[6],
                                           pool_size, pool_stride);
    acc_rows = tiled_conv_total_spad_rows(true,
                                          job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                          args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3],
                                          args_candidate[4], args_candidate[5], args_candidate[6],
                                          pool_size, pool_stride);

    if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows)
    {
      args[ocols_idx] = args_candidate[ocols_idx];
      not_increased = false;
    }
  }

  bool nothing_increased = false;
  while (!nothing_increased)
  {
    nothing_increased = true;

    for (size_t i = 0; i < sizeof(args) / sizeof(args[0]); i++)
    {
      int args_candidate[] = {args[0], args[1], args[2],
                              args[3], args[4], args[5], args[6]};

      if (i == out_channels_idx || i == in_channels_idx)
      {
        args_candidate[i] += minimum_ch;
      }
      else
      {
        args_candidate[i]++;
      }

      if (args_candidate[i] > max_args[i])
        continue;

      spad_rows = tiled_conv_total_spad_rows(false,
                                             job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                             args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3],
                                             args_candidate[4], args_candidate[5], args_candidate[6],
                                             pool_size, pool_stride);
      acc_rows = tiled_conv_total_spad_rows(true,
                                            job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                            args_candidate[0], args_candidate[1], args_candidate[2], args_candidate[3],
                                            args_candidate[4], args_candidate[5], args_candidate[6],
                                            pool_size, pool_stride);

      if (spad_rows <= max_spad_rows && acc_rows <= max_acc_rows)
      {
        args[i] = args_candidate[i];
        nothing_increased = false;
      }
    }
  }

  const int batches = args[0];
  const int porows = args[1];
  const int pocols = args[2];
  const int pochs = args[3];
  const int krows = args[4];
  const int kcols = args[5];
  const int kchs = args[6];

  spad_rows = tiled_conv_total_spad_rows(false,
                                         job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                         batches, porows, pocols, pochs, krows, kcols, kchs,
                                         pool_size, pool_stride);
  acc_rows = tiled_conv_total_spad_rows(true,
                                        job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                        batches, porows, pocols, pochs, krows, kcols, kchs,
                                        pool_size, pool_stride);

#ifdef PRINT_TILE
#if PRINT_TILE
  printf("batches = %d\n", batches);
  printf("orows   = %d\n", porows);
  printf("ocols   = %d\n", pocols);
  printf("ochs    = %d\n", pochs);
  printf("krows   = %d\n", krows);
  printf("kcols   = %d\n", kcols);
  printf("kchs    = %d\n\n", kchs);

  printf("total spad_rows reserved: %d\n", spad_rows);
  printf("total acc_rows reserved: %d\n\n", acc_rows);

  printf("scratchpad row utilization: %d%%\n", (spad_rows * 100) / max_spad_rows);
  printf("accumulator row utilization: %d%%\n\n", (acc_rows * 100) / max_acc_rows);

  printf("inner matmul size: i=%d, j=%d, k=%d\n\n", pocols, pochs, kchs);
#endif
#endif

  job->batches = batches;
  job->porows = porows;
  job->pocols = pocols;
  job->pochs = pochs;
  job->krows = krows;
  job->kcols = kcols;
  job->kchs = kchs;

  job->pool_size = pool_size;
  job->pool_stride = no_pool ? 0 : pool_stride;
  job->pool_padding = pool_padding;

  job->sp_addr_range = spad_rows * 2;
  job->acc_addr_range = acc_rows * 2;

  job->sp_addr_A_stacked = tiled_conv_total_spad_rows_test(false, true, false,
                                                           job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                                           batches, porows, pocols, pochs, krows, kcols, kchs,
                                                           pool_size, pool_stride);
  job->sp_addr_B_stacked = tiled_conv_total_spad_rows_test(false, false, true,
                                                           job->stride, job->input_dilation, job->kernel_dilation, downsample, job->trans_weight_0132, job->trans_input_3120,
                                                           batches, porows, pocols, pochs, krows, kcols, kchs,
                                                           pool_size, pool_stride);
  job->acc_addr_stacked = acc_rows;
}

void shared_multi_tiled_conv_job_init(shared_multi_conv_job_t *job)
{
#ifdef GEMMINI_ASSERTIONS
  if (job->trans_weight_1203 && job->trans_weight_0132)
  {
    printf("Only one weight transformation can be applied at a time\n");
    exit(1);
  }
#endif

  if (job->tiled_conv_type == CPU)
  {
    int pool_size = job->pool_size;
    int pool_stride = job->pool_stride;
    int pool_padding = job->pool_padding;

    if (pool_size == 1 && pool_stride == 1 && pool_padding == 0)
    {
      pool_stride = 0;
    }

    // assume in_dim_rows = in_dim_cols
    // and out_dim_rows = out_dim_cols for now
    conv_cpu(
        job->batch_size, job->in_row_dim, job->in_col_dim, job->in_channels,
        job->out_channels, job->out_row_dim, job->out_col_dim,
        job->stride, job->input_dilation, job->kernel_dilation, job->padding, job->kernel_dim,
        job->wrot180, job->trans_output_1203, job->trans_input_3120,
        job->trans_weight_1203, job->trans_weight_0132,
        job->input, job->weights, job->bias, job->output,
        job->act, job->scale,
        pool_size, pool_stride, pool_padding);
    return;
  }
  else if (job->tiled_conv_type == OS)
  {
    printf("Gemmini convs do not currently support OS\n");
    exit(1);
  }

  // TODO move everything below this into a tiled_conv_outer function to match the tiled_matmul function

  job->no_bias = false;
  if (job->bias == NULL)
  {
    job->bias = (acc_t *)1;
    job->no_bias = true;
  }

  job->no_pool = job->pool_stride == 0;
  if (job->no_pool)
  {
    job->pool_size = 1;
    job->pool_stride = 1;
    job->pool_padding = 0;
  }

  job->downsample = job->stride == 2 && job->kernel_dim == 1 && job->in_row_dim % 2 == 0 && job->in_col_dim % 2 == 0 && job->padding == 0 && job->no_pool && job->input_dilation == 1 && !job->trans_input_3120;

  job->input_dilated = job->input_dilation == 2;

#ifdef GEMMINI_ASSERTIONS
  {
    // const int orows = porows * pool_stride + pool_size - 1;
    // const int ocols = pocols * pool_stride + pool_size - 1;

    // Check that data will fit in scratchpad
    const int spad_rows = tiled_conv_total_spad_rows(false,
                                                     job->stride, job->input_dilation, job->kernel_dilation, job->downsample, job->trans_weight_0132, job->trans_input_3120,
                                                     job->batches, job->porows, job->pocols, job->pochs, job->krows, job->kcols, job->kchs, job->pool_size, job->pool_stride);
    const int acc_rows = tiled_conv_total_spad_rows(true,
                                                    job->stride, job->input_dilation, job->kernel_dilation, job->downsample, job->trans_weight_0132, job->trans_input_3120,
                                                    job->batches, job->porows, job->pocols, job->pochs, job->krows, job->kcols, job->kchs, job->pool_size, job->pool_stride);

    if (spad_rows > job->sp_addr_range / 2)
    {
      printf("not enough scratchpad space to store inputs and weights, %d\n", spad_rows);
      exit(1);
    }
    if (acc_rows > job->acc_addr_range / 2)
    {
      printf("not enough accumulator space to store outputs\n");
      exit(1);
    }
    if (job->kernel_dim <= job->padding)
    {
      printf("kernel_dim must be larger than padding\n");
      exit(1);
    }
    if (job->input_dilation > 2)
    {
      printf("input_dilation > 2 is only supported on CPU\n");
      exit(1);
    }
    if (job->input_dilation > 1 && job->stride > 1)
    {
      printf("input input_dilation is only supported when stride == 1\n");
      exit(1);
    }
    if (job->trans_output_1203 && !job->no_pool)
    {
      printf("Output can only be transposed when pooling is disabled\n");
      exit(1);
    }
    if (job->trans_input_3120 && job->trans_weight_0132)
    {
      printf("Cannot transpose innermost dimensions of both inputs and weights on WS.\n");
      exit(1);
    }
  }
#endif

  const size_t st_dram_stride = job->trans_output_1203 ? job->batch_size * job->out_channels * sizeof(elem_t) : job->out_channels * sizeof(elem_t);

  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((job->gemmini_list >> i) & 1)
    {
      switch (i)
      {
      case 0:
        gemmini_extended_config_st(custom0, st_dram_stride, job->act, job->scale);
        gemmini_extended3_config_ex(custom0, WEIGHT_STATIONARY, 0, 0, 0, job->input_dilation, job->stride >> job->downsample, job->trans_input_3120, job->trans_weight_0132, false);
        break;
      case 1:
        gemmini_extended_config_st(custom1, st_dram_stride, job->act, job->scale);
        gemmini_extended3_config_ex(custom1, WEIGHT_STATIONARY, 0, 0, 0, job->input_dilation, job->stride >> job->downsample, job->trans_input_3120, job->trans_weight_0132, false);
        break;
      case 2:
        gemmini_extended_config_st(custom2, st_dram_stride, job->act, job->scale);
        gemmini_extended3_config_ex(custom2, WEIGHT_STATIONARY, 0, 0, 0, job->input_dilation, job->stride >> job->downsample, job->trans_input_3120, job->trans_weight_0132, false);
        break;
      case 3:
        gemmini_extended_config_st(custom3, st_dram_stride, job->act, job->scale);
        gemmini_extended3_config_ex(custom3, WEIGHT_STATIONARY, 0, 0, 0, job->input_dilation, job->stride >> job->downsample, job->trans_input_3120, job->trans_weight_0132, false);
        break;
      }
    }
  }

  job->pool_out_row_dim = (job->out_row_dim + 2 * job->pool_padding - job->pool_size) / job->pool_stride + 1;
  job->pool_out_col_dim = (job->out_col_dim + 2 * job->pool_padding - job->pool_size) / job->pool_stride + 1;
  job->dilated_in_row_dim = job->in_row_dim + (job->input_dilation - 1) * (job->in_row_dim - 1);
  job->dilated_in_col_dim = job->in_col_dim + (job->input_dilation - 1) * (job->in_col_dim - 1);

  job->b = job->porow = job->pocol = job->poch = job->krow = job->kcol = job->kch = 0;
  job->inner_call_counter = 0;
  job->lastK_toggle = 1;
  job->done = false;
}

static void shared_multi_tiled_conv_job_step(shared_multi_conv_job_t *job)
{
  if (job->done)
    return;

  int batch_size = job->batch_size;
  int in_row_dim = job->in_row_dim;
  int in_col_dim = job->in_col_dim;
  int in_channels = job->in_channels;
  int out_channels = job->out_channels;
  int out_row_dim = job->out_row_dim;
  int out_col_dim = job->out_col_dim;
  int stride = job->stride;
  int input_dilation = job->input_dilation;
  int kernel_dilation = job->kernel_dilation;
  int padding = job->padding;
  int kernel_dim = job->kernel_dim;
  bool wrot180 = job->wrot180;
  bool trans_output_1203 = job->trans_output_1203;
  bool trans_input_3120 = job->trans_input_3120;
  bool trans_weight_1203 = job->trans_weight_1203;
  bool trans_weight_0132 = job->trans_weight_0132;

  int batches = job->batches;
  int porows = job->porows;
  int pocols = job->pocols;
  int pochs = job->pochs;
  int krows = job->krows;
  int kcols = job->kcols;
  int kchs = job->kchs;

  int pool_size = job->pool_size;
  int pool_stride = job->pool_stride;
  int pool_padding = job->pool_padding;
  int pool_out_row_dim = job->pool_out_row_dim;
  int pool_out_col_dim = job->pool_out_col_dim;
  int dilated_in_row_dim = job->dilated_in_row_dim;
  int dilated_in_col_dim = job->dilated_in_col_dim;
  int input_dilated = job->input_dilated;
  bool no_bias = job->no_bias;
  bool no_pool = job->no_pool;
  bool downsample = job->downsample;

  const elem_t *input = job->input;
  const elem_t *weights = job->weights;
  const acc_t *bias = job->bias;
  elem_t *output = job->output;

  int act = job->act;
  acc_scale_t scale = job->scale;

  int gemmini_list = job->gemmini_list;
  int gemmini_num = job->gemmini_num;
  size_t sp_addr_start_stack = job->sp_addr_start_stack;
  size_t sp_addr_end_stack = job->sp_addr_end_stack;
  size_t acc_addr_start_stack = job->acc_addr_start_stack;
  size_t sp_addr_range = job->sp_addr_range;
  size_t acc_addr_range = job->acc_addr_range;
  int tile_id = job->tile_id;

  int b = job->b;
  int porow = job->porow;
  int pocol = job->pocol;
  int poch = job->poch;
  int krow = job->krow;
  int kcol = job->kcol;
  int kch = job->kch;

  const int orow = porow * pool_stride - pool_padding;

  const int ocol = pocol * pool_stride - pool_padding;

  const int orow_floored = orow < 0 ? 0 : orow;
  int irow = orow_floored * stride + krow * kernel_dilation - padding;

  const int ocol_floored = ocol < 0 ? 0 : ocol;
  int icol = ocol_floored * stride + kcol * kernel_dilation - padding;

  elem_t *out = output + (b * pool_out_row_dim * pool_out_col_dim + porow * pool_out_col_dim + pocol) * out_channels + poch;
  if (trans_output_1203)
  {
    out = output + (porow * pool_out_col_dim * batch_size + pocol * batch_size + b) * out_channels + poch;
  }

  if (krow + krows < kernel_dim ||
      kcol + kcols < kernel_dim ||
      kch + kchs < in_channels)
  {
    out = NULL;
  }

  const acc_t *bias_ = bias + poch;
  if (krow > 0 ||
      kcol > 0 ||
      kch > 0)
  {
    bias_ = NULL;
  }

  const int batches_ = batch_size - b > batches ? batches : batch_size - b;
  const int porows_ = pool_out_row_dim - porow > porows ? porows : pool_out_row_dim - porow;
  const int pocols_ = pool_out_col_dim - pocol > pocols ? pocols : pool_out_col_dim - pocol;
  const int pochs_ = out_channels - poch > pochs ? pochs : out_channels - poch;
  const int krows_ = kernel_dim - krow > krows ? krows : kernel_dim - krow;
  const int kcols_ = kernel_dim - kcol > kcols ? kcols : kernel_dim - kcol;
  const int kchs_ = in_channels - kch > kchs ? kchs : in_channels - kch;

  const int ocols_ = pocols_ * pool_stride + pool_size - 1;
  const int orows_ = porows_ * pool_stride + pool_size - 1;

  const int plpad = ocol < 0 ? -ocol : 0;
  const int prpad = ocol + ocols_ > out_col_dim ? ocol + ocols_ - out_col_dim : 0;
  const int pupad = orow < 0 ? -orow : 0;
  const int pdpad = orow + orows_ > out_row_dim ? orow + orows_ - out_row_dim : 0;

  const int dilated_krows_ = krows_ + (kernel_dilation - 1) * (krows_ - 1);
  const int dilated_kcols_ = kcols_ + (kernel_dilation - 1) * (kcols_ - 1);

  const int icols_ = (ocols_ - plpad - prpad) * stride + dilated_kcols_ - 1;
  const int irows_ = (orows_ - pupad - pdpad) * stride + dilated_krows_ - 1;

  int lpad = icol < 0 ? -icol : 0;
  int rpad = icol + icols_ > dilated_in_col_dim ? icol + icols_ - dilated_in_col_dim : 0;
  int upad = irow < 0 ? -irow : 0;
  int dpad = irow + irows_ > dilated_in_row_dim ? irow + irows_ - dilated_in_row_dim : 0;

  if (input_dilated)
  {
    lpad += lpad == 0 && icol % 2 != 0;
    rpad += rpad == 0 && (icol + icols_) % 2 != 1;
    upad += upad == 0 && irow % 2 != 0;
    dpad += dpad == 0 && (irow + irows_) % 2 != 1;
  }

  int krow_ = krow;
  int kcol_ = kcol;
  if (wrot180)
  {
    krow_ = kernel_dim - krow - krows_;
    kcol_ = kernel_dim - kcol - kcols_;
  }

  const elem_t *weights_slice = weights + (krow_ * kernel_dim * in_channels + kcol_ * in_channels + kch) * out_channels + poch;
  if (trans_weight_1203)
  {
    weights_slice = weights + (kch * kernel_dim * kernel_dim + krow_ * kernel_dim + kcol_) * out_channels + poch;
  }
  else if (trans_weight_0132)
  {
    weights_slice = weights + (krow_ * kernel_dim * out_channels + kcol_ * out_channels + poch) * in_channels + kch;
  }

  const elem_t *in = input + (b * in_row_dim * in_col_dim + ((irow + upad) >> input_dilated) * in_col_dim + ((icol + lpad) >> input_dilated)) * in_channels + kch;
  if (trans_input_3120)
  {
    in = input + (kch * in_row_dim * in_col_dim + ((irow + upad) >> input_dilated) * in_col_dim + ((icol + lpad) >> input_dilated)) * batch_size + b;
  }

  const int kchs_Remain = kchs_ % DIM;
  const int kchs_DIM = kchs_ / DIM + (kchs_Remain != 0);

  const int pochs_Remain = pochs_ % DIM;
  const int pochs_DIM = pochs_ / DIM + (pochs_Remain != 0);

  int using_gemmini_num = gemmini_num;
  if (kchs_DIM < gemmini_num && pochs_DIM < gemmini_num)
  {
    using_gemmini_num = (kchs_DIM < pochs_DIM) ? pochs_DIM : kchs_DIM;
  }

  int group_list = gemmini_list;
  int shift_num = 0;
  for (int i = 0; i < total_gemmini_num; i++)
  {
    size_t shifted_list = (gemmini_list >> i);
    if (shift_num == using_gemmini_num)
    {
      group_list = group_list & ~(shifted_list << i);
      break;
    }
    if (shifted_list & 1)
    {
      shift_num++;
    }
  }

  // for LdInput
  const size_t kchs_DIM_div = kchs_DIM / using_gemmini_num;
  const size_t kchs_DIM_div_added = kchs_DIM_div + 1;
  const size_t kchs_added_gemmini_num = kchs_DIM % using_gemmini_num;

  // for LdWeights, LdBias, Execute, St
  const size_t ochs_DIM_div = pochs_DIM / using_gemmini_num;
  const size_t ochs_DIM_div_added = ochs_DIM_div + 1;
  const size_t ochs_added_gemmini_num = pochs_DIM % using_gemmini_num;

  // for ld, st dram address
  const elem_t *in_local = in;
  const elem_t *weights_slice_local = weights_slice;
  const elem_t *out_local = out;
  const acc_t *bias_local = bias_;

  // for local addr
  size_t laddrochs_offset = 0;
  size_t laddrkchs_offset = 0;

  const int t = (job->inner_call_counter & 1);
  const size_t local_sp_addr_start = (t == 0) ? sp_addr_start_stack : sp_addr_start_stack + BANK_NUM * BANK_ROWS / 2;
  const size_t local_sp_addr_end = (t == 0) ? BANK_NUM * BANK_ROWS / 2 - sp_addr_end_stack : BANK_NUM * BANK_ROWS - sp_addr_end_stack;
  const size_t local_acc_addr_start = job->lastK_toggle ? acc_addr_start_stack : acc_addr_start_stack + ACC_ROWS / 2;

  int activated_gemmini_num = 0;
  for (int i = 0; i < total_gemmini_num; i++)
  {
    if ((group_list >> i) & 1)
    {
      size_t this_ochs = (activated_gemmini_num < ochs_added_gemmini_num) ? ochs_DIM_div_added * DIM : ochs_DIM_div * DIM;
      int last_ochs_gemmini = (ochs_DIM_div == 0) ? ochs_added_gemmini_num - 1 : using_gemmini_num - 1;
      if (pochs_Remain != 0 && activated_gemmini_num == last_ochs_gemmini)
      {
        this_ochs = this_ochs - DIM + pochs_Remain;
      }

      size_t this_kchs = (activated_gemmini_num < kchs_added_gemmini_num) ? kchs_DIM_div_added * DIM : kchs_DIM_div * DIM;
      int last_kchs_gemmini = (kchs_DIM_div == 0) ? kchs_added_gemmini_num - 1 : using_gemmini_num - 1;
      if (kchs_Remain != 0 && activated_gemmini_num == last_kchs_gemmini)
      {
        this_kchs = this_kchs - DIM + kchs_Remain;
      }

      shared_multi_sp_tiled_conv_test(
          i, group_list, tile_id << 1 | t,
          local_sp_addr_start, local_sp_addr_end, local_acc_addr_start,
          batch_size, in_row_dim, in_col_dim, in_channels,
          out_channels, out_row_dim, out_col_dim,
          pool_out_row_dim, pool_out_col_dim,

          stride, padding, kernel_dim, kernel_dilation,

          pool_size, pool_stride, pool_padding,

          this_ochs, this_kchs, laddrochs_offset, laddrkchs_offset,

          batches_,
          porows_, pocols_, pochs_,
          krows_, kcols_, kchs_,

          lpad, rpad, upad, dpad,
          plpad, prpad, pupad, pdpad,

          in_local,
          weights_slice_local,
          out_local,
          bias_local,

          act, scale,

          wrot180, trans_output_1203, trans_input_3120,
          trans_weight_1203, trans_weight_0132,

          no_bias, no_pool, downsample, input_dilated,
          false);

      laddrochs_offset += this_ochs;
      laddrkchs_offset += this_kchs;

      if (trans_input_3120)
      {
        in_local += this_kchs * in_row_dim * in_col_dim * batch_size;
      }
      else
      {
        in_local += this_kchs;
      }

      if (trans_weight_0132)
      {
        weights_slice_local += this_ochs * in_channels;
      }
      else
      {
        weights_slice_local += this_ochs;
      }

      out_local += this_ochs;
      if (krow + krows < kernel_dim ||
          kcol + kcols < kernel_dim ||
          kch + kchs < in_channels)
      {
        out_local = NULL;
      }

      bias_local += this_ochs;
      if (krow > 0 ||
          kcol > 0 ||
          kch > 0)
      {
        bias_local = NULL;
      }

      activated_gemmini_num++;
    }
  }

  job->inner_call_counter++;
  if (krow + krows >= kernel_dim &&
      kcol + kcols >= kernel_dim &&
      kch + kchs >= in_channels)
    job->lastK_toggle ^= 1;

  job->kch += kchs;
  if (job->kch >= in_channels)
  {
    job->kch = 0;
    job->kcol += kcols;
    if (job->kcol >= kernel_dim)
    {
      job->kcol = 0;
      job->krow += krows;
      if (job->krow >= kernel_dim)
      {
        job->krow = 0;
        job->poch += pochs;
        if (job->poch >= out_channels)
        {
          job->poch = 0;
          job->pocol += pocols;
          if (job->pocol >= pool_out_col_dim)
          {
            job->pocol = 0;
            job->porow += porows;
            if (job->porow >= pool_out_row_dim)
            {
              job->porow = 0;
              job->b += batches;
              if (job->b >= batch_size)
              {
                job->done = true;
              }
            }
          }
        }
      }
    }
  }
}
// made end

#undef abs

#endif // SRC_MAIN_C_GEMMINI_H
