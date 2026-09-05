// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_INCLUDE_VPU_FLASHATTENTION_KERNEL_H_
#define GEMMINI_ROCC_TESTS_INCLUDE_VPU_FLASHATTENTION_KERNEL_H_

/*
 * Header-only causal FlashAttention scheduler for the BF16 Gemmini + FP32
 * fusion VPU configuration.
 *
 * The caller owns Q/K/V/output storage and one DIM-by-DIM FP32 causal-mask
 * workspace.  Shapes and row strides are runtime values (strides are in
 * elements).  The implementation keeps the smoke-test execution contract:
 *
 *   grouped Q*K^T -> the shared accumulator SRAM
 *   grouped VPU causal mask + online softmax + O rescale
 *   grouped in-place P*V from the accumulator SRAM
 *   grouped VPU final normalization -> standalone host store
 *
 * Gemmini child commands and VPU commands retain their SharedDeps ordering;
 * the VPU row/tile programs use its capture/replay hardware loop.
 * Calls are synchronous, but not concurrently re-entrant: the caller must
 * hold exclusive ownership of the selected Gemminis and the VPU until the
 * function returns.  Q/K/V are read-only and may alias one another; output
 * and causal_mask_workspace must be mutually disjoint and must not overlap
 * any Q/K/V range.
 */

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "gemmini_params.h"
#include "gemmini_matmul_job.h"
#include "gemmini_page_packed.h"
#include "gemmini_tiling.h"
#include "rocc-software/src/xcustom.h"
#include "vpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Keep unsupported arithmetic/storage combinations from silently emitting a
 * command stream with a different ABI.  This header is intentionally opt-in
 * from vpu_kernels.h so standalone-VPU builds never see these constraints. */
#if !defined(VPU_ENABLE_GEMMINI_FLASHATTENTION) ||                             \
    VPU_ENABLE_GEMMINI_FLASHATTENTION != 1
#error "define VPU_ENABLE_GEMMINI_FLASHATTENTION=1 for the fusion kernel"
#endif
#if !defined(ELEM_T_IS_LOWPREC_FLOAT) || !defined(ELEM_T_EXP_BITS) ||          \
    !defined(ELEM_T_SIG_BITS) || ELEM_T_EXP_BITS != 8 || ELEM_T_SIG_BITS != 8
#error "vpu_flashattention_kernel requires Gemmini BF16 elements"
#endif
#if !defined(ACC_T_EXP_BITS) || !defined(ACC_T_SIG_BITS) ||                    \
    ACC_T_EXP_BITS != 8 || ACC_T_SIG_BITS != 24
#error "vpu_flashattention_kernel requires Gemmini FP32 accumulation"
#endif
#if VPU_STORAGE_KIND != VPU_STORAGE_FP32
#error "vpu_flashattention_kernel requires FP32 VPU storage"
#endif
#if VPU_GROUPED_COMMANDS != 1 || VPU_SHARED_DEPS != 1 ||                       \
    VPU_MATRIX_PORTS < 1 || VPU_MATRIX_PORTS > 4
#error                                                                         \
    "vpu_flashattention_kernel requires grouped commands, SharedDeps, and one to four logical Gemmini endpoints"
#endif
#if !defined(VPU_MATRIX_ROW_ELEMENTS)
#error "vpu_flashattention_kernel requires generated VPU matrix-row geometry"
#endif
#if VPU_MATRIX_ROW_ELEMENTS != DIM || VPU_VLEN < DIM
#error                                                                         \
    "vpu_flashattention_kernel requires matrix rows matching DIM and VLEN>=DIM"
#endif
/* The shared-ACC backend owns the physical lane-word packing. In particular, a DIM=8
 * accumulator row may occupy half of a 16-lane VPU word.  Software continues
 * to address the shared memory in FP32 elements and never assumes that a
 * matrix row is an integral number of physical VPU words. */
#if VPU_FP_STATE_ENTRIES < 3 * DIM
#error "vpu_flashattention_kernel requires m/l/alpha FP state for each query row"
#endif
#if ACC_ROWS < 2 * DIM || (ACC_ROWS % 2) != 0
#error "vpu_flashattention_kernel requires two DIM-row accumulator halves"
#endif
#if VPU_VSPAD_ELEMENTS < ACC_ROWS * DIM
#error                                                                         \
    "vpu_flashattention_kernel requires VPU access to both accumulator halves"
#endif
#if VPU_LOOP_BUFFER_ENTRIES < 48 || VPU_LOOP_STACK_DEPTH < 2
#error "vpu_flashattention_kernel requires a 48-entry, depth-2 hardware loop"
#endif

#define VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS (DIM * DIM)

typedef struct {
  const elem_t *queries; /* [query_rows][query_stride] */
  const elem_t *keys;    /* [sequence][key_stride] */
  const elem_t *values;  /* [sequence][value_stride] */
  float *output;         /* [query_rows][output_stride], disjoint */

  size_t query_rows;
  size_t sequence;
  size_t q_dim;
  size_t k_dim;
  size_t value_dim;
  /* Positive finite multiplier applied to Q*K^T before softmax.  The current
   * causal-mask schedule relies on positivity to preserve masked -Infinity. */
  float score_scale;
  size_t query_base; /* global sequence index of queries[0] */

  /* Element strides. A zero stride requests the contiguous logical width.
   * A GEMMINI_PAGE_PACKED_STRIDE(width) value is valid only for the input
   * sources: queries use Gemmini's A layout and keys/values use its B source
   * layout. output_stride is always an ordinary linear FP32 row stride; the
   * VPU H_STORE path rejects a packed output encoding. */
  size_t query_stride;
  size_t key_stride;
  size_t value_stride;
  size_t output_stride;

  /* Logical Gemmini-member bitmap. Member 0 is issued through the generated
   * XCUSTOM_ACC opcode; selected members must be fused endpoints. */
  unsigned gemmini_mask;

  /* QK and PV are independent collective matmuls and may partition different
   * logical dimensions. Zero-initialized configurations preserve M/M. */
  gemmini_partition_axis_t qk_partition_axis;
  gemmini_partition_axis_t pv_partition_axis;

  /* Optional exact full-block overrides. q_block_rows and kv_block_rows are
   * element-row counts; nonzero values must be DIM-aligned and no larger than
   * their logical query/sequence dimensions. qk_tile_k is a count of DIM-wide
   * QK reduction tiles. Zero preserves automatic planning. Nonzero overrides
   * are never rounded, shrunk, or grown; an unsupported forced plan fails. */
  size_t q_block_rows;
  size_t kv_block_rows;
  unsigned qk_tile_k;

  /* Reusable caller-owned storage disjoint from Q/K/V/output; at least
   * VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS floats.  The kernel writes its
   * packed 0/-Inf triangle before publishing CPU writes. */
  float *causal_mask_workspace;
  size_t causal_mask_workspace_elements;
} vpu_flashattention_config_t;

typedef struct {
  unsigned q_tiles;
  unsigned kv_tiles;
  unsigned qk_tiles;
  unsigned q_rows;
  unsigned kv_rows;
  unsigned qk_depth;
  unsigned gemmini_mask;
  unsigned gemmini_count;
} vpu_flashattention_plan_t;

typedef struct {
  uint64_t qk_row_loop_regions;
  uint64_t qk_row_loop_rows;
  uint64_t mask_row_loop_regions;
  uint64_t mask_row_loop_rows;
  uint64_t mask_diagonal_vectors;
  uint64_t mask_future_vectors;
  uint64_t normalize_row_loop_regions;
  uint64_t normalize_row_loop_rows;
  uint64_t rectangular_blocks;
  uint64_t causal_blocks;
  uint64_t skipped_future_blocks;
  uint64_t qk_jobs;
  uint64_t pv_jobs;
  uint64_t gemmini_job_steps;
} vpu_flashattention_stats_t;

typedef enum {
  VPU_FLASHATTENTION_OK = 0,
  VPU_FLASHATTENTION_INVALID_ARGUMENT,
  VPU_FLASHATTENTION_INVALID_SHAPE,
  VPU_FLASHATTENTION_INVALID_STRIDE,
  VPU_FLASHATTENTION_INVALID_GEMMINI_MASK,
  VPU_FLASHATTENTION_INSUFFICIENT_MASK_WORKSPACE,
  VPU_FLASHATTENTION_ADDRESS_RANGE_OVERFLOW,
  VPU_FLASHATTENTION_NO_TILING,
  VPU_FLASHATTENTION_QK_JOB_UNSUPPORTED,
  VPU_FLASHATTENTION_PV_JOB_UNSUPPORTED,
  VPU_FLASHATTENTION_VPU_ERROR,
} vpu_flashattention_status_t;

typedef struct {
  vpu_flashattention_status_t status;
  uint64_t vpu_status;
  vpu_flashattention_plan_t plan;
  vpu_flashattention_stats_t stats;
} vpu_flashattention_result_t;

enum {
  VPU_FA_CUSTOM0 = 0,
  VPU_FA_CUSTOM1 = 1,
  VPU_FA_CUSTOM2 = 2,
  VPU_FA_CUSTOM3 = 3,
  VPU_FA_CONFIG_EX = 0,
  VPU_FA_CONFIG_LD = 1,
  VPU_FA_CONFIG_ST = 2,
  VPU_FA_WEIGHT_STATIONARY = 1,
  VPU_FA_NO_ACTIVATION = 0,
};

/* xcustom stringifies funct7, so these must be preprocessor numeric literals
 * rather than enum constants. */
#define VPU_FA_K_CONFIG 0
#define VPU_FA_K_FLUSH 7
#define VPU_FA_K_LOOP_WS 8
#define VPU_FA_K_LOOP_WS_CONFIG_BOUNDS 9
#define VPU_FA_K_LOOP_WS_CONFIG_ADDRS_AB 10
#define VPU_FA_K_LOOP_WS_CONFIG_ADDRS_DC 11
#define VPU_FA_K_LOOP_WS_CONFIG_STRIDES_AB 12
#define VPU_FA_K_LOOP_WS_CONFIG_STRIDES_DC 13
#define VPU_FA_K_LOOP_WS_CONFIG_PARTITION_BOUNDS 24
#define VPU_FA_K_LOOP_WS_CONFIG_MV_BOUNDS_1 \
  VPU_FA_K_LOOP_WS_CONFIG_PARTITION_BOUNDS
#define VPU_FA_K_LOOP_WS_CONFIG_SPADDR 25
#define VPU_FA_K_LOOP_WS_CONFIG_PAGE_OFFSETS 28

#define VPU_FA_PACK_PAGE_BLOCK_OFFSETS(first_row_, first_col_, second_row_,    \
                                       second_col_)                            \
  ((((uint64_t)(first_row_)&UINT64_C(0xffff)) << 0) |                          \
   (((uint64_t)(first_col_)&UINT64_C(0xffff)) << 16) |                         \
   (((uint64_t)(second_row_)&UINT64_C(0xffff)) << 32) |                        \
   (((uint64_t)(second_col_)&UINT64_C(0xffff)) << 48))

#define VPU_FA_TOTAL_SPAD_ROWS (BANK_NUM * BANK_ROWS)
#define VPU_FA_MATRIX_TILE_ELEMENTS (DIM * DIM)
#define VPU_FA_CAUSAL_MASK_ELEMENTS VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS
#define VPU_FA_ACC_HALF_ROWS (ACC_ROWS / 2u)
#define VPU_FA_SCORE_BASE_ROW 0u
#define VPU_FA_SCORE_BASE (VPU_FA_SCORE_BASE_ROW * DIM)
#define VPU_FA_CAUSAL_MASK_BASE_ROW (ACC_ROWS - DIM)
#define VPU_FA_CAUSAL_MASK_BASE                                                \
  (VPU_FA_CAUSAL_MASK_BASE_ROW * DIM)
#define VPU_FA_SPAD_END (VPU_FA_TOTAL_SPAD_ROWS / 2u)
#define VPU_FA_STATE_ROWS (VPU_FP_STATE_ENTRIES / 3u)
#define VPU_FA_STATE_M_BASE 0u
#define VPU_FA_STATE_L_BASE VPU_FA_STATE_ROWS
#define VPU_FA_STATE_ALPHA_BASE (2u * VPU_FA_STATE_ROWS)

enum {
  VPU_FA_FP_M_OLD = 0,
  VPU_FA_FP_L_OLD = 1,
  VPU_FA_FP_M_NEW = 2,
  VPU_FA_FP_ALPHA = 3,
  VPU_FA_FP_P_SUM = 4,
  VPU_FA_FP_SCALE = 5,
  VPU_FA_FP_NEG_INF = 6,
  VPU_FA_FP_TERMINATOR = 7,
  VPU_FA_GP_VECTOR = 0,
  VPU_FA_GP_OUTPUT = 1,
  VPU_FA_GP_SCORE_ROW = 2,
  VPU_FA_GP_STATE = 3,
  VPU_FA_GP_HOST_OFFSET = 4,
  VPU_FA_GP_STATE_L = 4,
  VPU_FA_GP_OUTPUT_ROW = 5,
  VPU_FA_GP_MASK_ROW = 6,
  VPU_FA_GP_MASK_HOST_OFFSET = 7,
  VPU_FA_GP_FUTURE_ROW = 8,
  VPU_FA_GP_STATE_ALPHA = 9,
  VPU_FA_GP_ROW_LOOP = 14,
  VPU_FA_GP_TILE_LOOP = 15,
  VPU_FA_H_OUTPUT = 0,
  VPU_FA_H_CAUSAL_MASK = 1,
};

#define VPU_FA_LOOP_PRODUCE_EVENT_ID_SHIFT 3u
#define VPU_FA_LOOP_PRODUCE_EVENT_VALID (UINT64_C(1) << 6)
#define VPU_FA_LOOP_PRODUCE_EVENT_SEAL (UINT64_C(1) << 7)
#define VPU_FA_LOOP_WAIT_EVENT_ID_SHIFT 35u
#define VPU_FA_LOOP_WAIT_EVENT_VALID (UINT64_C(1) << 38)

static inline uint32_t vpu_fa_float_bits(float value) {
  union {
    float value;
    uint32_t bits;
  } conversion = {value};
  return conversion.bits;
}

static inline bool vpu_fa_positive_finite(float value) {
  const uint32_t bits = vpu_fa_float_bits(value);
  const uint32_t magnitude = bits & UINT32_C(0x7fffffff);
  return (bits & UINT32_C(0x80000000)) == 0u && magnitude != 0u &&
         (magnitude & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

/* funct7 must remain an assembly-time literal, hence this switch/macro pair. */
#ifndef VPU_FA_GEMMINI_ISSUE
#define VPU_FA_GEMMINI_ISSUE(custom_, rs1_, rs2_, funct_)                      \
  do {                                                                         \
    switch (custom_) {                                                         \
    case VPU_FA_CUSTOM0:                                                       \
      ROCC_INSTRUCTION_0_R_R(XCUSTOM_ACC, rs1_, rs2_, funct_);                 \
      break;                                                                   \
    case VPU_FA_CUSTOM1:                                                       \
      ROCC_INSTRUCTION_0_R_R(1, rs1_, rs2_, funct_);                           \
      break;                                                                   \
    case VPU_FA_CUSTOM2:                                                       \
      ROCC_INSTRUCTION_0_R_R(2, rs1_, rs2_, funct_);                           \
      break;                                                                   \
    default:                                                                   \
      ROCC_INSTRUCTION_0_R_R(3, rs1_, rs2_, funct_);                           \
      break;                                                                   \
    }                                                                          \
  } while (0)
#endif

#if GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE
#define VPU_FA_ISSUE_PARTITION_BOUNDS(                                         \
    custom_, axis_, partition_extent_, partition_offset_, aux_extent_,         \
    aux_offset_, aux_pad_)                                                     \
  do {                                                                         \
    VPU_FA_GEMMINI_ISSUE(                                                      \
        custom_,                                                               \
        gemmini_shared_partition_pack_rs1(                                     \
            axis_, partition_offset_, aux_offset_),                            \
        gemmini_shared_partition_pack_rs2(                                     \
            partition_extent_, aux_extent_, aux_pad_),                         \
        VPU_FA_K_LOOP_WS_CONFIG_PARTITION_BOUNDS);                             \
  } while (0)
#else
#define VPU_FA_ISSUE_PARTITION_BOUNDS(                                         \
    custom_, axis_, partition_extent_, partition_offset_, aux_extent_,         \
    aux_offset_, aux_pad_)                                                     \
  do {                                                                         \
  } while (0)
#endif

static inline int vpu_fa_group_last_member(unsigned group_list) {
  for (int i = 3; i >= 0; --i) {
    if (((group_list >> i) & 1u) != 0u)
      return i;
  }
  return -1;
}

static inline bool vpu_fa_issue_group_run(unsigned custom,
                                          unsigned group_list) {
  return (int)custom == vpu_fa_group_last_member(group_list);
}

static inline void vpu_fa_gemmini_flush(unsigned custom) {
  VPU_FA_GEMMINI_ISSUE(custom, 0, 0, VPU_FA_K_FLUSH);
}

static inline void vpu_fa_config_ex(unsigned custom, size_t a_stride,
                                    bool b_transpose) {
  const uint64_t rs1 =
      ((uint64_t)vpu_fa_float_bits((float)ACC_SCALE_IDENTITY) << 32) |
      (UINT64_C(1) << 16) | ((uint64_t)b_transpose << 9) |
      ((uint64_t)VPU_FA_WEIGHT_STATIONARY << 2) | VPU_FA_CONFIG_EX;
  const uint64_t rs2 = UINT64_C(1) << 48;
  (void)a_stride;
  VPU_FA_GEMMINI_ISSUE(custom, rs1, rs2, VPU_FA_K_CONFIG);
}

static inline void vpu_fa_config_ld(unsigned custom, size_t stride,
                                    unsigned id) {
  const uint64_t rs1 =
      ((uint64_t)vpu_fa_float_bits((float)MVIN_SCALE_IDENTITY) << 32) |
      ((uint64_t)DIM << 16) | (UINT64_C(1) << 8) | ((uint64_t)id << 3) |
      VPU_FA_CONFIG_LD;
  VPU_FA_GEMMINI_ISSUE(custom, rs1, stride, VPU_FA_K_CONFIG);
}

static inline void vpu_fa_config_st(unsigned custom) {
  const uint64_t rs2 =
      ((uint64_t)vpu_fa_float_bits((float)ACC_SCALE_IDENTITY) << 32);
  VPU_FA_GEMMINI_ISSUE(custom, VPU_FA_CONFIG_ST, rs2, VPU_FA_K_CONFIG);
}

/* Shared LOOP_WS dialect. Configuration is sent to every selected member,
 * while only the last endpoint launches funct=8 for atomic router broadcast.
 * ABI 0 retains explicit SPAD/ACC bases, but omits the
 * funct=24 descriptor because only the legacy M axis is available. NULL A
 * selects an ACC-resident A operand, while NULL D leaves C
 * initialization or in-place accumulation to ex_accumulate and the SPADDR
 * accumulator bases. */
#define VPU_FA_SHARED_LOOP_WS_AXIS(                                            \
    custom_, group_list_, group_id_, sp_addr_start_, sp_addr_end_,             \
    acc_addr_start_, axis_, partition_extent_, aux_extent_, aux_pad_,           \
    partition_offset_, aux_offset_,                                             \
    I_, J_, K_, pad_I_, pad_J_, pad_K_, A_, B_, D_, C_, A_stride_, B_stride_,  \
    D_stride_, C_stride_, A_transpose_, B_transpose_, full_C_, low_D_,         \
    ex_accumulate_, act_, A_row_offset_, A_col_offset_, B_row_offset_,         \
    B_col_offset_, D_row_offset_, D_col_offset_, C_row_offset_, C_col_offset_, \
    produce_event_id_, produce_event_valid_, produce_event_seal_,             \
    wait_event_id_, wait_event_valid_)                                         \
  do {                                                                         \
    const gemmini_partition_axis_t vpu_fa_axis_ = (axis_);                     \
    const size_t vpu_fa_partition_extent_ = (partition_extent_);               \
    const size_t vpu_fa_partition_offset_ = (partition_offset_);               \
    const size_t vpu_fa_aux_extent_ = (aux_extent_);                           \
    const size_t vpu_fa_aux_offset_ = (aux_offset_);                           \
    const size_t vpu_fa_aux_pad_ = (aux_pad_);                                 \
    if (!gemmini_shared_partition_fields_valid(                                \
            vpu_fa_axis_, vpu_fa_partition_extent_,                            \
            vpu_fa_partition_offset_, vpu_fa_aux_extent_,                      \
            vpu_fa_aux_offset_, vpu_fa_aux_pad_) ||                            \
        !gemmini_partition_axis_supported_by_abi(                              \
            vpu_fa_axis_, gemmini_partition_generated_axis_abi_version()))     \
      abort();                                                                 \
    VPU_FA_GEMMINI_ISSUE(custom_, acc_addr_start_,                             \
                         ((uint64_t)(sp_addr_end_) << 16) |                    \
                             (uint64_t)(sp_addr_start_),                       \
                         VPU_FA_K_LOOP_WS_CONFIG_SPADDR);                      \
    VPU_FA_ISSUE_PARTITION_BOUNDS(                                             \
        custom_, vpu_fa_axis_, vpu_fa_partition_extent_,                      \
        vpu_fa_partition_offset_, vpu_fa_aux_extent_, vpu_fa_aux_offset_,     \
        vpu_fa_aux_pad_);                                                      \
    VPU_FA_GEMMINI_ISSUE(custom_,                                              \
                         ((uint64_t)(pad_K_) << 32) |                          \
                             ((uint64_t)(pad_J_) << 16) | (uint64_t)(pad_I_),  \
                         ((uint64_t)(K_) << 32) | ((uint64_t)(J_) << 16) |     \
                             (uint64_t)(I_),                                   \
                         VPU_FA_K_LOOP_WS_CONFIG_BOUNDS);                      \
    VPU_FA_GEMMINI_ISSUE(custom_, (uint64_t)(uintptr_t)(A_),                   \
                         (uint64_t)(uintptr_t)(B_),                            \
                         VPU_FA_K_LOOP_WS_CONFIG_ADDRS_AB);                    \
    VPU_FA_GEMMINI_ISSUE(custom_, (uint64_t)(uintptr_t)(D_),                   \
                         (uint64_t)(uintptr_t)(C_),                            \
                         VPU_FA_K_LOOP_WS_CONFIG_ADDRS_DC);                    \
    VPU_FA_GEMMINI_ISSUE(custom_, A_stride_, B_stride_,                        \
                         VPU_FA_K_LOOP_WS_CONFIG_STRIDES_AB);                  \
    VPU_FA_GEMMINI_ISSUE(custom_, D_stride_, C_stride_,                        \
                         VPU_FA_K_LOOP_WS_CONFIG_STRIDES_DC);                  \
    VPU_FA_GEMMINI_ISSUE(                                                      \
        custom_,                                                               \
        VPU_FA_PACK_PAGE_BLOCK_OFFSETS(A_row_offset_, A_col_offset_,           \
                                       B_row_offset_, B_col_offset_),          \
        VPU_FA_PACK_PAGE_BLOCK_OFFSETS(D_row_offset_, D_col_offset_,           \
                                       C_row_offset_, C_col_offset_),          \
        VPU_FA_K_LOOP_WS_CONFIG_PAGE_OFFSETS);                                 \
    if (vpu_fa_issue_group_run((unsigned)(custom_),                            \
                               (unsigned)(group_list_))) {                     \
      const uint64_t vpu_fa_run_rs1 =                                          \
          ((uint64_t)(group_list_) << 48) |                                    \
          (((uint64_t)(wait_event_id_) & UINT64_C(0x7)) <<                     \
           VPU_FA_LOOP_WAIT_EVENT_ID_SHIFT) |                                  \
          (((uint64_t)(group_id_) & UINT64_C(0x7)) << 32) |                    \
          ((uint64_t)(act_) << 8) | ((uint64_t)(low_D_) << 2) |                \
          ((uint64_t)(full_C_) << 1) | (uint64_t)(ex_accumulate_) |            \
          (((uint64_t)(produce_event_id_) & UINT64_C(0x7)) <<                  \
           VPU_FA_LOOP_PRODUCE_EVENT_ID_SHIFT) |                               \
          ((produce_event_valid_) ? VPU_FA_LOOP_PRODUCE_EVENT_VALID : 0) |     \
          ((produce_event_seal_) ? VPU_FA_LOOP_PRODUCE_EVENT_SEAL : 0) |       \
          ((wait_event_valid_) ? VPU_FA_LOOP_WAIT_EVENT_VALID : 0);            \
      VPU_FA_GEMMINI_ISSUE(custom_, vpu_fa_run_rs1,                            \
                           ((uint64_t)(B_transpose_) << 1) |                   \
                               (uint64_t)(A_transpose_),                       \
                           VPU_FA_K_LOOP_WS);                                  \
    }                                                                          \
  } while (0)

typedef enum {
  VPU_FA_MATMUL_QK,
  VPU_FA_MATMUL_PV,
} vpu_fa_matmul_kind_t;

typedef struct {
  /* The common job is the single source of truth for dimensions, tiling,
   * partitioning, cursor progress, membership, and completion. */
  shared_multi_matmul_job_t core;
  shared_multi_matmul_job_extension_t extension;
  const vpu_flashattention_config_t *config;
  bool initialized;
} vpu_fa_matmul_job_t;

static inline unsigned vpu_fa_ceil_div(unsigned value, unsigned divisor) {
  return value / divisor + (value % divisor != 0u);
}

static inline unsigned vpu_fa_popcount(unsigned mask) {
  unsigned count = 0u;
  for (; mask != 0u; mask >>= 1)
    count += mask & 1u;
  return count;
}

static inline unsigned
vpu_fa_matmul_nominal_steps(unsigned total_k_tiles, unsigned tile_k) {
  return vpu_fa_ceil_div(total_k_tiles, tile_k);
}

static inline unsigned vpu_fa_matmul_planned_steps(
    vpu_fa_matmul_kind_t kind, unsigned total_k_tiles, unsigned tile_k) {
  (void)kind;
  return vpu_fa_matmul_nominal_steps(total_k_tiles, tile_k);
}

static inline unsigned vpu_fa_matmul_job_total_k_tiles(
    const vpu_fa_matmul_job_t *job) {
  return (unsigned)(job->core.dim_K_padded / DIM);
}

static inline unsigned vpu_fa_matmul_job_planned_steps(
    const vpu_fa_matmul_job_t *job) {
  return vpu_fa_matmul_nominal_steps(
      vpu_fa_matmul_job_total_k_tiles(job),
      (unsigned)job->core.tile_K);
}

static inline unsigned vpu_fa_matmul_job_next_k_tile(
    const vpu_fa_matmul_job_t *job) {
  const unsigned total = vpu_fa_matmul_job_total_k_tiles(job);
  const size_t issued =
      (size_t)job->core.inner_call_counter * job->core.tile_K;
  return issued < total ? (unsigned)issued : total;
}

typedef unsigned vpu_fa_stage_events_t;

#define VPU_FA_STAGE_WAIT_ID_SHIFT 0u
#define VPU_FA_STAGE_WAIT_VALID_SHIFT 3u
#define VPU_FA_STAGE_PRODUCE_ID_SHIFT 4u
#define VPU_FA_STAGE_PRODUCE_VALID_SHIFT 7u

static inline vpu_fa_stage_events_t
vpu_fa_stage_events(unsigned wait_event_id, bool wait_event_valid,
                    unsigned produce_event_id, bool produce_event_valid) {
  const unsigned event_mask = (1u << VPU_EVENT_ID_BITS) - 1u;
  return ((wait_event_id & event_mask) << VPU_FA_STAGE_WAIT_ID_SHIFT) |
      ((unsigned)wait_event_valid << VPU_FA_STAGE_WAIT_VALID_SHIFT) |
      ((produce_event_id & event_mask) << VPU_FA_STAGE_PRODUCE_ID_SHIFT) |
      ((unsigned)produce_event_valid << VPU_FA_STAGE_PRODUCE_VALID_SHIFT);
}

static inline unsigned
vpu_fa_stage_wait_event_id(vpu_fa_stage_events_t events) {
  return (events >> VPU_FA_STAGE_WAIT_ID_SHIFT) &
      ((1u << VPU_EVENT_ID_BITS) - 1u);
}

static inline bool
vpu_fa_stage_wait_event_valid(vpu_fa_stage_events_t events) {
  return ((events >> VPU_FA_STAGE_WAIT_VALID_SHIFT) & 1u) != 0u;
}

static inline unsigned
vpu_fa_stage_produce_event_id(vpu_fa_stage_events_t events) {
  return (events >> VPU_FA_STAGE_PRODUCE_ID_SHIFT) &
      ((1u << VPU_EVENT_ID_BITS) - 1u);
}

static inline bool
vpu_fa_stage_produce_event_valid(vpu_fa_stage_events_t events) {
  return ((events >> VPU_FA_STAGE_PRODUCE_VALID_SHIFT) & 1u) != 0u;
}

static inline void vpu_fa_group_command(vpu_fa_stage_events_t events,
                                        bool last,
                                        unsigned opcode, unsigned rd,
                                        unsigned rs1, unsigned rs2,
                                        unsigned rs3, unsigned funct1,
                                        uint64_t payload) {
  vpu_issue_event(vpu_fa_stage_wait_event_id(events),
                  vpu_fa_stage_wait_event_valid(events), last,
                  vpu_fa_stage_produce_event_id(events),
                  vpu_fa_stage_produce_event_valid(events),
                  vpu_fa_stage_produce_event_valid(events),
                  vpu_micro_op(opcode, rd, rs1, rs2, rs3, funct1), payload);
}

static inline void vpu_fa_event_uop(vpu_fa_stage_events_t events, bool last,
                                    uint32_t uop, uint64_t payload) {
  vpu_issue_event(vpu_fa_stage_wait_event_id(events),
                  vpu_fa_stage_wait_event_valid(events), last,
                  vpu_fa_stage_produce_event_id(events),
                  vpu_fa_stage_produce_event_valid(events),
                  vpu_fa_stage_produce_event_valid(events), uop, payload);
}

static inline void vpu_fa_group_write_gp(vpu_fa_stage_events_t events,
                                         unsigned reg,
                                         uint32_t value) {
  vpu_fa_group_command(events, false, VPU_OP_C_WRITE_GP, reg, 0, 0, 0, 0,
                       value);
}

static inline void vpu_fa_group_addi_gp(vpu_fa_stage_events_t events,
                                        unsigned dst,
                                        unsigned src, uint32_t immediate) {
  vpu_fa_event_uop(events, false, vpu_s_addi_int_uop(dst, src, immediate), 0);
}

static inline void vpu_fa_group_loop_start(vpu_fa_stage_events_t events,
                                           unsigned loop_reg,
                                           uint32_t iterations) {
  const uint32_t checked =
      iterations > 0u && iterations <= VPU_LOOP_COUNT_MAX ? iterations : 0u;
  vpu_fa_event_uop(events, false,
                   vpu_loop_start_uop(loop_reg, checked), 0);
}

static inline void vpu_fa_group_loop_end(vpu_fa_stage_events_t events,
                                         unsigned loop_reg) {
  vpu_fa_event_uop(events, false, vpu_loop_end_uop(loop_reg), 0);
}

static inline void vpu_fa_group_write_fp(vpu_fa_stage_events_t events,
                                         unsigned reg,
                                         float value) {
  vpu_fa_group_command(events, false, VPU_OP_C_WRITE_FP, reg, 0, 0, 0, 0,
                       vpu_float_to_bits(value));
}

static inline void vpu_fa_group_terminator(vpu_fa_stage_events_t events) {
  vpu_fa_group_command(events, true, VPU_OP_S_MAX_FP, VPU_FA_FP_TERMINATOR,
                       VPU_FA_FP_TERMINATOR, VPU_FA_FP_TERMINATOR, 0, 0, 0);
}

static inline void vpu_fa_group_set_vl(vpu_fa_stage_events_t events,
                                       size_t vl) {
  vpu_fa_group_command(events, false, VPU_OP_C_SET_VL, 0, 0, 0, 0, 0, vl);
}

static inline void vpu_fa_group_set_vector_stride(vpu_fa_stage_events_t events,
                                                  size_t stride_elements) {
  vpu_fa_group_command(events, false, VPU_OP_C_SET_VSTRIDE, 0, 0, 0, 0, 0,
                       stride_elements);
}

static inline void vpu_fa_group_vector(vpu_fa_stage_events_t events,
                                       unsigned opcode,
                                       unsigned dst_gp, unsigned src0_gp,
                                       unsigned src1_or_fp) {
  vpu_fa_group_command(events, false, opcode, dst_gp, src0_gp, src1_or_fp, 0,
                       0, 0);
}

static inline void vpu_fa_group_scalar(vpu_fa_stage_events_t events,
                                       unsigned opcode,
                                       unsigned dst_fp, unsigned src0_fp,
                                       unsigned src1_fp) {
  vpu_fa_group_command(events, false, opcode, dst_fp, src0_fp, src1_fp, 0, 0,
                       0);
}

static inline void vpu_fa_group_state_load_current(vpu_fa_stage_events_t events,
                                                   unsigned dst_fp,
                                                   unsigned state_gp) {
  vpu_fa_group_command(events, false, VPU_OP_S_LOAD_STATE, dst_fp, state_gp,
                       0, 0, 0, 0);
}

static inline void vpu_fa_group_state_store_current(vpu_fa_stage_events_t events,
                                                    unsigned state_gp,
                                                    unsigned src_fp) {
  vpu_fa_group_command(events, false, VPU_OP_S_STORE_STATE, state_gp, src_fp,
                       0, 0, 0, 0);
}

static inline vpu_flashattention_config_t
vpu_fa_normalize_config(const vpu_flashattention_config_t *source) {
  vpu_flashattention_config_t config = *source;
  if (config.query_stride == 0u)
    config.query_stride = config.q_dim;
  if (config.key_stride == 0u)
    config.key_stride = config.k_dim;
  if (config.value_stride == 0u)
    config.value_stride = config.value_dim;
  if (config.output_stride == 0u)
    config.output_stride = config.value_dim;
  return config;
}

static inline size_t vpu_fa_stride_payload(size_t stride) {
  return gemmini_page_packed_stride_is_packed(stride)
             ? gemmini_page_packed_stride_payload(stride)
             : stride;
}

static inline bool vpu_fa_stride_encoding_valid(size_t stride) {
  return stride != 0u && (stride & ~((size_t)UINT32_MAX)) == 0u &&
         vpu_fa_stride_payload(stride) != 0u;
}

static inline bool vpu_fa_checked_mul_size(size_t lhs, size_t rhs,
                                           size_t *product) {
  if (product == NULL || (lhs != 0u && rhs > SIZE_MAX / lhs))
    return false;
  *product = lhs * rhs;
  return true;
}

static inline size_t vpu_fa_ceil_div_size(size_t numerator,
                                          size_t denominator) {
  return numerator / denominator + (numerator % denominator != 0u);
}

typedef enum {
  VPU_FA_PAGE_LAYOUT_A,
  VPU_FA_PAGE_LAYOUT_B,
} vpu_fa_page_layout_t;

static inline bool vpu_fa_page_packed_storage_bytes(vpu_fa_page_layout_t layout,
                                                    size_t rows, size_t stride,
                                                    size_t *bytes) {
  const size_t row_blocks = vpu_fa_ceil_div_size(rows, DIM);
  const size_t col_blocks = vpu_fa_ceil_div_size(stride, DIM);
  size_t row_pages = 0u;
  size_t col_pages = 0u;

  if (layout == VPU_FA_PAGE_LAYOUT_A) {
    row_pages = vpu_fa_ceil_div_size(row_blocks,
                                     gemmini_page_packed_a_i_blocks_per_page());
    col_pages = vpu_fa_ceil_div_size(col_blocks,
                                     gemmini_page_packed_a_k_blocks_per_page());
  } else if (layout == VPU_FA_PAGE_LAYOUT_B) {
    row_pages = row_blocks;
    col_pages = vpu_fa_ceil_div_size(col_blocks,
                                     gemmini_page_packed_b_j_blocks_per_page());
  } else {
    return false;
  }

  size_t pages = 0u;
  return vpu_fa_checked_mul_size(row_pages, col_pages, &pages) &&
         vpu_fa_checked_mul_size(pages, GEMMINI_PAGE_PACKED_PAGE_BYTES, bytes);
}

static inline bool vpu_fa_pointer_range_overflows(const void *base,
                                                  size_t bytes) {
  return bytes != 0u && (uintptr_t)base > UINTPTR_MAX - (uintptr_t)(bytes - 1u);
}

static inline bool vpu_fa_page_aligned(const void *base) {
  return ((uintptr_t)base % GEMMINI_PAGE_PACKED_PAGE_BYTES) == 0u;
}

static inline bool vpu_fa_page_offset_dimension_fits(size_t elements) {
  return vpu_fa_ceil_div_size(elements, DIM) <= UINT16_MAX;
}

static inline bool vpu_fa_mask_region_fits(unsigned q_tiles,
                                           unsigned value_tiles) {
  const size_t output_rows = (size_t)q_tiles * value_tiles * DIM;
  return output_rows <= VPU_FA_ACC_HALF_ROWS;
}

static inline unsigned vpu_fa_output_base_row(unsigned q_rows,
                                              unsigned value_tiles) {
  const size_t q_tiles = vpu_fa_ceil_div(q_rows, DIM);
  const size_t output_rows = q_tiles * value_tiles * DIM;
  return (unsigned)(VPU_FA_ACC_HALF_ROWS - output_rows);
}

static inline uint32_t vpu_fa_output_base(unsigned q_rows,
                                          unsigned value_tiles) {
  return vpu_fa_output_base_row(q_rows, value_tiles) * DIM;
}

static inline uint32_t vpu_fa_matrix_address(uint32_t matrix_base,
                                             unsigned tile_columns,
                                             unsigned row,
                                             unsigned column_tile) {
  /* Match the natural accumulator layout: all Gemminis share one base and
   * LoopMatmul's global I offset selects a contiguous row tile. */
  const unsigned row_tile = row / DIM;
  const unsigned row_in_tile = row % DIM;
  const unsigned tile = row_tile * tile_columns + column_tile;
  return matrix_base + tile * VPU_FA_MATRIX_TILE_ELEMENTS +
         row_in_tile * DIM;
}

static inline bool vpu_fa_matrix_row_run_bank_local(uint32_t matrix_base,
                                                     unsigned tile_columns,
                                                     unsigned first_row,
                                                     unsigned row_count,
                                                     unsigned columns) {
  if (tile_columns == 0u || row_count == 0u || columns == 0u ||
      columns > VPU_VLEN)
    return false;

  const size_t last_column = columns - 1u;
  const size_t footprint =
      (last_column / DIM) * VPU_FA_MATRIX_TILE_ELEMENTS +
      last_column % DIM + 1u;
  for (unsigned offset = 0u; offset < row_count; ++offset) {
    const size_t row_base = vpu_fa_matrix_address(
        matrix_base, tile_columns, first_row + offset, 0u);
    if (row_base / VPU_ELEMENTS_PER_BANK !=
        (row_base + footprint - 1u) / VPU_ELEMENTS_PER_BANK)
      return false;
  }
  return true;
}

static inline bool
vpu_fa_matrix_row_run_vlen_chunks_bank_local(uint32_t matrix_base,
                                              unsigned tile_columns,
                                              unsigned first_row,
                                              unsigned row_count,
                                              unsigned columns) {
  if (tile_columns == 0u || row_count == 0u || columns <= VPU_VLEN ||
      VPU_VLEN < DIM || VPU_VLEN % DIM != 0u)
    return false;

  for (unsigned row_offset = 0u; row_offset < row_count; ++row_offset) {
    for (unsigned column = 0u; column < columns; column += VPU_VLEN) {
      const unsigned remaining = columns - column;
      const unsigned chunk_columns =
          remaining < VPU_VLEN ? remaining : VPU_VLEN;
      const size_t last_column = chunk_columns - 1u;
      const size_t footprint =
          (last_column / DIM) * VPU_FA_MATRIX_TILE_ELEMENTS +
          last_column % DIM + 1u;
      const size_t chunk_base = vpu_fa_matrix_address(
          matrix_base, tile_columns, first_row + row_offset, column / DIM);
      if (chunk_base / VPU_ELEMENTS_PER_BANK !=
          (chunk_base + footprint - 1u) / VPU_ELEMENTS_PER_BANK)
        return false;
    }
  }
  return true;
}

static inline bool vpu_fa_acc_live_set_fits(unsigned q_tiles,
                                            unsigned kv_tiles,
                                            unsigned value_tiles) {
  const size_t score_rows = (size_t)q_tiles * kv_tiles * DIM;
  const size_t output_rows = (size_t)q_tiles * value_tiles * DIM;
  return score_rows <= VPU_FA_ACC_HALF_ROWS - DIM &&
         output_rows <= VPU_FA_ACC_HALF_ROWS - score_rows;
}

static inline bool vpu_fa_tiling_fits(unsigned q_tiles, unsigned kv_tiles,
                                      unsigned value_tiles) {
  const size_t max_operand_rows = VPU_FA_TOTAL_SPAD_ROWS / 4u;
  const size_t pv_b_rows = (size_t)kv_tiles * value_tiles * DIM;
  const size_t q_rows = (size_t)q_tiles * DIM;

  /* S is overwritten in-place by P. O occupies the tail of the lower half,
   * while the DIM-row causal mask occupies the tail of the upper half. The
   * two inequalities in vpu_fa_acc_live_set_fits() protect both halves. */
  return vpu_fa_acc_live_set_fits(q_tiles, kv_tiles, value_tiles) &&
         vpu_fa_mask_region_fits(q_tiles, value_tiles) &&
         pv_b_rows <= max_operand_rows &&
         q_rows <= VPU_FA_STATE_ROWS;
}

/* Validate the largest element index before any C pointer arithmetic or
 * byte-stride encoding is performed.  The caller still owns the usual C API
 * contract that each pointer names an allocation large enough for the
 * declared shape. */
static inline bool vpu_fa_element_extent_overflows(size_t rows, size_t stride,
                                                   size_t columns) {
  const size_t max_elements = SIZE_MAX / sizeof(elem_t);
  return stride > max_elements || columns - 1u > max_elements ||
         rows - 1u > (max_elements - (columns - 1u)) / stride;
}

static inline vpu_flashattention_status_t
vpu_fa_validate_config(const vpu_flashattention_config_t *config) {
  if (config == NULL || config->queries == NULL || config->keys == NULL ||
      config->values == NULL || config->output == NULL ||
      config->causal_mask_workspace == NULL) {
    return VPU_FLASHATTENTION_INVALID_ARGUMENT;
  }
  if (!vpu_fa_positive_finite(config->score_scale))
    return VPU_FLASHATTENTION_INVALID_ARGUMENT;
  if (config->query_rows == 0u || config->sequence == 0u ||
      config->q_dim == 0u || config->k_dim == 0u || config->value_dim == 0u ||
      config->q_dim != config->k_dim || config->query_base > config->sequence ||
      config->query_rows > config->sequence - config->query_base ||
      config->query_rows > UINT32_MAX || config->sequence > UINT32_MAX ||
      config->q_dim > UINT32_MAX || config->value_dim > UINT32_MAX) {
    return VPU_FLASHATTENTION_INVALID_SHAPE;
  }
  if ((config->q_block_rows != 0u &&
       (config->q_block_rows % DIM != 0u ||
        config->q_block_rows > config->query_rows)) ||
      (config->kv_block_rows != 0u &&
       (config->kv_block_rows % DIM != 0u ||
        config->kv_block_rows > config->sequence)) ||
      (config->qk_tile_k != 0u &&
       config->qk_tile_k > vpu_fa_ceil_div((unsigned)config->q_dim, DIM))) {
    return VPU_FLASHATTENTION_INVALID_SHAPE;
  }
  if (!vpu_fa_stride_encoding_valid(config->query_stride) ||
      !vpu_fa_stride_encoding_valid(config->key_stride) ||
      !vpu_fa_stride_encoding_valid(config->value_stride) ||
      !vpu_fa_stride_encoding_valid(config->output_stride)) {
    return VPU_FLASHATTENTION_INVALID_STRIDE;
  }

  const bool packed_query =
      gemmini_page_packed_stride_is_packed(config->query_stride);
  const bool packed_key =
      gemmini_page_packed_stride_is_packed(config->key_stride);
  const bool packed_value =
      gemmini_page_packed_stride_is_packed(config->value_stride);
  const bool packed_output =
      gemmini_page_packed_stride_is_packed(config->output_stride);
  const size_t query_stride = vpu_fa_stride_payload(config->query_stride);
  const size_t key_stride = vpu_fa_stride_payload(config->key_stride);
  const size_t value_stride = vpu_fa_stride_payload(config->value_stride);
  const size_t output_stride = vpu_fa_stride_payload(config->output_stride);

  if (query_stride < config->q_dim || key_stride < config->k_dim ||
      value_stride < config->value_dim || output_stride < config->value_dim ||
      packed_output ||
      (packed_query && config->query_rows <= 1u) ||
      (packed_key && config->q_dim <= 1u) ||
      (packed_value && config->sequence <= 1u)) {
    return VPU_FLASHATTENTION_INVALID_STRIDE;
  }
  if ((packed_query &&
       (!vpu_fa_page_aligned(config->queries) ||
        !vpu_fa_page_offset_dimension_fits(config->query_rows) ||
        !vpu_fa_page_offset_dimension_fits(config->q_dim))) ||
      (packed_key && (!vpu_fa_page_aligned(config->keys) ||
                      !vpu_fa_page_offset_dimension_fits(config->q_dim) ||
                      !vpu_fa_page_offset_dimension_fits(config->sequence))) ||
      (packed_value &&
       (!vpu_fa_page_aligned(config->values) ||
        !vpu_fa_page_offset_dimension_fits(config->sequence) ||
        !vpu_fa_page_offset_dimension_fits(config->value_dim)))) {
    return VPU_FLASHATTENTION_INVALID_ARGUMENT;
  }
  if ((config->gemmini_mask & 0xfu) == 0u ||
      (config->gemmini_mask & ~0xfu) != 0u ||
      (config->gemmini_mask >> VPU_MATRIX_PORTS) != 0u) {
    return VPU_FLASHATTENTION_INVALID_GEMMINI_MASK;
  }
  if (!gemmini_partition_axis_supported_by_abi(
          config->qk_partition_axis,
          gemmini_partition_generated_axis_abi_version()) ||
      !gemmini_partition_axis_supported_by_abi(
          config->pv_partition_axis,
          gemmini_partition_generated_axis_abi_version())) {
    return VPU_FLASHATTENTION_INVALID_ARGUMENT;
  }
  if (config->causal_mask_workspace_elements < VPU_FA_CAUSAL_MASK_ELEMENTS) {
    return VPU_FLASHATTENTION_INSUFFICIENT_MASK_WORKSPACE;
  }
  if ((!packed_query && vpu_fa_element_extent_overflows(
                            config->query_rows, query_stride, config->q_dim)) ||
      (!packed_key && vpu_fa_element_extent_overflows(
                          config->sequence, key_stride, config->k_dim)) ||
      (!packed_value &&
       vpu_fa_element_extent_overflows(config->sequence, value_stride,
                                       config->value_dim))) {
    return VPU_FLASHATTENTION_ADDRESS_RANGE_OVERFLOW;
  }

  size_t packed_bytes = 0u;
  if ((packed_query &&
       (!vpu_fa_page_packed_storage_bytes(VPU_FA_PAGE_LAYOUT_A,
                                          config->query_rows, query_stride,
                                          &packed_bytes) ||
        vpu_fa_pointer_range_overflows(config->queries, packed_bytes))) ||
      (packed_key &&
       (!vpu_fa_page_packed_storage_bytes(VPU_FA_PAGE_LAYOUT_B,
                                          config->sequence, key_stride,
                                          &packed_bytes) ||
        vpu_fa_pointer_range_overflows(config->keys, packed_bytes))) ||
      (packed_value &&
       (!vpu_fa_page_packed_storage_bytes(VPU_FA_PAGE_LAYOUT_B,
                                          config->sequence, value_stride,
                                          &packed_bytes) ||
        vpu_fa_pointer_range_overflows(config->values, packed_bytes)))) {
    return VPU_FLASHATTENTION_ADDRESS_RANGE_OVERFLOW;
  }

  if (output_stride > UINT32_MAX ||
      (config->query_rows - 1u) >
          (UINT32_MAX - (config->value_dim - 1u)) / output_stride) {
    return VPU_FLASHATTENTION_ADDRESS_RANGE_OVERFLOW;
  }
  if (VPU_FA_ACC_HALF_ROWS % DIM != 0u ||
      VPU_FA_CAUSAL_MASK_BASE_ROW > UINT16_MAX ||
      VPU_FA_CAUSAL_MASK_BASE % VPU_FA_MATRIX_TILE_ELEMENTS != 0u) {
    return VPU_FLASHATTENTION_INVALID_SHAPE;
  }
  return VPU_FLASHATTENTION_OK;
}

static inline vpu_flashattention_status_t
vpu_flashattention_make_plan(const vpu_flashattention_config_t *source,
                             vpu_flashattention_plan_t *plan) {
  if (source == NULL || plan == NULL)
    return VPU_FLASHATTENTION_INVALID_ARGUMENT;
  const vpu_flashattention_config_t config = vpu_fa_normalize_config(source);
  const vpu_flashattention_status_t validation =
      vpu_fa_validate_config(&config);
  if (validation != VPU_FLASHATTENTION_OK)
    return validation;

  const unsigned gemmini_count = vpu_fa_popcount(config.gemmini_mask & 0xfu);
  const gemmini_partition_axis_t qk_axis = config.qk_partition_axis;
  const gemmini_partition_axis_t pv_axis = config.pv_partition_axis;
  const size_t state_rows = VPU_FA_STATE_ROWS;
  const size_t planner_q_rows =
      config.query_rows < state_rows ? config.query_rows : state_rows;
  const bool q_block_forced = config.q_block_rows != 0u;
  const bool kv_block_forced = config.kv_block_rows != 0u;
  const gemmini_tiling_request_t request = {
      .dim_I = planner_q_rows,
      .dim_J = config.sequence,
      .dim_K = config.q_dim,
      .dim = DIM,
      .gemmini_count = gemmini_count,
      .sp_addr_range = VPU_FA_TOTAL_SPAD_ROWS,
      .acc_addr_range = ACC_ROWS,
      .double_buffered = true,
      .act = 0,
  };
  const gemmini_tiling_factors_t factors =
      gemmini_shared_multi_choose_tiling_axis(&request, qk_axis);
  unsigned q_tiles = q_block_forced
                         ? (unsigned)(config.q_block_rows / DIM)
                         : (unsigned)factors.tile_I;
  unsigned kv_tiles = kv_block_forced
                          ? (unsigned)(config.kv_block_rows / DIM)
                          : (unsigned)factors.tile_J;
  const unsigned value_tiles = vpu_fa_ceil_div((unsigned)config.value_dim, DIM);

  const bool q_partitioned = qk_axis == GEMMINI_PARTITION_AXIS_M ||
                             pv_axis == GEMMINI_PARTITION_AXIS_M;
  const bool q_auxiliary = qk_axis == GEMMINI_PARTITION_AXIS_N ||
                           pv_axis == GEMMINI_PARTITION_AXIS_N;
  const bool kv_partitioned = qk_axis == GEMMINI_PARTITION_AXIS_N ||
                              pv_axis == GEMMINI_PARTITION_AXIS_K;
  const bool kv_auxiliary = pv_axis == GEMMINI_PARTITION_AXIS_M;
  const unsigned max_q_tiles =
      vpu_fa_ceil_div((unsigned)planner_q_rows, DIM);
  const unsigned max_kv_tiles =
      vpu_fa_ceil_div((unsigned)config.sequence, DIM);
  const unsigned min_q_tiles = (q_partitioned || q_auxiliary)
      ? (gemmini_count < max_q_tiles ? gemmini_count : max_q_tiles) : 1u;
  const unsigned min_kv_tiles = (kv_partitioned || kv_auxiliary)
      ? (gemmini_count < max_kv_tiles ? gemmini_count : max_kv_tiles) : 1u;
  const unsigned q_step = q_partitioned && max_q_tiles >= gemmini_count
      ? gemmini_count : 1u;
  const unsigned kv_step = kv_partitioned && max_kv_tiles >= gemmini_count
      ? gemmini_count : 1u;

  if (q_block_forced) {
    if (q_tiles < min_q_tiles ||
        (q_step > 1u && q_tiles % q_step != 0u))
      return VPU_FLASHATTENTION_NO_TILING;
  } else {
    if (q_tiles < min_q_tiles)
      q_tiles = min_q_tiles;
    if (q_step > 1u && q_tiles % q_step != 0u)
      q_tiles = (q_tiles / q_step) * q_step;
  }
  if (kv_block_forced) {
    if (kv_tiles < min_kv_tiles ||
        (kv_step > 1u && kv_tiles % kv_step != 0u))
      return VPU_FLASHATTENTION_NO_TILING;
  } else {
    if (kv_tiles < min_kv_tiles)
      kv_tiles = min_kv_tiles;
    if (kv_step > 1u && kv_tiles % kv_step != 0u)
      kv_tiles = (kv_tiles / kv_step) * kv_step;
  }
  if (pv_axis == GEMMINI_PARTITION_AXIS_N &&
      value_tiles >= gemmini_count && value_tiles % gemmini_count != 0u)
    return VPU_FLASHATTENTION_NO_TILING;

  while (q_tiles != 0u && kv_tiles != 0u &&
         !vpu_fa_tiling_fits(q_tiles, kv_tiles, value_tiles)) {
    const bool output_or_state_too_large =
        !vpu_fa_mask_region_fits(q_tiles, value_tiles) ||
        (size_t)q_tiles * DIM > VPU_FA_STATE_ROWS;
    if (output_or_state_too_large && !q_block_forced &&
        q_tiles > min_q_tiles) {
      q_tiles -= q_step;
    } else if (!kv_block_forced && kv_tiles > min_kv_tiles) {
      kv_tiles -= kv_step;
    } else if (!q_block_forced && q_tiles > min_q_tiles) {
      q_tiles -= q_step;
    } else {
      return VPU_FLASHATTENTION_NO_TILING;
    }
  }
  if (q_tiles == 0u || kv_tiles == 0u || factors.tile_K == 0u)
    return VPU_FLASHATTENTION_NO_TILING;

  /* The generic planner sees only Gemmini's C tile. Grow the sequence block
   * back to the largest value which fits the complete attention live set
   * (S/P + O + mask) and the unchanged operand-SPAD constraints. */
  const unsigned sequence_tiles = max_kv_tiles;
  while (!kv_block_forced && kv_tiles + kv_step <= sequence_tiles &&
         vpu_fa_tiling_fits(q_tiles, kv_tiles + kv_step, value_tiles))
    kv_tiles += kv_step;

  gemmini_tiling_request_t final_qk_request = request;
  final_qk_request.dim_I = (size_t)q_tiles * DIM;
  final_qk_request.dim_J = (size_t)kv_tiles * DIM;
  const gemmini_tiling_factors_t final_qk_factors =
      gemmini_shared_multi_choose_tiling_axis(&final_qk_request, qk_axis);
  const unsigned total_qk_tiles =
      vpu_fa_ceil_div((unsigned)config.q_dim, DIM);
  const unsigned final_qk_tiles = config.qk_tile_k != 0u
                                       ? config.qk_tile_k
                                       : (unsigned)final_qk_factors.tile_K;
  if (final_qk_tiles == 0u || final_qk_tiles > total_qk_tiles ||
      !gemmini_tiling_split_spad_rows_fit(
          q_tiles, kv_tiles, final_qk_tiles, DIM,
          VPU_FA_TOTAL_SPAD_ROWS / 4u) ||
      (qk_axis == GEMMINI_PARTITION_AXIS_M &&
       total_qk_tiles >= gemmini_count &&
       final_qk_tiles < gemmini_count) ||
      (qk_axis == GEMMINI_PARTITION_AXIS_K &&
       total_qk_tiles >= gemmini_count &&
       final_qk_tiles % gemmini_count != 0u))
    return VPU_FLASHATTENTION_NO_TILING;

  *plan = (vpu_flashattention_plan_t){
      .q_tiles = q_tiles,
      .kv_tiles = kv_tiles,
      .qk_tiles = final_qk_tiles,
      .q_rows = q_tiles * DIM,
      .kv_rows = kv_tiles * DIM,
      .qk_depth = final_qk_tiles * DIM,
      .gemmini_mask = config.gemmini_mask & 0xfu,
      .gemmini_count = gemmini_count,
  };
  return VPU_FLASHATTENTION_OK;
}

static inline unsigned vpu_fa_spad_start(unsigned group_id) {
  return (group_id & 1u) != 0u ? VPU_FA_SPAD_END : 0u;
}

static inline unsigned vpu_fa_spad_end(unsigned group_id) {
  return vpu_fa_spad_start(group_id) + VPU_FA_SPAD_END;
}

/* Group zero and group one are pipeline slots, not stage identifiers. Score
   * tiles are ping-ponged across the two accumulator halves. The accumulated O
   * remains at the lower-half tail and the common causal mask at the upper-half
   * tail; vpu_fa_acc_live_set_fits() keeps both score slots disjoint from them. */
static inline unsigned vpu_fa_score_base_row_from_slot(unsigned memory_slot) {
  return (memory_slot & 1u) * VPU_FA_ACC_HALF_ROWS;
}

static inline uint32_t vpu_fa_score_base_from_slot(unsigned memory_slot) {
  return vpu_fa_score_base_row_from_slot(memory_slot) * DIM;
}

/* Synchronization groups protect the two SPAD ping/pong halves. Event IDs,
 * rather than extra group pairs, carry producer/consumer ordering. */
static inline unsigned vpu_fa_group_pair_base(unsigned block_index) {
  return block_index & 1u;
}

static inline unsigned vpu_fa_block_memory_slot(unsigned block_index) {
  return block_index & 1u;
}

static inline unsigned vpu_fa_qk_event(unsigned block_index) {
  return (3u * block_index) & ((1u << VPU_EVENT_ID_BITS) - 1u);
}

static inline unsigned vpu_fa_softmax_event(unsigned block_index) {
  return (3u * block_index + 1u) & ((1u << VPU_EVENT_ID_BITS) - 1u);
}

static inline unsigned vpu_fa_pv_event(unsigned block_index) {
  return (3u * block_index + 2u) & ((1u << VPU_EVENT_ID_BITS) - 1u);
}

static inline bool vpu_fa_matmul_job_prepare(
    vpu_fa_matmul_job_t *job, const vpu_flashattention_config_t *config,
    vpu_fa_matmul_kind_t kind,
    unsigned q_start, unsigned q_rows, unsigned kv_start, unsigned kv_rows) {
  if (job == NULL || config == NULL)
    return false;
  const gemmini_partition_axis_t partition_axis =
      kind == VPU_FA_MATMUL_QK ? config->qk_partition_axis
                               : config->pv_partition_axis;
  if (!gemmini_partition_axis_supported_by_abi(
          partition_axis, gemmini_partition_generated_axis_abi_version()))
    return false;

  /* Do not clear the common job here. Every field consumed by the attention
   * path is assigned below; clearing the full generic object on every
   * preflight/dispatch job creates visible Rocket issue bubbles. */
  job->initialized = false;

  const unsigned dim_j =
      kind == VPU_FA_MATMUL_QK ? kv_rows : (unsigned)config->value_dim;
  const unsigned dim_k =
      kind == VPU_FA_MATMUL_QK ? (unsigned)config->q_dim : kv_rows;
  const unsigned i_tiles = vpu_fa_ceil_div(q_rows, DIM);
  const unsigned j_tiles = vpu_fa_ceil_div(dim_j, DIM);
  const unsigned total_k_tiles = vpu_fa_ceil_div(dim_k, DIM);
  unsigned selected_k_tiles = total_k_tiles;
  if (kind == VPU_FA_MATMUL_QK) {
    selected_k_tiles = config->qk_tile_k;
    if (selected_k_tiles == 0u) {
      const gemmini_tiling_request_t request = {
          .dim_I = q_rows,
          .dim_J = dim_j,
          .dim_K = dim_k,
          .dim = DIM,
          .gemmini_count = vpu_fa_popcount(config->gemmini_mask & 0xfu),
          .sp_addr_range = VPU_FA_TOTAL_SPAD_ROWS,
          .acc_addr_range = ACC_ROWS,
          .double_buffered = true,
          .act = 0,
      };
      selected_k_tiles = (unsigned)gemmini_shared_multi_choose_tiling_axis(
          &request, partition_axis).tile_K;
    }
  }
  const size_t max_operand_spad_rows = VPU_FA_TOTAL_SPAD_ROWS / 4u;
  const size_t max_acc_rows = VPU_FA_ACC_HALF_ROWS;
  const bool operand_spad_fits =
      kind == VPU_FA_MATMUL_QK
          ? gemmini_tiling_split_spad_rows_fit(
                i_tiles, j_tiles, selected_k_tiles, DIM,
                max_operand_spad_rows)
          : gemmini_tiling_B_spad_rows(
                i_tiles, j_tiles, total_k_tiles, DIM) <=
                max_operand_spad_rows;

  /* The shared planner aligns a recurring main partition tile to the Gemmini
   * count. This job may instead be an exact Q/KV tail; let job_step distribute
   * that tail by quotient/remainder (for example, six tiles become 2/2/1/1). */
  if (selected_k_tiles == 0u || selected_k_tiles > total_k_tiles ||
      !operand_spad_fits ||
      gemmini_tiling_total_acc_rows(i_tiles, j_tiles, DIM) > max_acc_rows)
    return false;
  /* PV's A is already matrix-tile-major in ACC. It cannot be compacted into a
   * temporary tile between K chunks, so consume the whole K block in one
   * logical job step. */
  const unsigned tile_k = kind == VPU_FA_MATMUL_PV
                              ? total_k_tiles
                              : selected_k_tiles;
  if (vpu_fa_matmul_planned_steps(kind, total_k_tiles, tile_k) == 0u)
    return false;

  /* Install the attention block as an exact instance of the ordinary shared
   * matmul job. The FlashAttention planner has already accounted for the
   * complete S/P + O + mask live set, so only its selected factors are copied
   * here; running a second generic auto-tiling pass would discard that policy. */
  shared_multi_matmul_job_t *core = &job->core;
  job->config = config;
  core->gemmini_list = (int)(config->gemmini_mask & 0xfu);
  core->gemmini_num = (int)vpu_fa_popcount((unsigned)core->gemmini_list);
  core->dim_I = q_rows;
  core->dim_J = dim_j;
  core->dim_K = dim_k;
  const size_t query_stride = vpu_fa_stride_payload(config->query_stride);
  const size_t key_stride = vpu_fa_stride_payload(config->key_stride);
  const size_t value_stride = vpu_fa_stride_payload(config->value_stride);
  if (kind == VPU_FA_MATMUL_QK) {
    core->A = gemmini_page_packed_stride_is_packed(config->query_stride)
                  ? config->queries
                  : config->queries + (size_t)q_start * query_stride;
    core->B = gemmini_page_packed_stride_is_packed(config->key_stride)
                  ? config->keys
                  : config->keys + (size_t)kv_start * key_stride;
  } else {
    core->A = NULL;
    core->B = gemmini_page_packed_stride_is_packed(config->value_stride)
                  ? config->values
                  : config->values + (size_t)kv_start * value_stride;
  }
  core->D = NULL;
  core->stride_A = kind == VPU_FA_MATMUL_QK ? config->query_stride : dim_k;
  core->stride_B = kind == VPU_FA_MATMUL_QK ? config->key_stride
                                             : config->value_stride;
  core->stride_D = kind == VPU_FA_MATMUL_QK ? 0u : config->value_dim;
  core->stride_C = 0u;
  core->a_transpose = false;
  core->b_transpose = kind == VPU_FA_MATMUL_QK;
  core->full_C = false;
  core->low_D = false;
  core->partition_axis = partition_axis;
  core->inner_call_counter = 0u;
  core->done = false;
  if (!shared_multi_matmul_job_set_exact_tiling(
          core, i_tiles, j_tiles, tile_k))
    return false;

  job->extension.a_from_acc = kind == VPU_FA_MATMUL_PV;
  job->extension.a_acc_addr_start = 0u;
  job->extension.fixed_acc_addr_start = 0u;
  job->extension.first_k_accumulate =
      kind == VPU_FA_MATMUL_PV && kv_start != 0u;
  job->extension.A_row_offset_base =
      kind == VPU_FA_MATMUL_QK ? q_start / DIM : 0u;
  job->extension.A_col_offset_base = 0u;
  job->extension.B_row_offset_base =
      kind == VPU_FA_MATMUL_PV ? kv_start / DIM : 0u;
  job->extension.B_col_offset_base =
      kind == VPU_FA_MATMUL_QK ? kv_start / DIM : 0u;
  job->extension.wait_event_id = 0u;
  job->extension.produce_event_id = 0u;
  job->extension.wait_event_valid = false;
  job->extension.produce_event_valid = false;
  return true;
}

static inline void vpu_fa_matmul_job_init(vpu_fa_matmul_job_t *job) {
  if (!gemmini_partition_axis_supported_by_abi(
          job->core.partition_axis,
          gemmini_partition_generated_axis_abi_version())) {
    job->initialized = false;
    job->core.done = true;
    return;
  }
  if (!shared_multi_matmul_job_init_single_ij_state(&job->core) ||
      job->core.I0 != 1u || job->core.J0 != 1u) {
    job->initialized = false;
    job->core.done = true;
    return;
  }

  const bool qk = !job->extension.a_from_acc;
  const size_t a_stride = job->core.stride_A;
  const size_t b_stride = job->core.stride_B;
  const size_t d_stride = job->core.stride_D;
  const size_t a_dma_stride =
      qk ? gemmini_page_packed_a_dma_stride_bytes(a_stride)
         : a_stride * sizeof(elem_t);
  const size_t b_dma_stride = gemmini_page_packed_b_dma_stride_bytes(b_stride);

  for (unsigned custom = 0u; custom < 4u; ++custom) {
    if ((((unsigned)job->core.gemmini_list >> custom) & 1u) == 0u)
      continue;
    vpu_fa_config_ex(custom, 1u, qk);
    vpu_fa_config_st(custom);
    vpu_fa_config_ld(custom, a_dma_stride, 0u);
    vpu_fa_config_ld(custom, b_dma_stride, 1u);
    vpu_fa_config_ld(custom, d_stride * sizeof(float), 2u);
  }
  job->initialized = true;
}

static inline void vpu_fa_matmul_job_set_events(
    vpu_fa_matmul_job_t *job,
    unsigned wait_event_id, bool wait_event_valid,
    unsigned produce_event_id, bool produce_event_valid) {
  job->extension.wait_event_id = wait_event_id;
  job->extension.wait_event_valid = wait_event_valid;
  job->extension.produce_event_id = produce_event_id;
  job->extension.produce_event_valid = produce_event_valid;
}

static inline bool vpu_fa_matmul_job_step(vpu_fa_matmul_job_t *job,
                                          unsigned group_id,
                                          unsigned memory_slot) {
  if (!job->initialized || job->core.done)
    return false;

  shared_multi_matmul_job_step_t core_step;
  if (!shared_multi_matmul_job_plan_step(&job->core, &core_step))
    return false;

  const vpu_flashattention_config_t *config = job->config;
  const unsigned k_tiles = (unsigned)core_step.K;
  const unsigned k_element_start =
      (unsigned)(core_step.k0 * job->core.tile_K * DIM);
  const unsigned pad_i = (unsigned)core_step.pad_I;
  const unsigned pad_j = (unsigned)core_step.pad_J;
  const unsigned pad_k = (unsigned)core_step.pad_K;
  const bool final_step = core_step.final_job_step;

  const gemmini_partition_axis_t partition_axis = job->core.partition_axis;
  const unsigned active_count = (unsigned)core_step.active_gemmini_count;
  const unsigned group_list = (unsigned)core_step.group_list;
  group_id &= (1u << VPU_GROUP_ID_BITS) - 1u;
  memory_slot &= 1u;
  /* Preserve the generic Gemmini step invariant: the admission group and the
   * physical SPAD ping/pong half advance together.  The logical attention
   * slot remains independent because every K step must accumulate into the
  * same S/P half even while its SPAD operands alternate. */
  const unsigned spad_slot = group_id;
  const bool qk = !job->extension.a_from_acc;
  const unsigned score_local_row =
      vpu_fa_score_base_row_from_slot(memory_slot);
  const unsigned value_tiles =
      vpu_fa_ceil_div((unsigned)config->value_dim, DIM);
  const unsigned output_local_row =
      vpu_fa_output_base_row((unsigned)job->core.dim_I, value_tiles);
  job->extension.a_acc_addr_start = score_local_row;
  job->extension.fixed_acc_addr_start =
      qk ? score_local_row : output_local_row;
  const bool ex_accumulate =
      core_step.k0 != 0u || job->extension.first_k_accumulate;

  unsigned member_index = 0u;

  for (unsigned custom = 0u; custom < 4u; ++custom) {
    if (((group_list >> custom) & 1u) == 0u)
      continue;

    gemmini_partition_plan_t member_plan;
    if (!gemmini_partition_plan_member(
            partition_axis, core_step.I, core_step.J, core_step.K,
            pad_i, pad_j, pad_k, active_count, member_index,
            &member_plan)) {
      job->core.partition_status = SHARED_MULTI_PARTITION_FIELD_OVERFLOW;
      job->core.done = true;
      return false;
    }

    size_t member_pad_i = pad_i;
    size_t member_pad_j = pad_j;
    size_t member_pad_k = pad_k;
    size_t a_i_offset = 0u;
    size_t a_k_offset = 0u;
    size_t b_k_offset = 0u;
    size_t b_j_offset = 0u;
    if (partition_axis == GEMMINI_PARTITION_AXIS_M) {
      member_pad_i = member_plan.partition_pad;
      a_i_offset = member_plan.partition_offset;
      b_k_offset = member_plan.aux_offset;
    } else if (partition_axis == GEMMINI_PARTITION_AXIS_N) {
      member_pad_j = member_plan.partition_pad;
      a_i_offset = member_plan.aux_offset;
      b_j_offset = member_plan.partition_offset;
    } else {
      member_pad_k = member_plan.partition_pad;
      a_k_offset = member_plan.partition_offset;
      b_k_offset = member_plan.partition_offset;
    }

    const unsigned acc_base =
        (unsigned)job->extension.fixed_acc_addr_start;
    const unsigned a_local_base =
        qk ? vpu_fa_spad_start(spad_slot)
           : (unsigned)job->extension.a_acc_addr_start;

    const elem_t *a = NULL;
    const elem_t *b = NULL;
    size_t a_row_offset = 0u;
    size_t a_col_offset = 0u;
    size_t b_row_offset = 0u;
    size_t b_col_offset = 0u;
    if (qk) {
      const size_t query_stride =
          vpu_fa_stride_payload(job->core.stride_A);
      const size_t key_stride =
          vpu_fa_stride_payload(job->core.stride_B);
      a = gemmini_page_packed_stride_is_packed(job->core.stride_A)
              ? job->core.A
              : job->core.A + a_i_offset * DIM * query_stride +
                    k_element_start + a_k_offset * DIM;
      b = gemmini_page_packed_stride_is_packed(job->core.stride_B)
              ? job->core.B
              : job->core.B + b_j_offset * DIM * key_stride +
                    k_element_start + b_k_offset * DIM;
      a_row_offset = job->extension.A_row_offset_base;
      a_col_offset =
          job->extension.A_col_offset_base + k_element_start / DIM;
      /* B offsets are expressed in logical (K,J) coordinates.  The
       * LoopMatmul transpose path swaps them onto the physical [KV,D]
       * page-packed key source. */
      b_row_offset =
          job->extension.B_row_offset_base + k_element_start / DIM;
      b_col_offset = job->extension.B_col_offset_base;
    } else {
      const size_t value_stride =
          vpu_fa_stride_payload(job->core.stride_B);
      b = gemmini_page_packed_stride_is_packed(job->core.stride_B)
              ? job->core.B
              : job->core.B +
                    (k_element_start + b_k_offset * DIM) * value_stride +
                    b_j_offset * DIM;
      b_row_offset =
          job->extension.B_row_offset_base + k_element_start / DIM;
      b_col_offset = job->extension.B_col_offset_base;
    }

    VPU_FA_SHARED_LOOP_WS_AXIS(
        custom, group_list, group_id, a_local_base,
        vpu_fa_spad_end(spad_slot),
        acc_base, partition_axis,
        member_plan.partition_extent, member_plan.aux_extent,
        member_plan.aux_pad, member_plan.partition_offset,
        member_plan.aux_offset, core_step.I, core_step.J, core_step.K,
        member_pad_i, member_pad_j, member_pad_k,
        job->extension.a_from_acc ? NULL : a, b, NULL, NULL,
        job->core.stride_A, job->core.stride_B,
        job->core.stride_D, job->core.stride_C,
        job->core.a_transpose, job->core.b_transpose, false, false,
        ex_accumulate,
        VPU_FA_NO_ACTIVATION, a_row_offset, a_col_offset, b_row_offset,
        b_col_offset, 0u, 0u, 0u, 0u,
        job->extension.produce_event_id,
        job->extension.produce_event_valid,
        final_step && job->extension.produce_event_valid,
        job->extension.wait_event_id,
        core_step.first_job_step && job->extension.wait_event_valid);

    ++member_index;
  }

  if (member_index != active_count) {
    job->core.partition_status = SHARED_MULTI_PARTITION_EMPTY_MASK;
    job->core.done = true;
    return false;
  }

  if (final_step &&
      core_step.k0 * job->core.tile_K + k_tiles !=
          vpu_fa_matmul_job_total_k_tiles(job))
    abort();
  shared_multi_matmul_job_complete_step(&job->core);
  return true;
}

static inline void
vpu_fa_issue_matmul_job(vpu_fa_matmul_job_t *job,
                        unsigned group_pair_base, unsigned memory_slot,
                        vpu_flashattention_stats_t *stats) {
  vpu_fa_matmul_job_init(job);
  while (!job->core.done) {
    const unsigned step_group_id =
        group_pair_base ^ ((unsigned)job->core.inner_call_counter & 1u);
    (void)vpu_fa_matmul_job_step(job, step_group_id, memory_slot);
    ++stats->gemmini_job_steps;
  }
}

static inline bool
vpu_fa_matmul_job_is_single_step(const vpu_fa_matmul_job_t *job) {
  return vpu_fa_matmul_job_planned_steps(job) == 1u;
}

static inline bool
vpu_fa_matmul_job_replay_safe(const vpu_fa_matmul_job_t *job) {
  return vpu_fa_matmul_job_planned_steps(job) <= 3u;
}

static inline bool vpu_fa_cross_tile_coalescing_fits(unsigned elements) {
  /* `fragmentStride` preserves the tile-major address of every ACC-row
   * fragment in one logical VPU word.  A DIM=8 target can therefore coalesce
   * two rows in its 16 lanes without requiring VPU_NLANES <= DIM. */
  return elements != 0u && elements <= VPU_VLEN;
}

/* Admit the bounded two-chunk fabric-batched path known to fit in the
 * 48-entry outer row loop.  Keeping this predicate shared by the row setup
 * and both score phases also makes the vector-stride state explicit: the row
 * loop establishes tile-major gathering once, while the phase bodies spend
 * their entries only on useful fabric work. */
static inline bool
vpu_fa_score_two_chunk_chain(unsigned full_tiles, unsigned tail_columns,
                             bool chunks_bank_local) {
  const unsigned total_columns = full_tiles * DIM + tail_columns;
  if (!chunks_bank_local || VPU_VLEN < DIM || VPU_VLEN % DIM != 0u ||
      total_columns <= VPU_VLEN)
    return false;

  const unsigned full_chunks = total_columns / VPU_VLEN;
  const unsigned tail_chunk = total_columns % VPU_VLEN;
  return full_chunks + (tail_chunk != 0u ? 1u : 0u) == 2u;
}

static inline void vpu_fa_group_score_max_stream(unsigned group_id,
                                                 unsigned full_tiles,
                                                 unsigned tail_columns,
                                                 bool chunks_bank_local) {
  const bool expand_tiles =
      VPU_LOOP_BUFFER_ENTRIES >= 56u && tail_columns == 0u && full_tiles <= 4u;
  const unsigned total_columns = full_tiles * DIM + tail_columns;
  /* Keep Gemmini's tile-major matrix layout, but present one score row to the
   * VPU as a single logical vector. This removes per-DIM reduction drains
   * while retaining the shared QK/PV accumulator addresses. */
  if (vpu_fa_cross_tile_coalescing_fits(total_columns)) {
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_SCORE_ROW, 0u);
    vpu_fa_group_set_vl(group_id, total_columns);
    vpu_fa_group_set_vector_stride(group_id, VPU_FA_MATRIX_TILE_ELEMENTS);
    vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, VPU_FA_FP_SCALE);
    vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX, VPU_FA_FP_M_NEW,
                         VPU_FA_GP_VECTOR, 0, 0, 0, 0);
    vpu_fa_group_set_vector_stride(group_id, 0u);
    return;
  }
  if (vpu_fa_score_two_chunk_chain(full_tiles, tail_columns,
                                   chunks_bank_local)) {
    const unsigned tail_chunk = total_columns % VPU_VLEN;
    const unsigned chunk_stride =
        (VPU_VLEN / DIM) * VPU_FA_MATRIX_TILE_ELEMENTS;

    /* Keep both chunk addresses live so same-fabric vector commands are truly
     * adjacent for the two-full-chunk case.  A real tail only needs the
     * unavoidable VL change between them. */
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_SCORE_ROW,
                         chunk_stride);
    /* Base-FMA phase: finish both MUL chunks before changing fabrics. */
    vpu_fa_group_set_vl(group_id, VPU_VLEN);
    vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_SCORE_ROW,
                        VPU_FA_GP_SCORE_ROW, VPU_FA_FP_SCALE);
    if (tail_chunk != 0u)
      vpu_fa_group_set_vl(group_id, tail_chunk);
    vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, VPU_FA_FP_SCALE);

    /* Reduction phase: START retains the fabric accumulator; FINAL folds it
     * once and writes M_NEW. */
    if (tail_chunk != 0u)
      vpu_fa_group_set_vl(group_id, VPU_VLEN);
    vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX,
                         VPU_FA_FP_M_NEW, VPU_FA_GP_SCORE_ROW, 0, 0,
                         VPU_REDUCTION_START, 0);
    if (tail_chunk != 0u)
      vpu_fa_group_set_vl(group_id, tail_chunk);
    vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX,
                         VPU_FA_FP_M_NEW, VPU_FA_GP_VECTOR, 0, 0,
                         VPU_REDUCTION_FINAL, 0);
    return;
  }
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_SCORE_ROW, 0u);
  /* A score row can be wider than VLEN while still consisting of complete
   * tile rows.  Stream it in VLEN-sized, tile-aligned chunks so a DIM=8
   * matrix can use all 16 physical VPU lanes instead of reducing one
   * eight-element tile at a time. */
  if (chunks_bank_local && VPU_VLEN >= DIM && VPU_VLEN % DIM == 0u &&
      total_columns > VPU_VLEN) {
    const unsigned full_chunks = total_columns / VPU_VLEN;
    const unsigned tail_chunk = total_columns % VPU_VLEN;
    const unsigned chunk_stride =
        (VPU_VLEN / DIM) * VPU_FA_MATRIX_TILE_ELEMENTS;

    vpu_fa_group_set_vl(group_id, VPU_VLEN);
    vpu_fa_group_set_vector_stride(group_id, VPU_FA_MATRIX_TILE_ELEMENTS);
    if (full_chunks == 1u) {
      vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_SCALE);
      vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX,
                           VPU_FA_FP_M_NEW, VPU_FA_GP_VECTOR, 0, 0, 0, 0);
      if (tail_chunk != 0u)
        vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                             chunk_stride);
    } else {
      vpu_fa_group_loop_start(group_id, VPU_FA_GP_TILE_LOOP, full_chunks);
      vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_SCALE);
      vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX,
                           VPU_FA_FP_M_NEW, VPU_FA_GP_VECTOR, 0, 0, 0, 0);
      vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                           chunk_stride);
      vpu_fa_group_loop_end(group_id, VPU_FA_GP_TILE_LOOP);
    }
    if (tail_chunk != 0u) {
      vpu_fa_group_set_vl(group_id, tail_chunk);
      vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_SCALE);
      vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX,
                           VPU_FA_FP_M_NEW, VPU_FA_GP_VECTOR, 0, 0, 0, 0);
    }
    vpu_fa_group_set_vector_stride(group_id, 0u);
    return;
  }
  if (full_tiles != 0u) {
    vpu_fa_group_set_vl(group_id, DIM);
    if (expand_tiles) {
      for (unsigned tile = 0u; tile < full_tiles; ++tile) {
        vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                            VPU_FA_GP_VECTOR, VPU_FA_FP_SCALE);
        vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX, VPU_FA_FP_M_NEW,
                             VPU_FA_GP_VECTOR, 0, 0, 0, 0);
        if (tile + 1u < full_tiles)
          vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                               VPU_FA_MATRIX_TILE_ELEMENTS);
      }
    } else if (full_tiles == 1u) {
      vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_SCALE);
      vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX, VPU_FA_FP_M_NEW,
                           VPU_FA_GP_VECTOR, 0, 0, 0, 0);
      if (tail_columns != 0u)
        vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                             VPU_FA_MATRIX_TILE_ELEMENTS);
    } else {
      vpu_fa_group_loop_start(group_id, VPU_FA_GP_TILE_LOOP, full_tiles);
      vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_SCALE);
      vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX, VPU_FA_FP_M_NEW,
                           VPU_FA_GP_VECTOR, 0, 0, 0, 0);
      vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                           VPU_FA_MATRIX_TILE_ELEMENTS);
      vpu_fa_group_loop_end(group_id, VPU_FA_GP_TILE_LOOP);
    }
  }
  if (tail_columns != 0u) {
    vpu_fa_group_set_vl(group_id, tail_columns);
    vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, VPU_FA_FP_SCALE);
    vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX, VPU_FA_FP_M_NEW,
                         VPU_FA_GP_VECTOR, 0, 0, 0, 0);
  }
}

static inline void vpu_fa_group_score_exp_stream(unsigned group_id,
                                                 unsigned full_tiles,
                                                 unsigned tail_columns,
                                                 bool chunks_bank_local) {
  const bool expand_tiles =
      VPU_LOOP_BUFFER_ENTRIES >= 56u && tail_columns == 0u && full_tiles <= 4u;
  const unsigned total_columns = full_tiles * DIM + tail_columns;
  if (vpu_fa_cross_tile_coalescing_fits(total_columns)) {
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_SCORE_ROW, 0u);
    vpu_fa_group_set_vl(group_id, total_columns);
    vpu_fa_group_set_vector_stride(group_id, VPU_FA_MATRIX_TILE_ELEMENTS);
    vpu_fa_group_vector(group_id, VPU_OP_V_SUB_VF, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, VPU_FA_FP_M_NEW);
    vpu_fa_group_vector(group_id, VPU_OP_V_EXP_V, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, 0u);
    vpu_fa_group_command(group_id, false, VPU_OP_V_RED_SUM, VPU_FA_FP_P_SUM,
                         VPU_FA_GP_VECTOR, 0, 0, 0, 0);
    vpu_fa_group_set_vector_stride(group_id, 0u);
    return;
  }
  if (vpu_fa_score_two_chunk_chain(full_tiles, tail_columns,
                                   chunks_bank_local)) {
    const unsigned tail_chunk = total_columns % VPU_VLEN;
    const unsigned chunk_stride =
        (VPU_VLEN / DIM) * VPU_FA_MATRIX_TILE_ELEMENTS;

    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_SCORE_ROW,
                         chunk_stride);
    /* Base-FMA phase: normalize both chunks before entering EXP. */
    vpu_fa_group_set_vl(group_id, VPU_VLEN);
    vpu_fa_group_vector(group_id, VPU_OP_V_SUB_VF, VPU_FA_GP_SCORE_ROW,
                        VPU_FA_GP_SCORE_ROW, VPU_FA_FP_M_NEW);
    if (tail_chunk != 0u)
      vpu_fa_group_set_vl(group_id, tail_chunk);
    vpu_fa_group_vector(group_id, VPU_OP_V_SUB_VF, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, VPU_FA_FP_M_NEW);

    /* EXP phase. */
    if (tail_chunk != 0u)
      vpu_fa_group_set_vl(group_id, VPU_VLEN);
    vpu_fa_group_vector(group_id, VPU_OP_V_EXP_V, VPU_FA_GP_SCORE_ROW,
                        VPU_FA_GP_SCORE_ROW, 0u);
    if (tail_chunk != 0u)
      vpu_fa_group_set_vl(group_id, tail_chunk);
    vpu_fa_group_vector(group_id, VPU_OP_V_EXP_V, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, 0u);

    /* Reduction phase: fold the two exponent chunks only once. */
    if (tail_chunk != 0u)
      vpu_fa_group_set_vl(group_id, VPU_VLEN);
    vpu_fa_group_command(group_id, false, VPU_OP_V_RED_SUM,
                         VPU_FA_FP_P_SUM, VPU_FA_GP_SCORE_ROW, 0, 0,
                         VPU_REDUCTION_START, 0);
    if (tail_chunk != 0u)
      vpu_fa_group_set_vl(group_id, tail_chunk);
    vpu_fa_group_command(group_id, false, VPU_OP_V_RED_SUM,
                         VPU_FA_FP_P_SUM, VPU_FA_GP_VECTOR, 0, 0,
                         VPU_REDUCTION_FINAL, 0);
    return;
  }
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_SCORE_ROW, 0u);
  if (chunks_bank_local && VPU_VLEN >= DIM && VPU_VLEN % DIM == 0u &&
      total_columns > VPU_VLEN) {
    const unsigned full_chunks = total_columns / VPU_VLEN;
    const unsigned tail_chunk = total_columns % VPU_VLEN;
    const unsigned chunk_stride =
        (VPU_VLEN / DIM) * VPU_FA_MATRIX_TILE_ELEMENTS;

    vpu_fa_group_set_vl(group_id, VPU_VLEN);
    vpu_fa_group_set_vector_stride(group_id, VPU_FA_MATRIX_TILE_ELEMENTS);
    if (full_chunks == 1u) {
      vpu_fa_group_vector(group_id, VPU_OP_V_SUB_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_M_NEW);
      vpu_fa_group_vector(group_id, VPU_OP_V_EXP_V, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, 0u);
      vpu_fa_group_command(group_id, false, VPU_OP_V_RED_SUM,
                           VPU_FA_FP_P_SUM, VPU_FA_GP_VECTOR, 0, 0, 0, 0);
      if (tail_chunk != 0u)
        vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                             chunk_stride);
    } else {
      vpu_fa_group_loop_start(group_id, VPU_FA_GP_TILE_LOOP, full_chunks);
      vpu_fa_group_vector(group_id, VPU_OP_V_SUB_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_M_NEW);
      vpu_fa_group_vector(group_id, VPU_OP_V_EXP_V, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, 0u);
      vpu_fa_group_command(group_id, false, VPU_OP_V_RED_SUM,
                           VPU_FA_FP_P_SUM, VPU_FA_GP_VECTOR, 0, 0, 0, 0);
      vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                           chunk_stride);
      vpu_fa_group_loop_end(group_id, VPU_FA_GP_TILE_LOOP);
    }
    if (tail_chunk != 0u) {
      vpu_fa_group_set_vl(group_id, tail_chunk);
      vpu_fa_group_vector(group_id, VPU_OP_V_SUB_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_M_NEW);
      vpu_fa_group_vector(group_id, VPU_OP_V_EXP_V, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, 0u);
      vpu_fa_group_command(group_id, false, VPU_OP_V_RED_SUM,
                           VPU_FA_FP_P_SUM, VPU_FA_GP_VECTOR, 0, 0, 0, 0);
    }
    vpu_fa_group_set_vector_stride(group_id, 0u);
    return;
  }
  if (full_tiles != 0u) {
    vpu_fa_group_set_vl(group_id, DIM);
    if (expand_tiles) {
      for (unsigned tile = 0u; tile < full_tiles; ++tile) {
        vpu_fa_group_vector(group_id, VPU_OP_V_SUB_VF, VPU_FA_GP_VECTOR,
                            VPU_FA_GP_VECTOR, VPU_FA_FP_M_NEW);
        vpu_fa_group_vector(group_id, VPU_OP_V_EXP_V, VPU_FA_GP_VECTOR,
                            VPU_FA_GP_VECTOR, 0u);
        vpu_fa_group_command(group_id, false, VPU_OP_V_RED_SUM, VPU_FA_FP_P_SUM,
                             VPU_FA_GP_VECTOR, 0, 0, 0, 0);
        if (tile + 1u < full_tiles)
          vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                               VPU_FA_MATRIX_TILE_ELEMENTS);
      }
    } else if (full_tiles == 1u) {
      vpu_fa_group_vector(group_id, VPU_OP_V_SUB_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_M_NEW);
      vpu_fa_group_vector(group_id, VPU_OP_V_EXP_V, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, 0u);
      vpu_fa_group_command(group_id, false, VPU_OP_V_RED_SUM, VPU_FA_FP_P_SUM,
                           VPU_FA_GP_VECTOR, 0, 0, 0, 0);
      if (tail_columns != 0u)
        vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                             VPU_FA_MATRIX_TILE_ELEMENTS);
    } else {
      vpu_fa_group_loop_start(group_id, VPU_FA_GP_TILE_LOOP, full_tiles);
      vpu_fa_group_vector(group_id, VPU_OP_V_SUB_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_M_NEW);
      vpu_fa_group_vector(group_id, VPU_OP_V_EXP_V, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, 0u);
      vpu_fa_group_command(group_id, false, VPU_OP_V_RED_SUM, VPU_FA_FP_P_SUM,
                           VPU_FA_GP_VECTOR, 0, 0, 0, 0);
      vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                           VPU_FA_MATRIX_TILE_ELEMENTS);
      vpu_fa_group_loop_end(group_id, VPU_FA_GP_TILE_LOOP);
    }
  }
  if (tail_columns != 0u) {
    vpu_fa_group_set_vl(group_id, tail_columns);
    vpu_fa_group_vector(group_id, VPU_OP_V_SUB_VF, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, VPU_FA_FP_M_NEW);
    vpu_fa_group_vector(group_id, VPU_OP_V_EXP_V, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, 0u);
    vpu_fa_group_command(group_id, false, VPU_OP_V_RED_SUM, VPU_FA_FP_P_SUM,
                         VPU_FA_GP_VECTOR, 0, 0, 0, 0);
  }
}

static inline void vpu_fa_group_output_scale_stream(unsigned group_id,
                                                    unsigned scalar_fp,
                                                    unsigned value_dim,
                                                    bool bank_local) {
  const unsigned full_tiles = value_dim / DIM;
  const unsigned tail_columns = value_dim % DIM;
  const bool expand_tiles =
      VPU_LOOP_BUFFER_ENTRIES >= 56u && tail_columns == 0u && full_tiles <= 4u;
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_OUTPUT, VPU_FA_GP_OUTPUT_ROW, 0u);
  if (bank_local && vpu_fa_cross_tile_coalescing_fits(value_dim)) {
    vpu_fa_group_set_vl(group_id, value_dim);
    vpu_fa_group_set_vector_stride(group_id, VPU_FA_MATRIX_TILE_ELEMENTS);
    vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_OUTPUT,
                        VPU_FA_GP_OUTPUT, scalar_fp);
    vpu_fa_group_set_vector_stride(group_id, 0u);
    return;
  }
  if (full_tiles != 0u) {
    vpu_fa_group_set_vl(group_id, DIM);
    if (expand_tiles) {
      for (unsigned tile = 0u; tile < full_tiles; ++tile) {
        vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_OUTPUT,
                            VPU_FA_GP_OUTPUT, scalar_fp);
        if (tile + 1u < full_tiles)
          vpu_fa_group_addi_gp(group_id, VPU_FA_GP_OUTPUT, VPU_FA_GP_OUTPUT,
                               VPU_FA_MATRIX_TILE_ELEMENTS);
      }
    } else if (full_tiles == 1u) {
      vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_OUTPUT,
                          VPU_FA_GP_OUTPUT, scalar_fp);
      if (tail_columns != 0u)
        vpu_fa_group_addi_gp(group_id, VPU_FA_GP_OUTPUT, VPU_FA_GP_OUTPUT,
                             VPU_FA_MATRIX_TILE_ELEMENTS);
    } else {
      vpu_fa_group_loop_start(group_id, VPU_FA_GP_TILE_LOOP, full_tiles);
      vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_OUTPUT,
                          VPU_FA_GP_OUTPUT, scalar_fp);
      vpu_fa_group_addi_gp(group_id, VPU_FA_GP_OUTPUT, VPU_FA_GP_OUTPUT,
                           VPU_FA_MATRIX_TILE_ELEMENTS);
      vpu_fa_group_loop_end(group_id, VPU_FA_GP_TILE_LOOP);
    }
  }
  if (tail_columns != 0u) {
    vpu_fa_group_set_vl(group_id, tail_columns);
    vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_OUTPUT,
                        VPU_FA_GP_OUTPUT, scalar_fp);
  }
}

static inline void
vpu_fa_issue_causal_mask_row_run(vpu_flashattention_stats_t *stats,
                                 unsigned group_id, unsigned memory_slot,
                                 unsigned kv_start, unsigned kv_rows,
                                 unsigned first_row, unsigned row_count,
                                 unsigned global_query,
                                 bool zero_full_future) {
  const unsigned score_tiles = vpu_fa_ceil_div(kv_rows, DIM);
  const unsigned full_score_tiles = kv_rows / DIM;
  const unsigned tail_score_columns = kv_rows % DIM;
  const bool all_future = global_query < kv_start;
  const bool all_visible =
      !all_future && global_query >= kv_start + kv_rows - 1u;
  if (all_visible)
    return;

  const bool has_diagonal = !all_future;
  const unsigned diagonal_tile =
      has_diagonal ? (global_query - kv_start) / DIM : 0u;
  const unsigned diagonal_key_start = kv_start + diagonal_tile * DIM;
  const unsigned first_future_tile = has_diagonal ? diagonal_tile + 1u : 0u;
  const unsigned first_future_full = first_future_tile < full_score_tiles
                                         ? first_future_tile
                                         : full_score_tiles;
  const unsigned future_full_tiles = full_score_tiles - first_future_full;
  const bool future_tail =
      tail_score_columns != 0u && first_future_tile <= full_score_tiles;
  const unsigned future_columns =
      future_full_tiles * DIM + (future_tail ? tail_score_columns : 0u);
  const bool coalesce_future =
      vpu_fa_cross_tile_coalescing_fits(future_columns);

  ++stats->mask_row_loop_regions;
  stats->mask_row_loop_rows += row_count;
  if (has_diagonal) {
    stats->mask_diagonal_vectors += row_count;
    vpu_fa_group_write_gp(
        group_id, VPU_FA_GP_SCORE_ROW,
        vpu_fa_matrix_address(vpu_fa_score_base_from_slot(memory_slot),
                              score_tiles,
                              first_row,
                              diagonal_tile));
    vpu_fa_group_write_gp(group_id, VPU_FA_GP_MASK_ROW,
                          VPU_FA_CAUSAL_MASK_BASE +
                              (global_query - diagonal_key_start) * DIM);
  }
  if (future_full_tiles != 0u || future_tail) {
    stats->mask_future_vectors +=
        (uint64_t)row_count *
        (coalesce_future ? 1u : future_full_tiles + (future_tail ? 1u : 0u));
    vpu_fa_group_write_gp(group_id, VPU_FA_GP_FUTURE_ROW,
                          vpu_fa_matrix_address(
                              vpu_fa_score_base_from_slot(memory_slot),
                              score_tiles, first_row,
                              first_future_tile));
  }

  vpu_fa_group_loop_start(group_id, VPU_FA_GP_ROW_LOOP, row_count);
  if (has_diagonal) {
    const unsigned diagonal_columns =
        diagonal_tile < full_score_tiles ? DIM : tail_score_columns;
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_SCORE_ROW, 0u);
    vpu_fa_group_set_vl(group_id, diagonal_columns);
    vpu_fa_group_vector(group_id, VPU_OP_V_ADD_VV, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, VPU_FA_GP_MASK_ROW);
  }
  if (future_full_tiles != 0u || future_tail) {
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_FUTURE_ROW, 0u);
    if (coalesce_future) {
      vpu_fa_group_set_vl(group_id, future_columns);
      vpu_fa_group_set_vector_stride(group_id, VPU_FA_MATRIX_TILE_ELEMENTS);
      if (zero_full_future)
        vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                            VPU_FA_GP_VECTOR, VPU_FA_FP_TERMINATOR);
      else
        vpu_fa_group_vector(group_id, VPU_OP_V_MIN_VF, VPU_FA_GP_VECTOR,
                            VPU_FA_GP_VECTOR, VPU_FA_FP_NEG_INF);
      vpu_fa_group_set_vector_stride(group_id, 0u);
    } else if (future_full_tiles != 0u) {
      vpu_fa_group_set_vl(group_id, DIM);
      vpu_fa_group_loop_start(group_id, VPU_FA_GP_TILE_LOOP, future_full_tiles);
      if (zero_full_future)
        vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                            VPU_FA_GP_VECTOR, VPU_FA_FP_TERMINATOR);
      else
        vpu_fa_group_vector(group_id, VPU_OP_V_MIN_VF, VPU_FA_GP_VECTOR,
                            VPU_FA_GP_VECTOR, VPU_FA_FP_NEG_INF);
      vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                           VPU_FA_MATRIX_TILE_ELEMENTS);
      vpu_fa_group_loop_end(group_id, VPU_FA_GP_TILE_LOOP);
    }
    if (future_tail && !coalesce_future) {
      vpu_fa_group_set_vl(group_id, tail_score_columns);
      if (zero_full_future)
        vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                            VPU_FA_GP_VECTOR, VPU_FA_FP_TERMINATOR);
      else
        vpu_fa_group_vector(group_id, VPU_OP_V_MIN_VF, VPU_FA_GP_VECTOR,
                            VPU_FA_GP_VECTOR, VPU_FA_FP_NEG_INF);
    }
  }
  if (has_diagonal) {
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_SCORE_ROW, VPU_FA_GP_SCORE_ROW,
                         DIM);
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_MASK_ROW, VPU_FA_GP_MASK_ROW, DIM);
  }
  if (future_full_tiles != 0u || future_tail)
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_FUTURE_ROW, VPU_FA_GP_FUTURE_ROW,
                         DIM);
  vpu_fa_group_loop_end(group_id, VPU_FA_GP_ROW_LOOP);
}

static inline unsigned vpu_fa_causal_row_run(
    unsigned q_rows, unsigned row, unsigned global_query,
    unsigned kv_start) {
  const unsigned rows_to_local_boundary = DIM - row % DIM;
  const unsigned rows_to_mask_boundary =
      global_query < kv_start ? kv_start - global_query
                              : DIM - (global_query - kv_start) % DIM;
  const unsigned rows_left = q_rows - row;
  unsigned row_run = rows_left < rows_to_local_boundary
                         ? rows_left
                         : rows_to_local_boundary;
  if (rows_to_mask_boundary < row_run)
    row_run = rows_to_mask_boundary;
  return row_run;
}

static inline unsigned vpu_fa_causal_active_columns(
    unsigned kv_start, unsigned kv_rows, unsigned global_query,
    unsigned row_count) {
  const unsigned last_query = global_query + row_count - 1u;
  if (last_query < kv_start)
    return 0u;
  const unsigned visible = last_query - kv_start + 1u;
  if (visible >= kv_rows)
    return kv_rows;
  const unsigned rounded = vpu_fa_ceil_div(visible, DIM) * DIM;
  return rounded < kv_rows ? rounded : kv_rows;
}

static inline void vpu_fa_issue_softmax_prefix_row_run(
    const vpu_flashattention_config_t *config,
    vpu_flashattention_stats_t *stats, unsigned group_id,
    unsigned memory_slot, unsigned kv_rows, unsigned active_columns,
    unsigned q_block_rows, unsigned first_row, unsigned row_count,
    bool first_block) {
  const unsigned score_tiles = vpu_fa_ceil_div(kv_rows, DIM);
  const unsigned full_score_tiles = active_columns / DIM;
  const unsigned tail_score_columns = active_columns % DIM;
  const unsigned state_l_base = VPU_FA_STATE_L_BASE;
  const unsigned state_alpha_base = VPU_FA_STATE_ALPHA_BASE;
  const bool score_chunks_bank_local =
      vpu_fa_matrix_row_run_vlen_chunks_bank_local(
          vpu_fa_score_base_from_slot(memory_slot), score_tiles, first_row,
          row_count, active_columns);

  (void)config;
  (void)q_block_rows;

  ++stats->qk_row_loop_regions;
  stats->qk_row_loop_rows += row_count;
  vpu_fa_group_write_gp(
      group_id, VPU_FA_GP_SCORE_ROW,
      vpu_fa_matrix_address(vpu_fa_score_base_from_slot(memory_slot),
                            score_tiles,
                            first_row, 0u));
  vpu_fa_group_write_gp(group_id, VPU_FA_GP_STATE, first_row);
  vpu_fa_group_write_gp(group_id, VPU_FA_GP_STATE_L, state_l_base + first_row);
  if (!first_block)
    vpu_fa_group_write_gp(group_id, VPU_FA_GP_STATE_ALPHA,
                          state_alpha_base + first_row);

  const bool chain_score_chunks = vpu_fa_score_two_chunk_chain(
      full_score_tiles, tail_score_columns, score_chunks_bank_local);
  /* This command is deliberately before LOOP_START so replay does not repeat
   * invariant stride setup.  Leaving the last row's mode in place also avoids
   * an unnecessary reset immediately after LOOP_END. */
  vpu_fa_group_set_vector_stride(
      group_id, chain_score_chunks ? VPU_FA_MATRIX_TILE_ELEMENTS : 0u);
  vpu_fa_group_loop_start(group_id, VPU_FA_GP_ROW_LOOP, row_count);
  if (first_block) {
    vpu_fa_group_write_fp(group_id, VPU_FA_FP_M_OLD, -INFINITY);
    vpu_fa_group_write_fp(group_id, VPU_FA_FP_L_OLD, 0.0f);
  } else {
    vpu_fa_group_state_load_current(group_id, VPU_FA_FP_M_OLD, VPU_FA_GP_STATE);
    vpu_fa_group_state_load_current(group_id, VPU_FA_FP_L_OLD,
                                    VPU_FA_GP_STATE_L);
  }
  vpu_fa_group_write_fp(group_id, VPU_FA_FP_M_NEW, -INFINITY);
  vpu_fa_group_score_max_stream(group_id, full_score_tiles,
                                tail_score_columns,
                                score_chunks_bank_local);
  vpu_fa_group_scalar(group_id, VPU_OP_S_MAX_FP, VPU_FA_FP_M_NEW,
                      VPU_FA_FP_M_OLD, VPU_FA_FP_M_NEW);
  vpu_fa_group_scalar(group_id, VPU_OP_S_SUB_FP, VPU_FA_FP_ALPHA,
                      VPU_FA_FP_M_OLD, VPU_FA_FP_M_NEW);
  vpu_fa_group_scalar(group_id, VPU_OP_S_EXP_FP, VPU_FA_FP_ALPHA,
                      VPU_FA_FP_ALPHA, 0u);
  if (!first_block)
    vpu_fa_group_state_store_current(group_id, VPU_FA_GP_STATE_ALPHA,
                                     VPU_FA_FP_ALPHA);
  vpu_fa_group_write_fp(group_id, VPU_FA_FP_P_SUM, 0.0f);
  vpu_fa_group_score_exp_stream(group_id, full_score_tiles,
                                tail_score_columns,
                                score_chunks_bank_local);
  vpu_fa_group_scalar(group_id, VPU_OP_S_MUL_FP, VPU_FA_FP_L_OLD,
                      VPU_FA_FP_L_OLD, VPU_FA_FP_ALPHA);
  vpu_fa_group_scalar(group_id, VPU_OP_S_ADD_FP, VPU_FA_FP_L_OLD,
                      VPU_FA_FP_L_OLD, VPU_FA_FP_P_SUM);
  vpu_fa_group_state_store_current(group_id, VPU_FA_GP_STATE, VPU_FA_FP_M_NEW);
  vpu_fa_group_state_store_current(group_id, VPU_FA_GP_STATE_L,
                                   VPU_FA_FP_L_OLD);
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_SCORE_ROW, VPU_FA_GP_SCORE_ROW, DIM);
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_STATE, VPU_FA_GP_STATE, 1u);
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_STATE_L, VPU_FA_GP_STATE_L, 1u);
  if (!first_block)
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_STATE_ALPHA,
                         VPU_FA_GP_STATE_ALPHA, 1u);
  vpu_fa_group_loop_end(group_id, VPU_FA_GP_ROW_LOOP);
}

static inline void vpu_fa_issue_rescale_row_run(
    const vpu_flashattention_config_t *config,
    vpu_flashattention_stats_t *stats, unsigned group_id,
    unsigned q_block_rows, unsigned first_row, unsigned row_count) {
  const unsigned value_tiles =
      vpu_fa_ceil_div((unsigned)config->value_dim, DIM);
  const uint32_t output_base =
      vpu_fa_output_base(q_block_rows, value_tiles);

  (void)stats;
  vpu_fa_group_write_gp(
      group_id, VPU_FA_GP_OUTPUT_ROW,
      vpu_fa_matrix_address(output_base, value_tiles, first_row, 0u));
  vpu_fa_group_write_gp(group_id, VPU_FA_GP_STATE_ALPHA,
                        VPU_FA_STATE_ALPHA_BASE + first_row);
  vpu_fa_group_loop_start(group_id, VPU_FA_GP_ROW_LOOP, row_count);
  vpu_fa_group_state_load_current(group_id, VPU_FA_FP_ALPHA,
                                  VPU_FA_GP_STATE_ALPHA);
  const bool output_bank_local = vpu_fa_matrix_row_run_bank_local(
      output_base, value_tiles, first_row, row_count,
      (unsigned)config->value_dim);
  vpu_fa_group_output_scale_stream(group_id, VPU_FA_FP_ALPHA,
                                   (unsigned)config->value_dim,
                                   output_bank_local);
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_OUTPUT_ROW,
                       VPU_FA_GP_OUTPUT_ROW, DIM);
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_STATE_ALPHA,
                       VPU_FA_GP_STATE_ALPHA, 1u);
  vpu_fa_group_loop_end(group_id, VPU_FA_GP_ROW_LOOP);
}

static inline void vpu_fa_issue_output_rescale(
    const vpu_flashattention_config_t *config,
    vpu_flashattention_stats_t *stats, unsigned q_rows,
    unsigned group_id) {
  vpu_fa_group_set_vector_stride(group_id, 0u);
  for (unsigned row = 0u; row < q_rows;) {
    const unsigned rows_to_tile_boundary = DIM - row % DIM;
    const unsigned rows_left = q_rows - row;
    const unsigned row_run = rows_left < rows_to_tile_boundary
                                 ? rows_left
                                 : rows_to_tile_boundary;
    vpu_fa_issue_rescale_row_run(config, stats, group_id, q_rows, row, row_run);
    row += row_run;
  }
}

static inline void
vpu_fa_issue_softmax_stage(const vpu_flashattention_config_t *config,
                           vpu_flashattention_stats_t *stats,
                           unsigned q_start, unsigned q_rows,
                           unsigned kv_start, unsigned kv_rows,
                           vpu_fa_stage_events_t events,
                           unsigned memory_slot,
                           vpu_fa_matmul_job_t *lookahead_job,
                           unsigned lookahead_group_pair,
                           unsigned lookahead_memory_slot) {
  const bool first_block = kv_start == 0u;
  /* Make the kernel independent of any prior standalone/aborted VPU stream. */
  vpu_fa_group_set_vector_stride(events, 0u);
  vpu_fa_group_write_fp(events, VPU_FA_FP_SCALE, config->score_scale);
  vpu_fa_group_write_fp(events, VPU_FA_FP_NEG_INF, -INFINITY);
  vpu_fa_group_write_fp(events, VPU_FA_FP_TERMINATOR, 0.0f);

  bool lookahead_issued = false;
  for (unsigned row = 0u; row < q_rows;) {
    const unsigned global_query = (unsigned)config->query_base + q_start + row;
    const unsigned row_run =
        vpu_fa_causal_row_run(q_rows, row, global_query, kv_start);
    vpu_fa_issue_causal_mask_row_run(stats, events, memory_slot, kv_start,
                                     kv_rows, row, row_run, global_query,
                                     true);
    row += row_run;
  }
  for (unsigned row = 0u; row < q_rows;) {
    const unsigned global_query = (unsigned)config->query_base + q_start + row;
    const unsigned row_run =
        vpu_fa_causal_row_run(q_rows, row, global_query, kv_start);
    const unsigned active_columns = vpu_fa_causal_active_columns(
        kv_start, kv_rows, global_query, row_run);
    vpu_fa_issue_softmax_prefix_row_run(config, stats, events, memory_slot,
                                        kv_rows, active_columns, q_rows, row,
                                        row_run, first_block);
    /* LOOP_END above starts replay of the captured online-softmax row body.
     * Switch RoCC endpoints before submitting another VPU command so the next
     * half's disjoint QK is admitted and executes during that replay. */
    if (!lookahead_issued && lookahead_job != NULL) {
      vpu_fa_issue_matmul_job(lookahead_job, lookahead_group_pair,
                              lookahead_memory_slot, stats);
      lookahead_issued = true;
    }
    row += row_run;
  }
  if (lookahead_job != NULL && !lookahead_issued)
    abort();
  vpu_fa_group_terminator(events);
}

/* Register the next QK immediately after the first rescale row LOOP_END. The
 * hardware loop keeps replaying R while the disjoint Gemmini group accepts
 * QK; the stage terminator follows that lookahead submission. */
static inline void vpu_fa_issue_rescale_stage(
    const vpu_flashattention_config_t *config,
    vpu_flashattention_stats_t *stats, unsigned q_rows,
    vpu_fa_stage_events_t events,
    vpu_fa_matmul_job_t *lookahead_qk_job,
    unsigned lookahead_group_pair, unsigned lookahead_memory_slot) {
  vpu_fa_group_set_vector_stride(events, 0u);
  bool lookahead_issued = false;
  for (unsigned row = 0u; row < q_rows;) {
    const unsigned rows_to_tile_boundary = DIM - row % DIM;
    const unsigned rows_left = q_rows - row;
    const unsigned row_run = rows_left < rows_to_tile_boundary
                                 ? rows_left
                                 : rows_to_tile_boundary;
    vpu_fa_issue_rescale_row_run(config, stats, events, q_rows, row,
                                 row_run);
    if (!lookahead_issued && lookahead_qk_job != NULL) {
      vpu_fa_issue_matmul_job(lookahead_qk_job, lookahead_group_pair,
                              lookahead_memory_slot, stats);
      lookahead_issued = true;
    }
    row += row_run;
  }
  if (lookahead_qk_job != NULL && !lookahead_issued)
    abort();
  vpu_fa_group_terminator(events);
}

static inline void
vpu_fa_issue_final_normalize(const vpu_flashattention_config_t *config,
                             vpu_flashattention_stats_t *stats, unsigned q_rows,
                             bool last_causal_block,
                             vpu_fa_stage_events_t events) {
  /* Final normalization uses the same segmented layout only where requested
   * below; establish the legacy contiguous mode for all other commands. */
  vpu_fa_group_set_vector_stride(events, 0u);
  if (!last_causal_block) {
    vpu_fa_group_terminator(events);
    return;
  }
  const unsigned state_l_base = VPU_FA_STATE_L_BASE;
  const unsigned value_tiles =
      vpu_fa_ceil_div((unsigned)config->value_dim, DIM);
  const uint32_t output_base = vpu_fa_output_base(q_rows, value_tiles);
  for (unsigned row = 0u; row < q_rows;) {
    const unsigned rows_to_tile_boundary = DIM - row % DIM;
    const unsigned rows_left = q_rows - row;
    const unsigned row_run =
        rows_left < rows_to_tile_boundary ? rows_left : rows_to_tile_boundary;
    ++stats->normalize_row_loop_regions;
    stats->normalize_row_loop_rows += row_run;
    vpu_fa_group_write_gp(events, VPU_FA_GP_OUTPUT_ROW,
                          vpu_fa_matrix_address(output_base, value_tiles, row,
                                                0u));
    vpu_fa_group_write_gp(events, VPU_FA_GP_STATE_L, state_l_base + row);
    vpu_fa_group_loop_start(events, VPU_FA_GP_ROW_LOOP, row_run);
    vpu_fa_group_state_load_current(events, VPU_FA_FP_L_OLD,
                                    VPU_FA_GP_STATE_L);
    vpu_fa_group_scalar(events, VPU_OP_S_RECI_FP, VPU_FA_FP_L_OLD,
                        VPU_FA_FP_L_OLD, 0u);
    const bool output_bank_local = vpu_fa_matrix_row_run_bank_local(
        output_base, value_tiles, row, row_run,
        (unsigned)config->value_dim);
    vpu_fa_group_output_scale_stream(events, VPU_FA_FP_L_OLD,
                                     (unsigned)config->value_dim,
                                     output_bank_local);
    vpu_fa_group_addi_gp(events, VPU_FA_GP_OUTPUT_ROW, VPU_FA_GP_OUTPUT_ROW,
                         DIM);
    vpu_fa_group_addi_gp(events, VPU_FA_GP_STATE_L, VPU_FA_GP_STATE_L, 1u);
    vpu_fa_group_loop_end(events, VPU_FA_GP_ROW_LOOP);
    row += row_run;
  }
  vpu_fa_group_terminator(events);
}

static inline void
vpu_fa_enqueue_output_store(const vpu_flashattention_config_t *config,
                            unsigned q_start, unsigned q_rows) {
  const unsigned value_tiles =
      vpu_fa_ceil_div((unsigned)config->value_dim, DIM);
  const uint32_t output_base = vpu_fa_output_base(q_rows, value_tiles);
  const unsigned full_tiles = (unsigned)config->value_dim / DIM;
  const size_t output_stride = vpu_fa_stride_payload(config->output_stride);
  for (unsigned row = 0u; row < q_rows; ++row) {
    const uint32_t first_vector =
        vpu_fa_matrix_address(output_base, value_tiles, row, 0u);
    const size_t global_row = (size_t)q_start + row;
    const uint32_t first_host = (uint32_t)(global_row * output_stride);
    if (full_tiles >= 2u) {
      vpu_set_vl(DIM);
      vpu_write_gp(VPU_FA_GP_VECTOR,
                   first_vector - VPU_FA_MATRIX_TILE_ELEMENTS);
      vpu_write_gp(VPU_FA_GP_HOST_OFFSET, first_host - DIM);
      vpu_loop_start(VPU_FA_GP_TILE_LOOP, full_tiles);
      vpu_s_addi_int(VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                     VPU_FA_MATRIX_TILE_ELEMENTS);
      vpu_s_addi_int(VPU_FA_GP_HOST_OFFSET, VPU_FA_GP_HOST_OFFSET, DIM);
      vpu_h_store_v(VPU_FA_GP_VECTOR, VPU_FA_GP_HOST_OFFSET, VPU_FA_H_OUTPUT);
      vpu_loop_end(VPU_FA_GP_TILE_LOOP);
    } else if (full_tiles == 1u) {
      vpu_set_vl(DIM);
      vpu_write_gp(VPU_FA_GP_VECTOR, first_vector);
      vpu_write_gp(VPU_FA_GP_HOST_OFFSET, first_host);
      vpu_h_store_v(VPU_FA_GP_VECTOR, VPU_FA_GP_HOST_OFFSET, VPU_FA_H_OUTPUT);
    }
    if (full_tiles < value_tiles) {
      const unsigned column = full_tiles * DIM;
      vpu_set_vl(config->value_dim - column);
      vpu_write_gp(VPU_FA_GP_VECTOR,
                   vpu_fa_matrix_address(output_base, value_tiles, row,
                                         full_tiles));
      vpu_write_gp(VPU_FA_GP_HOST_OFFSET, first_host + column);
      vpu_h_store_v(VPU_FA_GP_VECTOR, VPU_FA_GP_HOST_OFFSET, VPU_FA_H_OUTPUT);
    }
  }
}

static inline void vpu_fa_initialize_causal_mask(float *workspace) {
  for (unsigned row = 0u; row < DIM; ++row)
    for (unsigned column = 0u; column < DIM; ++column)
      workspace[row * DIM + column] = column <= row ? 0.0f : -INFINITY;
}

static inline void
vpu_fa_prefetch_causal_mask(const vpu_flashattention_config_t *config) {
  vpu_write_h(VPU_FA_H_CAUSAL_MASK, (uintptr_t)config->causal_mask_workspace);
  for (unsigned offset = 0u; offset < VPU_FA_CAUSAL_MASK_ELEMENTS;
       offset += VPU_VLEN) {
    const unsigned remaining = VPU_FA_CAUSAL_MASK_ELEMENTS - offset;
    const unsigned elements = remaining < VPU_VLEN ? remaining : VPU_VLEN;
    vpu_set_vl(elements);
    vpu_write_gp(VPU_FA_GP_MASK_ROW, VPU_FA_CAUSAL_MASK_BASE + offset);
    vpu_write_gp(VPU_FA_GP_MASK_HOST_OFFSET, offset);
    vpu_h_prefetch_v(VPU_FA_GP_MASK_ROW, VPU_FA_GP_MASK_HOST_OFFSET,
                     VPU_FA_H_CAUSAL_MASK);
  }
}

static inline void vpu_fa_flush_gemminis(unsigned mask) {
  for (unsigned custom = 0u; custom < 4u; ++custom)
    if (((mask >> custom) & 1u) != 0u)
      vpu_fa_gemmini_flush(custom);
}

/* Preflight all runtime tail-block shapes before issuing any hardware
 * command.  A failure therefore never leaves a half-open fusion group. */
static inline vpu_flashattention_status_t
vpu_fa_preflight_jobs(const vpu_flashattention_config_t *config,
                      const vpu_flashattention_plan_t *plan) {
  for (size_t q_start = 0u; q_start < config->query_rows;) {
    const unsigned q_rows = config->query_rows - q_start < plan->q_rows
                                ? (unsigned)(config->query_rows - q_start)
                                : plan->q_rows;
    const size_t query_block_end = config->query_base + q_start + q_rows;
    const size_t causal_key_end =
        query_block_end < config->sequence ? query_block_end : config->sequence;
    for (size_t kv_start = 0u; kv_start < causal_key_end;) {
      const unsigned kv_rows = causal_key_end - kv_start < plan->kv_rows
                                   ? (unsigned)(causal_key_end - kv_start)
                                   : plan->kv_rows;
      vpu_fa_matmul_job_t job;
      if (!vpu_fa_matmul_job_prepare(&job, config, VPU_FA_MATMUL_QK,
                                     (unsigned)q_start, q_rows,
                                     (unsigned)kv_start, kv_rows))
        return VPU_FLASHATTENTION_QK_JOB_UNSUPPORTED;
      if (!vpu_fa_matmul_job_prepare(&job, config, VPU_FA_MATMUL_PV,
                                     (unsigned)q_start, q_rows,
                                     (unsigned)kv_start, kv_rows))
        return VPU_FLASHATTENTION_PV_JOB_UNSUPPORTED;
      kv_start += kv_rows;
    }
    q_start += q_rows;
  }
  return VPU_FLASHATTENTION_OK;
}

static inline vpu_flashattention_result_t
vpu_flashattention_auto(const vpu_flashattention_config_t *source) {
  vpu_flashattention_result_t result;
  memset(&result, 0, sizeof(result));
  result.status = VPU_FLASHATTENTION_INVALID_ARGUMENT;
  if (source == NULL)
    return result;

  vpu_flashattention_config_t config = vpu_fa_normalize_config(source);
  result.status = vpu_flashattention_make_plan(&config, &result.plan);
  if (result.status != VPU_FLASHATTENTION_OK)
    return result;
  /* The global planner has already selected and validated QK's K factor.
   * Reuse it for every full/tail block; smaller I/J tails cannot increase the
   * SPAD or ACC footprint. PV always consumes its whole ACC-resident K block,
   * so neither path needs to invoke the generic tiler again. */
  config.qk_tile_k = result.plan.qk_tiles;
  result.status = vpu_fa_preflight_jobs(&config, &result.plan);
  if (result.status != VPU_FLASHATTENTION_OK)
    return result;

  for (size_t q_start = 0u; q_start < config.query_rows;) {
    const unsigned q_rows = config.query_rows - q_start < result.plan.q_rows
                                ? (unsigned)(config.query_rows - q_start)
                                : result.plan.q_rows;
    const size_t query_block_end = config.query_base + q_start + q_rows;
    const size_t causal_key_end =
        query_block_end < config.sequence ? query_block_end : config.sequence;
    result.stats.rectangular_blocks +=
        vpu_fa_ceil_div((unsigned)config.sequence, result.plan.kv_rows);
    result.stats.causal_blocks +=
        vpu_fa_ceil_div((unsigned)causal_key_end, result.plan.kv_rows);
    q_start += q_rows;
  }
  result.stats.skipped_future_blocks =
      result.stats.rectangular_blocks - result.stats.causal_blocks;

  vpu_fa_initialize_causal_mask(config.causal_mask_workspace);
  vpu_publish_cpu_writes();
  vpu_fa_flush_gemminis(result.plan.gemmini_mask);
  vpu_clear_status(VPU_CLEAR_ALL);
  vpu_write_h(VPU_FA_H_OUTPUT, (uintptr_t)config.output);
  vpu_fa_prefetch_causal_mask(&config);

  for (size_t q_start = 0u; q_start < config.query_rows;) {
    const unsigned q_rows = config.query_rows - q_start < result.plan.q_rows
                                ? (unsigned)(config.query_rows - q_start)
                                : result.plan.q_rows;
    const size_t query_block_end = config.query_base + q_start + q_rows;
    const size_t causal_key_end =
        query_block_end < config.sequence ? query_block_end : config.sequence;

    /* Virgo pipeline. Synchronization groups only protect the two SPAD
     * ping/pong halves; the event tracker carries cross-stage ordering:
     *
     *   QK0 -> SM0 { QK1 }
     *   SM1 { PV0 } -> R1 { QK2 }
     *   SM2 { PV1 } -> R2 { QK3 }
     *
     * QKb produces E(3b), SMb waits E(3b) and produces E(3b+1), PVb waits
     * E(3b+1) and produces E(3b+2), and R/final waits E(3b+2). */
    size_t current_kv_start = 0u;
    unsigned current_kv_rows =
        causal_key_end < result.plan.kv_rows
            ? (unsigned)causal_key_end
            : result.plan.kv_rows;
    size_t previous_kv_start = 0u;
    unsigned previous_kv_rows = 0u;
    unsigned block_index = 0u;

    vpu_fa_matmul_job_t current_qk_job;
    (void)vpu_fa_matmul_job_prepare(
        &current_qk_job, &config, VPU_FA_MATMUL_QK,
        (unsigned)q_start, q_rows,
        (unsigned)current_kv_start, current_kv_rows);
    vpu_fa_matmul_job_set_events(
        &current_qk_job, 0u, false,
        vpu_fa_qk_event(block_index), true);
    ++result.stats.qk_jobs;
    vpu_fa_issue_matmul_job(
        &current_qk_job, vpu_fa_group_pair_base(block_index),
        vpu_fa_block_memory_slot(block_index), &result.stats);

    for (;;) {
      const bool has_previous = block_index != 0u;
      const bool has_next =
          current_kv_start + current_kv_rows < causal_key_end;
      const size_t next_kv_start = current_kv_start + current_kv_rows;
      const unsigned next_kv_rows = has_next
          ? (causal_key_end - next_kv_start < result.plan.kv_rows
                 ? (unsigned)(causal_key_end - next_kv_start)
                 : result.plan.kv_rows)
          : 0u;
      vpu_fa_matmul_job_t previous_pv_job;
      vpu_fa_matmul_job_t next_qk_job;

      if (has_previous) {
        (void)vpu_fa_matmul_job_prepare(
            &previous_pv_job, &config, VPU_FA_MATMUL_PV,
            (unsigned)q_start, q_rows,
            (unsigned)previous_kv_start, previous_kv_rows);
        vpu_fa_matmul_job_set_events(
            &previous_pv_job,
            vpu_fa_softmax_event(block_index - 1u), true,
            vpu_fa_pv_event(block_index - 1u), true);
        ++result.stats.pv_jobs;
      }
      if (has_next) {
        (void)vpu_fa_matmul_job_prepare(
            &next_qk_job, &config, VPU_FA_MATMUL_QK,
            (unsigned)q_start, q_rows,
            (unsigned)next_kv_start, next_kv_rows);
        vpu_fa_matmul_job_set_events(
            &next_qk_job, 0u, false,
            vpu_fa_qk_event(block_index + 1u), true);
        ++result.stats.qk_jobs;
      }

      const unsigned current_pair = vpu_fa_group_pair_base(block_index);
      const unsigned current_slot =
          vpu_fa_block_memory_slot(block_index);
      const vpu_fa_stage_events_t current_sm_events = vpu_fa_stage_events(
          vpu_fa_qk_event(block_index), true,
          vpu_fa_softmax_event(block_index), true);
      vpu_fa_matmul_job_t *softmax_lookahead =
          has_previous ? &previous_pv_job : (has_next ? &next_qk_job : NULL);
      const unsigned softmax_lookahead_block =
          has_previous ? block_index - 1u : block_index + 1u;
      vpu_fa_issue_softmax_stage(
          &config, &result.stats, (unsigned)q_start, q_rows,
          (unsigned)current_kv_start, current_kv_rows, current_sm_events,
          current_slot, softmax_lookahead,
          vpu_fa_group_pair_base(softmax_lookahead_block),
          vpu_fa_block_memory_slot(softmax_lookahead_block));

      if (has_previous) {
        const vpu_fa_stage_events_t rescale_events = vpu_fa_stage_events(
            vpu_fa_pv_event(block_index - 1u), true, 0u, false);
        vpu_fa_issue_rescale_stage(
            &config, &result.stats, q_rows, rescale_events,
            has_next ? &next_qk_job : NULL,
            vpu_fa_group_pair_base(block_index + 1u),
            vpu_fa_block_memory_slot(block_index + 1u));
      }

      if (!has_next) {
        vpu_fa_matmul_job_t final_pv_job;
        (void)vpu_fa_matmul_job_prepare(
            &final_pv_job, &config, VPU_FA_MATMUL_PV,
            (unsigned)q_start, q_rows,
            (unsigned)current_kv_start, current_kv_rows);
        vpu_fa_matmul_job_set_events(
            &final_pv_job, vpu_fa_softmax_event(block_index), true,
            vpu_fa_pv_event(block_index), true);
        ++result.stats.pv_jobs;
        vpu_fa_issue_matmul_job(&final_pv_job, current_pair, current_slot,
                                &result.stats);
        const vpu_fa_stage_events_t final_events = vpu_fa_stage_events(
            vpu_fa_pv_event(block_index), true, 0u, false);
        vpu_fa_issue_final_normalize(&config, &result.stats, q_rows, true,
                                     final_events);
        break;
      }

      previous_kv_start = current_kv_start;
      previous_kv_rows = current_kv_rows;
      current_kv_start = next_kv_start;
      current_kv_rows = next_kv_rows;
      ++block_index;
    }

    vpu_fa_enqueue_output_store(&config, (unsigned)q_start, q_rows);
    q_start += q_rows;
  }

  result.vpu_status = vpu_fence();
  result.status = (result.vpu_status & VPU_STATUS_ERROR_MASK) != 0u
                      ? VPU_FLASHATTENTION_VPU_ERROR
                      : VPU_FLASHATTENTION_OK;
  return result;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GEMMINI_ROCC_TESTS_INCLUDE_VPU_FLASHATTENTION_KERNEL_H_
