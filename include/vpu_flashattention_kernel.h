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
 *   grouped Q*K^T -> C_TO_VSRAM
 *   grouped VPU causal mask + online softmax + O rescale
 *   grouped P*V with A_FROM_VSRAM and optional D_FROM_VSRAM
 *   grouped VPU final normalization -> standalone VSRAM store
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
#include <string.h>

#include "gemmini_params.h"
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
    "vpu_flashattention_kernel requires grouped commands, SharedDeps, and matrix ports"
#endif
#if !defined(VPU_MATRIX_ROW_ELEMENTS)
#error "vpu_flashattention_kernel requires generated VPU matrix-row geometry"
#endif
#if VPU_MATRIX_ROW_ELEMENTS != DIM || VPU_NLANES > DIM ||                      \
    (DIM % VPU_NLANES) != 0 || VPU_VLEN < DIM
#error                                                                         \
    "vpu_flashattention_kernel requires matrix rows matching DIM, DIM divisible by VPU_NLANES, and VLEN>=DIM"
#endif
/* Matrix bridge rows and VPU commands share the architectural element-address
 * space.  A DIM-wide row may therefore span several physical lane words; the
 * bridge gathers/splits those words across sub-banks without changing the
 * DIM-by-DIM software tile layout below. */
#define VPU_FA_MATRIX_WORDS_PER_ROW (VPU_MATRIX_ROW_ELEMENTS / VPU_NLANES)
#if VPU_FA_MATRIX_WORDS_PER_ROW > VPU_VSPAD_SUBBANKS
#error                                                                         \
    "vpu_flashattention_kernel requires one VSRAM sub-bank per matrix-row lane word"
#endif
#if VPU_VSPAD_BANKS < 2
#error "vpu_flashattention_kernel requires compact score/output banks"
#endif
#if VPU_FP_STATE_ENTRIES < 2 * DIM
#error "vpu_flashattention_kernel requires at least 2*DIM FP-state entries"
#endif
#if VPU_ELEMENTS_PER_BANK < DIM * DIM
#error                                                                         \
    "vpu_flashattention_kernel requires at least one matrix tile per VSRAM bank"
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
   * XCUSTOM_ACC opcode; selected members must have matrix ports. */
  unsigned gemmini_mask;

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
#define VPU_FA_K_LOOP_WS_CONFIG_MV_BOUNDS_1 24
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
#define VPU_FA_SCORE_BANK_BASE 0u
#define VPU_FA_OUTPUT_BANK_BASE 1u
#define VPU_FA_VSRAM_MATRIX_ROWS_PER_BANK (VPU_ELEMENTS_PER_BANK / DIM)
#define VPU_FA_CAUSAL_MASK_ELEMENTS VPU_FLASHATTENTION_CAUSAL_MASK_ELEMENTS
#define VPU_FA_CAUSAL_MASK_BANK VPU_FA_OUTPUT_BANK_BASE
#define VPU_FA_CAUSAL_MASK_BASE                                                \
  (VPU_BANK_BASE(VPU_FA_CAUSAL_MASK_BANK) + VPU_ELEMENTS_PER_BANK -            \
   VPU_FA_CAUSAL_MASK_ELEMENTS)
#define VPU_FA_SPAD_END (VPU_FA_TOTAL_SPAD_ROWS / 2u)

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
  VPU_FA_GP_ROW_LOOP = 14,
  VPU_FA_GP_TILE_LOOP = 15,
  VPU_FA_H_OUTPUT = 0,
  VPU_FA_H_CAUSAL_MASK = 1,
};

#define VPU_FA_LOOP_A_FROM_VSRAM (UINT64_C(1) << 3)
#define VPU_FA_LOOP_C_TO_VSRAM (UINT64_C(1) << 4)
#define VPU_FA_LOOP_HAS_GEMV_FOLLOWUP (UINT64_C(1) << 5)
#define VPU_FA_LOOP_D_FROM_VSRAM (UINT64_C(1) << 6)

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

/* Byte-for-byte compatible with the fusion extension of LOOP_WS used by the
 * four-Gemmini test path.  Configuration is sent to every member, while only
 * the last selected endpoint launches funct=8 for atomic router broadcast. */
#define VPU_FA_SHARED_LOOP_WS(                                                 \
    custom_, group_list_, group_id_, sp_addr_start_, sp_addr_end_,             \
    acc_addr_start_, ex_I_, mv_K_, mv_pad_K_, laddrI_offset_, laddrK_offset_,  \
    I_, J_, K_, pad_I_, pad_J_, pad_K_, A_, B_, D_, C_, A_stride_, B_stride_,  \
    D_stride_, C_stride_, A_transpose_, B_transpose_, full_C_, low_D_,         \
    ex_accumulate_, act_, A_row_offset_, A_col_offset_, B_row_offset_,         \
    B_col_offset_, D_row_offset_, D_col_offset_, C_row_offset_, C_col_offset_, \
    a_from_vsram_, d_from_vsram_, c_to_vsram_, has_vpu_followup_)              \
  do {                                                                         \
    VPU_FA_GEMMINI_ISSUE(custom_, acc_addr_start_,                             \
                         ((uint64_t)(sp_addr_end_) << 16) |                    \
                             (uint64_t)(sp_addr_start_),                       \
                         VPU_FA_K_LOOP_WS_CONFIG_SPADDR);                      \
    VPU_FA_GEMMINI_ISSUE(                                                      \
        custom_,                                                               \
        ((uint64_t)(laddrK_offset_) << 16) | (uint64_t)(laddrI_offset_),       \
        ((uint64_t)(mv_K_) << 32) | ((uint64_t)(mv_pad_K_) << 16) |            \
            (uint64_t)(ex_I_),                                                 \
        VPU_FA_K_LOOP_WS_CONFIG_MV_BOUNDS_1);                                  \
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
          ((uint64_t)(group_list_) << 48) | ((uint64_t)(group_id_) << 32) |    \
          ((uint64_t)(act_) << 8) | ((uint64_t)(low_D_) << 2) |                \
          ((uint64_t)(full_C_) << 1) | (uint64_t)(ex_accumulate_) |            \
          ((a_from_vsram_) ? VPU_FA_LOOP_A_FROM_VSRAM : 0) |                   \
          ((d_from_vsram_) ? VPU_FA_LOOP_D_FROM_VSRAM : 0) |                   \
          ((c_to_vsram_) ? VPU_FA_LOOP_C_TO_VSRAM : 0) |                       \
          ((has_vpu_followup_) ? VPU_FA_LOOP_HAS_GEMV_FOLLOWUP : 0);           \
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
  const vpu_flashattention_config_t *config;
  vpu_fa_matmul_kind_t kind;
  unsigned q_start;
  unsigned kv_start;
  unsigned q_rows;
  unsigned kv_rows;
  unsigned dim_i;
  unsigned dim_j;
  unsigned dim_k;
  unsigned tile_i;
  unsigned tile_j;
  unsigned tile_k;
  unsigned total_k_tiles;
  unsigned next_k_tile;
  unsigned gemmini_mask;
  unsigned gemmini_count;
  unsigned last_group_id;
  bool first_attention_block;
  bool initialized;
  bool done;
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

static inline unsigned vpu_fa_take_members(unsigned mask, unsigned count) {
  unsigned selected = 0u;
  for (unsigned i = 0; i < 4u && count != 0u; ++i) {
    if (((mask >> i) & 1u) != 0u) {
      selected |= 1u << i;
      --count;
    }
  }
  return selected;
}

static inline void vpu_fa_group_command(unsigned group_id, bool last,
                                        unsigned opcode, unsigned rd,
                                        unsigned rs1, unsigned rs2,
                                        unsigned rs3, unsigned funct1,
                                        uint64_t payload) {
  vpu_issue_grouped(group_id, last,
                    vpu_micro_op(opcode, rd, rs1, rs2, rs3, funct1), payload);
}

static inline void vpu_fa_group_write_gp(unsigned group_id, unsigned reg,
                                         uint32_t value) {
  vpu_fa_group_command(group_id, false, VPU_OP_C_WRITE_GP, reg, 0, 0, 0, 0,
                       value);
}

static inline void vpu_fa_group_addi_gp(unsigned group_id, unsigned dst,
                                        unsigned src, uint32_t immediate) {
  vpu_group_s_addi_int(group_id, dst, src, immediate);
}

static inline void vpu_fa_group_loop_start(unsigned group_id, unsigned loop_reg,
                                           uint32_t iterations) {
  vpu_group_loop_start(group_id, loop_reg, iterations);
}

static inline void vpu_fa_group_loop_end(unsigned group_id, unsigned loop_reg) {
  vpu_group_loop_end(group_id, loop_reg);
}

static inline void vpu_fa_group_write_fp(unsigned group_id, unsigned reg,
                                         float value) {
  vpu_fa_group_command(group_id, false, VPU_OP_C_WRITE_FP, reg, 0, 0, 0, 0,
                       vpu_float_to_bits(value));
}

static inline void vpu_fa_group_terminator(unsigned group_id) {
  vpu_fa_group_command(group_id, true, VPU_OP_S_MAX_FP, VPU_FA_FP_TERMINATOR,
                       VPU_FA_FP_TERMINATOR, VPU_FA_FP_TERMINATOR, 0, 0, 0);
}

static inline void vpu_fa_group_set_vl(unsigned group_id, size_t vl) {
  vpu_fa_group_command(group_id, false, VPU_OP_C_SET_VL, 0, 0, 0, 0, 0, vl);
}

static inline void vpu_fa_group_set_vector_stride(unsigned group_id,
                                                  size_t stride_elements) {
  vpu_fa_group_command(group_id, false, VPU_OP_C_SET_VSTRIDE, 0, 0, 0, 0, 0,
                       stride_elements);
}

static inline void vpu_fa_group_vector(unsigned group_id, unsigned opcode,
                                       unsigned dst_gp, unsigned src0_gp,
                                       unsigned src1_or_fp) {
  vpu_fa_group_command(group_id, false, opcode, dst_gp, src0_gp, src1_or_fp, 0,
                       0, 0);
}

static inline void vpu_fa_group_scalar(unsigned group_id, unsigned opcode,
                                       unsigned dst_fp, unsigned src0_fp,
                                       unsigned src1_fp) {
  vpu_fa_group_command(group_id, false, opcode, dst_fp, src0_fp, src1_fp, 0, 0,
                       0);
}

static inline void vpu_fa_group_state_load_current(unsigned group_id,
                                                   unsigned dst_fp,
                                                   unsigned state_gp) {
  vpu_fa_group_command(group_id, false, VPU_OP_S_LOAD_STATE, dst_fp, state_gp,
                       0, 0, 0, 0);
}

static inline void vpu_fa_group_state_store_current(unsigned group_id,
                                                    unsigned state_gp,
                                                    unsigned src_fp) {
  vpu_fa_group_command(group_id, false, VPU_OP_S_STORE_STATE, state_gp, src_fp,
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
  if (VPU_FA_CAUSAL_MASK_ELEMENTS > VPU_ELEMENTS_PER_BANK)
    return false;
  const size_t output_elements =
      (size_t)q_tiles * value_tiles * VPU_FA_MATRIX_TILE_ELEMENTS;
  return output_elements <= VPU_ELEMENTS_PER_BANK - VPU_FA_CAUSAL_MASK_ELEMENTS;
}

static inline uint32_t vpu_fa_matrix_address(bool output, unsigned tile_columns,
                                             unsigned row,
                                             unsigned column_tile) {
  /* Match the natural accumulator layout: all Gemminis share one base and
   * LoopMatmul's global I offset selects a contiguous row tile. */
  const unsigned row_tile = row / DIM;
  const unsigned row_in_tile = row % DIM;
  const unsigned bank =
      output ? VPU_FA_OUTPUT_BANK_BASE : VPU_FA_SCORE_BANK_BASE;
  const unsigned tile = row_tile * tile_columns + column_tile;
  return VPU_BANK_BASE(bank) + tile * VPU_FA_MATRIX_TILE_ELEMENTS +
         row_in_tile * DIM;
}

static inline bool vpu_fa_tiling_fits(unsigned q_tiles, unsigned kv_tiles,
                                      unsigned value_tiles) {
  const size_t score_elements =
      (size_t)q_tiles * kv_tiles * VPU_FA_MATRIX_TILE_ELEMENTS;
  const size_t output_elements =
      (size_t)q_tiles * value_tiles * VPU_FA_MATRIX_TILE_ELEMENTS;
  const size_t max_operand_rows = VPU_FA_TOTAL_SPAD_ROWS / 4u;
  const size_t max_acc_rows = ACC_ROWS / 2u;
  const size_t pv_a_rows = (size_t)q_tiles * kv_tiles * DIM;
  const size_t pv_b_rows = (size_t)kv_tiles * value_tiles * DIM;
  const size_t pv_c_rows = (size_t)q_tiles * value_tiles * DIM;
  const size_t q_rows = (size_t)q_tiles * DIM;

  return score_elements <= VPU_ELEMENTS_PER_BANK &&
         output_elements <= VPU_ELEMENTS_PER_BANK &&
         vpu_fa_mask_region_fits(q_tiles, value_tiles) &&
         pv_a_rows <= max_operand_rows && pv_b_rows <= max_operand_rows &&
         pv_c_rows <= max_acc_rows && q_rows <= VPU_FP_STATE_ENTRIES / 2u;
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
  if (VPU_FA_CAUSAL_MASK_BANK >= VPU_VSPAD_BANKS ||
      2u * VPU_FA_VSRAM_MATRIX_ROWS_PER_BANK > ACC_ROWS ||
      VPU_ELEMENTS_PER_BANK % VPU_FA_MATRIX_TILE_ELEMENTS != 0u ||
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
  const size_t vsram_acc_rows = VPU_ELEMENTS_PER_BANK / DIM;
  const size_t planner_acc_range =
      2u * vsram_acc_rows < ACC_ROWS ? 2u * vsram_acc_rows : ACC_ROWS;
  const gemmini_tiling_request_t request = {
      .dim_I = config.query_rows,
      .dim_J = config.sequence,
      .dim_K = config.q_dim,
      .dim = DIM,
      .gemmini_count = gemmini_count,
      .sp_addr_range = VPU_FA_TOTAL_SPAD_ROWS,
      .acc_addr_range = planner_acc_range,
      .double_buffered = true,
      .act = 0,
  };
  const gemmini_tiling_factors_t factors =
      gemmini_shared_multi_choose_tiling(&request);
  unsigned q_tiles = (unsigned)factors.tile_I;
  unsigned kv_tiles = (unsigned)factors.tile_J;
  const unsigned value_tiles = vpu_fa_ceil_div((unsigned)config.value_dim, DIM);

  while (q_tiles != 0u && kv_tiles != 0u &&
         !vpu_fa_tiling_fits(q_tiles, kv_tiles, value_tiles)) {
    const bool output_or_state_too_large =
        (size_t)q_tiles * value_tiles * VPU_FA_MATRIX_TILE_ELEMENTS >
            VPU_ELEMENTS_PER_BANK ||
        !vpu_fa_mask_region_fits(q_tiles, value_tiles) ||
        (size_t)q_tiles * DIM > VPU_FP_STATE_ENTRIES / 2u;
    if (output_or_state_too_large && q_tiles > 1u) {
      --q_tiles;
    } else if (kv_tiles > 1u) {
      --kv_tiles;
    } else if (q_tiles > 1u) {
      --q_tiles;
    } else {
      return VPU_FLASHATTENTION_NO_TILING;
    }
  }
  if (q_tiles == 0u || kv_tiles == 0u || factors.tile_K == 0u)
    return VPU_FLASHATTENTION_NO_TILING;

  *plan = (vpu_flashattention_plan_t){
      .q_tiles = q_tiles,
      .kv_tiles = kv_tiles,
      .qk_tiles = (unsigned)factors.tile_K,
      .q_rows = q_tiles * DIM,
      .kv_rows = kv_tiles * DIM,
      .qk_depth = (unsigned)factors.tile_K * DIM,
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

static inline bool vpu_fa_matmul_job_prepare(
    vpu_fa_matmul_job_t *job, const vpu_flashattention_config_t *config,
    vpu_fa_matmul_kind_t kind, unsigned q_start, unsigned q_rows,
    unsigned kv_start, unsigned kv_rows) {
  const unsigned dim_j =
      kind == VPU_FA_MATMUL_QK ? kv_rows : (unsigned)config->value_dim;
  const unsigned dim_k =
      kind == VPU_FA_MATMUL_QK ? (unsigned)config->q_dim : kv_rows;
  const size_t vsram_acc_rows = VPU_ELEMENTS_PER_BANK / DIM;
  const size_t planner_acc_range =
      2u * vsram_acc_rows < ACC_ROWS ? 2u * vsram_acc_rows : ACC_ROWS;
  const gemmini_tiling_request_t request = {
      .dim_I = q_rows,
      .dim_J = dim_j,
      .dim_K = dim_k,
      .dim = DIM,
      .gemmini_count = vpu_fa_popcount(config->gemmini_mask & 0xfu),
      .sp_addr_range = VPU_FA_TOTAL_SPAD_ROWS,
      .acc_addr_range = planner_acc_range,
      .double_buffered = true,
      .act = 0,
  };
  const gemmini_tiling_factors_t factors =
      gemmini_shared_multi_choose_tiling(&request);
  const unsigned i_tiles = vpu_fa_ceil_div(q_rows, DIM);
  const unsigned j_tiles = vpu_fa_ceil_div(dim_j, DIM);
  const unsigned total_k_tiles = vpu_fa_ceil_div(dim_k, DIM);
  const size_t max_operand_spad_rows = VPU_FA_TOTAL_SPAD_ROWS / 4u;
  const size_t max_acc_rows = planner_acc_range / 2u;

  /* The shared planner aligns a recurring main I tile to the Gemmini count.
   * This job may instead be the final query tail.  Like the generic matmul
   * outer loop, keep that exact tail and let job_step distribute it by
   * quotient/remainder (for example, six tiles become 2/2/1/1). */
  if (factors.tile_J != j_tiles || factors.tile_K == 0u ||
      factors.tile_K > total_k_tiles ||
      !gemmini_tiling_split_spad_rows_fit(i_tiles, j_tiles, factors.tile_K, DIM,
                                          max_operand_spad_rows) ||
      gemmini_tiling_total_acc_rows(i_tiles, j_tiles, DIM) > max_acc_rows)
    return false;
  /* PV's A is already matrix-tile-major in VSRAM.  It cannot be compacted
   * into a temporary SPAD tile between K chunks, so consume the whole K block
   * in one logical job step. */
  if (kind == VPU_FA_MATMUL_PV && factors.tile_K != total_k_tiles)
    return false;

  memset(job, 0, sizeof(*job));
  job->config = config;
  job->kind = kind;
  job->q_start = q_start;
  job->kv_start = kv_start;
  job->q_rows = q_rows;
  job->kv_rows = kv_rows;
  job->dim_i = q_rows;
  job->dim_j = dim_j;
  job->dim_k = dim_k;
  job->tile_i = i_tiles;
  job->tile_j = (unsigned)factors.tile_J;
  job->tile_k = (unsigned)factors.tile_K;
  job->total_k_tiles = total_k_tiles;
  job->gemmini_mask = config->gemmini_mask & 0xfu;
  job->gemmini_count = vpu_fa_popcount(job->gemmini_mask);
  job->first_attention_block = kv_start == 0u;
  return true;
}

static inline void vpu_fa_matmul_job_init(vpu_fa_matmul_job_t *job) {
  const bool qk = job->kind == VPU_FA_MATMUL_QK;
  const size_t a_stride = qk ? job->config->query_stride : job->dim_k;
  const size_t b_stride =
      qk ? job->config->key_stride : job->config->value_stride;
  const size_t d_stride = qk ? 0u : job->config->value_dim;
  const size_t a_dma_stride =
      qk ? gemmini_page_packed_a_dma_stride_bytes(a_stride)
         : a_stride * sizeof(elem_t);
  const size_t b_dma_stride = gemmini_page_packed_b_dma_stride_bytes(b_stride);

  for (unsigned custom = 0u; custom < 4u; ++custom) {
    if (((job->gemmini_mask >> custom) & 1u) == 0u)
      continue;
    vpu_fa_config_ex(custom, 1u, qk);
    vpu_fa_config_st(custom);
    vpu_fa_config_ld(custom, a_dma_stride, 0u);
    vpu_fa_config_ld(custom, b_dma_stride, 1u);
    vpu_fa_config_ld(custom, d_stride * sizeof(float), 2u);
  }
  job->next_k_tile = 0u;
  job->initialized = true;
  job->done = false;
}

static inline bool vpu_fa_matmul_job_step(vpu_fa_matmul_job_t *job,
                                          unsigned *dispatch_sequence) {
  if (!job->initialized || job->done)
    return false;

  const vpu_flashattention_config_t *config = job->config;
  const unsigned remaining_k = job->total_k_tiles - job->next_k_tile;
  const unsigned k_tiles =
      remaining_k < job->tile_k ? remaining_k : job->tile_k;
  const unsigned k_element_start = job->next_k_tile * DIM;
  const unsigned k_elements_left = job->dim_k - k_element_start;
  const unsigned k_elements =
      k_elements_left < k_tiles * DIM ? k_elements_left : k_tiles * DIM;
  const unsigned pad_i = job->tile_i * DIM - job->dim_i;
  const unsigned pad_j = job->tile_j * DIM - job->dim_j;
  const unsigned pad_k = k_tiles * DIM - k_elements;
  const bool final_step = job->next_k_tile + k_tiles == job->total_k_tiles;

  unsigned active_count = job->gemmini_count;
  if (job->tile_i < active_count && k_tiles < active_count)
    active_count = job->tile_i > k_tiles ? job->tile_i : k_tiles;
  const unsigned group_list =
      vpu_fa_take_members(job->gemmini_mask, active_count);
  const unsigned group_id = (*dispatch_sequence) & 1u;
  ++(*dispatch_sequence);

  const unsigned i_base = job->tile_i / active_count;
  const unsigned i_extra = job->tile_i % active_count;
  const unsigned k_base = k_tiles / active_count;
  const unsigned k_extra = k_tiles % active_count;
  const unsigned execute_members =
      active_count < job->tile_i ? active_count : job->tile_i;
  const unsigned load_b_members =
      active_count < k_tiles ? active_count : k_tiles;
  unsigned member_index = 0u;
  unsigned i_offset = 0u;
  unsigned k_offset = 0u;

  for (unsigned custom = 0u; custom < 4u; ++custom) {
    if (((group_list >> custom) & 1u) == 0u)
      continue;
    const unsigned this_i = i_base + (member_index < i_extra ? 1u : 0u);
    const unsigned this_k = k_base + (member_index < k_extra ? 1u : 0u);
    const unsigned this_pad_i =
        member_index + 1u == execute_members ? pad_i : 0u;
    const unsigned this_pad_k =
        member_index + 1u == load_b_members ? pad_k : 0u;
    const bool qk = job->kind == VPU_FA_MATMUL_QK;
    const bool a_from_vsram = !qk;
    const bool d_from_vsram =
        !qk && !job->first_attention_block && job->next_k_tile == 0u;
    const bool ex_accumulate = job->next_k_tile != 0u || d_from_vsram;
    const unsigned score_local_row =
        VPU_BANK_BASE(VPU_FA_SCORE_BANK_BASE) / DIM;
    const unsigned output_local_row =
        VPU_BANK_BASE(VPU_FA_OUTPUT_BANK_BASE) / DIM;
    const unsigned acc_local_row = qk ? score_local_row : output_local_row;
    const unsigned acc_base = acc_local_row;
    const unsigned a_local_base =
        qk ? vpu_fa_spad_start(group_id) : score_local_row;
    const unsigned d_local_base = output_local_row;

    const elem_t *a = NULL;
    const elem_t *b = NULL;
    size_t a_row_offset = 0u;
    size_t a_col_offset = 0u;
    size_t b_row_offset = 0u;
    size_t b_col_offset = 0u;
    if (qk) {
      const unsigned a_row =
          this_i != 0u ? job->q_start + i_offset * DIM : job->q_start;
      const unsigned b_col =
          this_k != 0u ? k_element_start + k_offset * DIM : k_element_start;
      const size_t query_stride = vpu_fa_stride_payload(config->query_stride);
      const size_t key_stride = vpu_fa_stride_payload(config->key_stride);
      a = gemmini_page_packed_stride_is_packed(config->query_stride)
              ? config->queries
              : config->queries + (size_t)a_row * query_stride +
                    k_element_start;
      b = gemmini_page_packed_stride_is_packed(config->key_stride)
              ? config->keys
              : config->keys + (size_t)job->kv_start * key_stride + b_col;
      a_row_offset = job->q_start / DIM;
      a_col_offset = k_element_start / DIM;
      /* B offsets are expressed in logical (K,J) coordinates.  The
       * LoopMatmul transpose path swaps them onto the physical [KV,D]
       * page-packed key source. */
      b_row_offset = k_element_start / DIM;
      b_col_offset = job->kv_start / DIM;
    } else {
      const unsigned b_row =
          this_k != 0u ? job->kv_start + k_element_start + k_offset * DIM
                       : job->kv_start + k_element_start;
      const size_t value_stride = vpu_fa_stride_payload(config->value_stride);
      b = gemmini_page_packed_stride_is_packed(config->value_stride)
              ? config->values
              : config->values + (size_t)b_row * value_stride;
      b_row_offset = (job->kv_start + k_element_start) / DIM;
    }

    VPU_FA_SHARED_LOOP_WS(
        custom, group_list, group_id, a_local_base, vpu_fa_spad_end(group_id),
        acc_base, this_i, this_k, this_pad_k, i_offset, k_offset, job->tile_i,
        job->tile_j, k_tiles, this_pad_i, pad_j, pad_k, a, b,
        d_from_vsram ? (void *)(uintptr_t)d_local_base : NULL, NULL,
        qk ? config->query_stride : job->dim_k,
        qk ? config->key_stride : config->value_stride,
        qk ? 0u : config->value_dim, 0u, false, qk, false, false, ex_accumulate,
        VPU_FA_NO_ACTIVATION, a_row_offset, a_col_offset, b_row_offset,
        b_col_offset, 0u, 0u, 0u, 0u, a_from_vsram, d_from_vsram, final_step,
        final_step);

    i_offset += this_i;
    k_offset += this_k;
    ++member_index;
  }

  job->last_group_id = group_id;
  job->next_k_tile += k_tiles;
  job->done = final_step;
  return true;
}

static inline void vpu_fa_group_score_max_stream(unsigned group_id,
                                                 unsigned full_tiles,
                                                 unsigned tail_columns) {
  const bool expand_tiles =
      VPU_LOOP_BUFFER_ENTRIES >= 56u && tail_columns == 0u && full_tiles <= 4u;
  const unsigned total_columns = full_tiles * DIM + tail_columns;
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_SCORE_ROW, 0u);
  /* Keep Gemmini's tile-major matrix layout, but present one score row to the
   * VPU as a single logical vector. This removes per-DIM reduction drains
   * while retaining the existing QK/PV bridge addresses. */
  if (total_columns != 0u && total_columns <= VPU_VLEN) {
    vpu_fa_group_set_vl(group_id, total_columns);
    vpu_fa_group_set_vector_stride(group_id, VPU_FA_MATRIX_TILE_ELEMENTS);
    vpu_fa_group_vector(group_id, VPU_OP_V_MUL_VF, VPU_FA_GP_VECTOR,
                        VPU_FA_GP_VECTOR, VPU_FA_FP_SCALE);
    vpu_fa_group_command(group_id, false, VPU_OP_V_RED_MAX, VPU_FA_FP_M_NEW,
                         VPU_FA_GP_VECTOR, 0, 0, 0, 0);
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
                                                 unsigned tail_columns) {
  const bool expand_tiles =
      VPU_LOOP_BUFFER_ENTRIES >= 56u && tail_columns == 0u && full_tiles <= 4u;
  const unsigned total_columns = full_tiles * DIM + tail_columns;
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_SCORE_ROW, 0u);
  if (total_columns != 0u && total_columns <= VPU_VLEN) {
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
                                                    unsigned value_dim) {
  const unsigned full_tiles = value_dim / DIM;
  const unsigned tail_columns = value_dim % DIM;
  const bool expand_tiles =
      VPU_LOOP_BUFFER_ENTRIES >= 56u && tail_columns == 0u && full_tiles <= 4u;
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_OUTPUT, VPU_FA_GP_OUTPUT_ROW, 0u);
  if (value_dim != 0u && value_dim <= VPU_VLEN) {
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
                                 unsigned group_id, unsigned kv_start,
                                 unsigned kv_rows, unsigned first_row,
                                 unsigned row_count, unsigned global_query) {
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
      future_columns != 0u && future_columns <= VPU_VLEN;

  ++stats->mask_row_loop_regions;
  stats->mask_row_loop_rows += row_count;
  if (has_diagonal) {
    stats->mask_diagonal_vectors += row_count;
    vpu_fa_group_write_gp(
        group_id, VPU_FA_GP_SCORE_ROW,
        vpu_fa_matrix_address(false, score_tiles, first_row, diagonal_tile));
    vpu_fa_group_write_gp(group_id, VPU_FA_GP_MASK_ROW,
                          VPU_FA_CAUSAL_MASK_BASE +
                              (global_query - diagonal_key_start) * DIM);
  }
  if (future_full_tiles != 0u || future_tail) {
    stats->mask_future_vectors +=
        (uint64_t)row_count *
        (coalesce_future ? 1u : future_full_tiles + (future_tail ? 1u : 0u));
    vpu_fa_group_write_gp(group_id, VPU_FA_GP_FUTURE_ROW,
                          vpu_fa_matrix_address(false, score_tiles, first_row,
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
      vpu_fa_group_vector(group_id, VPU_OP_V_MIN_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_NEG_INF);
      vpu_fa_group_set_vector_stride(group_id, 0u);
    } else if (future_full_tiles != 0u) {
      vpu_fa_group_set_vl(group_id, DIM);
      vpu_fa_group_loop_start(group_id, VPU_FA_GP_TILE_LOOP, future_full_tiles);
      vpu_fa_group_vector(group_id, VPU_OP_V_MIN_VF, VPU_FA_GP_VECTOR,
                          VPU_FA_GP_VECTOR, VPU_FA_FP_NEG_INF);
      vpu_fa_group_addi_gp(group_id, VPU_FA_GP_VECTOR, VPU_FA_GP_VECTOR,
                           VPU_FA_MATRIX_TILE_ELEMENTS);
      vpu_fa_group_loop_end(group_id, VPU_FA_GP_TILE_LOOP);
    }
    if (future_tail && !coalesce_future) {
      vpu_fa_group_set_vl(group_id, tail_score_columns);
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

static inline void vpu_fa_issue_unmasked_row_run(
    const vpu_flashattention_config_t *config,
    vpu_flashattention_stats_t *stats, unsigned group_id, unsigned kv_rows,
    unsigned first_row, unsigned row_count, bool first_block) {
  const unsigned score_tiles = vpu_fa_ceil_div(kv_rows, DIM);
  const unsigned full_score_tiles = kv_rows / DIM;
  const unsigned tail_score_columns = kv_rows % DIM;
  const unsigned value_tiles =
      vpu_fa_ceil_div((unsigned)config->value_dim, DIM);
  const unsigned state_l_base = VPU_FP_STATE_ENTRIES / 2u;

  ++stats->qk_row_loop_regions;
  stats->qk_row_loop_rows += row_count;
  vpu_fa_group_write_gp(
      group_id, VPU_FA_GP_SCORE_ROW,
      vpu_fa_matrix_address(false, score_tiles, first_row, 0u));
  if (!first_block)
    vpu_fa_group_write_gp(
        group_id, VPU_FA_GP_OUTPUT_ROW,
        vpu_fa_matrix_address(true, value_tiles, first_row, 0u));
  vpu_fa_group_write_gp(group_id, VPU_FA_GP_STATE, first_row);
  vpu_fa_group_write_gp(group_id, VPU_FA_GP_STATE_L, state_l_base + first_row);

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
  vpu_fa_group_score_max_stream(group_id, full_score_tiles, tail_score_columns);
  vpu_fa_group_scalar(group_id, VPU_OP_S_MAX_FP, VPU_FA_FP_M_NEW,
                      VPU_FA_FP_M_OLD, VPU_FA_FP_M_NEW);
  vpu_fa_group_scalar(group_id, VPU_OP_S_SUB_FP, VPU_FA_FP_ALPHA,
                      VPU_FA_FP_M_OLD, VPU_FA_FP_M_NEW);
  vpu_fa_group_scalar(group_id, VPU_OP_S_EXP_FP, VPU_FA_FP_ALPHA,
                      VPU_FA_FP_ALPHA, 0u);
  vpu_fa_group_write_fp(group_id, VPU_FA_FP_P_SUM, 0.0f);
  vpu_fa_group_score_exp_stream(group_id, full_score_tiles, tail_score_columns);
  vpu_fa_group_scalar(group_id, VPU_OP_S_MUL_FP, VPU_FA_FP_L_OLD,
                      VPU_FA_FP_L_OLD, VPU_FA_FP_ALPHA);
  vpu_fa_group_scalar(group_id, VPU_OP_S_ADD_FP, VPU_FA_FP_L_OLD,
                      VPU_FA_FP_L_OLD, VPU_FA_FP_P_SUM);
  vpu_fa_group_state_store_current(group_id, VPU_FA_GP_STATE, VPU_FA_FP_M_NEW);
  vpu_fa_group_state_store_current(group_id, VPU_FA_GP_STATE_L,
                                   VPU_FA_FP_L_OLD);
  if (!first_block)
    vpu_fa_group_output_scale_stream(group_id, VPU_FA_FP_ALPHA,
                                     (unsigned)config->value_dim);
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_SCORE_ROW, VPU_FA_GP_SCORE_ROW, DIM);
  if (!first_block)
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_OUTPUT_ROW, VPU_FA_GP_OUTPUT_ROW,
                         DIM);
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_STATE, VPU_FA_GP_STATE, 1u);
  vpu_fa_group_addi_gp(group_id, VPU_FA_GP_STATE_L, VPU_FA_GP_STATE_L, 1u);
  vpu_fa_group_loop_end(group_id, VPU_FA_GP_ROW_LOOP);
}

static inline void
vpu_fa_issue_qk_vpu(const vpu_flashattention_config_t *config,
                    vpu_flashattention_stats_t *stats, unsigned q_start,
                    unsigned q_rows, unsigned kv_start, unsigned kv_rows,
                    unsigned group_id) {
  const bool first_block = kv_start == 0u;
  /* Make the kernel independent of any prior standalone/aborted VPU stream. */
  vpu_fa_group_set_vector_stride(group_id, 0u);
  vpu_fa_group_write_fp(group_id, VPU_FA_FP_SCALE, config->score_scale);
  vpu_fa_group_write_fp(group_id, VPU_FA_FP_NEG_INF, -INFINITY);

  for (unsigned row = 0u; row < q_rows;) {
    const unsigned global_query = (unsigned)config->query_base + q_start + row;
    const unsigned rows_to_local_boundary = DIM - row % DIM;
    const unsigned rows_to_mask_boundary =
        global_query < kv_start ? kv_start - global_query
                                : DIM - (global_query - kv_start) % DIM;
    const unsigned rows_left = q_rows - row;
    unsigned row_run =
        rows_left < rows_to_local_boundary ? rows_left : rows_to_local_boundary;
    if (rows_to_mask_boundary < row_run)
      row_run = rows_to_mask_boundary;
    vpu_fa_issue_causal_mask_row_run(stats, group_id, kv_start, kv_rows, row,
                                     row_run, global_query);
    row += row_run;
  }
  for (unsigned row = 0u; row < q_rows;) {
    const unsigned rows_to_tile_boundary = DIM - row % DIM;
    const unsigned rows_left = q_rows - row;
    const unsigned row_run =
        rows_left < rows_to_tile_boundary ? rows_left : rows_to_tile_boundary;
    vpu_fa_issue_unmasked_row_run(config, stats, group_id, kv_rows, row,
                                  row_run, first_block);
    row += row_run;
  }
  vpu_fa_group_terminator(group_id);
}

static inline void
vpu_fa_issue_final_normalize(const vpu_flashattention_config_t *config,
                             vpu_flashattention_stats_t *stats, unsigned q_rows,
                             bool last_causal_block, unsigned group_id) {
  /* Final normalization uses the same segmented layout only where requested
   * below; establish the legacy contiguous mode for all other commands. */
  vpu_fa_group_set_vector_stride(group_id, 0u);
  if (!last_causal_block) {
    vpu_fa_group_terminator(group_id);
    return;
  }
  const unsigned state_l_base = VPU_FP_STATE_ENTRIES / 2u;
  const unsigned value_tiles =
      vpu_fa_ceil_div((unsigned)config->value_dim, DIM);
  for (unsigned row = 0u; row < q_rows;) {
    const unsigned rows_to_tile_boundary = DIM - row % DIM;
    const unsigned rows_left = q_rows - row;
    const unsigned row_run =
        rows_left < rows_to_tile_boundary ? rows_left : rows_to_tile_boundary;
    ++stats->normalize_row_loop_regions;
    stats->normalize_row_loop_rows += row_run;
    vpu_fa_group_write_gp(group_id, VPU_FA_GP_OUTPUT_ROW,
                          vpu_fa_matrix_address(true, value_tiles, row, 0u));
    vpu_fa_group_write_gp(group_id, VPU_FA_GP_STATE_L, state_l_base + row);
    vpu_fa_group_loop_start(group_id, VPU_FA_GP_ROW_LOOP, row_run);
    vpu_fa_group_state_load_current(group_id, VPU_FA_FP_L_OLD,
                                    VPU_FA_GP_STATE_L);
    vpu_fa_group_scalar(group_id, VPU_OP_S_RECI_FP, VPU_FA_FP_L_OLD,
                        VPU_FA_FP_L_OLD, 0u);
    vpu_fa_group_output_scale_stream(group_id, VPU_FA_FP_L_OLD,
                                     (unsigned)config->value_dim);
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_OUTPUT_ROW, VPU_FA_GP_OUTPUT_ROW,
                         DIM);
    vpu_fa_group_addi_gp(group_id, VPU_FA_GP_STATE_L, VPU_FA_GP_STATE_L, 1u);
    vpu_fa_group_loop_end(group_id, VPU_FA_GP_ROW_LOOP);
    row += row_run;
  }
  vpu_fa_group_terminator(group_id);
}

static inline void
vpu_fa_enqueue_output_store(const vpu_flashattention_config_t *config,
                            unsigned q_start, unsigned q_rows) {
  const unsigned value_tiles =
      vpu_fa_ceil_div((unsigned)config->value_dim, DIM);
  const unsigned full_tiles = (unsigned)config->value_dim / DIM;
  const size_t output_stride = vpu_fa_stride_payload(config->output_stride);
  for (unsigned row = 0u; row < q_rows; ++row) {
    const uint32_t first_vector =
        vpu_fa_matrix_address(true, value_tiles, row, 0u);
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
                   vpu_fa_matrix_address(true, value_tiles, row, full_tiles));
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

  const vpu_flashattention_config_t config = vpu_fa_normalize_config(source);
  result.status = vpu_flashattention_make_plan(&config, &result.plan);
  if (result.status != VPU_FLASHATTENTION_OK)
    return result;
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

  unsigned dispatch_sequence = 0u;
  for (size_t q_start = 0u; q_start < config.query_rows;) {
    const unsigned q_rows = config.query_rows - q_start < result.plan.q_rows
                                ? (unsigned)(config.query_rows - q_start)
                                : result.plan.q_rows;
    const size_t query_block_end = config.query_base + q_start + q_rows;
    const size_t causal_key_end =
        query_block_end < config.sequence ? query_block_end : config.sequence;

    for (size_t kv_start = 0u; kv_start < causal_key_end;) {
      const unsigned kv_rows = causal_key_end - kv_start < result.plan.kv_rows
                                   ? (unsigned)(causal_key_end - kv_start)
                                   : result.plan.kv_rows;
      const bool last_causal_block = kv_start + kv_rows == causal_key_end;

      vpu_fa_matmul_job_t qk_job;
      (void)vpu_fa_matmul_job_prepare(&qk_job, &config, VPU_FA_MATMUL_QK,
                                      (unsigned)q_start, q_rows,
                                      (unsigned)kv_start, kv_rows);
      vpu_fa_matmul_job_init(&qk_job);
      ++result.stats.qk_jobs;
      while (!qk_job.done) {
        (void)vpu_fa_matmul_job_step(&qk_job, &dispatch_sequence);
        ++result.stats.gemmini_job_steps;
      }
      vpu_fa_issue_qk_vpu(&config, &result.stats, (unsigned)q_start, q_rows,
                          (unsigned)kv_start, kv_rows, qk_job.last_group_id);

      vpu_fa_matmul_job_t pv_job;
      (void)vpu_fa_matmul_job_prepare(&pv_job, &config, VPU_FA_MATMUL_PV,
                                      (unsigned)q_start, q_rows,
                                      (unsigned)kv_start, kv_rows);
      vpu_fa_matmul_job_init(&pv_job);
      ++result.stats.pv_jobs;
      while (!pv_job.done) {
        (void)vpu_fa_matmul_job_step(&pv_job, &dispatch_sequence);
        ++result.stats.gemmini_job_steps;
      }
      vpu_fa_issue_final_normalize(&config, &result.stats, q_rows,
                                   last_causal_block, pv_job.last_group_id);
      kv_start += kv_rows;
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
