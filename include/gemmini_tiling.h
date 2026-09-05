// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_INCLUDE_GEMMINI_TILING_H_
#define GEMMINI_ROCC_TESTS_INCLUDE_GEMMINI_TILING_H_

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The tiling planner is deliberately independent of elem_t, acc_t, and the
 * Gemmini command interface. All matrix dimensions and returned tile factors
 * are expressed in elements and systolic-array DIM blocks, respectively.
 */
enum {
  GEMMINI_TILING_ACT_LAYERNORM = 2,
  GEMMINI_TILING_ACT_SOFTMAX = 4,
};

/*
 * Encoding helpers for the shared LOOP_WS funct=24 partition descriptor.
 * Encoding version 1 adds the axis selector in rs1[33:32]. Axis M is zero, so
 * the legacy ex_I/mv_K command remains byte-for-byte identical.
 */
/* The encoding helper is always available. The generated hardware header
 * separately defines GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION; absence means
 * a legacy M-only bitstream. */
#define GEMMINI_SHARED_PARTITION_ENCODING_VERSION 1
#define GEMMINI_SHARED_PARTITION_FIELD_MAX UINT16_MAX

/* ABI 0 keeps the fusion LOOP_WS command/address dialect, but has no funct=24
 * partition descriptor.  It is therefore valid for the legacy M axis while
 * callers must omit that one configuration command. */
#if defined(GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION) &&                       \
    GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION ==                               \
        GEMMINI_SHARED_PARTITION_ENCODING_VERSION
#define GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE 1
#else
#define GEMMINI_SHARED_PARTITION_DESCRIPTOR_AVAILABLE 0
#endif

typedef enum {
  GEMMINI_PARTITION_AXIS_M = 0,
  GEMMINI_PARTITION_AXIS_N = 1,
  GEMMINI_PARTITION_AXIS_K = 2,
} gemmini_partition_axis_t;

typedef struct {
  gemmini_partition_axis_t axis;
  size_t partition_extent;
  size_t partition_offset;
  size_t partition_pad;
  size_t aux_extent;
  size_t aux_offset;
  size_t aux_pad;
} gemmini_partition_plan_t;

static inline bool gemmini_partition_axis_is_valid(
    gemmini_partition_axis_t axis) {
  return axis == GEMMINI_PARTITION_AXIS_M ||
         axis == GEMMINI_PARTITION_AXIS_N ||
         axis == GEMMINI_PARTITION_AXIS_K;
}

static inline unsigned gemmini_partition_generated_axis_abi_version(void) {
#if defined(GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION)
  return GEMMINI_SHARED_PARTITION_AXIS_ABI_VERSION;
#else
  return 0;
#endif
}

static inline bool gemmini_partition_axis_supported_by_abi(
    gemmini_partition_axis_t axis, unsigned generated_abi_version) {
  return axis == GEMMINI_PARTITION_AXIS_M ||
         (gemmini_partition_axis_is_valid(axis) &&
          generated_abi_version == GEMMINI_SHARED_PARTITION_ENCODING_VERSION);
}

static inline bool gemmini_shared_partition_fields_valid(
    gemmini_partition_axis_t axis,
    size_t partition_extent, size_t partition_offset,
    size_t aux_extent, size_t aux_offset, size_t aux_pad) {
  return gemmini_partition_axis_is_valid(axis) &&
         partition_extent <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         partition_offset <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         aux_extent <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         aux_offset <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         aux_pad <= GEMMINI_SHARED_PARTITION_FIELD_MAX;
}

static inline bool gemmini_loop_bounds_fields_valid(
    size_t I, size_t J, size_t K,
    size_t pad_I, size_t pad_J, size_t pad_K) {
  return I <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         J <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         K <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         pad_I <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         pad_J <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         pad_K <= GEMMINI_SHARED_PARTITION_FIELD_MAX;
}

static inline bool gemmini_page_offset_fields_valid(
    size_t A_row_offset, size_t A_col_offset,
    size_t B_row_offset, size_t B_col_offset,
    size_t D_row_offset, size_t D_col_offset,
    size_t C_row_offset, size_t C_col_offset) {
  return A_row_offset <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         A_col_offset <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         B_row_offset <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         B_col_offset <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         D_row_offset <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         D_col_offset <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         C_row_offset <= GEMMINI_SHARED_PARTITION_FIELD_MAX &&
         C_col_offset <= GEMMINI_SHARED_PARTITION_FIELD_MAX;
}

/* rs1[63:34] and rs2[63:48] are reserved and emitted as zero. */
static inline uint64_t gemmini_shared_partition_pack_rs1(
    gemmini_partition_axis_t axis,
    size_t partition_offset, size_t aux_offset) {
  return (((uint64_t)axis & UINT64_C(0x3)) << 32) |
         (((uint64_t)aux_offset & UINT64_C(0xffff)) << 16) |
         ((uint64_t)partition_offset & UINT64_C(0xffff));
}

static inline uint64_t gemmini_shared_partition_pack_rs2(
    size_t partition_extent, size_t aux_extent, size_t aux_pad) {
  return (((uint64_t)aux_extent & UINT64_C(0xffff)) << 32) |
         (((uint64_t)aux_pad & UINT64_C(0xffff)) << 16) |
         ((uint64_t)partition_extent & UINT64_C(0xffff));
}

static inline size_t gemmini_partition_min_size(size_t a, size_t b) {
  return a < b ? a : b;
}

/* Repeating bias advances only the accumulator destination, never DRAM D. */
static inline size_t gemmini_partition_d_memory_i_offset(
    size_t logical_i_offset, bool repeating_bias) {
  return repeating_bias ? 0 : logical_i_offset;
}

/* The first outer-K chunk initializes a no-bias accumulator; every other
 * chunk accumulates into an already initialized destination. */
static inline bool gemmini_partition_ex_accumulate(
    bool no_bias, size_t outer_k_index) {
  return !no_bias || outer_k_index != 0;
}

/* funct=28 carries page-block coordinates in 16-bit fields. Check the full
 * coordinate interval without overflowing size_t while forming base+offset. */
static inline bool gemmini_page_block_range_fits_u16(
    size_t base, size_t local_offset, size_t extent) {
  if (extent == 0) {
    return true;
  }
  if (base > UINT16_MAX || local_offset > UINT16_MAX - base) {
    return false;
  }
  const size_t first = base + local_offset;
  return extent - 1 <= UINT16_MAX - first;
}

/*
 * Validate the page-packed coordinates used by one shared LOOP_WS member.
 * Offsets and extents are logical (I/J/K) coordinates; transpose only swaps
 * the physical row/column wires and therefore does not change these bounds.
 */
static inline bool gemmini_shared_partition_page_offsets_valid(
    gemmini_partition_axis_t axis,
    size_t partition_extent, size_t partition_offset,
    size_t aux_extent, size_t aux_offset,
    size_t I, size_t J, size_t K,
    bool page_packed_A, bool page_packed_B,
    bool page_packed_D, bool page_packed_C,
    size_t A_row_offset, size_t A_col_offset,
    size_t B_row_offset, size_t B_col_offset,
    size_t D_row_offset, size_t D_col_offset,
    size_t C_row_offset, size_t C_col_offset) {
  if (!gemmini_partition_axis_is_valid(axis)) {
    return false;
  }

  size_t a_i_extent = I, a_i_offset = 0;
  size_t a_k_extent = K, a_k_offset = 0;
  size_t b_k_extent = K, b_k_offset = 0;
  size_t b_j_extent = J, b_j_offset = 0;
  size_t dc_i_extent = I, dc_i_offset = 0;
  size_t dc_j_extent = J, dc_j_offset = 0;

  if (axis == GEMMINI_PARTITION_AXIS_M) {
    a_i_extent = dc_i_extent = partition_extent;
    a_i_offset = dc_i_offset = partition_offset;
    b_k_extent = aux_extent;
    b_k_offset = aux_offset;
  } else if (axis == GEMMINI_PARTITION_AXIS_N) {
    a_i_extent = aux_extent;
    a_i_offset = aux_offset;
    b_j_extent = dc_j_extent = partition_extent;
    b_j_offset = dc_j_offset = partition_offset;
  } else {
    a_k_extent = b_k_extent = partition_extent;
    a_k_offset = b_k_offset = partition_offset;
    dc_i_extent = aux_extent;
    dc_i_offset = aux_offset;
  }

  const bool A_valid = !page_packed_A ||
      (gemmini_page_block_range_fits_u16(
           A_row_offset, a_i_offset, a_i_extent) &&
       gemmini_page_block_range_fits_u16(
           A_col_offset, a_k_offset, a_k_extent));
  const bool B_valid = !page_packed_B ||
      (gemmini_page_block_range_fits_u16(
           B_row_offset, b_k_offset, b_k_extent) &&
       gemmini_page_block_range_fits_u16(
           B_col_offset, b_j_offset, b_j_extent));
  const bool D_valid = !page_packed_D ||
      (gemmini_page_block_range_fits_u16(
           D_row_offset, dc_i_offset, dc_i_extent) &&
       gemmini_page_block_range_fits_u16(
           D_col_offset, dc_j_offset, dc_j_extent));
  const bool C_valid = !page_packed_C ||
      (gemmini_page_block_range_fits_u16(
           C_row_offset, dc_i_offset, dc_i_extent) &&
       gemmini_page_block_range_fits_u16(
           C_col_offset, dc_j_offset, dc_j_extent));

  return A_valid && B_valid && D_valid && C_valid;
}

/*
 * Preserve the legacy M policy and apply it symmetrically to N: a member may
 * have no work on the execution partition but still stream the auxiliary
 * operand. M therefore uses max(I,K), N uses max(J,I), and K needs only K.
 */
static inline size_t gemmini_partition_effective_member_count(
    gemmini_partition_axis_t axis, size_t requested_count,
    size_t I, size_t J, size_t K) {
  if (!gemmini_partition_axis_is_valid(axis) || requested_count == 0) {
    return 0;
  }

  size_t work_extent;
  if (axis == GEMMINI_PARTITION_AXIS_M) {
    work_extent = I > K ? I : K;
  } else if (axis == GEMMINI_PARTITION_AXIS_N) {
    work_extent = J > I ? J : I;
  } else {
    work_extent = K;
  }
  return gemmini_partition_min_size(requested_count, work_extent);
}

/* Retain the first member_count set bits, including non-contiguous masks. */
static inline uint64_t gemmini_partition_take_members(
    uint64_t requested_mask, size_t member_count) {
  uint64_t result = 0;
  size_t selected = 0;
  for (size_t bit = 0; bit < 64 && selected < member_count; ++bit) {
    const uint64_t member = UINT64_C(1) << bit;
    if ((requested_mask & member) != 0) {
      result |= member;
      ++selected;
    }
  }
  return result;
}

static inline void gemmini_partition_even_split(
    size_t total, size_t member_count, size_t rank,
    size_t *extent, size_t *offset) {
  const size_t base = total / member_count;
  const size_t extra = total % member_count;
  *extent = base + (rank < extra ? 1 : 0);
  *offset = rank * base + (rank < extra ? rank : extra);
}

/*
 * Build one member's forced-axis descriptor in DIM-block units. partition_pad
 * is routed through the corresponding global pad field; aux_pad is encoded in
 * funct=24. A tail pad belongs only to the last non-empty shard.
 */
static inline bool gemmini_partition_plan_member(
    gemmini_partition_axis_t axis,
    size_t I, size_t J, size_t K,
    size_t pad_I, size_t pad_J, size_t pad_K,
    size_t member_count, size_t rank,
    gemmini_partition_plan_t *plan) {
  if (plan == NULL || !gemmini_partition_axis_is_valid(axis) ||
      member_count == 0 || rank >= member_count) {
    return false;
  }

  size_t partition_total;
  size_t partition_global_pad;
  size_t aux_total;
  size_t aux_global_pad;
  if (axis == GEMMINI_PARTITION_AXIS_M) {
    partition_total = I;
    partition_global_pad = pad_I;
    aux_total = K;
    aux_global_pad = pad_K;
  } else if (axis == GEMMINI_PARTITION_AXIS_N) {
    partition_total = J;
    partition_global_pad = pad_J;
    aux_total = I;
    aux_global_pad = pad_I;
  } else {
    partition_total = K;
    partition_global_pad = pad_K;
    aux_total = I;
    aux_global_pad = pad_I;
  }

  plan->axis = axis;
  gemmini_partition_even_split(
      partition_total, member_count, rank,
      &plan->partition_extent, &plan->partition_offset);
  gemmini_partition_even_split(
      aux_total, member_count, rank,
      &plan->aux_extent, &plan->aux_offset);

  plan->partition_pad =
      plan->partition_extent != 0 &&
              plan->partition_offset + plan->partition_extent == partition_total
          ? partition_global_pad
          : 0;
  plan->aux_pad =
      plan->aux_extent != 0 &&
              plan->aux_offset + plan->aux_extent == aux_total
          ? aux_global_pad
          : 0;

  return gemmini_shared_partition_fields_valid(
      axis,
      plan->partition_extent, plan->partition_offset,
      plan->aux_extent, plan->aux_offset, plan->aux_pad);
}

typedef struct {
  size_t dim_I;
  size_t dim_J;
  size_t dim_K;

  /* Number of elements along one side of the systolic array. */
  size_t dim;

  size_t gemmini_count;
  size_t sp_addr_range;
  size_t acc_addr_range;
  bool double_buffered;
  int act;
} gemmini_tiling_request_t;

typedef struct {
  /* Tile factors are counts of dim-by-dim systolic-array blocks. */
  size_t tile_I;
  size_t tile_J;
  size_t tile_K;

  size_t dim_I_padded;
  size_t dim_J_padded;
  size_t dim_K_padded;
} gemmini_tiling_factors_t;

static inline size_t gemmini_tiling_round_up(size_t elements, size_t dim) {
  return (elements / dim + (elements % dim != 0)) * dim;
}

static inline size_t gemmini_tiling_total_spad_rows(
    size_t I, size_t J, size_t K, size_t dim) {
  return (I * K + K * J) * dim;
}

static inline size_t gemmini_tiling_A_spad_rows(
    size_t I, size_t J, size_t K, size_t dim) {
  (void)J;
  return (I * K) * dim;
}

static inline size_t gemmini_tiling_B_spad_rows(
    size_t I, size_t J, size_t K, size_t dim) {
  (void)I;
  return (K * J) * dim;
}

static inline size_t gemmini_tiling_total_acc_rows(
    size_t I, size_t J, size_t dim) {
  return (I * J) * dim;
}

static inline bool gemmini_tiling_split_spad_rows_fit(
    size_t I, size_t J, size_t K, size_t dim,
    size_t max_operand_spad_rows) {
  return gemmini_tiling_A_spad_rows(I, J, K, dim) <=
             max_operand_spad_rows &&
         gemmini_tiling_B_spad_rows(I, J, K, dim) <=
             max_operand_spad_rows;
}

static inline size_t gemmini_tiling_balance_score(
    size_t I, size_t J, size_t K) {
  const size_t max_IJ = I > J ? I : J;
  const size_t max_IJK = max_IJ > K ? max_IJ : K;
  const size_t min_IJ = I < J ? I : J;
  const size_t min_IJK = min_IJ < K ? min_IJ : K;
  return max_IJK - min_IJK;
}

static inline size_t gemmini_tiling_tail_gap(
    size_t total_tiles, size_t tile) {
  const size_t remainder = total_tiles % tile;
  return remainder == 0 ? 0 : tile - remainder;
}

static inline size_t gemmini_tiling_tail_score(
    size_t total_I, size_t total_J, size_t total_K,
    size_t I, size_t J, size_t K) {
  return gemmini_tiling_tail_gap(total_I, I) +
         gemmini_tiling_tail_gap(total_J, J) +
         gemmini_tiling_tail_gap(total_K, K);
}

static inline size_t gemmini_tiling_axis_preferred_min(
    bool constrained, size_t member_count, size_t maximum) {
  return constrained ? gemmini_partition_min_size(member_count, maximum) : 1;
}

static inline size_t gemmini_tiling_axis_quantum(
    bool partitioned, size_t member_count, size_t maximum) {
  return partitioned && member_count != 0 && member_count <= maximum
             ? member_count
             : 1;
}

static inline size_t gemmini_tiling_axis_normalize(
    size_t tile, size_t preferred_min, size_t quantum) {
  if (tile < preferred_min) {
    return preferred_min;
  }
  if (quantum > 1 && tile % quantum != 0) {
    const size_t aligned = (tile / quantum) * quantum;
    return aligned < preferred_min ? preferred_min : aligned;
  }
  return tile;
}

static inline size_t gemmini_tiling_axis_decrease(
    size_t tile, size_t preferred_min, size_t quantum) {
  if (tile <= preferred_min) {
    return tile;
  }
  const size_t distance = tile - preferred_min;
  return distance >= quantum ? tile - quantum : preferred_min;
}

static inline size_t gemmini_tiling_axis_next(
    size_t tile, size_t preferred_min, size_t quantum) {
  return tile < preferred_min ? tile + 1 : tile + quantum;
}

/*
 * Select the same factors as shared_multi_choose_tiling_factors() in
 * gemmini_all.h. Valid requests have non-zero dim, gemmini_count, dimensions,
 * and enough scratchpad/accumulator rows for at least one DIM block.
 */
static inline gemmini_tiling_factors_t gemmini_shared_multi_choose_tiling_axis(
    const gemmini_tiling_request_t *request,
    gemmini_partition_axis_t axis) {
  gemmini_tiling_factors_t result;

  const size_t dim = request->dim;
  const size_t partition_rows = request->sp_addr_range / 2;
  const size_t mats_in_partition = partition_rows / dim;
  const size_t mats_in_acc = request->acc_addr_range / dim;
  const size_t max_tile_i_j = (size_t)sqrt((double)mats_in_acc);
  const size_t max_tile_k = mats_in_partition / max_tile_i_j;

  const size_t db_partition_rows = (request->sp_addr_range / 2) / 2;
  const size_t db_mats_in_partition = db_partition_rows / dim;
  const size_t db_mats_in_acc = (request->acc_addr_range / 2) / dim;
  const size_t db_max_tile_i_j = (size_t)sqrt((double)db_mats_in_acc);
  const size_t db_max_tile_k =
      db_mats_in_partition / db_max_tile_i_j;

  result.dim_I_padded = gemmini_tiling_round_up(request->dim_I, dim);
  result.dim_J_padded = gemmini_tiling_round_up(request->dim_J, dim);
  result.dim_K_padded = gemmini_tiling_round_up(request->dim_K, dim);

  const size_t max_spad_rows = request->double_buffered
                                   ? request->sp_addr_range / 2
                                   : request->sp_addr_range;
  const size_t max_operand_spad_rows = max_spad_rows / 2;
  const size_t max_acc_rows = request->double_buffered
                                  ? request->acc_addr_range / 2
                                  : request->acc_addr_range;

  size_t tI;
  size_t tJ;
  size_t tK;

  if (request->act == GEMMINI_TILING_ACT_LAYERNORM ||
      request->act == GEMMINI_TILING_ACT_SOFTMAX) {
    tI = 1;
    tJ = result.dim_J_padded / dim;
    tK = 1;
  } else if (request->double_buffered) {
    const size_t max_i = result.dim_I_padded / dim;
    const size_t max_j = result.dim_J_padded / dim;
    const size_t max_k = result.dim_K_padded / dim;

    tI = max_i < db_max_tile_i_j ? max_i : db_max_tile_i_j;
    tJ = max_j < db_max_tile_i_j ? max_j : db_max_tile_i_j;
    tK = max_k < db_max_tile_k ? max_k : db_max_tile_k;
  } else {
    const size_t max_i = result.dim_I_padded / dim;
    const size_t max_j = result.dim_J_padded / dim;
    const size_t max_k = result.dim_K_padded / dim;

    tI = max_i < max_tile_i_j ? max_i : max_tile_i_j;
    tJ = max_j < max_tile_i_j ? max_j : max_tile_i_j;
    tK = max_k < max_tile_k ? max_k : max_tile_k;
  }

  const size_t max_i_tiles = result.dim_I_padded / dim;
  const size_t max_j_tiles = result.dim_J_padded / dim;
  const size_t max_k_tiles = result.dim_K_padded / dim;
  const size_t min_i_tiles = 1;
  const size_t min_j_tiles = 1;
  const size_t min_k_tiles = 1;
  const bool constrain_i = axis == GEMMINI_PARTITION_AXIS_M ||
                           axis == GEMMINI_PARTITION_AXIS_N;
  const bool constrain_j = axis == GEMMINI_PARTITION_AXIS_N;
  const bool constrain_k = axis == GEMMINI_PARTITION_AXIS_M ||
                           axis == GEMMINI_PARTITION_AXIS_K;
  const size_t preferred_min_i_tiles = gemmini_tiling_axis_preferred_min(
      constrain_i, request->gemmini_count, max_i_tiles);
  const size_t preferred_min_j_tiles = gemmini_tiling_axis_preferred_min(
      constrain_j, request->gemmini_count, max_j_tiles);
  const size_t preferred_min_k_tiles = gemmini_tiling_axis_preferred_min(
      constrain_k, request->gemmini_count, max_k_tiles);
  const size_t i_quantum = gemmini_tiling_axis_quantum(
      axis == GEMMINI_PARTITION_AXIS_M,
      request->gemmini_count, max_i_tiles);
  const size_t j_quantum = gemmini_tiling_axis_quantum(
      axis == GEMMINI_PARTITION_AXIS_N,
      request->gemmini_count, max_j_tiles);
  const size_t k_quantum = gemmini_tiling_axis_quantum(
      axis == GEMMINI_PARTITION_AXIS_K,
      request->gemmini_count, max_k_tiles);

  tI = gemmini_tiling_axis_normalize(
      tI, preferred_min_i_tiles, i_quantum);
  tJ = gemmini_tiling_axis_normalize(
      tJ, preferred_min_j_tiles, j_quantum);
  tK = gemmini_tiling_axis_normalize(
      tK, preferred_min_k_tiles, k_quantum);

  while (true) {
    bool decreased = false;

    if ((gemmini_tiling_B_spad_rows(tI, tJ, tK, dim) >
             max_operand_spad_rows ||
         gemmini_tiling_total_acc_rows(tI, tJ, dim) > max_acc_rows) &&
        tJ > preferred_min_j_tiles) {
      tJ = gemmini_tiling_axis_decrease(
          tJ, preferred_min_j_tiles, j_quantum);
      decreased = true;
    }

    if (!decreased &&
        !gemmini_tiling_split_spad_rows_fit(
            tI, tJ, tK, dim, max_operand_spad_rows) &&
        tK > preferred_min_k_tiles) {
      tK = gemmini_tiling_axis_decrease(
          tK, preferred_min_k_tiles, k_quantum);
      decreased = true;
    }

    if (!decreased &&
        (gemmini_tiling_A_spad_rows(tI, tJ, tK, dim) >
             max_operand_spad_rows ||
         gemmini_tiling_total_acc_rows(tI, tJ, dim) > max_acc_rows) &&
        tI > preferred_min_i_tiles) {
      tI = gemmini_tiling_axis_decrease(
          tI, preferred_min_i_tiles, i_quantum);
      decreased = true;
    }

    if (!decreased && axis == GEMMINI_PARTITION_AXIS_N &&
        (gemmini_tiling_B_spad_rows(tI, tJ, tK, dim) >
             max_operand_spad_rows ||
         gemmini_tiling_total_acc_rows(tI, tJ, dim) > max_acc_rows) &&
        tJ > min_j_tiles) {
      tJ--;
      decreased = true;
    }

    if (!decreased &&
        !gemmini_tiling_split_spad_rows_fit(
            tI, tJ, tK, dim, max_operand_spad_rows) &&
        tK > min_k_tiles) {
      tK--;
      decreased = true;
    }

    if (!decreased &&
        (gemmini_tiling_A_spad_rows(tI, tJ, tK, dim) >
             max_operand_spad_rows ||
         gemmini_tiling_total_acc_rows(tI, tJ, dim) > max_acc_rows) &&
        tI > min_i_tiles) {
      tI--;
      decreased = true;
    }

    if (!decreased) {
      break;
    }
  }

  while (true) {
    bool increased = false;
    size_t best_tI = tI;
    size_t best_tJ = tJ;
    size_t best_tK = tK;
    size_t best_score = 0;
    size_t best_tail_score = 0;
    size_t best_sum = 0;

    const size_t next_tJ = gemmini_tiling_axis_next(
        tJ, preferred_min_j_tiles, j_quantum);
    if (next_tJ <= max_j_tiles &&
        gemmini_tiling_split_spad_rows_fit(
            tI, next_tJ, tK, dim, max_operand_spad_rows) &&
        gemmini_tiling_total_acc_rows(tI, next_tJ, dim) <= max_acc_rows) {
      best_tJ = next_tJ;
      best_score = gemmini_tiling_balance_score(best_tI, best_tJ, best_tK);
      best_tail_score = gemmini_tiling_tail_score(
          max_i_tiles, max_j_tiles, max_k_tiles,
          best_tI, best_tJ, best_tK);
      best_sum = best_tI + best_tJ + best_tK;
      increased = true;
    }

    const size_t next_tI = gemmini_tiling_axis_next(
        tI, preferred_min_i_tiles, i_quantum);
    if (next_tI <= max_i_tiles &&
        gemmini_tiling_split_spad_rows_fit(
            next_tI, tJ, tK, dim, max_operand_spad_rows) &&
        gemmini_tiling_total_acc_rows(next_tI, tJ, dim) <= max_acc_rows) {
      const size_t score = gemmini_tiling_balance_score(next_tI, tJ, tK);
      const size_t tail_score = gemmini_tiling_tail_score(
          max_i_tiles, max_j_tiles, max_k_tiles, next_tI, tJ, tK);
      const size_t sum = next_tI + tJ + tK;
      if (!increased ||
          score < best_score ||
          (score == best_score && tail_score < best_tail_score) ||
          (score == best_score && tail_score == best_tail_score &&
           sum < best_sum)) {
        best_tI = next_tI;
        best_tJ = tJ;
        best_tK = tK;
        best_score = score;
        best_tail_score = tail_score;
        best_sum = sum;
      }
      increased = true;
    }

    const size_t next_tK = gemmini_tiling_axis_next(
        tK, preferred_min_k_tiles, k_quantum);
    if (next_tK <= max_k_tiles &&
        gemmini_tiling_split_spad_rows_fit(
            tI, tJ, next_tK, dim, max_operand_spad_rows)) {
      const size_t score = gemmini_tiling_balance_score(tI, tJ, next_tK);
      const size_t tail_score = gemmini_tiling_tail_score(
          max_i_tiles, max_j_tiles, max_k_tiles, tI, tJ, next_tK);
      const size_t sum = tI + tJ + next_tK;
      if (!increased ||
          score < best_score ||
          (score == best_score && tail_score < best_tail_score) ||
          (score == best_score && tail_score == best_tail_score &&
           sum < best_sum)) {
        best_tI = tI;
        best_tJ = tJ;
        best_tK = next_tK;
        best_score = score;
        best_tail_score = tail_score;
        best_sum = sum;
      }
      increased = true;
    }

    if (!increased) {
      break;
    }

    tI = best_tI;
    tJ = best_tJ;
    tK = best_tK;
  }

  result.tile_I = tI;
  result.tile_J = tJ;
  result.tile_K = tK;
  return result;
}

static inline gemmini_tiling_factors_t gemmini_shared_multi_choose_tiling(
    const gemmini_tiling_request_t *request) {
  return gemmini_shared_multi_choose_tiling_axis(
      request, GEMMINI_PARTITION_AXIS_M);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GEMMINI_ROCC_TESTS_INCLUDE_GEMMINI_TILING_H_
