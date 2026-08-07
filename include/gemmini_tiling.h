// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_INCLUDE_GEMMINI_TILING_H_
#define GEMMINI_ROCC_TESTS_INCLUDE_GEMMINI_TILING_H_

#include <math.h>
#include <stdbool.h>
#include <stddef.h>

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

/*
 * Select the same factors as shared_multi_choose_tiling_factors() in
 * gemmini_all.h. Valid requests have non-zero dim, gemmini_count, dimensions,
 * and enough scratchpad/accumulator rows for at least one DIM block.
 */
static inline gemmini_tiling_factors_t gemmini_shared_multi_choose_tiling(
    const gemmini_tiling_request_t *request) {
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
  const size_t min_k_tiles = 1;
  const size_t preferred_min_i_tiles =
      request->gemmini_count <= max_i_tiles
          ? request->gemmini_count
          : max_i_tiles;
  const size_t preferred_min_k_tiles =
      request->gemmini_count <= max_k_tiles
          ? request->gemmini_count
          : max_k_tiles;

  if (tI < preferred_min_i_tiles) {
    tI = preferred_min_i_tiles;
  } else if (tI >= request->gemmini_count &&
             tI % request->gemmini_count != 0) {
    tI = (tI / request->gemmini_count) * request->gemmini_count;
  }

  if (tK < preferred_min_k_tiles) {
    tK = preferred_min_k_tiles;
  }

  while (true) {
    bool decreased = false;

    if ((gemmini_tiling_B_spad_rows(tI, tJ, tK, dim) >
             max_operand_spad_rows ||
         gemmini_tiling_total_acc_rows(tI, tJ, dim) > max_acc_rows) &&
        tJ > 1) {
      tJ--;
      decreased = true;
    }

    if (!decreased &&
        !gemmini_tiling_split_spad_rows_fit(
            tI, tJ, tK, dim, max_operand_spad_rows) &&
        tK > preferred_min_k_tiles) {
      tK--;
      decreased = true;
    }

    if (!decreased &&
        (gemmini_tiling_A_spad_rows(tI, tJ, tK, dim) >
             max_operand_spad_rows ||
         gemmini_tiling_total_acc_rows(tI, tJ, dim) > max_acc_rows) &&
        tI > preferred_min_i_tiles) {
      const size_t next_tI = tI > request->gemmini_count
                                  ? tI - request->gemmini_count
                                  : preferred_min_i_tiles;
      tI = next_tI >= preferred_min_i_tiles ? next_tI
                                             : preferred_min_i_tiles;
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

    if (gemmini_tiling_split_spad_rows_fit(
            tI, tJ + 1, tK, dim, max_operand_spad_rows) &&
        gemmini_tiling_total_acc_rows(tI, tJ + 1, dim) <= max_acc_rows &&
        (tJ + 1) * dim <= result.dim_J_padded) {
      best_tJ = tJ + 1;
      best_score = gemmini_tiling_balance_score(best_tI, best_tJ, best_tK);
      best_tail_score = gemmini_tiling_tail_score(
          max_i_tiles, max_j_tiles, max_k_tiles,
          best_tI, best_tJ, best_tK);
      best_sum = best_tI + best_tJ + best_tK;
      increased = true;
    }

    const size_t next_tI = tI < preferred_min_i_tiles
                                ? tI + 1
                                : tI + request->gemmini_count;
    if (gemmini_tiling_split_spad_rows_fit(
            next_tI, tJ, tK, dim, max_operand_spad_rows) &&
        gemmini_tiling_total_acc_rows(next_tI, tJ, dim) <= max_acc_rows &&
        next_tI * dim <= result.dim_I_padded) {
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

    if (gemmini_tiling_split_spad_rows_fit(
            tI, tJ, tK + 1, dim, max_operand_spad_rows) &&
        (tK + 1) * dim <= result.dim_K_padded) {
      const size_t score = gemmini_tiling_balance_score(tI, tJ, tK + 1);
      const size_t tail_score = gemmini_tiling_tail_score(
          max_i_tiles, max_j_tiles, max_k_tiles, tI, tJ, tK + 1);
      const size_t sum = tI + tJ + tK + 1;
      if (!increased ||
          score < best_score ||
          (score == best_score && tail_score < best_tail_score) ||
          (score == best_score && tail_score == best_tail_score &&
           sum < best_sum)) {
        best_tI = tI;
        best_tJ = tJ;
        best_tK = tK + 1;
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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GEMMINI_ROCC_TESTS_INCLUDE_GEMMINI_TILING_H_
