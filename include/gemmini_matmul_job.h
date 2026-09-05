// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_INCLUDE_GEMMINI_MATMUL_JOB_H_
#define GEMMINI_ROCC_TESTS_INCLUDE_GEMMINI_MATMUL_JOB_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gemmini_params.h"
#include "gemmini_tiling.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  SHARED_MULTI_PARTITION_OK = 0,
  SHARED_MULTI_PARTITION_BAD_AXIS,
  SHARED_MULTI_PARTITION_EMPTY_MASK,
  SHARED_MULTI_PARTITION_UNSUPPORTED_MODE,
  SHARED_MULTI_PARTITION_FIELD_OVERFLOW,
  SHARED_MULTI_PARTITION_PLAN_FAILED,
} shared_multi_partition_status_t;

/*
 * Common software job state for a shared multi-Gemmini matmul.
 *
 * The ordinary matmul entry points in gemmini_all.h retain their existing
 * API.  Keeping the state in this lightweight header also lets fused kernels
 * use the same tiling, init and K/J/I step cursor without pulling the complete
 * Gemmini command library (and its RISC-V inline assembly) into host tests.
 */
typedef struct {
  /* Caller-supplied operation. */
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
  int dataflow;

  /* Values derived by tiling/init. */
  size_t dim_I_padded, dim_J_padded, dim_K_padded;
  size_t I0, J0, K0;
  size_t last_I, last_J, last_K;
  size_t padding_I, padding_J, padding_K;
  bool no_bias;
  size_t sizeof_D, sizeof_C;
  int gemmini_num;

  /* K -> J -> I outer-step cursor. */
  size_t i0, j0, k0;
  int inner_call_counter;
  int lastK_toggle;
  bool done;

  gemmini_partition_axis_t partition_axis;
  shared_multi_partition_status_t partition_status;
} shared_multi_matmul_job_t;

/*
 * Optional issue policy used by ACC-resident/fused consumers.  Ordinary
 * matmul never reads this object, so its established defaults and ABI remain
 * unchanged.  A fused wrapper supplies the policy explicitly for each job.
 */
typedef struct {
  bool a_from_acc;
  bool keep_output_in_acc;
  bool fixed_acc_addr;
  size_t a_acc_addr_start;
  size_t fixed_acc_addr_start;
  bool first_k_accumulate;

  size_t A_row_offset_base, A_col_offset_base;
  size_t B_row_offset_base, B_col_offset_base;

  unsigned wait_event_id;
  unsigned produce_event_id;
  bool wait_event_valid;
  bool produce_event_valid;
} shared_multi_matmul_job_extension_t;

typedef struct {
  size_t i0, j0, k0;
  size_t I, J, K;
  size_t pad_I, pad_J, pad_K;
  size_t active_gemmini_count;
  uint64_t group_list;
  bool first_job_step;
  bool final_k_step;
  bool final_job_step;
} shared_multi_matmul_job_step_t;

static inline size_t
shared_multi_matmul_job_count_members(uint64_t gemmini_list) {
  size_t count = 0;
  while (gemmini_list != 0) {
    count += gemmini_list & UINT64_C(1);
    gemmini_list >>= 1;
  }
  return count;
}

/* Install an already selected exact factor set.  Attention first applies its
 * complete live-set planner and then uses this helper; ordinary matmul uses it
 * after the existing generic selector. */
static inline bool shared_multi_matmul_job_set_exact_tiling(
    shared_multi_matmul_job_t *job,
    size_t tile_I, size_t tile_J, size_t tile_K) {
  if (job == NULL || job->dim_I == 0 || job->dim_J == 0 || job->dim_K == 0 ||
      tile_I == 0 || tile_J == 0 || tile_K == 0) {
    if (job != NULL) {
      job->partition_status = SHARED_MULTI_PARTITION_PLAN_FAILED;
      job->done = true;
    }
    return false;
  }

  job->dim_I_padded =
      (job->dim_I / DIM + (job->dim_I % DIM != 0)) * DIM;
  job->dim_J_padded =
      (job->dim_J / DIM + (job->dim_J % DIM != 0)) * DIM;
  job->dim_K_padded =
      (job->dim_K / DIM + (job->dim_K % DIM != 0)) * DIM;
  job->tile_I = tile_I;
  job->tile_J = tile_J;
  job->tile_K = tile_K;
  job->sp_addr_A_stacked =
      gemmini_tiling_A_spad_rows(tile_I, tile_J, tile_K, DIM);
  job->sp_addr_B_stacked =
      gemmini_tiling_B_spad_rows(tile_I, tile_J, tile_K, DIM);
  job->acc_addr_stacked =
      gemmini_tiling_total_acc_rows(tile_I, tile_J, DIM);
  job->partition_status = SHARED_MULTI_PARTITION_OK;
  return true;
}

/* Common init portion.  Command configuration remains at the caller because
 * ordinary matmul and fused attention use different load/store policies. */
static inline bool
shared_multi_matmul_job_init_state(shared_multi_matmul_job_t *job) {
  if (job == NULL || job->tile_I == 0 || job->tile_J == 0 ||
      job->tile_K == 0 || job->dim_I_padded == 0 ||
      job->dim_J_padded == 0 || job->dim_K_padded == 0) {
    if (job != NULL) {
      job->partition_status = SHARED_MULTI_PARTITION_PLAN_FAILED;
      job->done = true;
    }
    return false;
  }

  job->I0 = job->dim_I_padded / (job->tile_I * DIM) +
            (job->dim_I_padded % (job->tile_I * DIM) != 0);
  job->J0 = job->dim_J_padded / (job->tile_J * DIM) +
            (job->dim_J_padded % (job->tile_J * DIM) != 0);
  job->K0 = job->dim_K_padded / (job->tile_K * DIM) +
            (job->dim_K_padded % (job->tile_K * DIM) != 0);

  job->last_I = job->dim_I_padded % (job->tile_I * DIM) == 0
                    ? job->tile_I
                    : (job->dim_I_padded / DIM) % job->tile_I;
  job->last_J = job->dim_J_padded % (job->tile_J * DIM) == 0
                    ? job->tile_J
                    : (job->dim_J_padded / DIM) % job->tile_J;
  job->last_K = job->dim_K_padded % (job->tile_K * DIM) == 0
                    ? job->tile_K
                    : (job->dim_K_padded / DIM) % job->tile_K;

  job->padding_I = job->dim_I_padded - job->dim_I;
  job->padding_J = job->dim_J_padded - job->dim_J;
  job->padding_K = job->dim_K_padded - job->dim_K;
  job->no_bias = job->D == NULL;
  job->sizeof_D = job->low_D ? sizeof(elem_t) : sizeof(acc_t);
  job->sizeof_C = job->full_C ? sizeof(acc_t) : sizeof(elem_t);
  job->i0 = job->j0 = job->k0 = 0;
  job->inner_call_counter = 0;
  job->lastK_toggle = 1;
  job->done = false;
  return true;
}

/* Exact jobs whose selected I/J tiles cover the complete padded matrices do
 * not need the generic outer-count divisions. FlashAttention establishes this
 * invariant for every full and tail QK/PV block, while K may still span more
 * than one step. Keep this reset in the common job layer so fused users retain
 * the same cursor/completion semantics as ordinary matmul jobs. */
static inline bool shared_multi_matmul_job_init_single_ij_state(
    shared_multi_matmul_job_t *job) {
  if (job == NULL || job->tile_I == 0 || job->tile_J == 0 ||
      job->tile_K == 0 || job->dim_I_padded != job->tile_I * DIM ||
      job->dim_J_padded != job->tile_J * DIM ||
      job->dim_K_padded == 0 || job->dim_K_padded % DIM != 0) {
    if (job != NULL) {
      job->partition_status = SHARED_MULTI_PARTITION_PLAN_FAILED;
      job->done = true;
    }
    return false;
  }

  const size_t total_k_tiles = job->dim_K_padded / DIM;
  if (job->tile_K > total_k_tiles) {
    job->partition_status = SHARED_MULTI_PARTITION_PLAN_FAILED;
    job->done = true;
    return false;
  }

  job->I0 = 1;
  job->J0 = 1;
  job->last_I = job->tile_I;
  job->last_J = job->tile_J;
  if (job->tile_K == total_k_tiles) {
    job->K0 = 1;
    job->last_K = job->tile_K;
  } else {
    const size_t tail_k = total_k_tiles % job->tile_K;
    job->K0 = total_k_tiles / job->tile_K + (tail_k != 0);
    job->last_K = tail_k == 0 ? job->tile_K : tail_k;
  }

  job->padding_I = job->dim_I_padded - job->dim_I;
  job->padding_J = job->dim_J_padded - job->dim_J;
  job->padding_K = job->dim_K_padded - job->dim_K;
  job->no_bias = job->D == NULL;
  job->sizeof_D = job->low_D ? sizeof(elem_t) : sizeof(acc_t);
  job->sizeof_C = job->full_C ? sizeof(acc_t) : sizeof(elem_t);
  job->i0 = job->j0 = job->k0 = 0;
  job->inner_call_counter = 0;
  job->lastK_toggle = 1;
  job->done = false;
  return true;
}

/* Return the current outer tile and collective member plan.  Address policy
 * and actual command emission are intentionally outside this helper. */
static inline bool shared_multi_matmul_job_plan_step(
    shared_multi_matmul_job_t *job,
    shared_multi_matmul_job_step_t *step) {
  if (job == NULL || step == NULL || job->done)
    return false;

  const size_t I = job->i0 < job->I0 - 1 ? job->tile_I : job->last_I;
  const size_t J = job->j0 < job->J0 - 1 ? job->tile_J : job->last_J;
  const size_t K = job->k0 < job->K0 - 1 ? job->tile_K : job->last_K;
  const size_t pad_I = job->i0 == job->I0 - 1 ? job->padding_I : 0;
  const size_t pad_J = job->j0 == job->J0 - 1 ? job->padding_J : 0;
  const size_t pad_K = job->k0 == job->K0 - 1 ? job->padding_K : 0;
  const size_t active = gemmini_partition_effective_member_count(
      job->partition_axis, (size_t)job->gemmini_num, I, J, K);

  if (active == 0 || I > GEMMINI_SHARED_PARTITION_FIELD_MAX ||
      J > GEMMINI_SHARED_PARTITION_FIELD_MAX ||
      K > GEMMINI_SHARED_PARTITION_FIELD_MAX ||
      pad_I > GEMMINI_SHARED_PARTITION_FIELD_MAX ||
      pad_J > GEMMINI_SHARED_PARTITION_FIELD_MAX ||
      pad_K > GEMMINI_SHARED_PARTITION_FIELD_MAX) {
    job->partition_status = active == 0
                                ? SHARED_MULTI_PARTITION_PLAN_FAILED
                                : SHARED_MULTI_PARTITION_FIELD_OVERFLOW;
    job->done = true;
    return false;
  }

  step->i0 = job->i0;
  step->j0 = job->j0;
  step->k0 = job->k0;
  step->I = I;
  step->J = J;
  step->K = K;
  step->pad_I = pad_I;
  step->pad_J = pad_J;
  step->pad_K = pad_K;
  step->active_gemmini_count = active;
  step->group_list = gemmini_partition_take_members(
      (uint64_t)(unsigned int)job->gemmini_list, active);
  step->first_job_step = job->i0 == 0 && job->j0 == 0 && job->k0 == 0;
  step->final_k_step = job->k0 + 1 == job->K0;
  step->final_job_step = job->i0 + 1 == job->I0 &&
                         job->j0 + 1 == job->J0 &&
                         job->k0 + 1 == job->K0;
  return true;
}

static inline void
shared_multi_matmul_job_complete_step(shared_multi_matmul_job_t *job) {
  if (job == NULL || job->done)
    return;

  ++job->inner_call_counter;
  if (job->k0 + 1 == job->K0)
    job->lastK_toggle ^= 1;

  ++job->k0;
  if (job->k0 >= job->K0) {
    job->k0 = 0;
    ++job->j0;
    if (job->j0 >= job->J0) {
      job->j0 = 0;
      ++job->i0;
      if (job->i0 >= job->I0)
        job->done = true;
    }
  }
}

#ifdef __cplusplus
}
#endif

#endif /* GEMMINI_ROCC_TESTS_INCLUDE_GEMMINI_MATMUL_JOB_H_ */
