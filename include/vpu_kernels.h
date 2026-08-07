// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_INCLUDE_VPU_KERNELS_H_
#define GEMMINI_ROCC_TESTS_INCLUDE_VPU_KERNELS_H_

#include "vpu.h"

/*
 * Software-only nonlinear scheduling policy. This does not change generated
 * RTL. Applications may override it before including this header; the helper
 * below still clamps it to the VSRAM, DMA, and hardware-loop limits exported
 * by vpu_params.h.
 */
#ifndef VPU_NONLINEAR_CHUNK_ROWS
#define VPU_NONLINEAR_CHUNK_ROWS 4u
#endif

#if VPU_NONLINEAR_CHUNK_ROWS == 0
#error "VPU_NONLINEAR_CHUNK_ROWS must be at least one"
#endif

/*
 * These helpers emit only fine-grained VPU instructions. Inputs must already
 * reside in Vector SRAM and the caller remains responsible for DMA and fence.
 * Every GP argument names a GP register containing a VSRAM element address.
 */

static inline void vpu_emit_rmsnorm(
    unsigned input_gp, unsigned weight_gp, unsigned square_gp,
    unsigned normalized_gp, unsigned output_gp, unsigned accum_fp,
    unsigned inv_n_fp, unsigned epsilon_fp, size_t n, float epsilon) {
  if (n == 0) {
    return;
  }
  vpu_write_fp(accum_fp, 0.0f);
  vpu_write_fp(inv_n_fp, 1.0f / (float)n);
  vpu_write_fp(epsilon_fp, epsilon);
  vpu_v_mul_vv(square_gp, input_gp, input_gp);
  vpu_v_red_sum(accum_fp, square_gp);
  vpu_s_mul(accum_fp, accum_fp, inv_n_fp);
  vpu_s_add(accum_fp, accum_fp, epsilon_fp);
  vpu_s_sqrt(accum_fp, accum_fp);
  vpu_s_reci(accum_fp, accum_fp);
  vpu_v_mul_vf(normalized_gp, input_gp, accum_fp);
  vpu_v_mul_vv(output_gp, normalized_gp, weight_gp);
}

static inline void vpu_emit_silu(
    unsigned input_gp, unsigned neg_gp, unsigned denominator_gp,
    unsigned output_gp, unsigned zero_fp, unsigned one_fp) {
  vpu_write_fp(zero_fp, 0.0f);
  vpu_write_fp(one_fp, 1.0f);
  vpu_v_sub_vf(neg_gp, input_gp, zero_fp, true);
  vpu_v_exp_v(denominator_gp, neg_gp);
  vpu_v_add_vf(neg_gp, denominator_gp, one_fp);
  vpu_v_reci_v(denominator_gp, neg_gp);
  vpu_v_mul_vv(output_gp, input_gp, denominator_gp);
}

static inline void vpu_emit_swiglu(
    unsigned gate_gp, unsigned up_gp, unsigned neg_gp,
    unsigned denominator_gp, unsigned output_gp, unsigned zero_fp,
    unsigned one_fp) {
  vpu_write_fp(zero_fp, 0.0f);
  vpu_write_fp(one_fp, 1.0f);
  vpu_v_sub_vf(neg_gp, gate_gp, zero_fp, true);
  vpu_v_exp_v(denominator_gp, neg_gp);
  vpu_v_add_vf(neg_gp, denominator_gp, one_fp);
  vpu_v_reci_v(denominator_gp, neg_gp);
  vpu_v_mul_vv(neg_gp, gate_gp, denominator_gp);
  vpu_v_mul_vv(output_gp, neg_gp, up_gp);
}

static inline void vpu_emit_softmax(
    unsigned input_gp, unsigned shifted_gp, unsigned exponential_gp,
    unsigned output_gp, unsigned max_fp, unsigned sum_fp) {
  vpu_write_fp_bits(max_fp, 0xff800000u);  // -infinity
  vpu_write_fp(sum_fp, 0.0f);
  vpu_v_red_max(max_fp, input_gp);
  vpu_v_sub_vf(shifted_gp, input_gp, max_fp, false);
  vpu_v_exp_v(exponential_gp, shifted_gp);
  vpu_v_red_sum(sum_fp, exponential_gp);
  vpu_s_reci(sum_fp, sum_fp);
  vpu_v_mul_vf(output_gp, exponential_gp, sum_fp);
}

/*
 * Software-managed streaming layout used by the multi-tile helpers below.
 * A ping or pong tile owns four banks. The meaning of AUX/TEMP depends on the
 * kernel (weight, shifted value, exponential, or activation temporary).
 *
 * These helpers deliberately emit only the fine-grained instructions above;
 * they are not new architectural/coarse-grained operations. They clobber GP
 * registers 0..12, H registers 0..2, and FP registers 0..2.
 */
#if VPU_VSPAD_BANKS < 8
#error "The streaming VPU kernels require at least eight Vector SRAM banks"
#endif

#if VPU_VLEN > VPU_ADDI_INT_IMM_MAX
#error "The auto kernels require VLEN to fit the S_ADDI_INT immediate"
#endif

/*
 * Software schedule selected from the generated scratchpad geometry and a
 * kernel's live set.  A resident tile remains in VSRAM across global passes;
 * a partial schedule caches as many leading tiles as possible and streams
 * only the overflow.  The streaming schedule is the original ping/pong
 * implementation below.
 *
 * These are software plans, not new architectural operations.  They are
 * public so benchmarks can report which path an auto-scheduled kernel used.
 */
enum vpu_tiled_kernel {
  VPU_TILED_KERNEL_RMSNORM = 0,
  VPU_TILED_KERNEL_FINAL_NORM = 1,
  VPU_TILED_KERNEL_SILU = 2,
  VPU_TILED_KERNEL_SWIGLU = 3,
  VPU_TILED_KERNEL_SOFTMAX = 4,
};

enum vpu_tiled_schedule {
  VPU_TILED_SCHEDULE_EMPTY = 0,
  VPU_TILED_SCHEDULE_STREAMING = 1,
  VPU_TILED_SCHEDULE_RESIDENT_FULL = 2,
  VPU_TILED_SCHEDULE_RESIDENT_PARTIAL = 3,
  VPU_TILED_SCHEDULE_ILLEGAL = 4,
};

struct vpu_tiled_plan {
  enum vpu_tiled_kernel kernel;
  enum vpu_tiled_schedule schedule;
  size_t elements;
  size_t tile_count;
  size_t resident_tile_capacity;
  size_t resident_tiles;
  unsigned resident_banks;
  unsigned workspace_banks;
};

static inline struct vpu_tiled_plan vpu_tiled_plan_for(
    enum vpu_tiled_kernel kernel, size_t elements) {
  struct vpu_tiled_plan plan;
  plan.kernel = kernel;
  plan.schedule = VPU_TILED_SCHEDULE_STREAMING;
  plan.elements = elements;
  plan.tile_count = 0;
  plan.resident_tile_capacity = 0;
  plan.resident_tiles = 0;
  plan.resident_banks = 0;
  plan.workspace_banks = VPU_VSPAD_BANKS;

  if (kernel < VPU_TILED_KERNEL_RMSNORM ||
      kernel > VPU_TILED_KERNEL_SOFTMAX) {
    plan.schedule = VPU_TILED_SCHEDULE_ILLEGAL;
    return plan;
  }
  if (elements == 0) {
    plan.schedule = VPU_TILED_SCHEDULE_EMPTY;
    return plan;
  }
  if (elements > UINT32_MAX) {
    plan.schedule = VPU_TILED_SCHEDULE_ILLEGAL;
    return plan;
  }
  plan.tile_count = (elements + VPU_VLEN - 1u) / VPU_VLEN;

  /* SiLU/SwiGLU already consume each input once, so their standalone E2E
   * auto schedule keeps the load/execute/store-overlapped ping/pong path. */
  const int supports_residency =
      kernel == VPU_TILED_KERNEL_RMSNORM ||
      kernel == VPU_TILED_KERNEL_FINAL_NORM ||
      kernel == VPU_TILED_KERNEL_SOFTMAX;
  if (!supports_residency) {
    return plan;
  }

  /* Once a resident input tile is consumed, its slot can hold the result.
   * RMSNorm therefore reserves only weight and scratch banks for a full-fit
   * row; softmax reserves one shifted/output bank.  A partial plan reserves
   * one additional bank for overflow input/recompute. */
  const unsigned full_workspace =
      kernel == VPU_TILED_KERNEL_SOFTMAX ? 1u : 2u;
  if (VPU_VSPAD_BANKS > full_workspace) {
    const unsigned resident_banks = VPU_VSPAD_BANKS - full_workspace;
    const size_t capacity =
        (size_t)resident_banks * (size_t)VPU_SLOTS_PER_BANK;
    if (plan.tile_count <= capacity) {
      plan.schedule = VPU_TILED_SCHEDULE_RESIDENT_FULL;
      plan.resident_banks = resident_banks;
      plan.workspace_banks = full_workspace;
      plan.resident_tile_capacity = capacity;
      plan.resident_tiles = plan.tile_count;
      return plan;
    }
  }

  const unsigned partial_workspace =
      kernel == VPU_TILED_KERNEL_SOFTMAX ? 2u : 3u;
  if (VPU_VSPAD_BANKS > partial_workspace) {
    const unsigned resident_banks = VPU_VSPAD_BANKS - partial_workspace;
    const size_t capacity =
        (size_t)resident_banks * (size_t)VPU_SLOTS_PER_BANK;
    if (capacity != 0) {
      plan.schedule = VPU_TILED_SCHEDULE_RESIDENT_PARTIAL;
      plan.resident_banks = resident_banks;
      plan.workspace_banks = partial_workspace;
      plan.resident_tile_capacity = capacity;
      plan.resident_tiles =
          plan.tile_count < capacity ? plan.tile_count : capacity;
    }
  }
  return plan;
}

static inline int vpu_tiled_plan_is_resident(struct vpu_tiled_plan plan) {
  return plan.schedule == VPU_TILED_SCHEDULE_RESIDENT_FULL ||
         plan.schedule == VPU_TILED_SCHEDULE_RESIDENT_PARTIAL;
}

static inline int vpu_tiled_plan_is_fully_resident(
    struct vpu_tiled_plan plan) {
  return plan.schedule == VPU_TILED_SCHEDULE_RESIDENT_FULL;
}

static inline int vpu_tiled_resident_eligible(
    enum vpu_tiled_kernel kernel, size_t elements) {
  return vpu_tiled_plan_is_resident(vpu_tiled_plan_for(kernel, elements));
}

static inline size_t vpu_tiled_plan_resident_elements(
    struct vpu_tiled_plan plan) {
  const size_t capacity = plan.resident_tiles * (size_t)VPU_VLEN;
  return capacity < plan.elements ? capacity : plan.elements;
}

static inline size_t vpu_tiled_plan_overflow_elements(
    struct vpu_tiled_plan plan) {
  if (plan.schedule == VPU_TILED_SCHEDULE_EMPTY ||
      plan.schedule == VPU_TILED_SCHEDULE_ILLEGAL) {
    return 0;
  }
  if (!vpu_tiled_plan_is_resident(plan)) {
    return plan.elements;
  }
  return plan.elements - vpu_tiled_plan_resident_elements(plan);
}

/* Number of storage elements transferred by all DMA reads in the selected
 * schedule.  This includes weight/up reads, but not the one output write. */
static inline size_t vpu_tiled_plan_read_elements(
    struct vpu_tiled_plan plan) {
  if (plan.schedule == VPU_TILED_SCHEDULE_EMPTY ||
      plan.schedule == VPU_TILED_SCHEDULE_ILLEGAL) {
    return 0;
  }
  if (!vpu_tiled_plan_is_resident(plan)) {
    switch (plan.kernel) {
      case VPU_TILED_KERNEL_RMSNORM:
      case VPU_TILED_KERNEL_FINAL_NORM:
      case VPU_TILED_KERNEL_SOFTMAX:
        return 3u * plan.elements;
      case VPU_TILED_KERNEL_SWIGLU:
        return 2u * plan.elements;
      case VPU_TILED_KERNEL_SILU:
      default:
        return plan.elements;
    }
  }
  const size_t overflow = vpu_tiled_plan_overflow_elements(plan);
  if (plan.kernel == VPU_TILED_KERNEL_SOFTMAX) {
    return plan.elements + 2u * overflow;
  }
  /* RMSNorm reads x once, reloads only overflow x, and reads every weight. */
  return 2u * plan.elements + overflow;
}

static inline const char *vpu_tiled_schedule_name(
    enum vpu_tiled_schedule schedule) {
  switch (schedule) {
    case VPU_TILED_SCHEDULE_EMPTY: return "empty";
    case VPU_TILED_SCHEDULE_STREAMING: return "streaming";
    case VPU_TILED_SCHEDULE_RESIDENT_FULL: return "resident-full";
    case VPU_TILED_SCHEDULE_RESIDENT_PARTIAL: return "resident-partial";
    case VPU_TILED_SCHEDULE_ILLEGAL: return "illegal";
    default: return "unknown";
  }
}

/* Fill every slot in one resident bank before advancing to the next bank.
 * Callers use this helper only for tile < resident_tiles. */
static inline unsigned vpu_tiled_resident_address(
    struct vpu_tiled_plan plan, size_t tile) {
  (void)plan;  /* The caller-checked plan bounds tile to the resident prefix. */
  const unsigned bank = (unsigned)(tile / VPU_SLOTS_PER_BANK);
  const unsigned slot = (unsigned)(tile % VPU_SLOTS_PER_BANK);
  return VPU_SLOT_ADDR(bank, slot);
}

static inline unsigned vpu_tiled_stream_address(unsigned bank, size_t tile) {
  return VPU_SLOT_ADDR(bank, (unsigned)(tile % VPU_SLOTS_PER_BANK));
}

enum vpu_stream_register {
  VPU_STREAM_GP_PING_INPUT = 0,
  VPU_STREAM_GP_PING_AUX = 1,
  VPU_STREAM_GP_PING_TEMP = 2,
  VPU_STREAM_GP_PING_OUTPUT = 3,
  VPU_STREAM_GP_PONG_INPUT = 4,
  VPU_STREAM_GP_PONG_AUX = 5,
  VPU_STREAM_GP_PONG_TEMP = 6,
  VPU_STREAM_GP_PONG_OUTPUT = 7,
  VPU_STREAM_GP_PING_OFFSET = 8,
  VPU_STREAM_GP_PONG_OFFSET = 9,
  VPU_STREAM_GP_PING_ROWS = 10,
  VPU_STREAM_GP_PONG_ROWS = 11,
  VPU_STREAM_GP_LOOP = 12,
  VPU_STREAM_H_INPUT = 0,
  VPU_STREAM_H_AUX = 1,
  VPU_STREAM_H_OUTPUT = 2,
  VPU_STREAM_FP_ACCUM_OR_MAX = 0,
  VPU_STREAM_FP_CONSTANT_OR_SUM = 1,
  VPU_STREAM_FP_EPSILON = 2,
};

static inline unsigned vpu_stream_input_gp(unsigned buffer) {
  return buffer ? VPU_STREAM_GP_PONG_INPUT : VPU_STREAM_GP_PING_INPUT;
}

static inline unsigned vpu_stream_aux_gp(unsigned buffer) {
  return buffer ? VPU_STREAM_GP_PONG_AUX : VPU_STREAM_GP_PING_AUX;
}

static inline unsigned vpu_stream_temp_gp(unsigned buffer) {
  return buffer ? VPU_STREAM_GP_PONG_TEMP : VPU_STREAM_GP_PING_TEMP;
}

static inline unsigned vpu_stream_output_gp(unsigned buffer) {
  return buffer ? VPU_STREAM_GP_PONG_OUTPUT : VPU_STREAM_GP_PING_OUTPUT;
}

static inline unsigned vpu_stream_offset_gp(unsigned buffer) {
  return buffer ? VPU_STREAM_GP_PONG_OFFSET : VPU_STREAM_GP_PING_OFFSET;
}

static inline unsigned vpu_stream_rows_gp(unsigned buffer) {
  return buffer ? VPU_STREAM_GP_PONG_ROWS : VPU_STREAM_GP_PING_ROWS;
}

/* Physical rows available to one bank-local 2-D descriptor.  Rearrangement
 * kernels use this capacity because their large row counts already provide a
 * steady ping/pong pipeline and their gather/slide bodies dominate runtime. */
static inline size_t vpu_stream_physical_batch_capacity(void) {
  const size_t dma_capacity = VPU_SLOTS_PER_BANK < VPU_DMA_MAX_ROWS
      ? VPU_SLOTS_PER_BANK : VPU_DMA_MAX_ROWS;
  return dma_capacity < VPU_LOOP_COUNT_MAX
      ? dma_capacity : VPU_LOOP_COUNT_MAX;
}

/* Nonlinear kernels deliberately use a smaller dependency/completion unit than
 * the physical bank.  A long 2-D load retires only after its final SRAM row and
 * exposes one conservative write range to the reservation station; limiting
 * it here lets LD(chunk n+1), EX(chunk n), and ST(chunk n-1) overlap. */
static inline size_t vpu_stream_batch_capacity(void) {
  const size_t physical = vpu_stream_physical_batch_capacity();
  return physical < VPU_NONLINEAR_CHUNK_ROWS
      ? physical : (size_t)VPU_NONLINEAR_CHUNK_ROWS;
}

/* Limit a run both to the nonlinear chunk target and to the remaining slots in
 * the selected bank.  The bank-boundary term is required by the 2-D DMA ABI. */
static inline size_t vpu_tiled_chunk_rows(size_t first_tile,
                                          size_t remaining_tiles) {
  const size_t bank_remaining = VPU_SLOTS_PER_BANK -
      (first_tile % VPU_SLOTS_PER_BANK);
  size_t rows = remaining_tiles < bank_remaining
      ? remaining_tiles : bank_remaining;
  const size_t capacity = vpu_stream_batch_capacity();
  return rows < capacity ? rows : capacity;
}

static inline size_t vpu_stream_tile_count(size_t elements) {
  return elements == 0 ? 0 : (elements + VPU_VLEN - 1u) / VPU_VLEN;
}

static inline size_t vpu_stream_tile_elements(size_t elements, size_t tile) {
  const size_t offset = tile * VPU_VLEN;
  const size_t remaining = elements - offset;
  return remaining < VPU_VLEN ? remaining : VPU_VLEN;
}

static inline void vpu_stream_configure_spad(void) {
  vpu_write_gp(VPU_STREAM_GP_PING_INPUT, VPU_BANK_BASE(0));
  vpu_write_gp(VPU_STREAM_GP_PING_AUX, VPU_BANK_BASE(1));
  vpu_write_gp(VPU_STREAM_GP_PING_TEMP, VPU_BANK_BASE(2));
  vpu_write_gp(VPU_STREAM_GP_PING_OUTPUT, VPU_BANK_BASE(3));
  vpu_write_gp(VPU_STREAM_GP_PONG_INPUT, VPU_BANK_BASE(4));
  vpu_write_gp(VPU_STREAM_GP_PONG_AUX, VPU_BANK_BASE(5));
  vpu_write_gp(VPU_STREAM_GP_PONG_TEMP, VPU_BANK_BASE(6));
  vpu_write_gp(VPU_STREAM_GP_PONG_OUTPUT, VPU_BANK_BASE(7));
}

static inline void vpu_stream_configure_memory(const vpu_storage_t *input,
                                                const vpu_storage_t *aux,
                                                vpu_storage_t *output) {
  /* Publish all software-produced tiles once before asynchronous prefetches. */
  vpu_publish_cpu_writes();
  vpu_write_h(VPU_STREAM_H_INPUT, (uintptr_t)input);
  vpu_write_h(VPU_STREAM_H_AUX, (uintptr_t)aux);
  vpu_write_h(VPU_STREAM_H_OUTPUT, (uintptr_t)output);
}

static inline void vpu_stream_select_tile(size_t elements, size_t tile) {
  const unsigned buffer = (unsigned)(tile & 1u);
  vpu_write_gp(vpu_stream_offset_gp(buffer),
               (uint32_t)(tile * VPU_VLEN));
  vpu_set_vl(vpu_stream_tile_elements(elements, tile));
}

static inline void vpu_stream_prefetch_input(size_t elements, size_t tile) {
  const unsigned buffer = (unsigned)(tile & 1u);
  vpu_stream_select_tile(elements, tile);
  vpu_h_prefetch_v(vpu_stream_input_gp(buffer),
                   vpu_stream_offset_gp(buffer), VPU_STREAM_H_INPUT);
}

/*
 * A full-width streaming batch occupies consecutive VLEN slots in one bank.
 * One 2-D descriptor moves the complete batch and a hardware loop walks its
 * rows by adding VLEN to the VSRAM address registers.  The final partial tile
 * is deliberately peeled because every row in one descriptor snapshots the
 * same VL.
 */
static inline void vpu_stream_reset_batch_addresses(unsigned buffer) {
  vpu_write_gp(vpu_stream_input_gp(buffer),
               VPU_BANK_BASE(buffer ? 4u : 0u));
  vpu_write_gp(vpu_stream_aux_gp(buffer),
               VPU_BANK_BASE(buffer ? 5u : 1u));
  vpu_write_gp(vpu_stream_temp_gp(buffer),
               VPU_BANK_BASE(buffer ? 6u : 2u));
  vpu_write_gp(vpu_stream_output_gp(buffer),
               VPU_BANK_BASE(buffer ? 7u : 3u));
}

static inline void vpu_stream_prepare_full_batch(
    unsigned buffer, size_t first_tile, size_t rows) {
  vpu_stream_reset_batch_addresses(buffer);
  vpu_write_gp(vpu_stream_offset_gp(buffer),
               (uint32_t)(first_tile * (size_t)VPU_VLEN));
  vpu_write_gp(vpu_stream_rows_gp(buffer), (uint32_t)rows);
}

static inline void vpu_stream_prefetch_full_batch(
    unsigned buffer, size_t first_tile, size_t rows, int load_aux) {
  vpu_stream_prepare_full_batch(buffer, first_tile, rows);
  vpu_h_prefetch_v_2d(vpu_stream_input_gp(buffer),
                      vpu_stream_offset_gp(buffer), VPU_STREAM_H_INPUT,
                      vpu_stream_rows_gp(buffer));
  if (load_aux) {
    vpu_h_prefetch_v_2d(vpu_stream_aux_gp(buffer),
                        vpu_stream_offset_gp(buffer), VPU_STREAM_H_AUX,
                        vpu_stream_rows_gp(buffer));
  }
}

static inline void vpu_stream_store_full_batch(
    unsigned buffer, size_t first_tile, size_t rows) {
  /* Execute address induction leaves output_gp one-past the batch.  Restore
   * the descriptor base explicitly; descriptors snapshot it at admission. */
  vpu_write_gp(vpu_stream_output_gp(buffer),
               VPU_BANK_BASE(buffer ? 7u : 3u));
  vpu_write_gp(vpu_stream_offset_gp(buffer),
               (uint32_t)(first_tile * (size_t)VPU_VLEN));
  vpu_write_gp(vpu_stream_rows_gp(buffer), (uint32_t)rows);
  vpu_h_store_v_2d(vpu_stream_output_gp(buffer),
                   vpu_stream_offset_gp(buffer), VPU_STREAM_H_OUTPUT,
                   vpu_stream_rows_gp(buffer));
}

static inline uint64_t vpu_stream_empty(void) {
  vpu_set_vl(0);
  return vpu_fence();
}

/* Register contract used by the geometry-driven resident schedules. */
enum vpu_resident_register {
  VPU_RESIDENT_GP_INPUT = 0,
  VPU_RESIDENT_GP_AUX = 1,
  VPU_RESIDENT_GP_SCRATCH = 2,
  VPU_RESIDENT_GP_OUTPUT = 3,
  VPU_RESIDENT_GP_OFFSET = 4,
  VPU_RESIDENT_GP_ROWS = 5,
  VPU_RESIDENT_GP_LOOP = 6,
};

static inline size_t vpu_tiled_tile_offset(size_t tile) {
  return tile * (size_t)VPU_VLEN;
}

static inline void vpu_tiled_write_tile_address(
    unsigned gp, unsigned address) {
  vpu_write_gp(gp, address);
}

static inline void vpu_tiled_prefetch_address(
    unsigned address_gp, unsigned address, unsigned base_h,
    size_t elements, size_t tile) {
  vpu_tiled_write_tile_address(address_gp, address);
  vpu_write_gp(VPU_RESIDENT_GP_OFFSET,
               (uint32_t)vpu_tiled_tile_offset(tile));
  vpu_set_vl(vpu_stream_tile_elements(elements, tile));
  vpu_h_prefetch_v(address_gp, VPU_RESIDENT_GP_OFFSET, base_h);
}

static inline void vpu_tiled_store_address(
    unsigned address_gp, unsigned address, size_t elements, size_t tile) {
  vpu_tiled_write_tile_address(address_gp, address);
  vpu_write_gp(VPU_RESIDENT_GP_OFFSET,
               (uint32_t)vpu_tiled_tile_offset(tile));
  vpu_set_vl(vpu_stream_tile_elements(elements, tile));
  vpu_h_store_v(address_gp, VPU_RESIDENT_GP_OFFSET, VPU_STREAM_H_OUTPUT);
}

static inline void vpu_tiled_prepare_run(
    unsigned address_gp, unsigned address, size_t first_tile,
    size_t rows, size_t vl) {
  vpu_write_gp(address_gp, address);
  vpu_write_gp(VPU_RESIDENT_GP_OFFSET,
               (uint32_t)(first_tile * (size_t)VPU_VLEN));
  vpu_write_gp(VPU_RESIDENT_GP_ROWS, (uint32_t)rows);
  vpu_set_vl(vl);
}

static inline void vpu_tiled_prefetch_run(
    unsigned address_gp, unsigned address, unsigned base_h,
    size_t first_tile, size_t rows, size_t vl) {
  vpu_tiled_prepare_run(address_gp, address, first_tile, rows, vl);
  vpu_h_prefetch_v_2d(address_gp, VPU_RESIDENT_GP_OFFSET, base_h,
                      VPU_RESIDENT_GP_ROWS);
}

static inline void vpu_tiled_store_run(
    unsigned address_gp, unsigned address, size_t first_tile,
    size_t rows, size_t vl) {
  vpu_tiled_prepare_run(address_gp, address, first_tile, rows, vl);
  vpu_h_store_v_2d(address_gp, VPU_RESIDENT_GP_OFFSET,
                   VPU_STREAM_H_OUTPUT, VPU_RESIDENT_GP_ROWS);
}

static inline void vpu_tiled_emit_rms_reduce_run(
    unsigned input_address, unsigned scratch_address, size_t rows) {
  vpu_write_gp(VPU_RESIDENT_GP_INPUT, input_address);
  vpu_write_gp(VPU_RESIDENT_GP_SCRATCH, scratch_address);
#if VPU_LOOP_BUFFER_ENTRIES >= 6 && VPU_VLEN <= VPU_ADDI_INT_IMM_MAX
  if (rows > 1u) {
    vpu_loop_start(VPU_RESIDENT_GP_LOOP, (uint32_t)rows);
    vpu_v_mul_vv(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_RESIDENT_GP_INPUT);
    vpu_v_red_sum(VPU_STREAM_FP_ACCUM_OR_MAX,
                  VPU_RESIDENT_GP_SCRATCH);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_SCRATCH,
                   VPU_RESIDENT_GP_SCRATCH, VPU_VLEN);
    vpu_loop_end(VPU_RESIDENT_GP_LOOP);
    return;
  }
#endif
  for (size_t row = 0; row < rows; ++row) {
    vpu_v_mul_vv(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_RESIDENT_GP_INPUT);
    vpu_v_red_sum(VPU_STREAM_FP_ACCUM_OR_MAX,
                  VPU_RESIDENT_GP_SCRATCH);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_SCRATCH,
                   VPU_RESIDENT_GP_SCRATCH, VPU_VLEN);
  }
}

static inline void vpu_tiled_emit_rms_output_run(
    unsigned input_address, unsigned weight_address,
    unsigned scratch_address, unsigned output_address, size_t rows) {
  vpu_write_gp(VPU_RESIDENT_GP_INPUT, input_address);
  vpu_write_gp(VPU_RESIDENT_GP_AUX, weight_address);
  vpu_write_gp(VPU_RESIDENT_GP_SCRATCH, scratch_address);
  vpu_write_gp(VPU_RESIDENT_GP_OUTPUT, output_address);
#if VPU_LOOP_BUFFER_ENTRIES >= 8 && VPU_VLEN <= VPU_ADDI_INT_IMM_MAX
  if (rows > 1u) {
    vpu_loop_start(VPU_RESIDENT_GP_LOOP, (uint32_t)rows);
    vpu_v_mul_vf(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_ACCUM_OR_MAX);
    vpu_v_mul_vv(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_SCRATCH,
                 VPU_RESIDENT_GP_AUX);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_AUX,
                   VPU_RESIDENT_GP_AUX, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_SCRATCH,
                   VPU_RESIDENT_GP_SCRATCH, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_OUTPUT,
                   VPU_RESIDENT_GP_OUTPUT, VPU_VLEN);
    vpu_loop_end(VPU_RESIDENT_GP_LOOP);
    return;
  }
#endif
  for (size_t row = 0; row < rows; ++row) {
    vpu_v_mul_vf(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_ACCUM_OR_MAX);
    vpu_v_mul_vv(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_SCRATCH,
                 VPU_RESIDENT_GP_AUX);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_AUX,
                   VPU_RESIDENT_GP_AUX, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_SCRATCH,
                   VPU_RESIDENT_GP_SCRATCH, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_OUTPUT,
                   VPU_RESIDENT_GP_OUTPUT, VPU_VLEN);
  }
}

static inline void vpu_tiled_emit_red_max_run(
    unsigned input_address, size_t rows) {
  vpu_write_gp(VPU_RESIDENT_GP_INPUT, input_address);
#if VPU_LOOP_BUFFER_ENTRIES >= 4 && VPU_VLEN <= VPU_ADDI_INT_IMM_MAX
  if (rows > 1u) {
    vpu_loop_start(VPU_RESIDENT_GP_LOOP, (uint32_t)rows);
    vpu_v_red_max(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_RESIDENT_GP_INPUT);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_loop_end(VPU_RESIDENT_GP_LOOP);
    return;
  }
#endif
  for (size_t row = 0; row < rows; ++row) {
    vpu_v_red_max(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_RESIDENT_GP_INPUT);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
  }
}

static inline void vpu_tiled_emit_exp_sum_run(
    unsigned input_address, unsigned scratch_address,
    unsigned exponential_address, size_t rows) {
  vpu_write_gp(VPU_RESIDENT_GP_INPUT, input_address);
  vpu_write_gp(VPU_RESIDENT_GP_SCRATCH, scratch_address);
  vpu_write_gp(VPU_RESIDENT_GP_OUTPUT, exponential_address);
#if VPU_LOOP_BUFFER_ENTRIES >= 8 && VPU_VLEN <= VPU_ADDI_INT_IMM_MAX
  if (rows > 1u) {
    vpu_loop_start(VPU_RESIDENT_GP_LOOP, (uint32_t)rows);
    vpu_v_sub_vf(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_ACCUM_OR_MAX, false);
    vpu_v_exp_v(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_SCRATCH);
    vpu_v_red_sum(VPU_STREAM_FP_CONSTANT_OR_SUM,
                  VPU_RESIDENT_GP_OUTPUT);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_SCRATCH,
                   VPU_RESIDENT_GP_SCRATCH, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_OUTPUT,
                   VPU_RESIDENT_GP_OUTPUT, VPU_VLEN);
    vpu_loop_end(VPU_RESIDENT_GP_LOOP);
    return;
  }
#endif
  for (size_t row = 0; row < rows; ++row) {
    vpu_v_sub_vf(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_ACCUM_OR_MAX, false);
    vpu_v_exp_v(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_SCRATCH);
    vpu_v_red_sum(VPU_STREAM_FP_CONSTANT_OR_SUM,
                  VPU_RESIDENT_GP_OUTPUT);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_SCRATCH,
                   VPU_RESIDENT_GP_SCRATCH, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_OUTPUT,
                   VPU_RESIDENT_GP_OUTPUT, VPU_VLEN);
  }
}

static inline void vpu_tiled_emit_softmax_scale_run(
    unsigned exponential_address, unsigned output_address, size_t rows) {
  vpu_write_gp(VPU_RESIDENT_GP_INPUT, exponential_address);
  vpu_write_gp(VPU_RESIDENT_GP_OUTPUT, output_address);
#if VPU_LOOP_BUFFER_ENTRIES >= 5 && VPU_VLEN <= VPU_ADDI_INT_IMM_MAX
  if (rows > 1u) {
    vpu_loop_start(VPU_RESIDENT_GP_LOOP, (uint32_t)rows);
    vpu_v_mul_vf(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_CONSTANT_OR_SUM);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_OUTPUT,
                   VPU_RESIDENT_GP_OUTPUT, VPU_VLEN);
    vpu_loop_end(VPU_RESIDENT_GP_LOOP);
    return;
  }
#endif
  for (size_t row = 0; row < rows; ++row) {
    vpu_v_mul_vf(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_CONSTANT_OR_SUM);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_OUTPUT,
                   VPU_RESIDENT_GP_OUTPUT, VPU_VLEN);
  }
}

static inline void vpu_tiled_emit_softmax_recompute_run(
    unsigned input_address, unsigned scratch_address,
    unsigned output_address, size_t rows) {
  vpu_write_gp(VPU_RESIDENT_GP_INPUT, input_address);
  vpu_write_gp(VPU_RESIDENT_GP_SCRATCH, scratch_address);
  vpu_write_gp(VPU_RESIDENT_GP_OUTPUT, output_address);
#if VPU_LOOP_BUFFER_ENTRIES >= 8 && VPU_VLEN <= VPU_ADDI_INT_IMM_MAX
  if (rows > 1u) {
    vpu_loop_start(VPU_RESIDENT_GP_LOOP, (uint32_t)rows);
    vpu_v_sub_vf(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_ACCUM_OR_MAX, false);
    vpu_v_exp_v(VPU_RESIDENT_GP_INPUT, VPU_RESIDENT_GP_SCRATCH);
    vpu_v_mul_vf(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_CONSTANT_OR_SUM);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_SCRATCH,
                   VPU_RESIDENT_GP_SCRATCH, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_OUTPUT,
                   VPU_RESIDENT_GP_OUTPUT, VPU_VLEN);
    vpu_loop_end(VPU_RESIDENT_GP_LOOP);
    return;
  }
#endif
  for (size_t row = 0; row < rows; ++row) {
    vpu_v_sub_vf(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_ACCUM_OR_MAX, false);
    vpu_v_exp_v(VPU_RESIDENT_GP_INPUT, VPU_RESIDENT_GP_SCRATCH);
    vpu_v_mul_vf(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_CONSTANT_OR_SUM);
    vpu_s_addi_int(VPU_RESIDENT_GP_INPUT,
                   VPU_RESIDENT_GP_INPUT, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_SCRATCH,
                   VPU_RESIDENT_GP_SCRATCH, VPU_VLEN);
    vpu_s_addi_int(VPU_RESIDENT_GP_OUTPUT,
                   VPU_RESIDENT_GP_OUTPUT, VPU_VLEN);
  }
}

/*
 * Geometry-driven resident/hybrid RMSNorm.
 *
 * Full plan: all x tiles are packed over the resident banks.  The remaining
 * two banks stream weights and hold scratch rows.  If the row does not consume
 * every nominal resident bank, the first unused one becomes an output ring;
 * a maximum-capacity row safely falls back to overwriting each dead x slot.
 *
 * Partial plan: the leading resident_tiles remain packed in VSRAM.  Pass-one
 * overflow streams through the first workspace bank.  Pass two uses that bank
 * for weights, the next for scratch, and the last for output; overflow x is
 * reloaded into bank zero after the cached prefix has been consumed.  Hazard
 * tracking makes reuse safe even while the final cached commands drain.
 */
static inline uint64_t vpu_tiled_rmsnorm_resident(
    const vpu_storage_t *input, const vpu_storage_t *weight,
    vpu_storage_t *output, size_t elements, float epsilon,
    struct vpu_tiled_plan plan) {
  const int rms_plan = plan.kernel == VPU_TILED_KERNEL_RMSNORM ||
      plan.kernel == VPU_TILED_KERNEL_FINAL_NORM;
  if (!rms_plan || !vpu_tiled_plan_is_resident(plan) || elements == 0 ||
      elements > UINT32_MAX || plan.elements != elements) {
    return VPU_STATUS_ILLEGAL_COMMAND;
  }

  vpu_stream_configure_memory(input, weight, output);
  vpu_write_fp(VPU_STREAM_FP_ACCUM_OR_MAX, 0.0f);
  vpu_write_fp(VPU_STREAM_FP_CONSTANT_OR_SUM, 1.0f / (float)elements);
  vpu_write_fp(VPU_STREAM_FP_EPSILON, epsilon);
  vpu_set_stride_bytes((uint64_t)VPU_VLEN * VPU_STORAGE_BYTES);

  const unsigned first_workspace_bank = plan.resident_banks;
  const unsigned pass_one_stream_bank = first_workspace_bank;
  const unsigned scratch_bank = first_workspace_bank + 1u;
  const size_t full_tiles = elements / VPU_VLEN;
  const size_t tail = elements % VPU_VLEN;
  const size_t cached_full_tiles = plan.resident_tiles < full_tiles
      ? plan.resident_tiles : full_tiles;
  const unsigned resident_banks_used = (unsigned)(
      (plan.resident_tiles + VPU_SLOTS_PER_BANK - 1u) /
      VPU_SLOTS_PER_BANK);
  /* A short full-resident row often leaves one or more nominal resident banks
   * unused.  Borrow the first such bank for output so ST(chunk n) does not
   * contend with EX(chunk n+1) for the input bank's read port.  At the maximum
   * full-resident capacity there is no spare bank, so the safe in-place scheme
   * remains the fallback.  A partial plan already reserves an output bank. */
  const int has_separate_output =
      plan.schedule == VPU_TILED_SCHEDULE_RESIDENT_PARTIAL ||
      resident_banks_used < plan.resident_banks;
  const unsigned output_bank = plan.schedule ==
          VPU_TILED_SCHEDULE_RESIDENT_PARTIAL
      ? first_workspace_bank + 2u
      : (has_separate_output ? resident_banks_used : 0u);

  /* Retain the prefix and stream any overflow while accumulating sum(x*x).
   * Hardware-loop replay backpressures the RoCC input, so prime the current
   * descriptor and explicitly admit LD(next) before replaying EX(current).
   * Cached and overflow chunks use the same schedule across their boundary. */
  size_t tile = 0;
  if (full_tiles != 0) {
    const int cached = tile < cached_full_tiles;
    const size_t remaining = cached
        ? cached_full_tiles - tile : full_tiles - tile;
    const size_t rows = vpu_tiled_chunk_rows(tile, remaining);
    const unsigned input_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(pass_one_stream_bank, tile);
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                           VPU_STREAM_H_INPUT, tile, rows, VPU_VLEN);
  }
  while (tile < full_tiles) {
    const int cached = tile < cached_full_tiles;
    const size_t remaining = cached
        ? cached_full_tiles - tile : full_tiles - tile;
    const size_t rows = vpu_tiled_chunk_rows(tile, remaining);
    const size_t next_tile = tile + rows;
    const unsigned input_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(pass_one_stream_bank, tile);
    const unsigned scratch_address =
        vpu_tiled_stream_address(scratch_bank, tile);
    if (next_tile < full_tiles) {
      const int next_cached = next_tile < cached_full_tiles;
      const size_t next_remaining = next_cached
          ? cached_full_tiles - next_tile : full_tiles - next_tile;
      const size_t next_rows =
          vpu_tiled_chunk_rows(next_tile, next_remaining);
      const unsigned next_input_address = next_cached
          ? vpu_tiled_resident_address(plan, next_tile)
          : vpu_tiled_stream_address(pass_one_stream_bank, next_tile);
      vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, next_input_address,
                             VPU_STREAM_H_INPUT, next_tile, next_rows,
                             VPU_VLEN);
    }
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_rms_reduce_run(input_address, scratch_address, rows);
    tile = next_tile;
  }
  if (tail != 0) {
    const int cached = tile < plan.resident_tiles;
    const unsigned input_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(pass_one_stream_bank, tile);
    const unsigned scratch_address =
        vpu_tiled_stream_address(scratch_bank, tile);
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                           VPU_STREAM_H_INPUT, tile, 1, tail);
    vpu_tiled_emit_rms_reduce_run(input_address, scratch_address, 1);
  }

  vpu_s_mul(VPU_STREAM_FP_ACCUM_OR_MAX,
            VPU_STREAM_FP_ACCUM_OR_MAX,
            VPU_STREAM_FP_CONSTANT_OR_SUM);
  vpu_s_add(VPU_STREAM_FP_ACCUM_OR_MAX,
            VPU_STREAM_FP_ACCUM_OR_MAX, VPU_STREAM_FP_EPSILON);
  vpu_s_sqrt(VPU_STREAM_FP_ACCUM_OR_MAX,
             VPU_STREAM_FP_ACCUM_OR_MAX);
  vpu_s_reci(VPU_STREAM_FP_ACCUM_OR_MAX,
             VPU_STREAM_FP_ACCUM_OR_MAX);

  /* Pass two similarly primes its input/weight descriptors.  Cached input needs
   * only a weight load; overflow needs both descriptors.  All ring workspaces
   * use the same slot offset, avoiding false WAR/WAW edges between chunks. */
  tile = 0;
  if (full_tiles != 0) {
    const int cached = tile < cached_full_tiles;
    const size_t remaining = cached
        ? cached_full_tiles - tile : full_tiles - tile;
    const size_t rows = vpu_tiled_chunk_rows(tile, remaining);
    if (!cached) {
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_INPUT, vpu_tiled_stream_address(0u, tile),
          VPU_STREAM_H_INPUT, tile, rows, VPU_VLEN);
    }
    vpu_tiled_prefetch_run(
        VPU_RESIDENT_GP_AUX,
        vpu_tiled_stream_address(first_workspace_bank, tile),
        VPU_STREAM_H_AUX, tile, rows, VPU_VLEN);
  }
  while (tile < full_tiles) {
    const int cached = tile < cached_full_tiles;
    const size_t remaining = cached
        ? cached_full_tiles - tile : full_tiles - tile;
    const size_t rows = vpu_tiled_chunk_rows(tile, remaining);
    const size_t next_tile = tile + rows;
    const unsigned input_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(0u, tile);
    const unsigned weight_address =
        vpu_tiled_stream_address(first_workspace_bank, tile);
    const unsigned scratch_address =
        vpu_tiled_stream_address(scratch_bank, tile);
    const unsigned output_address =
        vpu_tiled_stream_address(output_bank, tile);
    const unsigned result_address = has_separate_output
        ? output_address : input_address;
    if (next_tile < full_tiles) {
      const int next_cached = next_tile < cached_full_tiles;
      const size_t next_remaining = next_cached
          ? cached_full_tiles - next_tile : full_tiles - next_tile;
      const size_t next_rows =
          vpu_tiled_chunk_rows(next_tile, next_remaining);
      if (!next_cached) {
        vpu_tiled_prefetch_run(
            VPU_RESIDENT_GP_INPUT,
            vpu_tiled_stream_address(0u, next_tile),
            VPU_STREAM_H_INPUT, next_tile, next_rows, VPU_VLEN);
      }
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_AUX,
          vpu_tiled_stream_address(first_workspace_bank, next_tile),
          VPU_STREAM_H_AUX, next_tile, next_rows, VPU_VLEN);
    }
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_rms_output_run(input_address, weight_address,
                                  scratch_address, result_address, rows);
    vpu_tiled_store_run(VPU_RESIDENT_GP_OUTPUT, result_address,
                        tile, rows, VPU_VLEN);
    tile = next_tile;
  }
  if (tail != 0) {
    const int cached = tile < plan.resident_tiles;
    const unsigned input_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(0u, tile);
    const unsigned weight_address =
        vpu_tiled_stream_address(first_workspace_bank, tile);
    const unsigned scratch_address =
        vpu_tiled_stream_address(scratch_bank, tile);
    const unsigned output_address =
        vpu_tiled_stream_address(output_bank, tile);
    const unsigned result_address = has_separate_output
        ? output_address : input_address;
    if (!cached) {
      vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                             VPU_STREAM_H_INPUT, tile, 1, tail);
    }
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_AUX, weight_address,
                           VPU_STREAM_H_AUX, tile, 1, tail);
    vpu_tiled_emit_rms_output_run(input_address, weight_address,
                                  scratch_address, result_address, 1);
    vpu_tiled_store_run(VPU_RESIDENT_GP_OUTPUT, result_address,
                        tile, 1, tail);
  }
  return vpu_fence();
}

/*
 * Two-pass RMSNorm/final-norm implementation:
 *   pass 1: stream x, materialize x*x per tile, accumulate FP32 sum;
 *   pass 2: reload x and weight, normalize, and store each output tile.
 */
static inline uint64_t vpu_tiled_rmsnorm_streaming(
    const vpu_storage_t *input, const vpu_storage_t *weight,
    vpu_storage_t *output, size_t elements, float epsilon) {
  if (elements > UINT32_MAX) {
    return VPU_STATUS_ILLEGAL_COMMAND;
  }
  vpu_stream_configure_spad();
  vpu_stream_configure_memory(input, weight, output);
  if (elements == 0) {
    return vpu_stream_empty();
  }

  const size_t full_tiles = elements / VPU_VLEN;
  const size_t tail = elements % VPU_VLEN;
  const size_t capacity = vpu_stream_batch_capacity();
  const size_t full_batches = full_tiles == 0
      ? 0 : 1u + (full_tiles - 1u) / capacity;
  vpu_write_fp(VPU_STREAM_FP_ACCUM_OR_MAX, 0.0f);
  vpu_write_fp(VPU_STREAM_FP_CONSTANT_OR_SUM, 1.0f / (float)elements);
  vpu_write_fp(VPU_STREAM_FP_EPSILON, epsilon);
  vpu_set_stride_bytes((uint64_t)VPU_VLEN * VPU_STORAGE_BYTES);

  if (full_batches != 0) {
    const size_t rows = full_tiles < capacity ? full_tiles : capacity;
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, VPU_BANK_BASE(0),
                           VPU_STREAM_H_INPUT, 0, rows, VPU_VLEN);
  }
  for (size_t batch = 0; batch < full_batches; ++batch) {
    const unsigned buffer = (unsigned)(batch & 1u);
    const size_t first_tile = batch * capacity;
    const size_t remaining = full_tiles - first_tile;
    const size_t rows = remaining < capacity ? remaining : capacity;
    if (batch + 1u < full_batches) {
      const size_t next_first = first_tile + rows;
      const size_t next_remaining = full_tiles - next_first;
      const size_t next_rows = next_remaining < capacity
          ? next_remaining : capacity;
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_INPUT, VPU_BANK_BASE(buffer ? 0u : 4u),
          VPU_STREAM_H_INPUT, next_first, next_rows, VPU_VLEN);
    }
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_rms_reduce_run(VPU_BANK_BASE(buffer ? 4u : 0u),
                                  VPU_BANK_BASE(buffer ? 6u : 2u), rows);
  }
  if (tail != 0) {
    const unsigned buffer = (unsigned)(full_batches & 1u);
    const unsigned input_address = VPU_BANK_BASE(buffer ? 4u : 0u);
    const unsigned scratch_address = VPU_BANK_BASE(buffer ? 6u : 2u);
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                           VPU_STREAM_H_INPUT, full_tiles, 1, tail);
    vpu_tiled_emit_rms_reduce_run(input_address, scratch_address, 1);
  }

  vpu_s_mul(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_STREAM_FP_ACCUM_OR_MAX,
            VPU_STREAM_FP_CONSTANT_OR_SUM);
  vpu_s_add(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_STREAM_FP_ACCUM_OR_MAX,
            VPU_STREAM_FP_EPSILON);
  vpu_s_sqrt(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_STREAM_FP_ACCUM_OR_MAX);
  vpu_s_reci(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_STREAM_FP_ACCUM_OR_MAX);

  if (full_batches != 0) {
    const size_t rows = full_tiles < capacity ? full_tiles : capacity;
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, VPU_BANK_BASE(0),
                           VPU_STREAM_H_INPUT, 0, rows, VPU_VLEN);
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_AUX, VPU_BANK_BASE(1),
                           VPU_STREAM_H_AUX, 0, rows, VPU_VLEN);
  }
  for (size_t batch = 0; batch < full_batches; ++batch) {
    const unsigned buffer = (unsigned)(batch & 1u);
    const size_t first_tile = batch * capacity;
    const size_t remaining = full_tiles - first_tile;
    const size_t rows = remaining < capacity ? remaining : capacity;
    if (batch + 1u < full_batches) {
      const size_t next_first = first_tile + rows;
      const size_t next_remaining = full_tiles - next_first;
      const size_t next_rows = next_remaining < capacity
          ? next_remaining : capacity;
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_INPUT, VPU_BANK_BASE(buffer ? 0u : 4u),
          VPU_STREAM_H_INPUT, next_first, next_rows, VPU_VLEN);
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_AUX, VPU_BANK_BASE(buffer ? 1u : 5u),
          VPU_STREAM_H_AUX, next_first, next_rows, VPU_VLEN);
    }
    const unsigned input_address = VPU_BANK_BASE(buffer ? 4u : 0u);
    const unsigned weight_address = VPU_BANK_BASE(buffer ? 5u : 1u);
    const unsigned scratch_address = VPU_BANK_BASE(buffer ? 6u : 2u);
    const unsigned output_address = VPU_BANK_BASE(buffer ? 7u : 3u);
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_rms_output_run(input_address, weight_address,
                                  scratch_address, output_address, rows);
    vpu_tiled_store_run(VPU_RESIDENT_GP_OUTPUT, output_address,
                        first_tile, rows, VPU_VLEN);
  }
  if (tail != 0) {
    const unsigned buffer = (unsigned)(full_batches & 1u);
    const unsigned input_address = VPU_BANK_BASE(buffer ? 4u : 0u);
    const unsigned weight_address = VPU_BANK_BASE(buffer ? 5u : 1u);
    const unsigned scratch_address = VPU_BANK_BASE(buffer ? 6u : 2u);
    const unsigned output_address = VPU_BANK_BASE(buffer ? 7u : 3u);
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                           VPU_STREAM_H_INPUT, full_tiles, 1, tail);
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_AUX, weight_address,
                           VPU_STREAM_H_AUX, full_tiles, 1, tail);
    vpu_tiled_emit_rms_output_run(input_address, weight_address,
                                  scratch_address, output_address, 1);
    vpu_tiled_store_run(VPU_RESIDENT_GP_OUTPUT, output_address,
                        full_tiles, 1, tail);
  }
  return vpu_fence();
}

static inline uint64_t vpu_tiled_rmsnorm(
    const vpu_storage_t *input, const vpu_storage_t *weight,
    vpu_storage_t *output, size_t elements, float epsilon) {
  const struct vpu_tiled_plan plan =
      vpu_tiled_plan_for(VPU_TILED_KERNEL_RMSNORM, elements);
  if (vpu_tiled_plan_is_resident(plan)) {
    return vpu_tiled_rmsnorm_resident(
        input, weight, output, elements, epsilon, plan);
  }
  return vpu_tiled_rmsnorm_streaming(
      input, weight, output, elements, epsilon);
}

static inline uint64_t vpu_tiled_rmsnorm_auto(
    const vpu_storage_t *input, const vpu_storage_t *weight,
    vpu_storage_t *output, size_t elements, float epsilon) {
  return vpu_tiled_rmsnorm(input, weight, output, elements, epsilon);
}

static inline uint64_t vpu_tiled_final_norm(
    const vpu_storage_t *input, const vpu_storage_t *weight,
    vpu_storage_t *output, size_t elements, float epsilon) {
  return vpu_tiled_rmsnorm(input, weight, output, elements, epsilon);
}

static inline uint64_t vpu_tiled_final_norm_auto(
    const vpu_storage_t *input, const vpu_storage_t *weight,
    vpu_storage_t *output, size_t elements, float epsilon) {
  return vpu_tiled_final_norm(input, weight, output, elements, epsilon);
}

static inline uint64_t vpu_stream_rmsnorm(
    const vpu_storage_t *input, const vpu_storage_t *weight,
    vpu_storage_t *output, size_t elements, float epsilon) {
  return vpu_tiled_rmsnorm(input, weight, output, elements, epsilon);
}

static inline uint64_t vpu_stream_final_norm(
    const vpu_storage_t *input, const vpu_storage_t *weight,
    vpu_storage_t *output, size_t elements, float epsilon) {
  return vpu_tiled_final_norm(input, weight, output, elements, epsilon);
}

enum vpu_stream_activation {
  VPU_STREAM_ACT_RELU = 0,
  VPU_STREAM_ACT_SIGMOID = 1,
  VPU_STREAM_ACT_TANH = 2,
  VPU_STREAM_ACT_GELU = 3,
  VPU_STREAM_ACT_SILU = 4,
  VPU_STREAM_ACT_SWIGLU = 5,
};

/* Emit one dynamic VLEN row. Constants are FP0=0, FP1=1, and FP2 is the
 * kernel coefficient (2 for tanh, 1.702 for sigmoid-GELU). */
static inline void vpu_stream_emit_activation_row(
    enum vpu_stream_activation activation, unsigned buffer) {
  const unsigned input_gp = vpu_stream_input_gp(buffer);
  const unsigned aux_gp = vpu_stream_aux_gp(buffer);
  const unsigned temp_gp = vpu_stream_temp_gp(buffer);
  const unsigned output_gp = vpu_stream_output_gp(buffer);

  switch (activation) {
    case VPU_STREAM_ACT_RELU:
      vpu_v_max_vf(output_gp, input_gp, VPU_STREAM_FP_ACCUM_OR_MAX);
      break;
    case VPU_STREAM_ACT_SIGMOID:
      vpu_v_sub_vf(aux_gp, input_gp, VPU_STREAM_FP_ACCUM_OR_MAX, true);
      vpu_v_exp_v(temp_gp, aux_gp);
      vpu_v_add_vf(aux_gp, temp_gp, VPU_STREAM_FP_CONSTANT_OR_SUM);
      vpu_v_reci_v(output_gp, aux_gp);
      break;
    case VPU_STREAM_ACT_TANH:
      /* tanh(x) = 2*sigmoid(2*x)-1. */
      vpu_v_mul_vf(aux_gp, input_gp, VPU_STREAM_FP_EPSILON);
      vpu_v_sub_vf(temp_gp, aux_gp, VPU_STREAM_FP_ACCUM_OR_MAX, true);
      vpu_v_exp_v(output_gp, temp_gp);
      vpu_v_add_vf(temp_gp, output_gp, VPU_STREAM_FP_CONSTANT_OR_SUM);
      vpu_v_reci_v(output_gp, temp_gp);
      vpu_v_mul_vf(output_gp, output_gp, VPU_STREAM_FP_EPSILON);
      vpu_v_sub_vf(output_gp, output_gp,
                   VPU_STREAM_FP_CONSTANT_OR_SUM, false);
      break;
    case VPU_STREAM_ACT_GELU:
      /* PLENA's inexpensive GELU approximation: x*sigmoid(1.702*x). */
      vpu_v_mul_vf(aux_gp, input_gp, VPU_STREAM_FP_EPSILON);
      vpu_v_sub_vf(temp_gp, aux_gp, VPU_STREAM_FP_ACCUM_OR_MAX, true);
      vpu_v_exp_v(output_gp, temp_gp);
      vpu_v_add_vf(aux_gp, output_gp, VPU_STREAM_FP_CONSTANT_OR_SUM);
      vpu_v_reci_v(output_gp, aux_gp);
      vpu_v_mul_vv(output_gp, input_gp, output_gp);
      break;
    case VPU_STREAM_ACT_SWIGLU:
      vpu_v_sub_vf(temp_gp, input_gp, VPU_STREAM_FP_ACCUM_OR_MAX, true);
      vpu_v_exp_v(output_gp, temp_gp);
      vpu_v_add_vf(temp_gp, output_gp, VPU_STREAM_FP_CONSTANT_OR_SUM);
      vpu_v_reci_v(output_gp, temp_gp);
      vpu_v_mul_vv(temp_gp, input_gp, output_gp);
      vpu_v_mul_vv(output_gp, temp_gp, aux_gp);
      break;
    case VPU_STREAM_ACT_SILU:
    default:
      vpu_v_sub_vf(aux_gp, input_gp, VPU_STREAM_FP_ACCUM_OR_MAX, true);
      vpu_v_exp_v(temp_gp, aux_gp);
      vpu_v_add_vf(aux_gp, temp_gp, VPU_STREAM_FP_CONSTANT_OR_SUM);
      vpu_v_reci_v(temp_gp, aux_gp);
      vpu_v_mul_vv(output_gp, input_gp, temp_gp);
      break;
  }
}

/* Advance the bank-local row pointers used by every multi-stage activation. */
static inline void vpu_stream_advance_activation_row(unsigned buffer) {
  vpu_s_addi_int(vpu_stream_input_gp(buffer),
                 vpu_stream_input_gp(buffer), VPU_VLEN);
  vpu_s_addi_int(vpu_stream_aux_gp(buffer),
                 vpu_stream_aux_gp(buffer), VPU_VLEN);
  vpu_s_addi_int(vpu_stream_temp_gp(buffer),
                 vpu_stream_temp_gp(buffer), VPU_VLEN);
  vpu_s_addi_int(vpu_stream_output_gp(buffer),
                 vpu_stream_output_gp(buffer), VPU_VLEN);
}

static inline void vpu_stream_emit_activation_batch(
    enum vpu_stream_activation activation, unsigned buffer, size_t rows) {
  vpu_stream_reset_batch_addresses(buffer);
#if VPU_LOOP_BUFFER_ENTRIES >= 5
  if (rows > 1u && activation == VPU_STREAM_ACT_RELU) {
    vpu_loop_start(VPU_STREAM_GP_LOOP, (uint32_t)rows);
    vpu_stream_emit_activation_row(activation, buffer);
    /* ReLU touches only input and output. Avoid two dead GP operations per
     * row so the replay frontend can feed useful work more often. */
    vpu_s_addi_int(vpu_stream_input_gp(buffer),
                   vpu_stream_input_gp(buffer), VPU_VLEN);
    vpu_s_addi_int(vpu_stream_output_gp(buffer),
                   vpu_stream_output_gp(buffer), VPU_VLEN);
    vpu_loop_end(VPU_STREAM_GP_LOOP);
    return;
  }
#endif
#if VPU_LOOP_BUFFER_ENTRIES >= 13
  if (rows > 1u) {
    vpu_loop_start(VPU_STREAM_GP_LOOP, (uint32_t)rows);
    vpu_stream_emit_activation_row(activation, buffer);
    vpu_stream_advance_activation_row(buffer);
    vpu_loop_end(VPU_STREAM_GP_LOOP);
    return;
  }
#endif
  for (size_t row = 0; row < rows; ++row) {
    vpu_stream_emit_activation_row(activation, buffer);
    if (activation == VPU_STREAM_ACT_RELU) {
      vpu_s_addi_int(vpu_stream_input_gp(buffer),
                     vpu_stream_input_gp(buffer), VPU_VLEN);
      vpu_s_addi_int(vpu_stream_output_gp(buffer),
                     vpu_stream_output_gp(buffer), VPU_VLEN);
    } else {
      vpu_stream_advance_activation_row(buffer);
    }
  }
}

static inline uint64_t vpu_stream_activation_auto(
    const vpu_storage_t *input, const vpu_storage_t *aux,
    vpu_storage_t *output, size_t elements,
    enum vpu_stream_activation activation) {
  if (elements > UINT32_MAX) {
    return VPU_STATUS_ILLEGAL_COMMAND;
  }
  const int load_aux = activation == VPU_STREAM_ACT_SWIGLU;
  vpu_stream_configure_spad();
  vpu_stream_configure_memory(input, load_aux ? aux : input, output);
  if (elements == 0) {
    return vpu_stream_empty();
  }

  vpu_write_fp(VPU_STREAM_FP_ACCUM_OR_MAX, 0.0f);
  vpu_write_fp(VPU_STREAM_FP_CONSTANT_OR_SUM, 1.0f);
  vpu_write_fp(VPU_STREAM_FP_EPSILON,
               activation == VPU_STREAM_ACT_GELU ? 1.702f : 2.0f);

  const size_t full_tiles = elements / VPU_VLEN;
  const size_t tail = elements % VPU_VLEN;
  const size_t capacity = vpu_stream_batch_capacity();
  const size_t full_batches = full_tiles == 0
      ? 0 : 1u + (full_tiles - 1u) / capacity;
  vpu_set_stride_bytes((uint64_t)VPU_VLEN * VPU_STORAGE_BYTES);
  vpu_set_vl(VPU_VLEN);

  if (full_batches != 0) {
    const size_t first_rows = full_tiles < capacity ? full_tiles : capacity;
    vpu_stream_prefetch_full_batch(0, 0, first_rows, load_aux);
  }
  for (size_t batch = 0; batch < full_batches; ++batch) {
    const unsigned buffer = (unsigned)(batch & 1u);
    const size_t first_tile = batch * capacity;
    const size_t remaining = full_tiles - first_tile;
    const size_t rows = remaining < capacity ? remaining : capacity;
    if (batch + 1u < full_batches) {
      const size_t next_first = first_tile + rows;
      const size_t next_remaining = full_tiles - next_first;
      const size_t next_rows = next_remaining < capacity
          ? next_remaining : capacity;
      vpu_stream_prefetch_full_batch(buffer ^ 1u, next_first, next_rows,
                                     load_aux);
    }
    vpu_set_vl(VPU_VLEN);
    vpu_stream_emit_activation_batch(activation, buffer, rows);
    vpu_set_vl(VPU_VLEN);
    vpu_stream_store_full_batch(buffer, first_tile, rows);
  }

  if (tail != 0) {
    const unsigned buffer = (unsigned)(full_batches & 1u);
    const size_t tile = full_tiles;
    vpu_stream_reset_batch_addresses(buffer);
    vpu_write_gp(vpu_stream_offset_gp(buffer),
                 (uint32_t)(tile * (size_t)VPU_VLEN));
    vpu_set_vl(tail);
    vpu_h_prefetch_v(vpu_stream_input_gp(buffer),
                     vpu_stream_offset_gp(buffer), VPU_STREAM_H_INPUT);
    if (load_aux) {
      vpu_h_prefetch_v(vpu_stream_aux_gp(buffer),
                       vpu_stream_offset_gp(buffer), VPU_STREAM_H_AUX);
    }
    vpu_stream_emit_activation_row(activation, buffer);
    vpu_write_gp(vpu_stream_output_gp(buffer),
                 VPU_BANK_BASE(buffer ? 7u : 3u));
    vpu_h_store_v(vpu_stream_output_gp(buffer),
                  vpu_stream_offset_gp(buffer), VPU_STREAM_H_OUTPUT);
  }
  return vpu_fence();
}

static inline uint64_t vpu_stream_relu(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_stream_activation_auto(
      input, input, output, elements, VPU_STREAM_ACT_RELU);
}

static inline uint64_t vpu_stream_sigmoid(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_stream_activation_auto(
      input, input, output, elements, VPU_STREAM_ACT_SIGMOID);
}

static inline uint64_t vpu_stream_tanh(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_stream_activation_auto(
      input, input, output, elements, VPU_STREAM_ACT_TANH);
}

static inline uint64_t vpu_stream_gelu(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_stream_activation_auto(
      input, input, output, elements, VPU_STREAM_ACT_GELU);
}

static inline uint64_t vpu_stream_silu(const vpu_storage_t *input,
                                       vpu_storage_t *output,
                                       size_t elements) {
  return vpu_stream_activation_auto(
      input, input, output, elements, VPU_STREAM_ACT_SILU);
}

static inline uint64_t vpu_stream_swiglu(
    const vpu_storage_t *gate, const vpu_storage_t *up,
    vpu_storage_t *output, size_t elements) {
  return vpu_stream_activation_auto(
      gate, up, output, elements, VPU_STREAM_ACT_SWIGLU);
}

/*
 * Stable softmax uses three streaming passes. Exponentials are recomputed in
 * the output pass so no hidden-size-sized temporary buffer is required:
 *   1. global maximum, 2. global shifted-exp sum, 3. normalized output.
 * This is the literal IEEE sequence: finite logits are the supported numeric
 * contract; NaN, +Inf, or an all-negative-infinity row propagates canonical
 * NaN and the corresponding sticky flags.
 */
static inline uint64_t vpu_tiled_softmax_streaming(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  if (elements > UINT32_MAX) {
    return VPU_STATUS_ILLEGAL_COMMAND;
  }
  vpu_stream_configure_spad();
  vpu_stream_configure_memory(input, input, output);
  if (elements == 0) {
    return vpu_stream_empty();
  }

  const size_t full_tiles = elements / VPU_VLEN;
  const size_t tail = elements % VPU_VLEN;
  const size_t capacity = vpu_stream_batch_capacity();
  const size_t full_batches = full_tiles == 0
      ? 0 : 1u + (full_tiles - 1u) / capacity;
  vpu_set_stride_bytes((uint64_t)VPU_VLEN * VPU_STORAGE_BYTES);
  vpu_write_fp_bits(VPU_STREAM_FP_ACCUM_OR_MAX, 0xff800000u);
  if (full_batches != 0) {
    const size_t rows = full_tiles < capacity ? full_tiles : capacity;
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, VPU_BANK_BASE(0),
                           VPU_STREAM_H_INPUT, 0, rows, VPU_VLEN);
  }
  for (size_t batch = 0; batch < full_batches; ++batch) {
    const unsigned buffer = (unsigned)(batch & 1u);
    const size_t first_tile = batch * capacity;
    const size_t remaining = full_tiles - first_tile;
    const size_t rows = remaining < capacity ? remaining : capacity;
    if (batch + 1u < full_batches) {
      const size_t next_first = first_tile + rows;
      const size_t next_remaining = full_tiles - next_first;
      const size_t next_rows = next_remaining < capacity
          ? next_remaining : capacity;
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_INPUT, VPU_BANK_BASE(buffer ? 0u : 4u),
          VPU_STREAM_H_INPUT, next_first, next_rows, VPU_VLEN);
    }
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_red_max_run(VPU_BANK_BASE(buffer ? 4u : 0u), rows);
  }
  if (tail != 0) {
    const unsigned buffer = (unsigned)(full_batches & 1u);
    const unsigned input_address = VPU_BANK_BASE(buffer ? 4u : 0u);
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                           VPU_STREAM_H_INPUT, full_tiles, 1, tail);
    vpu_tiled_emit_red_max_run(input_address, 1);
  }

  vpu_write_fp(VPU_STREAM_FP_CONSTANT_OR_SUM, 0.0f);
  if (full_batches != 0) {
    const size_t rows = full_tiles < capacity ? full_tiles : capacity;
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, VPU_BANK_BASE(0),
                           VPU_STREAM_H_INPUT, 0, rows, VPU_VLEN);
  }
  for (size_t batch = 0; batch < full_batches; ++batch) {
    const unsigned buffer = (unsigned)(batch & 1u);
    const size_t first_tile = batch * capacity;
    const size_t remaining = full_tiles - first_tile;
    const size_t rows = remaining < capacity ? remaining : capacity;
    if (batch + 1u < full_batches) {
      const size_t next_first = first_tile + rows;
      const size_t next_remaining = full_tiles - next_first;
      const size_t next_rows = next_remaining < capacity
          ? next_remaining : capacity;
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_INPUT, VPU_BANK_BASE(buffer ? 0u : 4u),
          VPU_STREAM_H_INPUT, next_first, next_rows, VPU_VLEN);
    }
    const unsigned input_address = VPU_BANK_BASE(buffer ? 4u : 0u);
    const unsigned scratch_address = VPU_BANK_BASE(buffer ? 5u : 1u);
    const unsigned exp_address = VPU_BANK_BASE(buffer ? 6u : 2u);
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_exp_sum_run(input_address, scratch_address,
                               exp_address, rows);
  }
  if (tail != 0) {
    const unsigned buffer = (unsigned)(full_batches & 1u);
    const unsigned input_address = VPU_BANK_BASE(buffer ? 4u : 0u);
    const unsigned scratch_address = VPU_BANK_BASE(buffer ? 5u : 1u);
    const unsigned exp_address = VPU_BANK_BASE(buffer ? 6u : 2u);
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                           VPU_STREAM_H_INPUT, full_tiles, 1, tail);
    vpu_tiled_emit_exp_sum_run(input_address, scratch_address,
                               exp_address, 1);
  }
  vpu_s_reci(VPU_STREAM_FP_CONSTANT_OR_SUM,
             VPU_STREAM_FP_CONSTANT_OR_SUM);

  if (full_batches != 0) {
    const size_t rows = full_tiles < capacity ? full_tiles : capacity;
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, VPU_BANK_BASE(0),
                           VPU_STREAM_H_INPUT, 0, rows, VPU_VLEN);
  }
  for (size_t batch = 0; batch < full_batches; ++batch) {
    const unsigned buffer = (unsigned)(batch & 1u);
    const size_t first_tile = batch * capacity;
    const size_t remaining = full_tiles - first_tile;
    const size_t rows = remaining < capacity ? remaining : capacity;
    if (batch + 1u < full_batches) {
      const size_t next_first = first_tile + rows;
      const size_t next_remaining = full_tiles - next_first;
      const size_t next_rows = next_remaining < capacity
          ? next_remaining : capacity;
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_INPUT, VPU_BANK_BASE(buffer ? 0u : 4u),
          VPU_STREAM_H_INPUT, next_first, next_rows, VPU_VLEN);
    }
    const unsigned input_address = VPU_BANK_BASE(buffer ? 4u : 0u);
    const unsigned scratch_address = VPU_BANK_BASE(buffer ? 5u : 1u);
    const unsigned output_address = VPU_BANK_BASE(buffer ? 7u : 3u);
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_softmax_recompute_run(
        input_address, scratch_address, output_address, rows);
    vpu_tiled_store_run(VPU_RESIDENT_GP_OUTPUT, output_address,
                        first_tile, rows, VPU_VLEN);
  }
  if (tail != 0) {
    const unsigned buffer = (unsigned)(full_batches & 1u);
    const unsigned input_address = VPU_BANK_BASE(buffer ? 4u : 0u);
    const unsigned scratch_address = VPU_BANK_BASE(buffer ? 5u : 1u);
    const unsigned output_address = VPU_BANK_BASE(buffer ? 7u : 3u);
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                           VPU_STREAM_H_INPUT, full_tiles, 1, tail);
    vpu_tiled_emit_softmax_recompute_run(
        input_address, scratch_address, output_address, 1);
    vpu_tiled_store_run(VPU_RESIDENT_GP_OUTPUT, output_address,
                        full_tiles, 1, tail);
  }
  return vpu_fence();
}

/*
 * Resident/hybrid stable softmax.  Pass one retains the planned input prefix.
 * Pass two writes EXP back over the now-dead input slot, so a full plan uses
 * every bank except one shifted/output workspace.  A partial plan reserves an
 * additional transient bank: overflow input is reloaded for passes two and
 * three and EXP overwrites that transient input slot.  Cached EXP is reused in
 * pass three, eliminating both reload and recomputation for resident tiles.
 */
static inline uint64_t vpu_tiled_softmax_resident(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements,
    struct vpu_tiled_plan plan) {
  if (plan.kernel != VPU_TILED_KERNEL_SOFTMAX ||
      !vpu_tiled_plan_is_resident(plan) || elements == 0 ||
      elements > UINT32_MAX || plan.elements != elements) {
    return VPU_STATUS_ILLEGAL_COMMAND;
  }

  vpu_stream_configure_memory(input, input, output);
  const unsigned first_workspace_bank = plan.resident_banks;
  const unsigned transient_bank = first_workspace_bank;
  const unsigned shifted_output_bank = plan.schedule ==
          VPU_TILED_SCHEDULE_RESIDENT_FULL
      ? first_workspace_bank
      : first_workspace_bank + 1u;
  const size_t full_tiles = elements / VPU_VLEN;
  const size_t tail = elements % VPU_VLEN;
  vpu_set_stride_bytes((uint64_t)VPU_VLEN * VPU_STORAGE_BYTES);

  /* Pass 1: retain the resident prefix while reducing the global maximum.
   * Prime the first chunk, then admit the next disjoint load before replaying
   * the current reduction loop.  Hardware-loop replay blocks new RoCC input,
   * so this explicit lookahead is what creates the LD(n+1)/EX(n) overlap. */
  vpu_write_fp_bits(VPU_STREAM_FP_ACCUM_OR_MAX, 0xff800000u);
  size_t tile = 0;
  const size_t cached_full_tiles = plan.resident_tiles < full_tiles
      ? plan.resident_tiles : full_tiles;
  if (tile < cached_full_tiles) {
    const size_t rows = vpu_tiled_chunk_rows(
        tile, cached_full_tiles - tile);
    vpu_tiled_prefetch_run(
        VPU_RESIDENT_GP_INPUT, vpu_tiled_resident_address(plan, tile),
        VPU_STREAM_H_INPUT, tile, rows, VPU_VLEN);
  }
  while (tile < cached_full_tiles) {
    const size_t rows = vpu_tiled_chunk_rows(
        tile, cached_full_tiles - tile);
    const size_t next_tile = tile + rows;
    const unsigned input_address =
        vpu_tiled_resident_address(plan, tile);
    if (next_tile < cached_full_tiles) {
      const size_t next_rows = vpu_tiled_chunk_rows(
          next_tile, cached_full_tiles - next_tile);
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_INPUT,
          vpu_tiled_resident_address(plan, next_tile),
          VPU_STREAM_H_INPUT, next_tile, next_rows, VPU_VLEN);
    }
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_red_max_run(input_address, rows);
    tile = next_tile;
  }

  /* The partial-plan suffix uses the transient bank as a slot-indexed ring.
   * Adjacent chunks therefore have disjoint dependency ranges even though
   * they share one physical 1R1W bank. */
  if (tile < full_tiles) {
    const size_t rows = vpu_tiled_chunk_rows(tile, full_tiles - tile);
    vpu_tiled_prefetch_run(
        VPU_RESIDENT_GP_INPUT,
        vpu_tiled_stream_address(transient_bank, tile),
        VPU_STREAM_H_INPUT, tile, rows, VPU_VLEN);
  }
  while (tile < full_tiles) {
    const size_t rows = vpu_tiled_chunk_rows(tile, full_tiles - tile);
    const size_t next_tile = tile + rows;
    const unsigned input_address =
        vpu_tiled_stream_address(transient_bank, tile);
    if (next_tile < full_tiles) {
      const size_t next_rows = vpu_tiled_chunk_rows(
          next_tile, full_tiles - next_tile);
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_INPUT,
          vpu_tiled_stream_address(transient_bank, next_tile),
          VPU_STREAM_H_INPUT, next_tile, next_rows, VPU_VLEN);
    }
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_red_max_run(input_address, rows);
    tile = next_tile;
  }
  if (tail != 0) {
    const int cached = tile < plan.resident_tiles;
    const unsigned input_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(transient_bank, tile);
    vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                           VPU_STREAM_H_INPUT, tile, 1, tail);
    vpu_tiled_emit_red_max_run(input_address, 1);
  }

  /* Pass 2: turn resident logits into cached exponentials.  The resident
   * prefix has no DMA, while the overflow suffix repeats the same explicit
   * prefetch lookahead and overwrites only its current transient slot. */
  vpu_write_fp(VPU_STREAM_FP_CONSTANT_OR_SUM, 0.0f);
  tile = 0;
  while (tile < cached_full_tiles) {
    const size_t rows = vpu_tiled_chunk_rows(
        tile, cached_full_tiles - tile);
    const unsigned input_address = vpu_tiled_resident_address(plan, tile);
    const unsigned scratch_address =
        vpu_tiled_stream_address(shifted_output_bank, tile);
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_exp_sum_run(input_address, scratch_address,
                               input_address, rows);
    tile += rows;
  }
  if (tile < full_tiles) {
    const size_t rows = vpu_tiled_chunk_rows(tile, full_tiles - tile);
    vpu_tiled_prefetch_run(
        VPU_RESIDENT_GP_INPUT,
        vpu_tiled_stream_address(transient_bank, tile),
        VPU_STREAM_H_INPUT, tile, rows, VPU_VLEN);
  }
  while (tile < full_tiles) {
    const size_t rows = vpu_tiled_chunk_rows(tile, full_tiles - tile);
    const size_t next_tile = tile + rows;
    const unsigned input_address =
        vpu_tiled_stream_address(transient_bank, tile);
    const unsigned scratch_address =
        vpu_tiled_stream_address(shifted_output_bank, tile);
    if (next_tile < full_tiles) {
      const size_t next_rows = vpu_tiled_chunk_rows(
          next_tile, full_tiles - next_tile);
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_INPUT,
          vpu_tiled_stream_address(transient_bank, next_tile),
          VPU_STREAM_H_INPUT, next_tile, next_rows, VPU_VLEN);
    }
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_exp_sum_run(input_address, scratch_address,
                               input_address, rows);
    tile = next_tile;
  }
  if (tail != 0) {
    const int cached = tile < plan.resident_tiles;
    const unsigned input_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(transient_bank, tile);
    const unsigned scratch_address =
        vpu_tiled_stream_address(shifted_output_bank, tile);
    if (!cached) {
      vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                             VPU_STREAM_H_INPUT, tile, 1, tail);
    }
    vpu_set_vl(tail);
    vpu_tiled_emit_exp_sum_run(input_address, scratch_address,
                               input_address, 1);
  }
  vpu_s_reci(VPU_STREAM_FP_CONSTANT_OR_SUM,
             VPU_STREAM_FP_CONSTANT_OR_SUM);

  /* Pass 3: write normalized values through the workspace output ring instead
   * of overwriting the cached EXP row.  ST(n) can then read the workspace bank
   * while EX(n+1) reads a resident/transient input bank. */
  tile = 0;
  while (tile < cached_full_tiles) {
    const size_t rows = vpu_tiled_chunk_rows(
        tile, cached_full_tiles - tile);
    const unsigned exp_address = vpu_tiled_resident_address(plan, tile);
    const unsigned output_address =
        vpu_tiled_stream_address(shifted_output_bank, tile);
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_softmax_scale_run(exp_address, output_address, rows);
    vpu_tiled_store_run(VPU_RESIDENT_GP_OUTPUT, output_address,
                        tile, rows, VPU_VLEN);
    tile += rows;
  }
  if (tile < full_tiles) {
    const size_t rows = vpu_tiled_chunk_rows(tile, full_tiles - tile);
    vpu_tiled_prefetch_run(
        VPU_RESIDENT_GP_INPUT,
        vpu_tiled_stream_address(transient_bank, tile),
        VPU_STREAM_H_INPUT, tile, rows, VPU_VLEN);
  }
  while (tile < full_tiles) {
    const size_t rows = vpu_tiled_chunk_rows(tile, full_tiles - tile);
    const size_t next_tile = tile + rows;
    const unsigned input_address =
        vpu_tiled_stream_address(transient_bank, tile);
    const unsigned output_address =
        vpu_tiled_stream_address(shifted_output_bank, tile);
    if (next_tile < full_tiles) {
      const size_t next_rows = vpu_tiled_chunk_rows(
          next_tile, full_tiles - next_tile);
      vpu_tiled_prefetch_run(
          VPU_RESIDENT_GP_INPUT,
          vpu_tiled_stream_address(transient_bank, next_tile),
          VPU_STREAM_H_INPUT, next_tile, next_rows, VPU_VLEN);
    }
    vpu_set_vl(VPU_VLEN);
    vpu_tiled_emit_softmax_recompute_run(
        input_address, output_address, output_address, rows);
    vpu_tiled_store_run(VPU_RESIDENT_GP_OUTPUT, output_address,
                        tile, rows, VPU_VLEN);
    tile = next_tile;
  }
  if (tail != 0) {
    const int cached = tile < plan.resident_tiles;
    const unsigned input_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(transient_bank, tile);
    const unsigned output_address =
        vpu_tiled_stream_address(shifted_output_bank, tile);
    if (cached) {
      vpu_set_vl(tail);
      vpu_tiled_emit_softmax_scale_run(input_address, output_address, 1);
    } else {
      vpu_tiled_prefetch_run(VPU_RESIDENT_GP_INPUT, input_address,
                             VPU_STREAM_H_INPUT, tile, 1, tail);
      vpu_tiled_emit_softmax_recompute_run(
          input_address, output_address, output_address, 1);
    }
    vpu_tiled_store_run(VPU_RESIDENT_GP_OUTPUT, output_address,
                        tile, 1, tail);
  }
  return vpu_fence();
}

static inline uint64_t vpu_tiled_softmax(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  const struct vpu_tiled_plan plan =
      vpu_tiled_plan_for(VPU_TILED_KERNEL_SOFTMAX, elements);
  if (vpu_tiled_plan_is_resident(plan)) {
    return vpu_tiled_softmax_resident(input, output, elements, plan);
  }
  return vpu_tiled_softmax_streaming(input, output, elements);
}

static inline uint64_t vpu_tiled_softmax_auto(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_tiled_softmax(input, output, elements);
}

static inline uint64_t vpu_stream_softmax(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_tiled_softmax(input, output, elements);
}

static inline uint64_t vpu_tiled_silu(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_stream_silu(input, output, elements);
}

static inline uint64_t vpu_tiled_silu_auto(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_tiled_silu(input, output, elements);
}

static inline uint64_t vpu_tiled_swiglu(
    const vpu_storage_t *gate, const vpu_storage_t *up,
    vpu_storage_t *output, size_t elements) {
  return vpu_stream_swiglu(gate, up, output, elements);
}

static inline uint64_t vpu_tiled_swiglu_auto(
    const vpu_storage_t *gate, const vpu_storage_t *up,
    vpu_storage_t *output, size_t elements) {
  return vpu_tiled_swiglu(gate, up, output, elements);
}

/*
 * Rotary-position-embedding layouts supported by vpu_rope_auto().  Both
 * layouts consume expanded cosine and sine tables with one value per input
 * element.  For INTERLEAVED the table convention is normally
 * [c0,c0,c1,c1,...]; for NEOX the two table halves normally match.
 */
enum vpu_rope_layout {
  VPU_ROPE_INTERLEAVED = 0,
  VPU_ROPE_NEOX = 1,
};

/* Private register allocation for the synchronous rearrangement kernels.
 * Cosine/sine (or gather indices) remain resident while input and temporary
 * vectors alternate between disjoint ping/pong banks. */
enum vpu_rearrange_kernel_register {
  VPU_REARRANGE_GP_TABLE0 = 0,
  VPU_REARRANGE_GP_TABLE1 = 1,
  VPU_REARRANGE_GP_PING_INPUT = 2,
  VPU_REARRANGE_GP_PING_TEMP = 3,
  VPU_REARRANGE_GP_PONG_INPUT = 4,
  VPU_REARRANGE_GP_PONG_TEMP = 5,
  VPU_REARRANGE_GP_PING_OFFSET = 6,
  VPU_REARRANGE_GP_PONG_OFFSET = 7,
  VPU_REARRANGE_GP_SHIFT = 8,
  VPU_REARRANGE_GP_PING_ROWS = 9,
  VPU_REARRANGE_GP_PONG_ROWS = 10,
  VPU_REARRANGE_GP_TABLE_OFFSET = 11,

  VPU_REARRANGE_H_INPUT = 0,
  VPU_REARRANGE_H_TABLE0 = 1,
  VPU_REARRANGE_H_TABLE1 = 2,
  VPU_REARRANGE_H_OUTPUT = 3,

  VPU_REARRANGE_FP_ZERO = 0,
};

static inline unsigned vpu_rearrange_input_gp(unsigned buffer) {
  return buffer ? VPU_REARRANGE_GP_PONG_INPUT
                : VPU_REARRANGE_GP_PING_INPUT;
}

static inline unsigned vpu_rearrange_temp_gp(unsigned buffer) {
  return buffer ? VPU_REARRANGE_GP_PONG_TEMP
                : VPU_REARRANGE_GP_PING_TEMP;
}

static inline unsigned vpu_rearrange_offset_gp(unsigned buffer) {
  return buffer ? VPU_REARRANGE_GP_PONG_OFFSET
                : VPU_REARRANGE_GP_PING_OFFSET;
}

static inline unsigned vpu_rearrange_rows_gp(unsigned buffer) {
  return buffer ? VPU_REARRANGE_GP_PONG_ROWS
                : VPU_REARRANGE_GP_PING_ROWS;
}

static inline unsigned vpu_rearrange_input_base(unsigned buffer) {
  return VPU_BANK_BASE(buffer ? 4u : 2u);
}

static inline size_t vpu_rearrange_batch_capacity(void) {
  return vpu_stream_physical_batch_capacity();
}

static inline size_t vpu_rearrange_batch_rows(size_t rows, size_t batch) {
  const size_t first = batch * vpu_rearrange_batch_capacity();
  const size_t remaining = rows - first;
  return remaining < vpu_rearrange_batch_capacity()
      ? remaining : vpu_rearrange_batch_capacity();
}

static inline void vpu_rearrange_configure_spad(void) {
  vpu_write_gp(VPU_REARRANGE_GP_TABLE0, VPU_BANK_BASE(0));
  vpu_write_gp(VPU_REARRANGE_GP_TABLE1, VPU_BANK_BASE(1));
  vpu_write_gp(VPU_REARRANGE_GP_PING_INPUT, VPU_BANK_BASE(2));
  vpu_write_gp(VPU_REARRANGE_GP_PING_TEMP, VPU_BANK_BASE(3));
  vpu_write_gp(VPU_REARRANGE_GP_PONG_INPUT, VPU_BANK_BASE(4));
  vpu_write_gp(VPU_REARRANGE_GP_PONG_TEMP, VPU_BANK_BASE(5));
  vpu_write_gp(VPU_REARRANGE_GP_TABLE_OFFSET, 0);
}

/* Configure and enqueue one contiguous-row input batch.  Keep every address
 * write adjacent to descriptor admission: per-row execute commands advance
 * input_gp, so relying on its value from a previous buffer use is unsafe. */
static inline void vpu_rearrange_prefetch_batch(
    unsigned buffer, size_t first_row, size_t row_count,
    size_t row_elements) {
  vpu_write_gp(vpu_rearrange_input_gp(buffer),
               vpu_rearrange_input_base(buffer));
  vpu_write_gp(vpu_rearrange_offset_gp(buffer),
               (uint32_t)(first_row * row_elements));
  vpu_write_gp(vpu_rearrange_rows_gp(buffer), (uint32_t)row_count);
  vpu_h_prefetch_v_2d(vpu_rearrange_input_gp(buffer),
                      vpu_rearrange_offset_gp(buffer),
                      VPU_REARRANGE_H_INPUT,
                      vpu_rearrange_rows_gp(buffer));
}

static inline int vpu_rearrange_total_valid(size_t groups,
                                             size_t group_elements) {
  return group_elements == 0 || groups <= UINT32_MAX / group_elements;
}

static inline void vpu_rope_build_masks(
    uint64_t *first, uint64_t *second, size_t rotary_dim,
    enum vpu_rope_layout layout) {
  for (unsigned chunk = 0; chunk < VPU_VMASK_CHUNKS; ++chunk) {
    first[chunk] = 0;
    second[chunk] = 0;
  }
  const size_t half = rotary_dim / 2u;
  for (size_t element = 0; element < rotary_dim; ++element) {
    const int in_first = layout == VPU_ROPE_INTERLEAVED
        ? (element & 1u) == 0
        : element < half;
    uint64_t *mask = in_first ? first : second;
    mask[element / VPU_VMASK_CHUNK_BITS] |=
        UINT64_C(1) << (element % VPU_VMASK_CHUNK_BITS);
  }
}

/*
 * Apply RoPE to `rows` contiguous vectors of `rotary_dim` elements.
 *
 * The cosine and sine arrays are one expanded table shared by every row.
 * Angles are deliberately not generated here because VPU v1 has no SIN/COS
 * operation.  Input and output may alias exactly.  Each row is independent,
 * rotary_dim must be even and no larger than VLEN, and all addressed element
 * offsets must fit the VPU's 32-bit GP/address interface.
 *
 * This is a synchronous E2E helper: it publishes CPU writes, loads the tables
 * once, streams rows through software-managed ping/pong banks, stores every
 * result, restores an all-enabled architectural mask, and fences before
 * returning status.
 */
static inline uint64_t vpu_rope_auto(
    const vpu_storage_t *input, const vpu_storage_t *cosine,
    const vpu_storage_t *sine, vpu_storage_t *output, size_t rows,
    size_t rotary_dim, enum vpu_rope_layout layout) {
  if ((layout != VPU_ROPE_INTERLEAVED && layout != VPU_ROPE_NEOX) ||
      rotary_dim > VPU_VLEN || (rotary_dim & 1u) != 0 ||
      !vpu_rearrange_total_valid(rows, rotary_dim)) {
    return VPU_STATUS_ILLEGAL_COMMAND;
  }
  if (rows == 0 || rotary_dim == 0) {
    vpu_set_vmask_all();
    return vpu_stream_empty();
  }

  uint64_t first_mask[VPU_VMASK_CHUNKS];
  uint64_t second_mask[VPU_VMASK_CHUNKS];
  vpu_rope_build_masks(first_mask, second_mask, rotary_dim, layout);

  vpu_rearrange_configure_spad();
  vpu_publish_cpu_writes();
  vpu_write_h(VPU_REARRANGE_H_INPUT, (uintptr_t)input);
  vpu_write_h(VPU_REARRANGE_H_TABLE0, (uintptr_t)cosine);
  vpu_write_h(VPU_REARRANGE_H_TABLE1, (uintptr_t)sine);
  vpu_write_h(VPU_REARRANGE_H_OUTPUT, (uintptr_t)output);
  vpu_write_gp(VPU_REARRANGE_GP_SHIFT,
               layout == VPU_ROPE_INTERLEAVED
                   ? 1u : (uint32_t)(rotary_dim / 2u));
  vpu_write_fp(VPU_REARRANGE_FP_ZERO, 0.0f);

  /* The expanded angle tables remain resident for all rows. */
  vpu_set_vl(rotary_dim);
  vpu_h_prefetch_v(VPU_REARRANGE_GP_TABLE0,
                   VPU_REARRANGE_GP_TABLE_OFFSET,
                   VPU_REARRANGE_H_TABLE0);
  vpu_h_prefetch_v(VPU_REARRANGE_GP_TABLE1,
                   VPU_REARRANGE_GP_TABLE_OFFSET,
                   VPU_REARRANGE_H_TABLE1);

  /* Batch contiguous host rows into consecutive VSRAM slots with one 2-D DMA
   * descriptor. Ping/pong still overlaps the next batch's load with current
   * execution, while reducing load/store command count by up to one bank's
   * slot capacity. */
  const size_t batches =
      1u + (rows - 1u) / vpu_rearrange_batch_capacity();
  vpu_set_stride_bytes((uint64_t)rotary_dim * VPU_STORAGE_BYTES);
  vpu_rearrange_prefetch_batch(
      0, 0, vpu_rearrange_batch_rows(rows, 0), rotary_dim);
  for (size_t batch = 0; batch < batches; ++batch) {
    const unsigned buffer = (unsigned)(batch & 1u);
    const unsigned input_gp = vpu_rearrange_input_gp(buffer);
    const unsigned temp_gp = vpu_rearrange_temp_gp(buffer);
    const size_t batch_rows = vpu_rearrange_batch_rows(rows, batch);
    if (batch + 1u < batches) {
      const unsigned next_buffer = buffer ^ 1u;
      const size_t next_first =
          (batch + 1u) * vpu_rearrange_batch_capacity();
      vpu_rearrange_prefetch_batch(
          next_buffer, next_first,
          vpu_rearrange_batch_rows(rows, batch + 1u), rotary_dim);
    }

    const unsigned input_base = vpu_rearrange_input_base(buffer);
    const unsigned temp_base = VPU_BANK_BASE(buffer ? 5u : 3u);
    vpu_write_gp(input_gp, input_base);
    vpu_write_gp(temp_gp, temp_base);
#if VPU_LOOP_BUFFER_ENTRIES >= (2 * VPU_VMASK_CHUNKS + 10) && \
    VPU_VLEN <= VPU_ADDI_INT_IMM_MAX
    if (batch_rows > 1u) {
      vpu_loop_start(VPU_REARRANGE_GP_TABLE_OFFSET,
                     (uint32_t)batch_rows);
      /* Build rotate_half(x) in temp without a gather. Masked slide writes
       * preserve the other half; negating while the first mask is still
       * active avoids another architectural-mask transition. */
      vpu_write_vmask(first_mask);
      vpu_v_slide_v_masked(temp_gp, input_gp, VPU_REARRANGE_GP_SHIFT,
                           VPU_SLIDE_LEFT);
      vpu_v_sub_vf_masked(temp_gp, temp_gp, VPU_REARRANGE_FP_ZERO, true);
      vpu_write_vmask(second_mask);
      vpu_v_slide_v_masked(temp_gp, input_gp, VPU_REARRANGE_GP_SHIFT,
                           VPU_SLIDE_RIGHT);
      /* These arithmetic commands are unmasked, so the second-half mask may
       * stay installed until the next row. */
      vpu_v_mul_vv(temp_gp, temp_gp, VPU_REARRANGE_GP_TABLE1);
      vpu_v_mul_vv(input_gp, input_gp, VPU_REARRANGE_GP_TABLE0);
      vpu_v_add_vv(input_gp, input_gp, temp_gp);
      vpu_s_addi_int(input_gp, input_gp, VPU_VLEN);
      vpu_s_addi_int(temp_gp, temp_gp, VPU_VLEN);
      vpu_loop_end(VPU_REARRANGE_GP_TABLE_OFFSET);
    } else
#endif
    {
      for (size_t row = 0; row < batch_rows; ++row) {
        vpu_write_vmask(first_mask);
        vpu_v_slide_v_masked(temp_gp, input_gp, VPU_REARRANGE_GP_SHIFT,
                             VPU_SLIDE_LEFT);
        vpu_v_sub_vf_masked(temp_gp, temp_gp, VPU_REARRANGE_FP_ZERO, true);
        vpu_write_vmask(second_mask);
        vpu_v_slide_v_masked(temp_gp, input_gp, VPU_REARRANGE_GP_SHIFT,
                             VPU_SLIDE_RIGHT);
        vpu_v_mul_vv(temp_gp, temp_gp, VPU_REARRANGE_GP_TABLE1);
        vpu_v_mul_vv(input_gp, input_gp, VPU_REARRANGE_GP_TABLE0);
        vpu_v_add_vv(input_gp, input_gp, temp_gp);
        vpu_s_addi_int(input_gp, input_gp, VPU_VLEN);
        vpu_s_addi_int(temp_gp, temp_gp, VPU_VLEN);
      }
    }
    vpu_write_gp(input_gp, input_base);
    vpu_h_store_v_2d(input_gp, vpu_rearrange_offset_gp(buffer),
                     VPU_REARRANGE_H_OUTPUT,
                     vpu_rearrange_rows_gp(buffer));
  }
  vpu_set_vmask_all();
  return vpu_fence();
}

/*
 * Apply one tile-local permutation to every contiguous input group.  `indices`
 * contains group_elements raw vpu_index_t values and is shared by all groups;
 * index i selects input[group][indices[i]].  Hardware gather semantics supply
 * +0 for an index outside the current group.  Input and output may alias.
 *
 * VPU v1 gather is local to one VL-sized source window, so this API does not
 * claim to implement a global cross-group permutation.  It keeps the index
 * vector resident and streams source/destination ping-pong buffers.  Like the
 * other public auto helpers, it fences before returning.
 */
static inline uint64_t vpu_permute_auto(
    const vpu_storage_t *input, const vpu_index_t *indices,
    vpu_storage_t *output, size_t groups, size_t group_elements) {
  if (group_elements > VPU_VLEN ||
      !vpu_rearrange_total_valid(groups, group_elements)) {
    return VPU_STATUS_ILLEGAL_COMMAND;
  }
  if (groups == 0 || group_elements == 0) {
    return vpu_stream_empty();
  }

  vpu_rearrange_configure_spad();
  vpu_publish_cpu_writes();
  vpu_write_h(VPU_REARRANGE_H_INPUT, (uintptr_t)input);
  vpu_write_h(VPU_REARRANGE_H_TABLE0, (uintptr_t)indices);
  vpu_write_h(VPU_REARRANGE_H_OUTPUT, (uintptr_t)output);
  vpu_set_vl(group_elements);
  vpu_h_prefetch_v(VPU_REARRANGE_GP_TABLE0,
                   VPU_REARRANGE_GP_TABLE_OFFSET,
                   VPU_REARRANGE_H_TABLE0);
  const size_t batches =
      1u + (groups - 1u) / vpu_rearrange_batch_capacity();
  vpu_set_stride_bytes((uint64_t)group_elements * VPU_STORAGE_BYTES);
  vpu_rearrange_prefetch_batch(
      0, 0, vpu_rearrange_batch_rows(groups, 0), group_elements);

  for (size_t batch = 0; batch < batches; ++batch) {
    const unsigned buffer = (unsigned)(batch & 1u);
    const unsigned input_gp = vpu_rearrange_input_gp(buffer);
    const unsigned output_gp = vpu_rearrange_temp_gp(buffer);
    const size_t batch_rows = vpu_rearrange_batch_rows(groups, batch);
    if (batch + 1u < batches) {
      const unsigned next_buffer = buffer ^ 1u;
      const size_t next_first =
          (batch + 1u) * vpu_rearrange_batch_capacity();
      vpu_rearrange_prefetch_batch(
          next_buffer, next_first,
          vpu_rearrange_batch_rows(groups, batch + 1u), group_elements);
    }
    const unsigned input_base = vpu_rearrange_input_base(buffer);
    const unsigned output_base = VPU_BANK_BASE(buffer ? 5u : 3u);
    vpu_write_gp(input_gp, input_base);
    vpu_write_gp(output_gp, output_base);
#if VPU_LOOP_BUFFER_ENTRIES >= 5 && VPU_VLEN <= VPU_ADDI_INT_IMM_MAX
    if (batch_rows > 1u) {
      vpu_loop_start(VPU_REARRANGE_GP_TABLE_OFFSET,
                     (uint32_t)batch_rows);
      vpu_v_gather_vv(output_gp, input_gp, VPU_REARRANGE_GP_TABLE0);
      vpu_s_addi_int(input_gp, input_gp, VPU_VLEN);
      vpu_s_addi_int(output_gp, output_gp, VPU_VLEN);
      vpu_loop_end(VPU_REARRANGE_GP_TABLE_OFFSET);
    } else
#endif
    {
      for (size_t group = 0; group < batch_rows; ++group) {
        vpu_v_gather_vv(output_gp, input_gp, VPU_REARRANGE_GP_TABLE0);
        vpu_s_addi_int(input_gp, input_gp, VPU_VLEN);
        vpu_s_addi_int(output_gp, output_gp, VPU_VLEN);
      }
    }
    vpu_write_gp(output_gp, output_base);
    vpu_h_store_v_2d(output_gp, vpu_rearrange_offset_gp(buffer),
                     VPU_REARRANGE_H_OUTPUT,
                     vpu_rearrange_rows_gp(buffer));
  }
  return vpu_fence();
}

/* Short public spellings analogous to Gemmini's auto helpers. */
static inline uint64_t vpu_rmsnorm_auto(
    const vpu_storage_t *input, const vpu_storage_t *weight,
    vpu_storage_t *output, size_t elements, float epsilon) {
  return vpu_tiled_rmsnorm_auto(input, weight, output, elements, epsilon);
}

static inline uint64_t vpu_final_norm_auto(
    const vpu_storage_t *input, const vpu_storage_t *weight,
    vpu_storage_t *output, size_t elements, float epsilon) {
  return vpu_tiled_final_norm_auto(input, weight, output, elements, epsilon);
}

static inline uint64_t vpu_silu_auto(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_tiled_silu_auto(input, output, elements);
}

static inline uint64_t vpu_relu_auto(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_stream_relu(input, output, elements);
}

static inline uint64_t vpu_sigmoid_auto(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_stream_sigmoid(input, output, elements);
}

static inline uint64_t vpu_tanh_auto(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_stream_tanh(input, output, elements);
}

/* Approximate GELU matching PLENA: x * sigmoid(1.702*x). */
static inline uint64_t vpu_gelu_auto(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_stream_gelu(input, output, elements);
}

static inline uint64_t vpu_swiglu_auto(
    const vpu_storage_t *gate, const vpu_storage_t *up,
    vpu_storage_t *output, size_t elements) {
  return vpu_tiled_swiglu_auto(gate, up, output, elements);
}

static inline uint64_t vpu_softmax_auto(
    const vpu_storage_t *input, vpu_storage_t *output, size_t elements) {
  return vpu_tiled_softmax_auto(input, output, elements);
}

/*
 * Gemmini--VPU FlashAttention needs Gemmini's generated BF16 ABI and its
 * fused LOOP_WS transport in addition to the standalone VPU interface above.
 * Keep that dependency opt-in so programs such as vpu_nonlinear continue to
 * compile for VPU-only configurations which do not define elem_t or DIM.
 *
 * A fusion caller must include its generated Gemmini parameter header before
 * this file and define VPU_ENABLE_GEMMINI_FLASHATTENTION.
 */
#if defined(VPU_ENABLE_GEMMINI_FLASHATTENTION)
#include "vpu_flashattention_kernel.h"
#endif

#endif  // GEMMINI_ROCC_TESTS_INCLUDE_VPU_KERNELS_H_
