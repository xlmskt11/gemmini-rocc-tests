// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_INCLUDE_VPU_KERNELS_H_
#define GEMMINI_ROCC_TESTS_INCLUDE_VPU_KERNELS_H_

#include "vpu.h"

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
 * registers 0..9, H registers 0..2, and FP registers 0..2.
 */
#if VPU_VSPAD_BANKS < 8
#error "The streaming VPU kernels require at least eight Vector SRAM banks"
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

/*
 * Geometry-driven resident/hybrid RMSNorm.
 *
 * Full plan: all x tiles are packed over the resident banks.  The remaining
 * two banks stream weights and hold a fixed scratch vector.  After x has been
 * normalized, the final multiply overwrites that dead x slot before storing.
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

  const unsigned first_workspace_bank = plan.resident_banks;
  const unsigned pass_one_stream_bank = first_workspace_bank;
  const unsigned scratch_bank = first_workspace_bank + 1u;
  const unsigned output_bank = plan.schedule ==
          VPU_TILED_SCHEDULE_RESIDENT_PARTIAL
      ? first_workspace_bank + 2u
      : 0u;
  const unsigned scratch_address = vpu_tiled_stream_address(scratch_bank, 0);
  vpu_tiled_write_tile_address(VPU_RESIDENT_GP_SCRATCH, scratch_address);

  /* Load every x tile exactly once, reduce it, and retain the planned prefix.
   * Queue/hazard backpressure overlaps later loads with current execution. */
  for (size_t tile = 0; tile < plan.tile_count; ++tile) {
    const unsigned input_address = tile < plan.resident_tiles
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(
              pass_one_stream_bank, tile - plan.resident_tiles);
    vpu_tiled_prefetch_address(
        VPU_RESIDENT_GP_INPUT, input_address, VPU_STREAM_H_INPUT,
        elements, tile);
    vpu_tiled_write_tile_address(VPU_RESIDENT_GP_INPUT, input_address);
    vpu_set_vl(vpu_stream_tile_elements(elements, tile));
    vpu_v_mul_vv(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_RESIDENT_GP_INPUT);
    vpu_v_red_sum(VPU_STREAM_FP_ACCUM_OR_MAX,
                  VPU_RESIDENT_GP_SCRATCH);
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

  const unsigned weight_bank = first_workspace_bank;
  for (size_t tile = 0; tile < plan.tile_count; ++tile) {
    const int cached = tile < plan.resident_tiles;
    const unsigned input_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(0u, tile - plan.resident_tiles);
    const unsigned weight_address =
        vpu_tiled_stream_address(weight_bank, tile);
    const unsigned result_address = plan.schedule ==
            VPU_TILED_SCHEDULE_RESIDENT_FULL
        ? input_address
        : vpu_tiled_stream_address(output_bank, tile);

    if (!cached) {
      vpu_tiled_prefetch_address(
          VPU_RESIDENT_GP_INPUT, input_address, VPU_STREAM_H_INPUT,
          elements, tile);
    }
    vpu_tiled_prefetch_address(
        VPU_RESIDENT_GP_AUX, weight_address, VPU_STREAM_H_AUX,
        elements, tile);

    /* Re-establish current addresses after prefetch descriptors snapshot GP,
     * H, and VL; subsequent register writes cannot change those descriptors. */
    vpu_tiled_write_tile_address(VPU_RESIDENT_GP_INPUT, input_address);
    vpu_tiled_write_tile_address(VPU_RESIDENT_GP_AUX, weight_address);
    vpu_tiled_write_tile_address(VPU_RESIDENT_GP_OUTPUT, result_address);
    vpu_set_vl(vpu_stream_tile_elements(elements, tile));
    vpu_v_mul_vf(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_ACCUM_OR_MAX);
    vpu_v_mul_vv(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_SCRATCH,
                 VPU_RESIDENT_GP_AUX);
    vpu_tiled_store_address(
        VPU_RESIDENT_GP_OUTPUT, result_address, elements, tile);
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

  const size_t tiles = vpu_stream_tile_count(elements);
  vpu_write_fp(VPU_STREAM_FP_ACCUM_OR_MAX, 0.0f);
  vpu_write_fp(VPU_STREAM_FP_CONSTANT_OR_SUM, 1.0f / (float)elements);
  vpu_write_fp(VPU_STREAM_FP_EPSILON, epsilon);

  vpu_stream_prefetch_input(elements, 0);
  for (size_t tile = 0; tile < tiles; ++tile) {
    const unsigned buffer = (unsigned)(tile & 1u);
    if (tile + 1u < tiles) {
      vpu_stream_prefetch_input(elements, tile + 1u);
    }
    vpu_set_vl(vpu_stream_tile_elements(elements, tile));
    vpu_v_mul_vv(vpu_stream_temp_gp(buffer), vpu_stream_input_gp(buffer),
                 vpu_stream_input_gp(buffer));
    vpu_v_red_sum(VPU_STREAM_FP_ACCUM_OR_MAX, vpu_stream_temp_gp(buffer));
  }

  vpu_s_mul(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_STREAM_FP_ACCUM_OR_MAX,
            VPU_STREAM_FP_CONSTANT_OR_SUM);
  vpu_s_add(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_STREAM_FP_ACCUM_OR_MAX,
            VPU_STREAM_FP_EPSILON);
  vpu_s_sqrt(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_STREAM_FP_ACCUM_OR_MAX);
  vpu_s_reci(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_STREAM_FP_ACCUM_OR_MAX);

  vpu_stream_prefetch_input(elements, 0);
  for (size_t tile = 0; tile < tiles; ++tile) {
    const unsigned buffer = (unsigned)(tile & 1u);
    const size_t count = vpu_stream_tile_elements(elements, tile);
    vpu_set_vl(count);
    vpu_h_prefetch_v(vpu_stream_aux_gp(buffer),
                     vpu_stream_offset_gp(buffer), VPU_STREAM_H_AUX);
    vpu_v_mul_vf(vpu_stream_temp_gp(buffer), vpu_stream_input_gp(buffer),
                 VPU_STREAM_FP_ACCUM_OR_MAX);
    if (tile + 1u < tiles) {
      vpu_stream_prefetch_input(elements, tile + 1u);
    }
    vpu_set_vl(count);
    vpu_v_mul_vv(vpu_stream_output_gp(buffer), vpu_stream_temp_gp(buffer),
                 vpu_stream_aux_gp(buffer));
    vpu_h_store_v(vpu_stream_output_gp(buffer),
                  vpu_stream_offset_gp(buffer), VPU_STREAM_H_OUTPUT);
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

static inline uint64_t vpu_stream_silu(const vpu_storage_t *input,
                                       vpu_storage_t *output,
                                       size_t elements) {
  if (elements > UINT32_MAX) {
    return VPU_STATUS_ILLEGAL_COMMAND;
  }
  vpu_stream_configure_spad();
  vpu_stream_configure_memory(input, input, output);
  if (elements == 0) {
    return vpu_stream_empty();
  }

  const size_t tiles = vpu_stream_tile_count(elements);
  vpu_write_fp(VPU_STREAM_FP_ACCUM_OR_MAX, 0.0f);
  vpu_write_fp(VPU_STREAM_FP_CONSTANT_OR_SUM, 1.0f);
  vpu_stream_prefetch_input(elements, 0);
  for (size_t tile = 0; tile < tiles; ++tile) {
    const unsigned buffer = (unsigned)(tile & 1u);
    const size_t count = vpu_stream_tile_elements(elements, tile);
    if (tile + 1u < tiles) {
      vpu_stream_prefetch_input(elements, tile + 1u);
    }
    vpu_set_vl(count);
    vpu_v_sub_vf(vpu_stream_aux_gp(buffer), vpu_stream_input_gp(buffer),
                 VPU_STREAM_FP_ACCUM_OR_MAX, true);
    vpu_v_exp_v(vpu_stream_temp_gp(buffer), vpu_stream_aux_gp(buffer));
    vpu_v_add_vf(vpu_stream_aux_gp(buffer), vpu_stream_temp_gp(buffer),
                 VPU_STREAM_FP_CONSTANT_OR_SUM);
    vpu_v_reci_v(vpu_stream_temp_gp(buffer), vpu_stream_aux_gp(buffer));
    vpu_v_mul_vv(vpu_stream_output_gp(buffer), vpu_stream_input_gp(buffer),
                 vpu_stream_temp_gp(buffer));
    vpu_h_store_v(vpu_stream_output_gp(buffer),
                  vpu_stream_offset_gp(buffer), VPU_STREAM_H_OUTPUT);
  }
  return vpu_fence();
}

static inline uint64_t vpu_stream_swiglu(
    const vpu_storage_t *gate, const vpu_storage_t *up,
    vpu_storage_t *output, size_t elements) {
  if (elements > UINT32_MAX) {
    return VPU_STATUS_ILLEGAL_COMMAND;
  }
  vpu_stream_configure_spad();
  vpu_stream_configure_memory(gate, up, output);
  if (elements == 0) {
    return vpu_stream_empty();
  }

  const size_t tiles = vpu_stream_tile_count(elements);
  vpu_write_fp(VPU_STREAM_FP_ACCUM_OR_MAX, 0.0f);
  vpu_write_fp(VPU_STREAM_FP_CONSTANT_OR_SUM, 1.0f);
  vpu_stream_prefetch_input(elements, 0);
  for (size_t tile = 0; tile < tiles; ++tile) {
    const unsigned buffer = (unsigned)(tile & 1u);
    const size_t count = vpu_stream_tile_elements(elements, tile);
    vpu_set_vl(count);
    vpu_h_prefetch_v(vpu_stream_aux_gp(buffer),
                     vpu_stream_offset_gp(buffer), VPU_STREAM_H_AUX);
    vpu_v_sub_vf(vpu_stream_temp_gp(buffer), vpu_stream_input_gp(buffer),
                 VPU_STREAM_FP_ACCUM_OR_MAX, true);
    vpu_v_exp_v(vpu_stream_output_gp(buffer), vpu_stream_temp_gp(buffer));
    vpu_v_add_vf(vpu_stream_temp_gp(buffer), vpu_stream_output_gp(buffer),
                 VPU_STREAM_FP_CONSTANT_OR_SUM);
    vpu_v_reci_v(vpu_stream_output_gp(buffer), vpu_stream_temp_gp(buffer));
    if (tile + 1u < tiles) {
      vpu_stream_prefetch_input(elements, tile + 1u);
    }
    vpu_set_vl(count);
    vpu_v_mul_vv(vpu_stream_temp_gp(buffer), vpu_stream_input_gp(buffer),
                 vpu_stream_output_gp(buffer));
    vpu_v_mul_vv(vpu_stream_output_gp(buffer), vpu_stream_temp_gp(buffer),
                 vpu_stream_aux_gp(buffer));
    vpu_h_store_v(vpu_stream_output_gp(buffer),
                  vpu_stream_offset_gp(buffer), VPU_STREAM_H_OUTPUT);
  }
  return vpu_fence();
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

  const size_t tiles = vpu_stream_tile_count(elements);
  vpu_write_fp_bits(VPU_STREAM_FP_ACCUM_OR_MAX, 0xff800000u);
  vpu_stream_prefetch_input(elements, 0);
  for (size_t tile = 0; tile < tiles; ++tile) {
    const unsigned buffer = (unsigned)(tile & 1u);
    if (tile + 1u < tiles) {
      vpu_stream_prefetch_input(elements, tile + 1u);
    }
    vpu_set_vl(vpu_stream_tile_elements(elements, tile));
    vpu_v_red_max(VPU_STREAM_FP_ACCUM_OR_MAX,
                  vpu_stream_input_gp(buffer));
  }

  vpu_write_fp(VPU_STREAM_FP_CONSTANT_OR_SUM, 0.0f);
  vpu_stream_prefetch_input(elements, 0);
  for (size_t tile = 0; tile < tiles; ++tile) {
    const unsigned buffer = (unsigned)(tile & 1u);
    if (tile + 1u < tiles) {
      vpu_stream_prefetch_input(elements, tile + 1u);
    }
    vpu_set_vl(vpu_stream_tile_elements(elements, tile));
    vpu_v_sub_vf(vpu_stream_aux_gp(buffer), vpu_stream_input_gp(buffer),
                 VPU_STREAM_FP_ACCUM_OR_MAX, false);
    vpu_v_exp_v(vpu_stream_temp_gp(buffer), vpu_stream_aux_gp(buffer));
    vpu_v_red_sum(VPU_STREAM_FP_CONSTANT_OR_SUM,
                  vpu_stream_temp_gp(buffer));
  }
  vpu_s_reci(VPU_STREAM_FP_CONSTANT_OR_SUM,
             VPU_STREAM_FP_CONSTANT_OR_SUM);

  vpu_stream_prefetch_input(elements, 0);
  for (size_t tile = 0; tile < tiles; ++tile) {
    const unsigned buffer = (unsigned)(tile & 1u);
    const size_t count = vpu_stream_tile_elements(elements, tile);
    if (tile + 1u < tiles) {
      vpu_stream_prefetch_input(elements, tile + 1u);
    }
    vpu_set_vl(count);
    vpu_v_sub_vf(vpu_stream_aux_gp(buffer), vpu_stream_input_gp(buffer),
                 VPU_STREAM_FP_ACCUM_OR_MAX, false);
    vpu_v_exp_v(vpu_stream_temp_gp(buffer), vpu_stream_aux_gp(buffer));
    vpu_v_mul_vf(vpu_stream_output_gp(buffer), vpu_stream_temp_gp(buffer),
                 VPU_STREAM_FP_CONSTANT_OR_SUM);
    vpu_h_store_v(vpu_stream_output_gp(buffer),
                  vpu_stream_offset_gp(buffer), VPU_STREAM_H_OUTPUT);
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

  vpu_write_fp_bits(VPU_STREAM_FP_ACCUM_OR_MAX, 0xff800000u);
  for (size_t tile = 0; tile < plan.tile_count; ++tile) {
    const unsigned input_address = tile < plan.resident_tiles
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(
              transient_bank, tile - plan.resident_tiles);
    vpu_tiled_prefetch_address(
        VPU_RESIDENT_GP_INPUT, input_address, VPU_STREAM_H_INPUT,
        elements, tile);
    vpu_tiled_write_tile_address(VPU_RESIDENT_GP_INPUT, input_address);
    vpu_set_vl(vpu_stream_tile_elements(elements, tile));
    vpu_v_red_max(VPU_STREAM_FP_ACCUM_OR_MAX, VPU_RESIDENT_GP_INPUT);
  }

  vpu_write_fp(VPU_STREAM_FP_CONSTANT_OR_SUM, 0.0f);
  for (size_t tile = 0; tile < plan.tile_count; ++tile) {
    const int cached = tile < plan.resident_tiles;
    const unsigned input_exp_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(
              transient_bank, tile - plan.resident_tiles);
    const unsigned shifted_address =
        vpu_tiled_stream_address(shifted_output_bank, tile);
    if (!cached) {
      vpu_tiled_prefetch_address(
          VPU_RESIDENT_GP_INPUT, input_exp_address, VPU_STREAM_H_INPUT,
          elements, tile);
    }
    vpu_tiled_write_tile_address(
        VPU_RESIDENT_GP_INPUT, input_exp_address);
    vpu_tiled_write_tile_address(
        VPU_RESIDENT_GP_SCRATCH, shifted_address);
    vpu_tiled_write_tile_address(
        VPU_RESIDENT_GP_OUTPUT, input_exp_address);
    vpu_set_vl(vpu_stream_tile_elements(elements, tile));
    vpu_v_sub_vf(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_ACCUM_OR_MAX, false);
    vpu_v_exp_v(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_SCRATCH);
    vpu_v_red_sum(VPU_STREAM_FP_CONSTANT_OR_SUM,
                  VPU_RESIDENT_GP_OUTPUT);
  }
  vpu_s_reci(VPU_STREAM_FP_CONSTANT_OR_SUM,
             VPU_STREAM_FP_CONSTANT_OR_SUM);

  for (size_t tile = 0; tile < plan.tile_count; ++tile) {
    const int cached = tile < plan.resident_tiles;
    const unsigned input_exp_address = cached
        ? vpu_tiled_resident_address(plan, tile)
        : vpu_tiled_stream_address(
              transient_bank, tile - plan.resident_tiles);
    const unsigned output_address =
        vpu_tiled_stream_address(shifted_output_bank, tile);
    if (!cached) {
      vpu_tiled_prefetch_address(
          VPU_RESIDENT_GP_INPUT, input_exp_address, VPU_STREAM_H_INPUT,
          elements, tile);
    }
    vpu_tiled_write_tile_address(
        VPU_RESIDENT_GP_INPUT, input_exp_address);
    vpu_tiled_write_tile_address(
        VPU_RESIDENT_GP_SCRATCH, output_address);
    vpu_tiled_write_tile_address(
        VPU_RESIDENT_GP_OUTPUT, output_address);
    vpu_set_vl(vpu_stream_tile_elements(elements, tile));
    if (!cached) {
      vpu_v_sub_vf(VPU_RESIDENT_GP_SCRATCH, VPU_RESIDENT_GP_INPUT,
                   VPU_STREAM_FP_ACCUM_OR_MAX, false);
      vpu_v_exp_v(VPU_RESIDENT_GP_INPUT, VPU_RESIDENT_GP_SCRATCH);
    }
    vpu_v_mul_vf(VPU_RESIDENT_GP_OUTPUT, VPU_RESIDENT_GP_INPUT,
                 VPU_STREAM_FP_CONSTANT_OR_SUM);
    vpu_tiled_store_address(
        VPU_RESIDENT_GP_OUTPUT, output_address, elements, tile);
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

#endif  // GEMMINI_ROCC_TESTS_INCLUDE_VPU_KERNELS_H_
