// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_INCLUDE_VPU_H_
#define GEMMINI_ROCC_TESTS_INCLUDE_VPU_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "vpu_params.h"

#ifdef __cplusplus
extern "C" {
#endif

/* custom0 is shared with Gemmini0; funct7=64 is routed exclusively to VPU. */
#define VPU_XCUSTOM 0
#define VPU_ROCC_FUNCT 64

/* six-bit micro-opcodes implemented by VPU v1. */
enum vpu_opcode {
  VPU_OP_V_ADD_VV = 0x0d,
  VPU_OP_V_ADD_VF = 0x0e,
  VPU_OP_V_SUB_VV = 0x0f,
  VPU_OP_V_SUB_VF = 0x10,
  VPU_OP_V_MUL_VV = 0x11,
  VPU_OP_V_MUL_VF = 0x12,
  VPU_OP_V_EXP_V = 0x13,
  VPU_OP_V_RECI_V = 0x14,
  VPU_OP_V_RED_SUM = 0x15,
  VPU_OP_V_RED_MAX = 0x16,

  VPU_OP_S_ADD_FP = 0x17,
  VPU_OP_S_SUB_FP = 0x18,
  VPU_OP_S_MAX_FP = 0x19,
  VPU_OP_S_MUL_FP = 0x1a,
  VPU_OP_S_EXP_FP = 0x1b,
  VPU_OP_S_RECI_FP = 0x1c,
  VPU_OP_S_SQRT_FP = 0x1d,

  VPU_OP_H_PREFETCH_V = 0x29,
  VPU_OP_H_STORE_V = 0x2a,

  VPU_OP_C_SET_STRIDE = 0x2d,
  VPU_OP_C_WRITE_VMASK = 0x2e,

  VPU_OP_V_GATHER_VV = 0x31,
  VPU_OP_V_SLIDE_V = 0x32,

  VPU_OP_V_MAX_VF = 0x35,
  VPU_OP_V_MIN_VF = 0x36,

  VPU_OP_C_WRITE_GP = 0x38,
  VPU_OP_C_WRITE_FP = 0x39,
  VPU_OP_C_WRITE_H = 0x3a,
  VPU_OP_C_SET_VL = 0x3b,
  VPU_OP_C_WAIT = 0x3c,
  VPU_OP_C_FENCE = 0x3d,
  VPU_OP_C_READ = 0x3e,
  VPU_OP_C_CLEAR_STATUS = 0x3f,
};

enum vpu_wait_mask {
  VPU_WAIT_LD = 1u << 0,
  VPU_WAIT_EX = 1u << 1,
  VPU_WAIT_ST = 1u << 2,
  VPU_WAIT_ALL = VPU_WAIT_LD | VPU_WAIT_EX | VPU_WAIT_ST,
};

enum vpu_clear_mask {
  VPU_CLEAR_FFLAGS = 1u << 0,
  VPU_CLEAR_FAULT_ILLEGAL = 1u << 1,
  VPU_CLEAR_PERF = 1u << 2,
  /* Compatibility names: ERRORS clears both numerical and fault status. */
  VPU_CLEAR_ERRORS = VPU_CLEAR_FFLAGS | VPU_CLEAR_FAULT_ILLEGAL,
  VPU_CLEAR_ALL = VPU_CLEAR_ERRORS | VPU_CLEAR_PERF,
};

enum vpu_read_selector {
  VPU_READ_STATUS = 0,
  VPU_READ_GP = 1,
  VPU_READ_FP = 2,
  VPU_READ_H = 3,
  VPU_READ_PERF = 4,
  VPU_READ_FAULT_ADDRESS = 5,
  VPU_READ_FAULT_INFO = 6,
};

enum vpu_fault_info {
  VPU_FAULT_CAUSE_MASK = 0x3u,
  VPU_FAULT_CAUSE_NONE = 0u,
  VPU_FAULT_CAUSE_TRANSLATION = 1u,
  VPU_FAULT_CAUSE_ACCESS = 2u,
  VPU_FAULT_CAUSE_STREAM_PROTOCOL = 3u,
  VPU_FAULT_IS_WRITE = 1u << 2,
  VPU_FAULT_VALID = 1u << 3,
};

enum vpu_perf_counter {
  VPU_PERF_CYCLES = 0,
  VPU_PERF_BUSY_CYCLES = 1,
  VPU_PERF_DMA_READ_BYTES = 2,
  VPU_PERF_DMA_WRITE_BYTES = 3,
  VPU_PERF_DMA_EXEC_OVERLAP_CYCLES = 4,
  VPU_PERF_BANK_CONFLICT_STALL_CYCLES = 5,
  VPU_PERF_HAZARD_STALL_CYCLES = 6,
  VPU_PERF_SFU_BUSY_CYCLES = 7,
  VPU_PERF_FAULTS = 8,
};

enum vpu_status_bits {
  VPU_STATUS_ILLEGAL_COMMAND = 1ull << 0,
  VPU_STATUS_DMA_FAULT = 1ull << 1,
  VPU_STATUS_DMA_HALTED = 1ull << 2,
  VPU_STATUS_FFLAG_NX = 1ull << 8,
  VPU_STATUS_FFLAG_UF = 1ull << 9,
  VPU_STATUS_FFLAG_OF = 1ull << 10,
  VPU_STATUS_FFLAG_DZ = 1ull << 11,
  VPU_STATUS_FFLAG_NV = 1ull << 12,
  VPU_STATUS_BUSY = 1ull << 16,
};

#define VPU_STATUS_ERROR_MASK \
  (VPU_STATUS_ILLEGAL_COMMAND | VPU_STATUS_DMA_FAULT)
#define VPU_STATUS_FFLAGS_MASK (0x1full << 8)

/*
 * [5:0] opcode, [9:6] rd, [13:10] rs1, [17:14] rs2,
 * [21:18] rs3/rmask, [25:22] funct1, [31:26] zero.
 */
#define VPU_MICRO_OP(op_, rd_, rs1_, rs2_, rs3_, funct1_)             \
  ((((uint32_t)(op_) & 0x3fu) << 0) |                                 \
   (((uint32_t)(rd_) & 0x0fu) << 6) |                                 \
   (((uint32_t)(rs1_) & 0x0fu) << 10) |                               \
   (((uint32_t)(rs2_) & 0x0fu) << 14) |                               \
   (((uint32_t)(rs3_) & 0x0fu) << 18) |                               \
   (((uint32_t)(funct1_) & 0x0fu) << 22))

static inline uint32_t vpu_micro_op(unsigned op, unsigned rd,
                                    unsigned rs1, unsigned rs2,
                                    unsigned rs3, unsigned funct1) {
  return VPU_MICRO_OP(op, rd, rs1, rs2, rs3, funct1);
}

static inline unsigned vpu_micro_opcode(uint32_t uop) {
  return (unsigned)(uop & 0x3fu);
}

static inline unsigned vpu_micro_rd(uint32_t uop) {
  return (unsigned)((uop >> 6) & 0x0fu);
}

static inline unsigned vpu_micro_rs1(uint32_t uop) {
  return (unsigned)((uop >> 10) & 0x0fu);
}

static inline unsigned vpu_micro_rs2(uint32_t uop) {
  return (unsigned)((uop >> 14) & 0x0fu);
}

static inline unsigned vpu_micro_rs3(uint32_t uop) {
  return (unsigned)((uop >> 18) & 0x0fu);
}

static inline unsigned vpu_micro_funct1(uint32_t uop) {
  return (unsigned)((uop >> 22) & 0x0fu);
}

static inline uint32_t vpu_float_to_bits(float value) {
  union {
    float value;
    uint32_t bits;
  } u = {value};
  return u.bits;
}

static inline float vpu_bits_to_float(uint32_t bits) {
  union {
    uint32_t bits;
    float value;
  } u = {bits};
  return u.value;
}

static inline uint16_t vpu_float_to_bf16(float value) {
  uint32_t bits = vpu_float_to_bits(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) {
    return (uint16_t)0x7fc0u;
  }
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

static inline float vpu_bf16_to_float(uint16_t value) {
  return vpu_bits_to_float((uint32_t)value << 16);
}

#if VPU_STORAGE_KIND == VPU_STORAGE_FP32
typedef float vpu_storage_t;
typedef uint32_t vpu_index_t;
static inline vpu_storage_t vpu_float_to_storage(float value) {
  return value;
}
static inline float vpu_storage_to_float(vpu_storage_t value) {
  return value;
}
#else
typedef uint16_t vpu_storage_t;
typedef uint16_t vpu_index_t;
static inline vpu_storage_t vpu_float_to_storage(float value) {
  return vpu_float_to_bf16(value);
}
static inline float vpu_storage_to_float(vpu_storage_t value) {
  return vpu_bf16_to_float(value);
}
#endif

#if defined(__riscv)
#include "rocc-software/src/xcustom.h"

static inline void vpu_cpu_memory_fence(void) {
  __asm__ volatile("fence rw, rw" ::: "memory");
}

static inline void vpu_rocc_issue(uint32_t uop, uint64_t payload) {
  __asm__ volatile("" ::: "memory");
  ROCC_INSTRUCTION_0_R_R(VPU_XCUSTOM, (uint64_t)uop, payload,
                         VPU_ROCC_FUNCT);
  __asm__ volatile("" ::: "memory");
}

static inline uint64_t vpu_rocc_issue_result(uint32_t uop,
                                             uint64_t payload) {
  uint64_t result;
  __asm__ volatile("" ::: "memory");
  ROCC_INSTRUCTION_R_R_R(VPU_XCUSTOM, result, (uint64_t)uop, payload,
                         VPU_ROCC_FUNCT);
  __asm__ volatile("" ::: "memory");
  return result;
}
#else
/* Host-side encoders can provide these hooks to emulate or trace commands. */
void vpu_host_issue(uint32_t uop, uint64_t payload);
uint64_t vpu_host_issue_result(uint32_t uop, uint64_t payload);

static inline void vpu_cpu_memory_fence(void) {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline void vpu_rocc_issue(uint32_t uop, uint64_t payload) {
  vpu_host_issue(uop, payload);
}

static inline uint64_t vpu_rocc_issue_result(uint32_t uop,
                                             uint64_t payload) {
  return vpu_host_issue_result(uop, payload);
}
#endif

/*
 * Publish host/CPU stores before enqueueing a batch of VPU DMA reads.  Call
 * this once after producing the input buffers, rather than before every
 * prefetch: on Rocket a RISC-V fence also waits for all RoCCs to become idle
 * and would otherwise serialize software-managed double buffering.
 */
static inline void vpu_publish_cpu_writes(void) {
  vpu_cpu_memory_fence();
}

static inline void vpu_write_gp(unsigned index, uint32_t value) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_C_WRITE_GP, index, 0, 0, 0, 0),
                 value);
}

static inline void vpu_write_fp_bits(unsigned index, uint32_t bits) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_C_WRITE_FP, index, 0, 0, 0, 0),
                 bits);
}

static inline void vpu_write_fp(unsigned index, float value) {
  vpu_write_fp_bits(index, vpu_float_to_bits(value));
}

static inline void vpu_write_h(unsigned index, uintptr_t address) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_C_WRITE_H, index, 0, 0, 0, 0),
                 (uint64_t)address);
}

/*
 * The architectural vector mask contains one bit per VLEN element.  Because
 * the RoCC payload is 64 bits, C_WRITE_VMASK selects one 64-bit chunk with
 * rd.  Vector commands snapshot all chunks when they are dispatched, so a
 * later write cannot change an already accepted operation.
 */
#ifndef VPU_VMASK_CHUNK_BITS
#define VPU_VMASK_CHUNK_BITS 64u
#endif
#ifndef VPU_VMASK_CHUNKS
#define VPU_VMASK_CHUNKS \
  ((VPU_VLEN + VPU_VMASK_CHUNK_BITS - 1u) / VPU_VMASK_CHUNK_BITS)
#endif

#if VPU_VMASK_CHUNKS > 16
#error "C_WRITE_VMASK's four-bit rd field supports at most 1024 mask bits"
#endif

static inline uint64_t vpu_vmask_valid_bits(unsigned chunk) {
  const size_t first = (size_t)chunk * VPU_VMASK_CHUNK_BITS;
  if (first >= VPU_VLEN) {
    return 0;
  }
  const size_t remaining = (size_t)VPU_VLEN - first;
  return remaining >= VPU_VMASK_CHUNK_BITS
      ? UINT64_MAX
      : (UINT64_C(1) << remaining) - UINT64_C(1);
}

static inline void vpu_write_vmask_chunk(unsigned chunk, uint64_t bits) {
  vpu_rocc_issue(
      vpu_micro_op(VPU_OP_C_WRITE_VMASK, chunk, 0, 0, 0, 0), bits);
}

static inline void vpu_write_vmask(const uint64_t *chunks) {
  for (unsigned chunk = 0; chunk < VPU_VMASK_CHUNKS; ++chunk) {
    vpu_write_vmask_chunk(chunk,
                          chunks[chunk] & vpu_vmask_valid_bits(chunk));
  }
}

static inline void vpu_clear_vmask(void) {
  for (unsigned chunk = 0; chunk < VPU_VMASK_CHUNKS; ++chunk) {
    vpu_write_vmask_chunk(chunk, 0);
  }
}

static inline void vpu_set_vmask_all(void) {
  for (unsigned chunk = 0; chunk < VPU_VMASK_CHUNKS; ++chunk) {
    vpu_write_vmask_chunk(chunk, vpu_vmask_valid_bits(chunk));
  }
}

/* Host stride is expressed in bytes, matching Gemmini config_mvin and the
 * external-memory interface. Each accepted 2-D memory command snapshots it. */
static inline void vpu_set_stride_bytes(uint64_t stride_bytes) {
  vpu_rocc_issue(
      vpu_micro_op(VPU_OP_C_SET_STRIDE, 0, 0, 0, 0, 0), stride_bytes);
}

static inline void vpu_set_vl(size_t vl) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_C_SET_VL, 0, 0, 0, 0, 0),
                 (uint64_t)vl);
}

static inline void vpu_wait(unsigned mask) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_C_WAIT, 0, 0, 0, 0, 0), mask);
}

static inline uint64_t vpu_fence(void) {
  const uint64_t status = vpu_rocc_issue_result(
      vpu_micro_op(VPU_OP_C_FENCE, 0, 0, 0, 0, 0), 0);
  /* Order CPU loads after all VPU DMA writes drained by C_FENCE. */
  vpu_cpu_memory_fence();
  return status;
}

static inline uint64_t vpu_read(unsigned selector, unsigned index) {
  return vpu_rocc_issue_result(
      vpu_micro_op(VPU_OP_C_READ, index, selector, 0, 0, 0), 0);
}

static inline uint64_t vpu_read_status(void) {
  return vpu_read(VPU_READ_STATUS, 0);
}

static inline uint32_t vpu_read_gp(unsigned index) {
  return (uint32_t)vpu_read(VPU_READ_GP, index);
}

static inline float vpu_read_fp(unsigned index) {
  return vpu_bits_to_float((uint32_t)vpu_read(VPU_READ_FP, index));
}

static inline uint64_t vpu_read_h(unsigned index) {
  return vpu_read(VPU_READ_H, index);
}

static inline uint64_t vpu_read_perf(unsigned index) {
  return vpu_read(VPU_READ_PERF, index);
}

static inline uint64_t vpu_read_fault_address(void) {
  return vpu_read(VPU_READ_FAULT_ADDRESS, 0);
}

static inline uint64_t vpu_read_fault_info(void) {
  return vpu_read(VPU_READ_FAULT_INFO, 0);
}

static inline void vpu_clear_status(unsigned mask) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_C_CLEAR_STATUS, 0, 0, 0, 0, 0),
                 mask);
}

/*
 * H[base_h] + GP[external_offset_gp] * VPU_STORAGE_BYTES is the external
 * address. GP[vector_address_gp] is the element address in Vector SRAM.
 */
static inline void vpu_h_prefetch_v(unsigned vector_address_gp,
                                    unsigned external_offset_gp,
                                    unsigned base_h) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_H_PREFETCH_V, vector_address_gp,
                              external_offset_gp, base_h, 0, 0),
                 0);
}

static inline void vpu_h_store_v(unsigned vector_address_gp,
                                 unsigned external_offset_gp,
                                 unsigned base_h) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_H_STORE_V, vector_address_gp,
                              external_offset_gp, base_h, 0, 0),
                 0);
}

/*
 * 2-D DMA keeps the ordinary address operands and names a separate GP
 * register containing rowCount in rs3.  Row r transfers current VL elements:
 *
 *   host  = H[base_h] + GP[offset_gp]*storageBytes + r*hostStrideBytes
 *   VSRAM = GP[vector_address_gp] + r*VPU_VLEN
 *
 * Keep row_count_gp distinct from all three address registers: descriptors
 * snapshot values at dispatch, but sharing registers makes surrounding
 * software updates unnecessarily error-prone.
 */
static inline void vpu_h_prefetch_v_2d(unsigned vector_address_gp,
                                       unsigned external_offset_gp,
                                       unsigned base_h,
                                       unsigned row_count_gp) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_H_PREFETCH_V, vector_address_gp,
                              external_offset_gp, base_h, row_count_gp, 1),
                 0);
}

static inline void vpu_h_store_v_2d(unsigned vector_address_gp,
                                    unsigned external_offset_gp,
                                    unsigned base_h,
                                    unsigned row_count_gp) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_H_STORE_V, vector_address_gp,
                              external_offset_gp, base_h, row_count_gp, 1),
                 0);
}

#define VPU_DEFINE_VV(name_, opcode_)                                      \
  static inline void name_(unsigned dst_gp, unsigned lhs_gp,               \
                           unsigned rhs_gp) {                               \
    vpu_rocc_issue(vpu_micro_op((opcode_), dst_gp, lhs_gp, rhs_gp, 0, 0), 0); \
  }

#define VPU_DEFINE_VF(name_, opcode_)                                      \
  static inline void name_(unsigned dst_gp, unsigned src_gp,               \
                           unsigned scalar_fp) {                            \
    vpu_rocc_issue(                                                         \
        vpu_micro_op((opcode_), dst_gp, src_gp, scalar_fp, 0, 0), 0);       \
  }

VPU_DEFINE_VV(vpu_v_add_vv, VPU_OP_V_ADD_VV)
VPU_DEFINE_VF(vpu_v_add_vf, VPU_OP_V_ADD_VF)
VPU_DEFINE_VV(vpu_v_sub_vv, VPU_OP_V_SUB_VV)
VPU_DEFINE_VV(vpu_v_mul_vv, VPU_OP_V_MUL_VV)
VPU_DEFINE_VF(vpu_v_mul_vf, VPU_OP_V_MUL_VF)
VPU_DEFINE_VF(vpu_v_max_vf, VPU_OP_V_MAX_VF)
VPU_DEFINE_VF(vpu_v_min_vf, VPU_OP_V_MIN_VF)

#undef VPU_DEFINE_VV
#undef VPU_DEFINE_VF

#define VPU_DEFINE_MASKED_VV(name_, opcode_)                              \
  static inline void name_(unsigned dst_gp, unsigned lhs_gp,              \
                           unsigned rhs_gp) {                              \
    vpu_rocc_issue(                                                        \
        vpu_micro_op((opcode_), dst_gp, lhs_gp, rhs_gp, 1, 0), 0);         \
  }

#define VPU_DEFINE_MASKED_VF(name_, opcode_)                              \
  static inline void name_(unsigned dst_gp, unsigned src_gp,              \
                           unsigned scalar_fp) {                           \
    vpu_rocc_issue(                                                        \
        vpu_micro_op((opcode_), dst_gp, src_gp, scalar_fp, 1, 0), 0);      \
  }

VPU_DEFINE_MASKED_VV(vpu_v_add_vv_masked, VPU_OP_V_ADD_VV)
VPU_DEFINE_MASKED_VF(vpu_v_add_vf_masked, VPU_OP_V_ADD_VF)
VPU_DEFINE_MASKED_VV(vpu_v_sub_vv_masked, VPU_OP_V_SUB_VV)
VPU_DEFINE_MASKED_VV(vpu_v_mul_vv_masked, VPU_OP_V_MUL_VV)
VPU_DEFINE_MASKED_VF(vpu_v_mul_vf_masked, VPU_OP_V_MUL_VF)
VPU_DEFINE_MASKED_VF(vpu_v_max_vf_masked, VPU_OP_V_MAX_VF)
VPU_DEFINE_MASKED_VF(vpu_v_min_vf_masked, VPU_OP_V_MIN_VF)

#undef VPU_DEFINE_MASKED_VV
#undef VPU_DEFINE_MASKED_VF

static inline void vpu_v_sub_vf(unsigned dst_gp, unsigned vector_gp,
                                unsigned scalar_fp,
                                bool scalar_minus_vector) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_V_SUB_VF, dst_gp, vector_gp,
                              scalar_fp, 0,
                              scalar_minus_vector ? 1u : 0u),
                 0);
}

static inline void vpu_v_sub_vf_masked(unsigned dst_gp, unsigned vector_gp,
                                       unsigned scalar_fp,
                                       bool scalar_minus_vector) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_V_SUB_VF, dst_gp, vector_gp,
                              scalar_fp, 1,
                              scalar_minus_vector ? 1u : 0u),
                 0);
}

static inline void vpu_v_exp_v(unsigned dst_gp, unsigned src_gp) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_V_EXP_V, dst_gp, src_gp, 0, 0, 0), 0);
}

static inline void vpu_v_exp_v_masked(unsigned dst_gp, unsigned src_gp) {
  vpu_rocc_issue(
      vpu_micro_op(VPU_OP_V_EXP_V, dst_gp, src_gp, 0, 1, 0), 0);
}

static inline void vpu_v_reci_v(unsigned dst_gp, unsigned src_gp) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_V_RECI_V, dst_gp, src_gp, 0, 0, 0), 0);
}

static inline void vpu_v_reci_v_masked(unsigned dst_gp, unsigned src_gp) {
  vpu_rocc_issue(
      vpu_micro_op(VPU_OP_V_RECI_V, dst_gp, src_gp, 0, 1, 0), 0);
}

static inline void vpu_v_red_sum(unsigned dst_fp, unsigned src_gp) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_V_RED_SUM, dst_fp, src_gp, 0, 0, 0), 0);
}

static inline void vpu_v_red_sum_masked(unsigned dst_fp, unsigned src_gp) {
  vpu_rocc_issue(
      vpu_micro_op(VPU_OP_V_RED_SUM, dst_fp, src_gp, 0, 1, 0), 0);
}

static inline void vpu_v_red_max(unsigned dst_fp, unsigned src_gp) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_V_RED_MAX, dst_fp, src_gp, 0, 0, 0), 0);
}

static inline void vpu_v_red_max_masked(unsigned dst_fp, unsigned src_gp) {
  vpu_rocc_issue(
      vpu_micro_op(VPU_OP_V_RED_MAX, dst_fp, src_gp, 0, 1, 0), 0);
}

/*
 * Gather indices are raw unsigned storage-width values.  An index outside the
 * current VL produces +0.  Destination aliasing either source is illegal.
 */
static inline void vpu_v_gather_vv(unsigned dst_gp, unsigned src_gp,
                                   unsigned index_gp) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_V_GATHER_VV, dst_gp, src_gp,
                              index_gp, 0, 0),
                 0);
}

static inline void vpu_v_gather_vv_masked(unsigned dst_gp, unsigned src_gp,
                                          unsigned index_gp) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_V_GATHER_VV, dst_gp, src_gp,
                              index_gp, 1, 0),
                 0);
}

enum vpu_slide_direction {
  /* dst[i] = i >= shift ? src[i - shift] : +0 */
  VPU_SLIDE_RIGHT = 0,
  /* dst[i] = i + shift < VL ? src[i + shift] : +0 */
  VPU_SLIDE_LEFT = 1,
};

static inline void vpu_v_slide_v(unsigned dst_gp, unsigned src_gp,
                                 unsigned shift_gp,
                                 enum vpu_slide_direction direction) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_V_SLIDE_V, dst_gp, src_gp,
                              shift_gp, 0, (unsigned)direction),
                 0);
}

static inline void vpu_v_slide_v_masked(
    unsigned dst_gp, unsigned src_gp, unsigned shift_gp,
    enum vpu_slide_direction direction) {
  vpu_rocc_issue(vpu_micro_op(VPU_OP_V_SLIDE_V, dst_gp, src_gp,
                              shift_gp, 1, (unsigned)direction),
                 0);
}

#define VPU_DEFINE_SS(name_, opcode_)                                      \
  static inline void name_(unsigned dst_fp, unsigned lhs_fp,               \
                           unsigned rhs_fp) {                               \
    vpu_rocc_issue(vpu_micro_op((opcode_), dst_fp, lhs_fp, rhs_fp, 0, 0), 0); \
  }

#define VPU_DEFINE_S(name_, opcode_)                                       \
  static inline void name_(unsigned dst_fp, unsigned src_fp) {             \
    vpu_rocc_issue(vpu_micro_op((opcode_), dst_fp, src_fp, 0, 0, 0), 0);   \
  }

VPU_DEFINE_SS(vpu_s_add, VPU_OP_S_ADD_FP)
VPU_DEFINE_SS(vpu_s_sub, VPU_OP_S_SUB_FP)
VPU_DEFINE_SS(vpu_s_max, VPU_OP_S_MAX_FP)
VPU_DEFINE_SS(vpu_s_mul, VPU_OP_S_MUL_FP)
VPU_DEFINE_S(vpu_s_exp, VPU_OP_S_EXP_FP)
VPU_DEFINE_S(vpu_s_reci, VPU_OP_S_RECI_FP)
VPU_DEFINE_S(vpu_s_sqrt, VPU_OP_S_SQRT_FP)

#undef VPU_DEFINE_SS
#undef VPU_DEFINE_S

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GEMMINI_ROCC_TESTS_INCLUDE_VPU_H_
