// See LICENSE for license details.

#ifndef GEMMINI_ROCC_TESTS_INCLUDE_VPU_PARAMS_H_
#define GEMMINI_ROCC_TESTS_INCLUDE_VPU_PARAMS_H_

/*
 * Stable software defaults for the standalone VPU.
 *
 * The default FP32 SoC writes include/vpu_params_generated.h, which is
 * selected automatically.  Configurations with a different software ABI use
 * a distinct generated artifact (the supplied BF16-storage config writes
 * vpu_params_bf16_generated.h). Select such a header, or a relocated one,
 * explicitly with, for example:
 *
 *   -DVPU_PARAMS_GENERATED_HEADER='"vpu_params_bf16_generated.h"'
 *
 * Every fallback below is guarded so values from the selected generated
 * hardware header take precedence over the documented FP32 defaults.
 */
#if defined(VPU_PARAMS_GENERATED_HEADER)
#include VPU_PARAMS_GENERATED_HEADER
#define VPU_HAS_GENERATED_PARAMS 1
#elif defined(__has_include)
#if __has_include("vpu_params_generated.h")
#include "vpu_params_generated.h"
#define VPU_HAS_GENERATED_PARAMS 1
#endif
#endif

#ifndef VPU_STORAGE_FP32
#define VPU_STORAGE_FP32 0
#endif
#ifndef VPU_STORAGE_BF16
#define VPU_STORAGE_BF16 1
#endif

#ifndef VPU_VLEN
#define VPU_VLEN 128u
#define VPU_USING_DEFAULT_PARAMS 1
#endif

#ifndef VPU_NLANES
#define VPU_NLANES 16u
#endif

#ifndef VPU_SFU_LANES
#define VPU_SFU_LANES 4u
#endif

#ifndef VPU_RECIPROCAL_LANES
#define VPU_RECIPROCAL_LANES (VPU_NLANES / 4u)
#endif

#ifndef VPU_RECIPROCAL_FMAS_PER_LANE
#define VPU_RECIPROCAL_FMAS_PER_LANE 4u
#endif

#ifndef VPU_RECIPROCAL_LATENCY
#define VPU_RECIPROCAL_LATENCY 13u
#endif

#ifndef VPU_VSPAD_KIB
#define VPU_VSPAD_KIB 64u
#endif

#ifndef VPU_VSPAD_BANKS
#define VPU_VSPAD_BANKS 8u
#endif

#ifndef VPU_STORAGE_KIND
#define VPU_STORAGE_KIND VPU_STORAGE_FP32
#endif

#ifndef VPU_COMPUTE_KIND
#define VPU_COMPUTE_KIND VPU_STORAGE_FP32
#endif

#if VPU_COMPUTE_KIND != VPU_STORAGE_FP32
#error "VPU v1 compute type must be FP32"
#endif

#ifndef VPU_DMA_BUS_BITS
#define VPU_DMA_BUS_BITS 128u
#endif

#ifndef VPU_DMA_MAX_BYTES
#define VPU_DMA_MAX_BYTES 64u
#endif

#ifndef VPU_DMA_MAX_IN_FLIGHT
#define VPU_DMA_MAX_IN_FLIGHT 8u
#endif

#ifndef VPU_TLB_ENTRIES
#define VPU_TLB_ENTRIES 4u
#endif

#ifndef VPU_LOAD_QUEUE_ENTRIES
#define VPU_LOAD_QUEUE_ENTRIES 4u
#endif

#ifndef VPU_EXEC_QUEUE_ENTRIES
#define VPU_EXEC_QUEUE_ENTRIES 8u
#endif

#ifndef VPU_STORE_QUEUE_ENTRIES
#define VPU_STORE_QUEUE_ENTRIES 4u
#endif

#ifndef VPU_HAZARD_ENTRIES
#define VPU_HAZARD_ENTRIES 17u
#endif

#ifndef VPU_EXP_TABLE_ENTRIES
#define VPU_EXP_TABLE_ENTRIES 64u
#endif

#ifndef VPU_RECIPROCAL_REFINE_ITERS
#define VPU_RECIPROCAL_REFINE_ITERS 2u
#endif

#ifndef VPU_FMA_PIPE_DEPTH
#define VPU_FMA_PIPE_DEPTH 4u
#endif

#if VPU_STORAGE_KIND == VPU_STORAGE_FP32
#define VPU_STORAGE_BITS 32u
#elif VPU_STORAGE_KIND == VPU_STORAGE_BF16
#define VPU_STORAGE_BITS 16u
#else
#error "VPU_STORAGE_KIND must be VPU_STORAGE_FP32 or VPU_STORAGE_BF16"
#endif

#define VPU_STORAGE_BYTES (VPU_STORAGE_BITS / 8u)
#define VPU_VSPAD_BYTES (VPU_VSPAD_KIB * 1024u)
#define VPU_VSPAD_ELEMENTS (VPU_VSPAD_BYTES / VPU_STORAGE_BYTES)
#define VPU_ELEMENTS_PER_BANK (VPU_VSPAD_ELEMENTS / VPU_VSPAD_BANKS)
#define VPU_VECTOR_BYTES (VPU_VLEN * VPU_STORAGE_BYTES)
#define VPU_SLOTS_PER_BANK (VPU_ELEMENTS_PER_BANK / VPU_VLEN)

#ifndef VPU_DMA_MAX_ROWS
#define VPU_DMA_MAX_ROWS VPU_SLOTS_PER_BANK
#endif

/* VSRAM addresses are element addresses, not byte addresses. */
#define VPU_BANK_BASE(bank_) ((unsigned)(bank_) * VPU_ELEMENTS_PER_BANK)
#define VPU_SLOT_ADDR(bank_, slot_)                                      \
  (VPU_BANK_BASE(bank_) + (unsigned)(slot_) * VPU_VLEN)

/* Recommended software-managed double-buffer layout. */
#define VPU_PING_INPUT_ADDR VPU_BANK_BASE(0u)
#define VPU_PING_TEMP0_ADDR VPU_BANK_BASE(1u)
#define VPU_PING_TEMP1_ADDR VPU_BANK_BASE(2u)
#define VPU_PING_OUTPUT_ADDR VPU_BANK_BASE(3u)
#define VPU_PONG_INPUT_ADDR VPU_BANK_BASE(4u)
#define VPU_PONG_TEMP0_ADDR VPU_BANK_BASE(5u)
#define VPU_PONG_TEMP1_ADDR VPU_BANK_BASE(6u)
#define VPU_PONG_OUTPUT_ADDR VPU_BANK_BASE(7u)

#if (VPU_VLEN == 0) || ((VPU_VLEN % VPU_NLANES) != 0)
#error "VPU_VLEN must be a non-zero multiple of VPU_NLANES"
#endif

#if (VPU_SFU_LANES == 0) || ((VPU_NLANES % VPU_SFU_LANES) != 0)
#error "VPU_SFU_LANES must be a non-zero divisor of VPU_NLANES"
#endif

#if (VPU_RECIPROCAL_LANES == 0) || \
    ((VPU_NLANES % VPU_RECIPROCAL_LANES) != 0) || \
    ((VPU_RECIPROCAL_LANES * VPU_RECIPROCAL_FMAS_PER_LANE) != VPU_NLANES)
#error "VPU reciprocal lanes must exactly partition the shared FMA lanes"
#endif

#if (VPU_VSPAD_BANKS == 0) || ((VPU_VSPAD_BANKS & (VPU_VSPAD_BANKS - 1)) != 0)
#error "VPU_VSPAD_BANKS must be a power of two"
#endif

#if ((VPU_VSPAD_BYTES % VPU_VSPAD_BANKS) != 0) || \
    ((VPU_VSPAD_BYTES % VPU_STORAGE_BYTES) != 0)
#error "VPU scratchpad size must divide evenly into banks and elements"
#endif

#if (VPU_ELEMENTS_PER_BANK < VPU_VLEN) || \
    ((VPU_ELEMENTS_PER_BANK % VPU_VLEN) != 0)
#error "Each VPU bank must contain an integral number of architectural vectors"
#endif

#if VPU_SLOTS_PER_BANK == 0
#error "Each VPU bank must expose at least one architectural vector slot"
#endif

#if (VPU_SLOTS_PER_BANK * VPU_VLEN) != VPU_ELEMENTS_PER_BANK
#error "VPU slot geometry must cover each bank exactly"
#endif

#if VPU_VSPAD_BANKS < 8
#error "The public ping/pong VSRAM layout requires at least eight banks"
#endif

#if VPU_FMA_PIPE_DEPTH != 4
#error "The VPU v1 software ABI requires VPU_FMA_PIPE_DEPTH=4"
#endif

#endif  // GEMMINI_ROCC_TESTS_INCLUDE_VPU_PARAMS_H_
