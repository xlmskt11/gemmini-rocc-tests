// See LICENSE for license details.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "include/vpu.h"

#define VPU_DMA_TEST_TILES 8u
#define VPU_DMA_TEST_ELEMENTS (VPU_DMA_TEST_TILES * VPU_VLEN)
#define VPU_DMA_PAGE_BYTES 4096u
#define VPU_DMA_PAGE_COUNT 3u
#define VPU_DMA_PAGE_CROSS_VL 17u
#define VPU_DMA_PAGE_ELEMENTS (VPU_DMA_PAGE_BYTES / VPU_STORAGE_BYTES)
#define VPU_DMA_PAGE_BUFFER_ELEMENTS \
  (VPU_DMA_PAGE_COUNT * VPU_DMA_PAGE_ELEMENTS)
#define VPU_DMA_PAGE_CROSS_BASE \
  (2u * VPU_DMA_PAGE_ELEMENTS - 2u)

#ifndef VPU_DMA_TEST_REQUIRE_OVERLAP
#define VPU_DMA_TEST_REQUIRE_OVERLAP 1
#endif

enum {
  GP_PING_ADDRESS = 0,
  GP_PONG_ADDRESS = 1,
  GP_CURRENT_OFFSET = 2,
  GP_NEXT_OFFSET = 3,
  FP_ONE = 0,
  H_INPUT = 0,
  H_OUTPUT = 1,
};

/* The +1 payload makes both DMA bases element-aligned but cache-line unaligned. */
static vpu_storage_t input[VPU_DMA_TEST_ELEMENTS + 2]
    __attribute__((aligned(64)));
static vpu_storage_t output[VPU_DMA_TEST_ELEMENTS + 2]
    __attribute__((aligned(64)));

/*
 * Separate, page-aligned objects make the virtual 4 KiB boundary independent
 * of the linker placement of the ordinary ping/pong buffers.  The transfer
 * begins two elements before the second page boundary and continues into the
 * third page; all elements outside the payload act as surrounding guards.
 */
static vpu_storage_t page_cross_input[VPU_DMA_PAGE_BUFFER_ELEMENTS]
    __attribute__((aligned(VPU_DMA_PAGE_BYTES)));
static vpu_storage_t page_cross_output[VPU_DMA_PAGE_BUFFER_ELEMENTS]
    __attribute__((aligned(VPU_DMA_PAGE_BYTES)));

#if (VPU_DMA_PAGE_BYTES % VPU_STORAGE_BYTES) != 0
#error "VPU storage elements must divide a 4 KiB page"
#endif

#if VPU_VLEN < VPU_DMA_PAGE_CROSS_VL
#error "vpu_dma page-crossing test requires VPU_VLEN >= 17"
#endif

static int test_zero_vl(void) {
  vpu_clear_status(VPU_CLEAR_ALL);
  vpu_set_vl(0);
  vpu_write_gp(GP_PING_ADDRESS, VPU_PING_INPUT_ADDR);
  vpu_write_gp(GP_CURRENT_OFFSET, 0);
  vpu_write_h(H_INPUT, 0);
  vpu_write_h(H_OUTPUT, 0);
  vpu_h_prefetch_v(GP_PING_ADDRESS, GP_CURRENT_OFFSET, H_INPUT);
  vpu_h_store_v(GP_PING_ADDRESS, GP_CURRENT_OFFSET, H_OUTPUT);
  const uint64_t status = vpu_fence();
  if ((status & VPU_STATUS_ERROR_MASK) != 0) {
    printf("VL=0 unexpectedly accessed memory: status=0x%llx\n",
           (unsigned long long)status);
    return 1;
  }
  return 0;
}

static int test_page_crossing(void) {
  const vpu_storage_t guard = vpu_float_to_storage(77.0f);
  const vpu_storage_t output_sentinel = vpu_float_to_storage(-321.0f);
  const size_t payload_begin = VPU_DMA_PAGE_CROSS_BASE;
  const size_t payload_end = payload_begin + VPU_DMA_PAGE_CROSS_VL;
  vpu_storage_t *const input_base = &page_cross_input[payload_begin];
  vpu_storage_t *const output_base = &page_cross_output[payload_begin];

  for (size_t i = 0; i < VPU_DMA_PAGE_BUFFER_ELEMENTS; ++i) {
    page_cross_input[i] = guard;
    page_cross_output[i] = guard;
  }
  for (size_t i = 0; i < VPU_DMA_PAGE_CROSS_VL; ++i) {
    const float value = (float)((int)i - 8) * 0.5f;
    input_base[i] = vpu_float_to_storage(value);
    output_base[i] = output_sentinel;
  }

  const uintptr_t input_first = (uintptr_t)input_base;
  const uintptr_t input_last =
      (uintptr_t)&input_base[VPU_DMA_PAGE_CROSS_VL - 1u] +
      VPU_STORAGE_BYTES - 1u;
  const uintptr_t output_first = (uintptr_t)output_base;
  const uintptr_t output_last =
      (uintptr_t)&output_base[VPU_DMA_PAGE_CROSS_VL - 1u] +
      VPU_STORAGE_BYTES - 1u;
  const uintptr_t page_mask = (uintptr_t)VPU_DMA_PAGE_BYTES - 1u;
  if ((((uintptr_t)page_cross_input | (uintptr_t)page_cross_output) &
       page_mask) != 0u ||
      (input_first % VPU_STORAGE_BYTES) != 0u ||
      (output_first % VPU_STORAGE_BYTES) != 0u ||
      (input_first & ~page_mask) == (input_last & ~page_mask) ||
      (output_first & ~page_mask) == (output_last & ~page_mask)) {
    printf("VPU page-crossing buffers do not cross an aligned 4 KiB "
           "boundary\n");
    return 1;
  }

  vpu_publish_cpu_writes();
  vpu_clear_status(VPU_CLEAR_ALL);
  vpu_set_vl(VPU_DMA_PAGE_CROSS_VL);
  vpu_write_gp(GP_PING_ADDRESS, VPU_PING_INPUT_ADDR);
  vpu_write_gp(GP_CURRENT_OFFSET, 0);
  vpu_write_fp(FP_ONE, 1.0f);
  vpu_write_h(H_INPUT, (uintptr_t)input_base);
  vpu_write_h(H_OUTPUT, (uintptr_t)output_base);
  vpu_h_prefetch_v(GP_PING_ADDRESS, GP_CURRENT_OFFSET, H_INPUT);
  vpu_v_add_vf(GP_PING_ADDRESS, GP_PING_ADDRESS, FP_ONE);
  vpu_h_store_v(GP_PING_ADDRESS, GP_CURRENT_OFFSET, H_OUTPUT);

  const uint64_t status = vpu_fence();
  if ((status & VPU_STATUS_ERROR_MASK) != 0) {
    printf("VPU page-crossing DMA failed with status 0x%llx\n",
           (unsigned long long)status);
    return 1;
  }

  for (size_t i = 0; i < VPU_DMA_PAGE_CROSS_VL; ++i) {
    const vpu_storage_t expected_input =
        vpu_float_to_storage((float)((int)i - 8) * 0.5f);
    const vpu_storage_t expected =
        vpu_float_to_storage(vpu_storage_to_float(expected_input) + 1.0f);
    if (input_base[i] != expected_input || output_base[i] != expected) {
      printf("VPU page-crossing mismatch at %u: input=0x%x got=0x%x "
             "expected=0x%x\n",
             (unsigned)i,
             vpu_float_to_bits(vpu_storage_to_float(input_base[i])),
             vpu_float_to_bits(vpu_storage_to_float(output_base[i])),
             vpu_float_to_bits(vpu_storage_to_float(expected)));
      return 1;
    }
  }

  for (size_t i = 0; i < VPU_DMA_PAGE_BUFFER_ELEMENTS; ++i) {
    if (i >= payload_begin && i < payload_end) {
      continue;
    }
    if (page_cross_input[i] != guard || page_cross_output[i] != guard) {
      printf("VPU page-crossing DMA corrupted a guard at element %u "
             "(input=%u output=%u)\n",
             (unsigned)i, page_cross_input[i] != guard,
             page_cross_output[i] != guard);
      return 1;
    }
  }

  const uint64_t expected_bytes =
      (uint64_t)VPU_DMA_PAGE_CROSS_VL * VPU_STORAGE_BYTES;
  const uint64_t read_bytes = vpu_read_perf(VPU_PERF_DMA_READ_BYTES);
  const uint64_t write_bytes = vpu_read_perf(VPU_PERF_DMA_WRITE_BYTES);
  const uint64_t fault_info = vpu_read_fault_info();
  if (read_bytes != expected_bytes || write_bytes != expected_bytes ||
      (fault_info & VPU_FAULT_VALID) != 0) {
    printf("VPU page-crossing accounting failed: read=%llu write=%llu "
           "expected=%llu fault=0x%llx\n",
           (unsigned long long)read_bytes,
           (unsigned long long)write_bytes,
           (unsigned long long)expected_bytes,
           (unsigned long long)fault_info);
    return 1;
  }

  return 0;
}

int main(void) {
  if (test_zero_vl()) {
    return 1;
  }

  const vpu_storage_t guard = vpu_float_to_storage(123.0f);
  input[0] = guard;
  input[VPU_DMA_TEST_ELEMENTS + 1] = guard;
  output[0] = guard;
  output[VPU_DMA_TEST_ELEMENTS + 1] = guard;
  for (size_t i = 0; i < VPU_DMA_TEST_ELEMENTS; ++i) {
    const float value = (float)((int)(i % 251u) - 125) * 0.25f;
    input[i + 1] = vpu_float_to_storage(value);
    output[i + 1] = vpu_float_to_storage(-321.0f);
  }

  vpu_publish_cpu_writes();

  vpu_clear_status(VPU_CLEAR_ALL);
  vpu_set_vl(VPU_VLEN);
  vpu_write_fp(FP_ONE, 1.0f);
  vpu_write_h(H_INPUT, (uintptr_t)&input[1]);
  vpu_write_h(H_OUTPUT, (uintptr_t)&output[1]);
  vpu_write_gp(GP_PING_ADDRESS, VPU_PING_INPUT_ADDR);
  vpu_write_gp(GP_PONG_ADDRESS, VPU_PONG_INPUT_ADDR);

  /*
   * Prime ping, then enqueue load(next) before execute/store(current).  The
   * VSRAM scoreboard holds execute until the priming load completes, while a
   * later load to the other bank can run concurrently with current execute.
   * Address and offset registers are distinct so changing the next tile never
   * changes a descriptor which has already been issued.
   */
  vpu_write_gp(GP_CURRENT_OFFSET, 0);
  vpu_h_prefetch_v(GP_PING_ADDRESS, GP_CURRENT_OFFSET, H_INPUT);

  for (size_t tile = 0; tile < VPU_DMA_TEST_TILES; ++tile) {
    const unsigned current_address_gp =
        (tile & 1u) ? GP_PONG_ADDRESS : GP_PING_ADDRESS;
    if (tile + 1u < VPU_DMA_TEST_TILES) {
      const unsigned next_address_gp =
          (tile & 1u) ? GP_PING_ADDRESS : GP_PONG_ADDRESS;
      vpu_write_gp(GP_NEXT_OFFSET,
                   (uint32_t)((tile + 1u) * VPU_VLEN));
      vpu_h_prefetch_v(next_address_gp, GP_NEXT_OFFSET, H_INPUT);
    }
    vpu_write_gp(GP_CURRENT_OFFSET, (uint32_t)(tile * VPU_VLEN));
    /* Multiplication by one creates an execute stage without changing data. */
    vpu_v_mul_vf(current_address_gp, current_address_gp, FP_ONE);
    vpu_h_store_v(current_address_gp, GP_CURRENT_OFFSET, H_OUTPUT);
  }

  const uint64_t status = vpu_fence();
  if ((status & VPU_STATUS_ERROR_MASK) != 0) {
    printf("VPU DMA test failed with status 0x%llx\n",
           (unsigned long long)status);
    return 1;
  }

  if (output[0] != guard || output[VPU_DMA_TEST_ELEMENTS + 1] != guard) {
    printf("VPU DMA wrote outside the requested range\n");
    return 1;
  }
  for (size_t i = 0; i < VPU_DMA_TEST_ELEMENTS; ++i) {
    const float actual = vpu_storage_to_float(output[i + 1]);
    const float expected = vpu_storage_to_float(input[i + 1]);
    if (vpu_float_to_bits(actual) != vpu_float_to_bits(expected)) {
      printf("VPU DMA mismatch at %u: got=0x%x expected=0x%x\n",
             (unsigned)i, vpu_float_to_bits(actual),
             vpu_float_to_bits(expected));
      return 1;
    }
  }

  const uint64_t expected_bytes =
      (uint64_t)VPU_DMA_TEST_ELEMENTS * VPU_STORAGE_BYTES;
  const uint64_t read_bytes = vpu_read_perf(VPU_PERF_DMA_READ_BYTES);
  const uint64_t write_bytes = vpu_read_perf(VPU_PERF_DMA_WRITE_BYTES);
  const uint64_t overlap =
      vpu_read_perf(VPU_PERF_DMA_EXEC_OVERLAP_CYCLES);
  const uint64_t fault_info = vpu_read_fault_info();
  (void)vpu_read_fault_address();
  const uint64_t status_after_reads = vpu_read_status();
  if ((fault_info & VPU_FAULT_VALID) != 0 ||
      (status_after_reads & VPU_STATUS_ERROR_MASK) != 0) {
    printf("VPU unexpectedly latched a DMA fault: info=0x%llx status=0x%llx\n",
           (unsigned long long)fault_info,
           (unsigned long long)status_after_reads);
    return 1;
  }
  if (read_bytes != expected_bytes || write_bytes != expected_bytes) {
    printf("VPU DMA byte counters mismatch: read=%llu write=%llu expected=%llu\n",
           (unsigned long long)read_bytes, (unsigned long long)write_bytes,
           (unsigned long long)expected_bytes);
    return 1;
  }
#if VPU_DMA_TEST_REQUIRE_OVERLAP
  if (overlap == 0) {
    printf("VPU DMA/execute overlap counter did not increment\n");
    return 1;
  }
#endif

  /* A legal minimum-size bank has only one slot, so run this geometry-specific
   * conflict check only when a distinct second row exists in the same bank. */
#if VPU_SLOTS_PER_BANK >= 2
  /* Put a second vector row in bank 0, then read it with the bank base. */
  vpu_write_gp(GP_NEXT_OFFSET, VPU_PING_INPUT_ADDR + VPU_VLEN);
  vpu_write_gp(GP_CURRENT_OFFSET, 0);
  vpu_set_vl(VPU_VLEN);
  vpu_h_prefetch_v(GP_NEXT_OFFSET, GP_CURRENT_OFFSET, H_INPUT);
  vpu_v_add_vv(GP_PONG_ADDRESS, GP_PING_ADDRESS, GP_NEXT_OFFSET);
  const uint64_t conflict_status = vpu_fence();
  const uint64_t bank_conflicts =
      vpu_read_perf(VPU_PERF_BANK_CONFLICT_STALL_CYCLES);
  if ((conflict_status & VPU_STATUS_ERROR_MASK) != 0 ||
      bank_conflicts == 0) {
    printf("VPU same-bank test failed: status=0x%llx conflicts=%llu\n",
           (unsigned long long)conflict_status,
           (unsigned long long)bank_conflicts);
    return 1;
  }
#endif

  if (test_page_crossing()) {
    return 1;
  }

  printf("VPU unaligned DMA, page crossing, ping/pong overlap, and "
         "bank-conflict test passed\n");
  return 0;
}
