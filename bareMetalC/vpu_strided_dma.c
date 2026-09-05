// See LICENSE for license details.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "include/vpu.h"

/*
 * Address registers and rowCount intentionally never alias:
 *
 *   GP0: VSRAM base, GP1: host element offset, GP2: 2-D row count
 *   H0: source base, H1: destination base
 *
 * The 2-D encoding places GP2's register index in rs3.  Keeping it separate
 * also makes it obvious that changing the row count cannot change either
 * address calculation.
 */
enum {
  GP_VSRAM_BASE = 0,
  GP_HOST_OFFSET = 1,
  GP_ROW_COUNT = 2,
  H_SOURCE = 0,
  H_DESTINATION = 1,
};

#define TWO_D_VL (VPU_VLEN - 3u)
#define SOURCE_PITCH (VPU_VLEN + 7u)
#define DESTINATION_PITCH (VPU_VLEN + 11u)
#ifndef VPU_STRIDED_DMA_ROWS
#if VPU_EXTERNAL_MEMORY
/* The shared-ACC path deliberately serializes through the physical ACC write
 * pipeline. Keep the ordinary RTL smoke bounded while retaining an override
 * which can request the architectural maximum for long-running regressions. */
#define VPU_STRIDED_DMA_ROWS (2u * VPU_VSPAD_SUBBANKS)
#else
#define VPU_STRIDED_DMA_ROWS VPU_DMA_MAX_ROWS
#endif
#endif
#define SOURCE_2D_ELEMENTS (VPU_STRIDED_DMA_ROWS * SOURCE_PITCH + 2u)
#define DESTINATION_2D_ELEMENTS \
  (VPU_STRIDED_DMA_ROWS * DESTINATION_PITCH + 2u)
#define ONE_D_VL 17u
#define ONE_D_ELEMENTS (VPU_VLEN + 2u)

#if VPU_VLEN < ONE_D_VL
#error "vpu_strided_dma requires VPU_VLEN >= 17"
#endif

#if VPU_STRIDED_DMA_ROWS == 0 || VPU_STRIDED_DMA_ROWS > VPU_DMA_MAX_ROWS
#error "The 2-D test row count must fit the architectural descriptor"
#endif

#if !VPU_EXTERNAL_MEMORY && VPU_DMA_MAX_ROWS != VPU_SLOTS_PER_BANK
#error "A maximum-size 2-D descriptor must cover exactly one VSRAM bank"
#endif

static vpu_storage_t source_2d[SOURCE_2D_ELEMENTS]
    __attribute__((aligned(64)));
static vpu_storage_t destination_2d[DESTINATION_2D_ELEMENTS]
    __attribute__((aligned(64)));
static vpu_storage_t source_1d[ONE_D_ELEMENTS]
    __attribute__((aligned(64)));
static vpu_storage_t destination_1d[ONE_D_ELEMENTS]
    __attribute__((aligned(64)));

static vpu_storage_t guard_value(void) {
  return vpu_float_to_storage(123.0f);
}

static vpu_storage_t destination_sentinel(void) {
  return vpu_float_to_storage(-321.0f);
}

static vpu_storage_t source_value(size_t row, size_t column) {
  const int value = (int)((row * 19u + column * 3u) % 127u) - 63;
  return vpu_float_to_storage((float)value);
}

static int status_has_error(const char *name, uint64_t status) {
  if ((status & VPU_STATUS_ERROR_MASK) == 0) {
    return 0;
  }
  printf("%s returned VPU status 0x%llx\n", name,
         (unsigned long long)status);
  return 1;
}

static int check_perf(const char *name, uint64_t expected_read,
                      uint64_t expected_write) {
  const uint64_t read = vpu_read_perf(VPU_PERF_DMA_READ_BYTES);
  const uint64_t write = vpu_read_perf(VPU_PERF_DMA_WRITE_BYTES);
  if (read == expected_read && write == expected_write) {
    return 0;
  }
  printf("%s DMA counters: read=%llu expected=%llu write=%llu "
         "expected=%llu\n",
         name, (unsigned long long)read, (unsigned long long)expected_read,
         (unsigned long long)write, (unsigned long long)expected_write);
  return 1;
}

static void initialize_2d_buffers(void) {
  const vpu_storage_t guard = guard_value();
  const vpu_storage_t sentinel = destination_sentinel();
  for (size_t i = 0; i < SOURCE_2D_ELEMENTS; ++i) {
    source_2d[i] = guard;
  }
  for (size_t i = 0; i < DESTINATION_2D_ELEMENTS; ++i) {
    destination_2d[i] = sentinel;
  }
  source_2d[0] = source_2d[SOURCE_2D_ELEMENTS - 1u] = guard;
  destination_2d[0] =
      destination_2d[DESTINATION_2D_ELEMENTS - 1u] = guard;

  for (size_t row = 0; row < VPU_STRIDED_DMA_ROWS; ++row) {
    for (size_t column = 0; column < TWO_D_VL; ++column) {
      source_2d[1u + row * SOURCE_PITCH + column] =
          source_value(row, column);
    }
  }
}

static int test_strided_2d_round_trip(void) {
  const char *name = "strided_2d_max_rows";
  initialize_2d_buffers();
  vpu_publish_cpu_writes();
  vpu_clear_status(VPU_CLEAR_ALL);

  vpu_write_gp(GP_VSRAM_BASE, VPU_BANK_BASE(0));
  vpu_write_gp(GP_HOST_OFFSET, 0);
  vpu_write_gp(GP_ROW_COUNT, VPU_STRIDED_DMA_ROWS);
  vpu_write_h(H_SOURCE, (uintptr_t)&source_2d[1]);
  vpu_write_h(H_DESTINATION, (uintptr_t)&destination_2d[1]);
  vpu_set_vl(TWO_D_VL);

  /* The load snapshots SOURCE_PITCH before the global stride register is
   * changed for the store. This simultaneously checks distinct source and
   * destination strides and dispatch-time descriptor snapshotting. */
  vpu_set_stride_bytes((uint64_t)SOURCE_PITCH * VPU_STORAGE_BYTES);
  vpu_h_prefetch_v_2d(GP_VSRAM_BASE, GP_HOST_OFFSET, H_SOURCE,
                      GP_ROW_COUNT);
  vpu_set_stride_bytes((uint64_t)DESTINATION_PITCH * VPU_STORAGE_BYTES);
  vpu_h_store_v_2d(GP_VSRAM_BASE, GP_HOST_OFFSET, H_DESTINATION,
                   GP_ROW_COUNT);
  const uint64_t status = vpu_fence();
  if (status_has_error(name, status)) {
    return 1;
  }

  const vpu_storage_t guard = guard_value();
  const vpu_storage_t sentinel = destination_sentinel();
  if (source_2d[0] != guard ||
      source_2d[SOURCE_2D_ELEMENTS - 1u] != guard ||
      destination_2d[0] != guard ||
      destination_2d[DESTINATION_2D_ELEMENTS - 1u] != guard) {
    printf("%s overwrote an outer guard\n", name);
    return 1;
  }

  for (size_t row = 0; row < VPU_STRIDED_DMA_ROWS; ++row) {
    for (size_t column = 0; column < DESTINATION_PITCH; ++column) {
      const vpu_storage_t actual =
          destination_2d[1u + row * DESTINATION_PITCH + column];
      const vpu_storage_t expected = column < TWO_D_VL
          ? source_value(row, column)
          : sentinel;
      if (actual != expected) {
        printf("%s mismatch row=%u column=%u got=0x%x expected=0x%x\n",
               name, (unsigned)row, (unsigned)column, (unsigned)actual,
               (unsigned)expected);
        return 1;
      }
    }
  }

  const uint64_t bytes = (uint64_t)VPU_STRIDED_DMA_ROWS * TWO_D_VL *
                         VPU_STORAGE_BYTES;
  return check_perf(name, bytes, bytes);
}

static void initialize_1d_buffers(void) {
  const vpu_storage_t guard = guard_value();
  const vpu_storage_t sentinel = destination_sentinel();
  source_1d[0] = source_1d[VPU_VLEN + 1u] = guard;
  destination_1d[0] = destination_1d[VPU_VLEN + 1u] = guard;
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    source_1d[i + 1u] = vpu_float_to_storage((float)((int)i - 8));
    destination_1d[i + 1u] = sentinel;
  }
}

static int test_legacy_1d_compatibility(void) {
  const char *name = "legacy_1d";
  initialize_1d_buffers();
  vpu_publish_cpu_writes();
  vpu_clear_status(VPU_CLEAR_ALL);

  /* Exercise a second bank when the generated geometry has one, while
   * keeping the compatibility case legal for a one-bank configuration. */
  const unsigned legacy_bank = VPU_VSPAD_BANKS > 1u ? 1u : 0u;
  vpu_write_gp(GP_VSRAM_BASE, VPU_BANK_BASE(legacy_bank));
  vpu_write_gp(GP_HOST_OFFSET, 0);
  /* A non-one rowCount and unrelated configured stride must have no effect
   * when legacy H_* wrappers encode funct1=0 and rs3=0. */
  vpu_write_gp(GP_ROW_COUNT, VPU_DMA_MAX_ROWS);
  vpu_write_h(H_SOURCE, (uintptr_t)&source_1d[1]);
  vpu_write_h(H_DESTINATION, (uintptr_t)&destination_1d[1]);
  vpu_set_stride_bytes((uint64_t)(VPU_VLEN + 29u) * VPU_STORAGE_BYTES);
  vpu_set_vl(ONE_D_VL);
  vpu_h_prefetch_v(GP_VSRAM_BASE, GP_HOST_OFFSET, H_SOURCE);
  vpu_h_store_v(GP_VSRAM_BASE, GP_HOST_OFFSET, H_DESTINATION);
  const uint64_t status = vpu_fence();
  if (status_has_error(name, status)) {
    return 1;
  }

  const vpu_storage_t guard = guard_value();
  const vpu_storage_t sentinel = destination_sentinel();
  if (destination_1d[0] != guard ||
      destination_1d[VPU_VLEN + 1u] != guard) {
    printf("%s overwrote an outer guard\n", name);
    return 1;
  }
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    const vpu_storage_t expected =
        i < ONE_D_VL ? source_1d[i + 1u] : sentinel;
    if (destination_1d[i + 1u] != expected) {
      printf("%s mismatch at %u got=0x%x expected=0x%x\n", name,
             (unsigned)i, (unsigned)destination_1d[i + 1u],
             (unsigned)expected);
      return 1;
    }
  }

  const uint64_t bytes = (uint64_t)ONE_D_VL * VPU_STORAGE_BYTES;
  return check_perf(name, bytes, bytes);
}

int main(void) {
  if (test_strided_2d_round_trip() || test_legacy_1d_compatibility()) {
    return 1;
  }
  printf("VPU strided 2-D load/store, VL-tail guards, "
         "source/destination stride, and legacy 1-D tests passed\n");
  return 0;
}
