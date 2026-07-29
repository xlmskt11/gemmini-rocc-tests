// See LICENSE for license details.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "include/vpu.h"

#define TEST_ACTIVE_VL (VPU_VLEN - 3u)
#define TEST_ELEMENTS (VPU_VLEN + 2u)

#if VPU_VLEN < 8
#error "vpu_mask_rearrange requires VPU_VLEN >= 8"
#endif

typedef char vpu_index_storage_width_must_match[
    sizeof(vpu_index_t) == sizeof(vpu_storage_t) ? 1 : -1];

enum {
  GP_SRC = 0,
  GP_RHS = 1,
  GP_DST = 2,
  GP_INDEX = 3,
  GP_OFFSET = 4,
  GP_SHIFT = 5,
  FP_SEED = 0,
  H_SRC = 0,
  H_RHS = 1,
  H_DST = 2,
  H_INDEX = 3,
  H_OUTPUT = 4,
};

static vpu_storage_t source[TEST_ELEMENTS] __attribute__((aligned(64)));
static vpu_storage_t rhs[TEST_ELEMENTS] __attribute__((aligned(64)));
static vpu_storage_t destination[TEST_ELEMENTS]
    __attribute__((aligned(64)));
static vpu_index_t indices[TEST_ELEMENTS] __attribute__((aligned(64)));
static vpu_storage_t output[TEST_ELEMENTS] __attribute__((aligned(64)));
static uint64_t mask_chunks[VPU_VMASK_CHUNKS];

#if VPU_STORAGE_KIND == VPU_STORAGE_FP32
#define TEST_SNAN_RAW UINT32_C(0x7f800001)
#else
#define TEST_SNAN_RAW UINT16_C(0x7f81)
#endif

static vpu_storage_t storage_from_raw(uint32_t raw) {
  return (vpu_storage_t)raw;
}

static vpu_storage_t guard_value(void) {
  return vpu_float_to_storage(123.0f);
}

static vpu_storage_t destination_value(size_t i) {
  return vpu_float_to_storage(-64.0f + (float)(i % 29u));
}

static int mask_enabled(size_t i) {
  return (i % 3u) != 1u;
}

static void build_mask(void) {
  for (unsigned chunk = 0; chunk < VPU_VMASK_CHUNKS; ++chunk) {
    mask_chunks[chunk] = 0;
  }
  /* Deliberately set some bits above TEST_ACTIVE_VL as well: execution must
   * use mask AND current VL, never the mask alone. */
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    if (mask_enabled(i)) {
      mask_chunks[i / VPU_VMASK_CHUNK_BITS] |=
          UINT64_C(1) << (i % VPU_VMASK_CHUNK_BITS);
    }
  }
}

static void initialize_vectors(void) {
  const vpu_storage_t guard = guard_value();
  source[0] = source[VPU_VLEN + 1u] = guard;
  rhs[0] = rhs[VPU_VLEN + 1u] = guard;
  destination[0] = destination[VPU_VLEN + 1u] = guard;
  indices[0] = indices[VPU_VLEN + 1u] = (vpu_index_t)0x55u;
  output[0] = output[VPU_VLEN + 1u] = guard;
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    source[i + 1u] =
        vpu_float_to_storage((float)((int)(i % 31u) - 15));
    rhs[i + 1u] = vpu_float_to_storage(2.0f);
    destination[i + 1u] = destination_value(i);
    indices[i + 1u] = (vpu_index_t)i;
    output[i + 1u] = vpu_float_to_storage(-321.0f);
  }
  build_mask();
}

static void configure_architectural_addresses(void) {
  vpu_write_gp(GP_SRC, VPU_BANK_BASE(0));
  vpu_write_gp(GP_RHS, VPU_BANK_BASE(1));
  vpu_write_gp(GP_DST, VPU_BANK_BASE(2));
  vpu_write_gp(GP_INDEX, VPU_BANK_BASE(3));
  vpu_write_gp(GP_OFFSET, 0);
  vpu_write_h(H_SRC, (uintptr_t)&source[1]);
  vpu_write_h(H_RHS, (uintptr_t)&rhs[1]);
  vpu_write_h(H_DST, (uintptr_t)&destination[1]);
  vpu_write_h(H_INDEX, (uintptr_t)&indices[1]);
  vpu_write_h(H_OUTPUT, (uintptr_t)&output[1]);
}

static int status_has_error(const char *name, uint64_t status) {
  if ((status & VPU_STATUS_ERROR_MASK) == 0) {
    return 0;
  }
  printf("%s returned VPU status 0x%llx\n", name,
         (unsigned long long)status);
  return 1;
}

static int prefetch_vectors(int load_src, int load_rhs, int load_dst,
                            int load_indices) {
  vpu_publish_cpu_writes();
  vpu_set_vl(VPU_VLEN);
  if (load_src) {
    vpu_h_prefetch_v(GP_SRC, GP_OFFSET, H_SRC);
  }
  if (load_rhs) {
    vpu_h_prefetch_v(GP_RHS, GP_OFFSET, H_RHS);
  }
  if (load_dst) {
    vpu_h_prefetch_v(GP_DST, GP_OFFSET, H_DST);
  }
  if (load_indices) {
    vpu_h_prefetch_v(GP_INDEX, GP_OFFSET, H_INDEX);
  }
  return status_has_error("prefetch", vpu_fence());
}

static uint64_t store_full(unsigned source_gp) {
  vpu_set_vl(VPU_VLEN);
  vpu_h_store_v(source_gp, GP_OFFSET, H_OUTPUT);
  return vpu_fence();
}

static int check_output_guards(const char *name) {
  const vpu_storage_t guard = guard_value();
  if (output[0] != guard || output[VPU_VLEN + 1u] != guard) {
    printf("%s overwrote an external-memory guard\n", name);
    return 1;
  }
  return 0;
}

static int expect_raw(const char *name, size_t i, vpu_storage_t expected) {
  const vpu_storage_t actual = output[i + 1u];
  if (actual == expected) {
    return 0;
  }
  printf("%s mismatch at %u: got=0x%x expected=0x%x\n", name,
         (unsigned)i, (unsigned)actual, (unsigned)expected);
  return 1;
}

static int test_masked_add_and_reductions(void) {
  const char *name = "masked_add";
  initialize_vectors();
  /* A masked-off signaling NaN must not update the destination or raise NV. */
  source[2] = storage_from_raw(TEST_SNAN_RAW);  // element 1 is masked off
  vpu_clear_status(VPU_CLEAR_ALL);
  if (prefetch_vectors(1, 1, 1, 0)) {
    return 1;
  }
  vpu_write_vmask(mask_chunks);
  vpu_set_vl(TEST_ACTIVE_VL);
  vpu_v_add_vv_masked(GP_DST, GP_SRC, GP_RHS);
  const uint64_t status = store_full(GP_DST);
  if (status_has_error(name, status) ||
      (status & VPU_STATUS_FFLAG_NV) != 0 || check_output_guards(name)) {
    printf("%s did not suppress a masked-off signaling NaN\n", name);
    return 1;
  }
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    vpu_storage_t expected = destination[i + 1u];
    if (i < TEST_ACTIVE_VL && mask_enabled(i)) {
      const float sum = vpu_storage_to_float(source[i + 1u]) +
                        vpu_storage_to_float(rhs[i + 1u]);
      expected = vpu_float_to_storage(sum);
    }
    if (expect_raw(name, i, expected)) {
      return 1;
    }
  }

  /* Use exactly representable ones so RED_SUM is independent of tree order.
   * Masked-off values include sNaN and a very large value; neither may affect
   * SUM/MAX or numerical flags. */
  size_t selected = 0;
  float selected_max = -1000.0f;
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    if (i < TEST_ACTIVE_VL && mask_enabled(i)) {
      source[i + 1u] = vpu_float_to_storage(1.0f);
      ++selected;
      if ((float)i > selected_max) {
        selected_max = (float)i;
      }
    } else {
      source[i + 1u] = vpu_float_to_storage(10000.0f);
    }
  }
  source[2] = storage_from_raw(TEST_SNAN_RAW);
  vpu_clear_status(VPU_CLEAR_ALL);
  if (prefetch_vectors(1, 0, 0, 0)) {
    return 1;
  }
  vpu_write_vmask(mask_chunks);
  vpu_set_vl(TEST_ACTIVE_VL);
  vpu_write_fp(FP_SEED, 3.0f);
  vpu_v_red_sum_masked(FP_SEED, GP_SRC);
  const uint64_t sum_status = vpu_fence();
  if (status_has_error("masked_red_sum", sum_status) ||
      (sum_status & VPU_STATUS_FFLAG_NV) != 0) {
    printf("masked_red_sum did not suppress masked-off sNaN\n");
    return 1;
  }
  const float sum = vpu_read_fp(FP_SEED);
  if (vpu_float_to_bits(sum) != vpu_float_to_bits(3.0f + (float)selected)) {
    printf("masked_red_sum got=0x%x expected=0x%x\n",
           vpu_float_to_bits(sum),
           vpu_float_to_bits(3.0f + (float)selected));
    return 1;
  }

  /* Give each selected element its index and retain the masked sNaN. */
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    source[i + 1u] = (i < TEST_ACTIVE_VL && mask_enabled(i))
        ? vpu_float_to_storage((float)i)
        : vpu_float_to_storage(10000.0f);
  }
  source[2] = storage_from_raw(TEST_SNAN_RAW);
  vpu_clear_status(VPU_CLEAR_ALL);
  if (prefetch_vectors(1, 0, 0, 0)) {
    return 1;
  }
  vpu_write_vmask(mask_chunks);
  vpu_set_vl(TEST_ACTIVE_VL);
  vpu_write_fp_bits(FP_SEED, UINT32_C(0xff800000));
  vpu_v_red_max_masked(FP_SEED, GP_SRC);
  const uint64_t max_status = vpu_fence();
  const float maximum = vpu_read_fp(FP_SEED);
  if (status_has_error("masked_red_max", max_status) ||
      (max_status & VPU_STATUS_FFLAG_NV) != 0 ||
      vpu_float_to_bits(maximum) != vpu_float_to_bits(selected_max)) {
    printf("masked_red_max got=0x%x expected=0x%x status=0x%llx\n",
           vpu_float_to_bits(maximum), vpu_float_to_bits(selected_max),
           (unsigned long long)max_status);
    return 1;
  }
  return 0;
}

static int test_slide_case(enum vpu_slide_direction direction, size_t shift,
                           int in_place) {
  const char *name = direction == VPU_SLIDE_RIGHT
      ? (in_place ? "slide_right_in_place" : "slide_right")
      : (in_place ? "slide_left_in_place" : "slide_left");
  initialize_vectors();
  vpu_clear_status(VPU_CLEAR_ALL);
  if (prefetch_vectors(1, 0, !in_place, 0)) {
    return 1;
  }
  vpu_write_gp(GP_SHIFT, (uint32_t)shift);
  vpu_set_vl(TEST_ACTIVE_VL);
  vpu_v_slide_v(in_place ? GP_SRC : GP_DST, GP_SRC, GP_SHIFT, direction);
  const uint64_t status = store_full(in_place ? GP_SRC : GP_DST);
  if (status_has_error(name, status) || check_output_guards(name)) {
    return 1;
  }
  const vpu_storage_t zero = vpu_float_to_storage(0.0f);
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    vpu_storage_t expected = in_place ? source[i + 1u]
                                      : destination[i + 1u];
    if (i < TEST_ACTIVE_VL) {
      if (direction == VPU_SLIDE_RIGHT) {
        expected = i >= shift ? source[i - shift + 1u] : zero;
      } else {
        expected = i + shift < TEST_ACTIVE_VL
            ? source[i + shift + 1u]
            : zero;
      }
    }
    if (expect_raw(name, i, expected)) {
      printf("%s shift=%u\n", name, (unsigned)shift);
      return 1;
    }
  }
  return 0;
}

static int test_slides(void) {
  const size_t shifts[] = {
      0u, 1u, VPU_NLANES - 1u, VPU_NLANES, VPU_NLANES + 1u,
      TEST_ACTIVE_VL - 1u, TEST_ACTIVE_VL, TEST_ACTIVE_VL + 1u,
  };
  for (size_t i = 0; i < sizeof(shifts) / sizeof(shifts[0]); ++i) {
    if (test_slide_case(VPU_SLIDE_RIGHT, shifts[i], 0) ||
        test_slide_case(VPU_SLIDE_LEFT, shifts[i], 0)) {
      return 1;
    }
  }
  /* Cross-word in-place behavior is the case most likely to be corrupted by
   * write-before-read sequencing in a serialized implementation. */
  return test_slide_case(VPU_SLIDE_RIGHT, VPU_NLANES + 1u, 1) ||
         test_slide_case(VPU_SLIDE_LEFT, VPU_NLANES + 1u, 1);
}

static void initialize_gather_indices(void) {
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    size_t index;
    if ((i % 7u) == 0u) {
      index = TEST_ACTIVE_VL + 5u;  // architecturally out of range
    } else if ((i % 5u) == 0u) {
      index = 3u;                   // duplicate selection
    } else {
      index = (i * (VPU_NLANES + 1u) + 11u) % TEST_ACTIVE_VL;
    }
    indices[i + 1u] = (vpu_index_t)index;
  }
}

static int run_gather(int masked) {
  const char *name = masked ? "gather_masked" : "gather";
  initialize_vectors();
  initialize_gather_indices();
  vpu_clear_status(VPU_CLEAR_ALL);
  if (prefetch_vectors(1, 0, 1, 1)) {
    return 1;
  }
  if (masked) {
    vpu_write_vmask(mask_chunks);
  }
  vpu_set_vl(TEST_ACTIVE_VL);
  if (masked) {
    vpu_v_gather_vv_masked(GP_DST, GP_SRC, GP_INDEX);
  } else {
    vpu_v_gather_vv(GP_DST, GP_SRC, GP_INDEX);
  }
  const uint64_t status = store_full(GP_DST);
  if (status_has_error(name, status) || check_output_guards(name)) {
    return 1;
  }
  const vpu_storage_t zero = vpu_float_to_storage(0.0f);
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    vpu_storage_t expected = destination[i + 1u];
    const int active = i < TEST_ACTIVE_VL && (!masked || mask_enabled(i));
    if (active) {
      const size_t index = (size_t)indices[i + 1u];
      expected = index < TEST_ACTIVE_VL ? source[index + 1u] : zero;
    }
    if (expect_raw(name, i, expected)) {
      return 1;
    }
  }
  return 0;
}

static int test_gather_alias_is_illegal(void) {
  const char *name = "gather_alias";
  initialize_vectors();
  initialize_gather_indices();
  vpu_clear_status(VPU_CLEAR_ALL);
  if (prefetch_vectors(1, 0, 0, 1)) {
    return 1;
  }
  vpu_set_vl(TEST_ACTIVE_VL);
  vpu_v_gather_vv(GP_SRC, GP_SRC, GP_INDEX);
  const uint64_t status = store_full(GP_SRC);
  if ((status & VPU_STATUS_ILLEGAL_COMMAND) == 0 ||
      (status & VPU_STATUS_DMA_FAULT) != 0 || check_output_guards(name)) {
    printf("%s did not report only illegalCommand: status=0x%llx\n", name,
           (unsigned long long)status);
    return 1;
  }
  for (size_t i = 0; i < VPU_VLEN; ++i) {
    if (expect_raw(name, i, source[i + 1u])) {
      return 1;
    }
  }
  vpu_clear_status(VPU_CLEAR_FAULT_ILLEGAL);
  return 0;
}

int main(void) {
  initialize_vectors();
  configure_architectural_addresses();
  if (test_masked_add_and_reductions() || test_slides() ||
      run_gather(0) || run_gather(1) || test_gather_alias_is_illegal()) {
    return 1;
  }
  printf("VPU per-element mask, masked reductions, cross-word/in-place slide, "
         "and gather tests passed\n");
  return 0;
}
