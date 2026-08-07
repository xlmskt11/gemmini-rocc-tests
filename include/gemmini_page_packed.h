#ifndef GEMMINI_PAGE_PACKED_H
#define GEMMINI_PAGE_PACKED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef GEMMINI_PAGE_PACKED_MATMUL
#define GEMMINI_PAGE_PACKED_MATMUL 0
#endif

#define GEMMINI_PAGE_PACKED_PAGE_BYTES ((size_t)4096)
#define GEMMINI_PAGE_PACKED_STRIDE_FLAG ((size_t)UINT64_C(1) << 31)
#define GEMMINI_PAGE_PACKED_STRIDE_MASK (GEMMINI_PAGE_PACKED_STRIDE_FLAG - 1)
// When LOOP_WS transposes an operand, page-pack its original source layout
// before the transpose and encode that source layout's row width here.
#define GEMMINI_PAGE_PACKED_STRIDE(stride) \
  (GEMMINI_PAGE_PACKED_STRIDE_FLAG | ((size_t)(stride) & GEMMINI_PAGE_PACKED_STRIDE_MASK))

static inline bool gemmini_page_packed_stride_is_packed(size_t stride) {
  return (stride & GEMMINI_PAGE_PACKED_STRIDE_FLAG) != 0;
}

static inline size_t gemmini_page_packed_stride_payload(size_t stride) {
  return stride & GEMMINI_PAGE_PACKED_STRIDE_MASK;
}

static inline size_t gemmini_ceil_div_size(size_t numerator, size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

static inline size_t gemmini_page_packed_input_blocks_per_page(void) {
  return GEMMINI_PAGE_PACKED_PAGE_BYTES / (DIM * DIM * sizeof(elem_t));
}

static inline size_t gemmini_page_packed_a_k_blocks_per_page(void) {
  const size_t blocks_per_page = gemmini_page_packed_input_blocks_per_page();
  return MAX_BLOCK_LEN < blocks_per_page ? MAX_BLOCK_LEN : blocks_per_page;
}

static inline size_t gemmini_page_packed_a_i_blocks_per_page(void) {
  return gemmini_page_packed_input_blocks_per_page() /
      gemmini_page_packed_a_k_blocks_per_page();
}

static inline size_t gemmini_page_packed_b_j_blocks_per_page(void) {
  return gemmini_page_packed_input_blocks_per_page();
}

static inline size_t gemmini_page_packed_acc_j_blocks_per_page(size_t elem_size) {
  const size_t blocks_per_page =
      GEMMINI_PAGE_PACKED_PAGE_BYTES / (DIM * DIM * elem_size);
  const size_t max_blocks = MAX_BYTES / (DIM * elem_size);
  const size_t bounded_max_blocks = max_blocks == 0 ? 1 : max_blocks;
  return bounded_max_blocks < blocks_per_page
      ? bounded_max_blocks
      : blocks_per_page;
}

static inline size_t gemmini_page_packed_acc_i_blocks_per_page(size_t elem_size) {
  const size_t blocks_per_page =
      GEMMINI_PAGE_PACKED_PAGE_BYTES / (DIM * DIM * elem_size);
  const size_t page_j_blocks =
      gemmini_page_packed_acc_j_blocks_per_page(elem_size);
  return blocks_per_page / page_j_blocks;
}

static inline size_t gemmini_page_packed_a_page_count(size_t rows, size_t cols) {
  const size_t page_k_blocks = gemmini_page_packed_a_k_blocks_per_page();
  const size_t page_i_blocks = gemmini_page_packed_a_i_blocks_per_page();
  const size_t i_pages = gemmini_ceil_div_size(gemmini_ceil_div_size(rows, DIM), page_i_blocks);
  const size_t k_pages = gemmini_ceil_div_size(gemmini_ceil_div_size(cols, DIM), page_k_blocks);
  return i_pages * k_pages;
}

static inline size_t gemmini_page_packed_b_page_count(size_t rows, size_t cols) {
  const size_t k_blocks = gemmini_ceil_div_size(rows, DIM);
  const size_t j_pages = gemmini_ceil_div_size(
      gemmini_ceil_div_size(cols, DIM), gemmini_page_packed_b_j_blocks_per_page());
  return k_blocks * j_pages;
}

static inline size_t gemmini_page_packed_acc_page_count(
    size_t rows, size_t cols, size_t elem_size) {
  const size_t page_j_blocks =
      gemmini_page_packed_acc_j_blocks_per_page(elem_size);
  const size_t page_i_blocks =
      gemmini_page_packed_acc_i_blocks_per_page(elem_size);
  const size_t i_pages =
      gemmini_ceil_div_size(gemmini_ceil_div_size(rows, DIM), page_i_blocks);
  const size_t j_pages =
      gemmini_ceil_div_size(gemmini_ceil_div_size(cols, DIM), page_j_blocks);
  return i_pages * j_pages;
}

static inline const elem_t *gemmini_page_packed_a_block_addr(
    const elem_t *base, size_t i_block, size_t k_block, size_t stride_elems) {
  const size_t page_k_blocks = gemmini_page_packed_a_k_blocks_per_page();
  const size_t page_i_blocks = gemmini_page_packed_a_i_blocks_per_page();
  const size_t k_blocks = gemmini_ceil_div_size(stride_elems, DIM);
  const size_t k_pages = gemmini_ceil_div_size(k_blocks, page_k_blocks);
  const size_t i_page = i_block / page_i_blocks;
  const size_t k_page = k_block / page_k_blocks;
  const size_t i_in_page = i_block % page_i_blocks;
  const size_t k_in_page = k_block % page_k_blocks;
  const size_t page_index = i_page * k_pages + k_page;
  const size_t page_cols = page_k_blocks * DIM;
  const size_t page_offset = page_index * GEMMINI_PAGE_PACKED_PAGE_BYTES;
  const size_t local_offset =
      (i_in_page * DIM * page_cols + k_in_page * DIM) * sizeof(elem_t);
  return (const elem_t *)((const uint8_t *)base + page_offset + local_offset);
}

static inline elem_t *gemmini_page_packed_a_block_addr_mut(
    elem_t *base, size_t i_block, size_t k_block, size_t stride_elems) {
  const elem_t *addr = gemmini_page_packed_a_block_addr(
      base, i_block, k_block, stride_elems);
  return base + (addr - base);
}

static inline const elem_t *gemmini_page_packed_b_block_addr(
    const elem_t *base, size_t k_block, size_t j_block, size_t stride_elems) {
  const size_t page_j_blocks = gemmini_page_packed_b_j_blocks_per_page();
  const size_t j_blocks = gemmini_ceil_div_size(stride_elems, DIM);
  const size_t j_pages = gemmini_ceil_div_size(j_blocks, page_j_blocks);
  const size_t j_page = j_block / page_j_blocks;
  const size_t j_in_page = j_block % page_j_blocks;
  const size_t page_index = k_block * j_pages + j_page;
  const size_t page_offset = page_index * GEMMINI_PAGE_PACKED_PAGE_BYTES;
  const size_t local_offset = j_in_page * DIM * sizeof(elem_t);
  return (const elem_t *)((const uint8_t *)base + page_offset + local_offset);
}

static inline elem_t *gemmini_page_packed_b_block_addr_mut(
    elem_t *base, size_t k_block, size_t j_block, size_t stride_elems) {
  const elem_t *addr = gemmini_page_packed_b_block_addr(
      base, k_block, j_block, stride_elems);
  return base + (addr - base);
}

static inline size_t gemmini_page_packed_acc_block_offset_bytes(
    size_t i_block, size_t j_block,
    size_t stride_elems, size_t elem_size) {
  const size_t page_j_blocks =
      gemmini_page_packed_acc_j_blocks_per_page(elem_size);
  const size_t page_i_blocks =
      gemmini_page_packed_acc_i_blocks_per_page(elem_size);
  const size_t j_blocks = gemmini_ceil_div_size(stride_elems, DIM);
  const size_t j_pages = gemmini_ceil_div_size(j_blocks, page_j_blocks);
  const size_t i_page = i_block / page_i_blocks;
  const size_t j_page = j_block / page_j_blocks;
  const size_t i_in_page = i_block % page_i_blocks;
  const size_t j_in_page = j_block % page_j_blocks;
  const size_t page_index = i_page * j_pages + j_page;
  const size_t page_cols = page_j_blocks * DIM;
  const size_t page_offset = page_index * GEMMINI_PAGE_PACKED_PAGE_BYTES;
  const size_t local_offset =
      (i_in_page * DIM * page_cols + j_in_page * DIM) * elem_size;
  return page_offset + local_offset;
}

static inline void *gemmini_page_packed_acc_block_addr_mut(
    void *base, size_t i_block, size_t j_block,
    size_t stride_elems, size_t elem_size) {
  return (void *)((uint8_t *)base + gemmini_page_packed_acc_block_offset_bytes(
      i_block, j_block, stride_elems, elem_size));
}

static inline const void *gemmini_page_packed_acc_block_addr(
    const void *base, size_t i_block, size_t j_block,
    size_t stride_elems, size_t elem_size) {
  return (const void *)((const uint8_t *)base +
      gemmini_page_packed_acc_block_offset_bytes(
          i_block, j_block, stride_elems, elem_size));
}

static inline size_t gemmini_page_packed_a_dma_stride_bytes(size_t stride) {
  if (!gemmini_page_packed_stride_is_packed(stride)) {
    return gemmini_page_packed_stride_payload(stride) * sizeof(elem_t);
  }
  const size_t page_k_blocks = gemmini_page_packed_a_k_blocks_per_page();
  return page_k_blocks * DIM * sizeof(elem_t);
}

static inline size_t gemmini_page_packed_b_dma_stride_bytes(size_t stride) {
  if (!gemmini_page_packed_stride_is_packed(stride)) {
    return gemmini_page_packed_stride_payload(stride) * sizeof(elem_t);
  }
  return gemmini_page_packed_b_j_blocks_per_page() * DIM * sizeof(elem_t);
}

static inline size_t gemmini_page_packed_acc_dma_stride_bytes(
    size_t stride, size_t elem_size) {
  if (!gemmini_page_packed_stride_is_packed(stride)) {
    return gemmini_page_packed_stride_payload(stride) * elem_size;
  }
  return gemmini_page_packed_acc_j_blocks_per_page(elem_size) * DIM * elem_size;
}

#endif
