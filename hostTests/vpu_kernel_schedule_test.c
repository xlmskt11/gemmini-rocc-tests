// See LICENSE for license details.

static void record_batch_publish(void);
#define VPU_BATCH_PUBLISH_CPU_WRITES() record_batch_publish()
#include "include/vpu_kernels.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

struct traced_command {
  uint32_t uop;
  uint64_t payload;
};

static struct traced_command trace[4096];
static size_t trace_count;
static size_t publish_count;

static void record_batch_publish(void) {
  ++publish_count;
}

static void reset_trace(void) {
  trace_count = 0;
  publish_count = 0;
}

static void record_command(uint64_t transport, uint64_t payload) {
  assert(trace_count < sizeof(trace) / sizeof(trace[0]));
  trace[trace_count].uop = (uint32_t)transport;
  trace[trace_count].payload = payload;
  ++trace_count;
}

void vpu_host_issue(uint64_t transport, uint64_t payload) {
  record_command(transport, payload);
}

uint64_t vpu_host_issue_result(uint64_t transport, uint64_t payload) {
  record_command(transport, payload);
  return 0;
}

static size_t count_opcode(unsigned opcode) {
  size_t count = 0;
  for (size_t i = 0; i < trace_count; ++i) {
    count += (trace[i].uop & 0x3fu) == opcode;
  }
  return count;
}

static size_t count_memory_form(unsigned opcode, unsigned funct1) {
  size_t count = 0;
  for (size_t i = 0; i < trace_count; ++i) {
    count += (trace[i].uop & 0x3fu) == opcode &&
        ((trace[i].uop >> 22) & 0x0fu) == funct1;
  }
  return count;
}

static size_t count_gp_writes(unsigned rd) {
  size_t count = 0;
  for (size_t i = 0; i < trace_count; ++i) {
    count += vpu_micro_opcode(trace[i].uop) == VPU_OP_C_WRITE_GP &&
        vpu_micro_rd(trace[i].uop) == rd;
  }
  return count;
}

static int contains_gp_write(unsigned rd, uint64_t payload) {
  for (size_t i = 0; i < trace_count; ++i) {
    if (vpu_micro_opcode(trace[i].uop) == VPU_OP_C_WRITE_GP &&
        vpu_micro_rd(trace[i].uop) == rd &&
        trace[i].payload == payload) {
      return 1;
    }
  }
  return 0;
}

static void check_chunk_row_writes(size_t expected_count,
                                   size_t expected_capacity) {
  size_t count = 0;
  for (size_t i = 0; i < trace_count; ++i) {
    if (vpu_micro_opcode(trace[i].uop) != VPU_OP_C_WRITE_GP) {
      continue;
    }
    const unsigned rd = vpu_micro_rd(trace[i].uop);
    if (rd != VPU_STREAM_GP_PING_ROWS &&
        rd != VPU_STREAM_GP_PONG_ROWS) {
      continue;
    }
    assert(trace[i].payload >= 1u);
    assert(trace[i].payload <= expected_capacity);
    assert(trace[i].payload <= VPU_NONLINEAR_CHUNK_ROWS);
    ++count;
  }
  assert(count == expected_count);
  assert(count == count_gp_writes(VPU_STREAM_GP_PING_ROWS) +
                      count_gp_writes(VPU_STREAM_GP_PONG_ROWS));
}

static size_t find_memory_form(unsigned opcode, unsigned funct1,
                               size_t occurrence) {
  for (size_t i = 0; i < trace_count; ++i) {
    if (vpu_micro_opcode(trace[i].uop) == opcode &&
        vpu_micro_funct1(trace[i].uop) == funct1) {
      if (occurrence == 0) {
        return i;
      }
      --occurrence;
    }
  }
  return trace_count;
}

static size_t find_opcode(unsigned opcode, size_t occurrence) {
  for (size_t i = 0; i < trace_count; ++i) {
    if (vpu_micro_opcode(trace[i].uop) == opcode) {
      if (occurrence == 0) {
        return i;
      }
      --occurrence;
    }
  }
  return trace_count;
}

static int contains_vl(size_t vl) {
  for (size_t i = 0; i < trace_count; ++i) {
    if ((trace[i].uop & 0x3fu) == VPU_OP_C_SET_VL &&
        trace[i].payload == vl) {
      return 1;
    }
  }
  return 0;
}

static size_t tiled_run_count(size_t full_tiles) {
  size_t runs = 0;
  for (size_t tile = 0; tile < full_tiles;) {
    const size_t rows = vpu_tiled_chunk_rows(tile, full_tiles - tile);
    assert(rows >= 1u);
    assert(rows <= vpu_stream_batch_capacity());
    assert(rows <= VPU_SLOTS_PER_BANK - tile % VPU_SLOTS_PER_BANK);
    tile += rows;
    ++runs;
  }
  return runs;
}

struct descriptor_summary {
  size_t loads;
  size_t stores;
  size_t tail_loads;
  size_t tail_stores;
  size_t short_full_loads;
  size_t short_full_stores;
};

/* Decode the register snapshots visible at each memory command.  Resident
 * kernels intentionally use the 2-D form even for the one-row partial-VLEN
 * tail, unlike the streaming activation path which peels a legacy 1-D tail. */
static struct descriptor_summary check_resident_descriptors(
    size_t tail, size_t expected_loads, size_t expected_stores) {
  uint64_t gp[16] = {0};
  size_t vl = 0;
  const size_t capacity = vpu_stream_batch_capacity();
  struct descriptor_summary summary = {0};

  for (size_t i = 0; i < trace_count; ++i) {
    const uint32_t uop = trace[i].uop;
    const unsigned opcode = vpu_micro_opcode(uop);
    if (opcode == VPU_OP_C_WRITE_GP) {
      gp[vpu_micro_rd(uop)] = trace[i].payload;
      continue;
    }
    if (opcode == VPU_OP_C_SET_VL) {
      vl = (size_t)trace[i].payload;
      continue;
    }
    if (opcode != VPU_OP_H_PREFETCH_V && opcode != VPU_OP_H_STORE_V) {
      continue;
    }

    assert(vpu_micro_funct1(uop) == 1u);
    const size_t rows = (size_t)gp[vpu_micro_rs3(uop)];
    const uint64_t base = gp[vpu_micro_rd(uop)];
    assert(rows >= 1u && rows <= capacity);
    assert(vl >= 1u && vl <= VPU_VLEN);
    assert((base % VPU_VLEN) == 0u);
    const uint64_t end = base +
        (uint64_t)(rows - 1u) * VPU_VLEN + vl;
    assert(end <= VPU_VSPAD_ELEMENTS);
    assert(base / VPU_ELEMENTS_PER_BANK ==
           (end - 1u) / VPU_ELEMENTS_PER_BANK);

    const int is_load = opcode == VPU_OP_H_PREFETCH_V;
    if (is_load) {
      ++summary.loads;
    } else {
      ++summary.stores;
    }
    if (vl == tail) {
      assert(rows == 1u);
      if (is_load) {
        ++summary.tail_loads;
      } else {
        ++summary.tail_stores;
      }
    } else {
      assert(vl == VPU_VLEN);
      if (rows < capacity) {
        if (is_load) {
          ++summary.short_full_loads;
        } else {
          ++summary.short_full_stores;
        }
      }
    }
  }

  assert(summary.loads == expected_loads);
  assert(summary.stores == expected_stores);
  return summary;
}

static void test_ping_pong_tail_schedule(void) {
  const size_t capacity = vpu_stream_batch_capacity();
  assert(capacity >= 1u);
  const size_t last_rows = capacity > 1u ? 2u : 1u;
  const size_t tail = VPU_VLEN > 17u ? 17u : VPU_VLEN - 1u;
  const size_t loop_batches =
      (capacity > 1u ? 2u : 0u) + (last_rows > 1u ? 1u : 0u);

  /* Three full-row batches force ping -> pong -> ping reuse. The third batch
   * has two rows when the geometry permits it.  With one physical row per
   * bank all three batches remain valid one-row runs.  The final partial tile
   * exercises the separately peeled legacy 1-D descriptor in either case. */
  const size_t elements =
      (2u * capacity + last_rows) * (size_t)VPU_VLEN + tail;
  const vpu_storage_t *input =
      (const vpu_storage_t *)(uintptr_t)0x100000u;
  vpu_storage_t *output = (vpu_storage_t *)(uintptr_t)0x200000u;

  reset_trace();
  assert(vpu_relu_auto(input, output, elements) == 0u);

  assert(count_opcode(VPU_OP_C_LOOP_START) == loop_batches);
  assert(count_opcode(VPU_OP_C_LOOP_END) == loop_batches);
  assert(count_opcode(VPU_OP_S_ADDI_INT) == 6u);
  assert(count_opcode(VPU_OP_V_MAX_VF) == 4u);

  assert(count_opcode(VPU_OP_H_PREFETCH_V) == 4u);
  assert(count_memory_form(VPU_OP_H_PREFETCH_V, 1u) == 3u);
  assert(count_memory_form(VPU_OP_H_PREFETCH_V, 0u) == 1u);
  assert(count_opcode(VPU_OP_H_STORE_V) == 4u);
  assert(count_memory_form(VPU_OP_H_STORE_V, 1u) == 3u);
  assert(count_memory_form(VPU_OP_H_STORE_V, 0u) == 1u);

  assert(contains_vl(VPU_VLEN));
  assert(contains_vl(tail));
  assert(count_opcode(VPU_OP_C_FENCE) == 1u);
}

static void test_fixed_2048_chunk_schedule(void) {
  const size_t elements = 2048u;
  const size_t full_tiles = elements / VPU_VLEN;
  const size_t tail = elements % VPU_VLEN;
  const size_t capacity = vpu_stream_batch_capacity();
  const size_t full_batches = full_tiles == 0u
      ? 0u : 1u + (full_tiles - 1u) / capacity;
  const size_t last_rows = full_batches == 0u
      ? 0u : full_tiles - (full_batches - 1u) * capacity;
  const size_t loop_batches = capacity == 1u ? 0u : full_batches -
      (full_batches != 0u && last_rows == 1u ? 1u : 0u);
  const size_t tail_descriptors = tail == 0u ? 0u : 1u;
  const vpu_storage_t *input =
      (const vpu_storage_t *)(uintptr_t)0x300000u;
  vpu_storage_t *output = (vpu_storage_t *)(uintptr_t)0x400000u;

  reset_trace();
  assert(vpu_relu_auto(input, output, elements) == 0u);

  /* Every full-width transfer is one small 2-D descriptor.  The optional
   * partial VLEN tail remains a separately peeled legacy 1-D descriptor. */
  assert(count_opcode(VPU_OP_H_PREFETCH_V) ==
         full_batches + tail_descriptors);
  assert(count_memory_form(VPU_OP_H_PREFETCH_V, 1u) == full_batches);
  assert(count_memory_form(VPU_OP_H_PREFETCH_V, 0u) == tail_descriptors);
  assert(count_opcode(VPU_OP_H_STORE_V) ==
         full_batches + tail_descriptors);
  assert(count_memory_form(VPU_OP_H_STORE_V, 1u) == full_batches);
  assert(count_memory_form(VPU_OP_H_STORE_V, 0u) == tail_descriptors);

  assert(count_opcode(VPU_OP_C_LOOP_START) == loop_batches);
  assert(count_opcode(VPU_OP_C_LOOP_END) == loop_batches);
  assert(count_opcode(VPU_OP_V_MAX_VF) ==
         full_batches + tail_descriptors);
  assert(count_opcode(VPU_OP_S_ADDI_INT) == 2u * full_batches);
  check_chunk_row_writes(2u * full_batches, capacity);

  /* The scheduler must put the next independent load in front of the current
   * batch's captured execute body.  Otherwise loop replay backpressures the
   * RoCC frontend and recreates the old load-then-execute serialization. */
  if (full_batches >= 2u) {
    const size_t second_load =
        find_memory_form(VPU_OP_H_PREFETCH_V, 1u, 1u);
    const size_t first_execute = find_opcode(VPU_OP_V_MAX_VF, 0u);
    assert(second_load < trace_count);
    assert(first_execute < trace_count);
    assert(second_load < first_execute);
  }

  assert(contains_vl(VPU_VLEN));
  if (tail != 0u) {
    assert(contains_vl(tail));
  }
  assert(count_opcode(VPU_OP_C_FENCE) == 1u);
}

typedef uint64_t (*vpu_binary_auto_fn)(
    const vpu_storage_t *, const vpu_storage_t *, vpu_storage_t *, size_t);

static void check_binary_elementwise_schedule(
    vpu_binary_auto_fn kernel, unsigned opcode, unsigned other_opcode) {
  const size_t capacity = vpu_stream_batch_capacity();
  assert(capacity >= 1u);
  const size_t last_rows = capacity > 1u ? 2u : 1u;
  const size_t full_tiles = 2u * capacity + last_rows;
  const size_t tail = VPU_VLEN > 17u ? 17u : VPU_VLEN - 1u;
  const size_t tail_descriptors = tail == 0u ? 0u : 1u;
  const size_t full_batches = 3u;
#if VPU_LOOP_BUFFER_ENTRIES >= 6
  const size_t loop_batches =
      (capacity > 1u ? 2u : 0u) + (last_rows > 1u ? 1u : 0u);
  const size_t execute_commands = full_batches + tail_descriptors;
  const size_t induction_commands = 3u * full_batches;
#else
  const size_t loop_batches = 0u;
  const size_t execute_commands = full_tiles + tail_descriptors;
  const size_t induction_commands = 3u * full_tiles;
#endif
  const size_t elements = full_tiles * (size_t)VPU_VLEN + tail;
  const vpu_storage_t *lhs =
      (const vpu_storage_t *)(uintptr_t)0x800000u;
  const vpu_storage_t *rhs =
      (const vpu_storage_t *)(uintptr_t)0x900000u;
  vpu_storage_t *output = (vpu_storage_t *)(uintptr_t)0xa00000u;

  reset_trace();
  assert(kernel(lhs, rhs, output, elements) == 0u);

  /* Every full batch has one 2-D load for each operand and one 2-D store.
   * A partial-VLEN tail uses the corresponding three legacy descriptors. */
  assert(count_opcode(VPU_OP_H_PREFETCH_V) ==
         2u * (full_batches + tail_descriptors));
  assert(count_memory_form(VPU_OP_H_PREFETCH_V, 1u) ==
         2u * full_batches);
  assert(count_memory_form(VPU_OP_H_PREFETCH_V, 0u) ==
         2u * tail_descriptors);
  assert(count_opcode(VPU_OP_H_STORE_V) ==
         full_batches + tail_descriptors);
  assert(count_memory_form(VPU_OP_H_STORE_V, 1u) == full_batches);
  assert(count_memory_form(VPU_OP_H_STORE_V, 0u) == tail_descriptors);

  assert(count_opcode(opcode) == execute_commands);
  assert(count_opcode(other_opcode) == 0u);
  assert(count_opcode(VPU_OP_C_LOOP_START) == loop_batches);
  assert(count_opcode(VPU_OP_C_LOOP_END) == loop_batches);
  assert(count_opcode(VPU_OP_S_ADDI_INT) == induction_commands);
  assert(count_opcode(VPU_OP_C_WRITE_FP) == 0u);
  check_chunk_row_writes(2u * full_batches, capacity);

  /* The arithmetic body must read INPUT/AUX and write OUTPUT.  Both loads for
   * the next pong batch are admitted before the first ping execution body. */
  const size_t first_execute = find_opcode(opcode, 0u);
  assert(first_execute < trace_count);
  assert(vpu_micro_rd(trace[first_execute].uop) ==
         VPU_STREAM_GP_PING_OUTPUT);
  assert(vpu_micro_rs1(trace[first_execute].uop) ==
         VPU_STREAM_GP_PING_INPUT);
  assert(vpu_micro_rs2(trace[first_execute].uop) ==
         VPU_STREAM_GP_PING_AUX);
  const size_t second_aux_load =
      find_memory_form(VPU_OP_H_PREFETCH_V, 1u, 3u);
  assert(second_aux_load < first_execute);

  assert(contains_vl(VPU_VLEN));
  if (tail != 0u) {
    assert(contains_vl(tail));
  }
  assert(count_opcode(VPU_OP_C_FENCE) == 1u);
}

static void test_binary_elementwise_schedule(void) {
  check_binary_elementwise_schedule(
      vpu_add_auto, VPU_OP_V_ADD_VV, VPU_OP_V_MUL_VV);
  check_binary_elementwise_schedule(
      vpu_mul_auto, VPU_OP_V_MUL_VV, VPU_OP_V_ADD_VV);

  const vpu_storage_t *lhs =
      (const vpu_storage_t *)(uintptr_t)0xb00000u;
  const vpu_storage_t *rhs =
      (const vpu_storage_t *)(uintptr_t)0xc00000u;
  vpu_storage_t *output = (vpu_storage_t *)(uintptr_t)0xd00000u;

  reset_trace();
  assert(vpu_add_auto(lhs, rhs, output, 0u) == 0u);
  assert(count_opcode(VPU_OP_H_PREFETCH_V) == 0u);
  assert(count_opcode(VPU_OP_H_STORE_V) == 0u);
  assert(count_opcode(VPU_OP_V_ADD_VV) == 0u);
  assert(contains_vl(0u));
  assert(count_opcode(VPU_OP_C_FENCE) == 1u);

#if SIZE_MAX > UINT32_MAX
  reset_trace();
  assert(vpu_mul_auto(
             lhs, rhs, output, (size_t)UINT32_MAX + 1u) ==
         VPU_STATUS_ILLEGAL_COMMAND);
  assert(trace_count == 0u);
#endif
}

static void test_resident_reduction_chunk_schedule(void) {
  const size_t capacity = vpu_stream_batch_capacity();
  const size_t remainder_rows = capacity > 1u ? capacity - 1u : 1u;
  const size_t full_tiles = 2u * capacity + remainder_rows;
  const size_t tail = VPU_VLEN - 1u;
  const size_t elements = full_tiles * (size_t)VPU_VLEN + tail;
  const size_t full_runs = tiled_run_count(full_tiles);
  const size_t pass_descriptors = full_runs + 1u;
  const vpu_storage_t *input =
      (const vpu_storage_t *)(uintptr_t)0x500000u;
  const vpu_storage_t *weight =
      (const vpu_storage_t *)(uintptr_t)0x600000u;
  vpu_storage_t *output = (vpu_storage_t *)(uintptr_t)0x700000u;
  const struct vpu_tiled_plan rms_plan =
      vpu_tiled_plan_for(VPU_TILED_KERNEL_RMSNORM, elements);
  const struct vpu_tiled_plan softmax_plan =
      vpu_tiled_plan_for(VPU_TILED_KERNEL_SOFTMAX, elements);

  assert(rms_plan.schedule == VPU_TILED_SCHEDULE_RESIDENT_FULL);
  assert(softmax_plan.schedule == VPU_TILED_SCHEDULE_RESIDENT_FULL);

  reset_trace();
  assert(vpu_tiled_rmsnorm_auto(
             input, weight, output, elements, 1.0e-5f) == 0u);
  assert(count_memory_form(VPU_OP_H_PREFETCH_V, 0u) == 0u);
  assert(count_memory_form(VPU_OP_H_STORE_V, 0u) == 0u);
  const struct descriptor_summary rms = check_resident_descriptors(
      tail, 2u * pass_descriptors, pass_descriptors);
  assert(rms.tail_loads == 2u);
  assert(rms.tail_stores == 1u);
  if (capacity > 1u) {
    assert(rms.short_full_loads >= 2u);
    assert(rms.short_full_stores >= 1u);
  }
  /* Pass one admits LD(run 1) before its first reduction.  After the tail and
   * scalar fold, pass two likewise admits weight LD(run 1) before output EX. */
  assert(find_memory_form(VPU_OP_H_PREFETCH_V, 1u, 1u) <
         find_opcode(VPU_OP_V_MUL_VV, 0u));
  assert(find_memory_form(VPU_OP_H_PREFETCH_V, 1u,
                          pass_descriptors + 1u) <
         find_opcode(VPU_OP_V_MUL_VF, 0u));

  /* When the resident row does not fill all nominal input banks, RMSNorm uses
   * the first spare one as its result ring.  This separates ST(current) reads
   * from EX(next) input reads without reducing resident capacity. */
  const unsigned resident_banks_used = (unsigned)(
      (rms_plan.resident_tiles + VPU_SLOTS_PER_BANK - 1u) /
      VPU_SLOTS_PER_BANK);
  if (resident_banks_used < rms_plan.resident_banks) {
    assert(contains_gp_write(VPU_RESIDENT_GP_OUTPUT,
                             VPU_BANK_BASE(resident_banks_used)));
  }

  reset_trace();
  assert(vpu_tiled_softmax_auto(input, output, elements) == 0u);
  assert(count_memory_form(VPU_OP_H_PREFETCH_V, 0u) == 0u);
  assert(count_memory_form(VPU_OP_H_STORE_V, 0u) == 0u);
  const struct descriptor_summary softmax = check_resident_descriptors(
      tail, pass_descriptors, pass_descriptors);
  assert(softmax.tail_loads == 1u);
  assert(softmax.tail_stores == 1u);
  if (capacity > 1u) {
    assert(softmax.short_full_loads >= 1u);
    assert(softmax.short_full_stores >= 1u);
  }
  assert(find_memory_form(VPU_OP_H_PREFETCH_V, 1u, 1u) <
         find_opcode(VPU_OP_V_RED_MAX, 0u));
  assert(contains_gp_write(VPU_RESIDENT_GP_OUTPUT,
                           VPU_BANK_BASE(softmax_plan.resident_banks)));
}

static void test_synchronous_auto_boundaries(void) {
  const size_t elements = VPU_VLEN;
  const vpu_storage_t *input =
      (const vpu_storage_t *)(uintptr_t)0x1000000u;
  const vpu_storage_t *aux =
      (const vpu_storage_t *)(uintptr_t)0x1100000u;
  const vpu_storage_t *cosine =
      (const vpu_storage_t *)(uintptr_t)0x1200000u;
  const vpu_storage_t *sine =
      (const vpu_storage_t *)(uintptr_t)0x1300000u;
  vpu_storage_t *output = (vpu_storage_t *)(uintptr_t)0x1400000u;

#define CHECK_SYNCHRONOUS_AUTO(call_) do { \
    reset_trace();                         \
    assert((call_) == 0u);                 \
    assert(publish_count == 1u);           \
    assert(count_opcode(VPU_OP_C_FENCE) == 1u); \
  } while (0)

  CHECK_SYNCHRONOUS_AUTO(
      vpu_rmsnorm_auto(input, aux, output, elements, 1.0e-5f));
  CHECK_SYNCHRONOUS_AUTO(
      vpu_final_norm_auto(input, aux, output, elements, 1.0e-5f));
  CHECK_SYNCHRONOUS_AUTO(vpu_softmax_auto(input, output, elements));
  CHECK_SYNCHRONOUS_AUTO(vpu_rope_auto(
      input, cosine, sine, output, 2u, VPU_VLEN, VPU_ROPE_NEOX));
  CHECK_SYNCHRONOUS_AUTO(vpu_add_auto(input, aux, output, elements));
  CHECK_SYNCHRONOUS_AUTO(vpu_mul_auto(input, aux, output, elements));
  CHECK_SYNCHRONOUS_AUTO(vpu_silu_auto(input, output, elements));
  CHECK_SYNCHRONOUS_AUTO(vpu_relu_auto(input, output, elements));
  CHECK_SYNCHRONOUS_AUTO(vpu_sigmoid_auto(input, output, elements));
  CHECK_SYNCHRONOUS_AUTO(vpu_tanh_auto(input, output, elements));
  CHECK_SYNCHRONOUS_AUTO(vpu_gelu_auto(input, output, elements));
  CHECK_SYNCHRONOUS_AUTO(vpu_swiglu_auto(input, aux, output, elements));

#undef CHECK_SYNCHRONOUS_AUTO
}

static void test_multi_enqueue_single_boundary(void) {
  const size_t elements = VPU_VLEN;
  const vpu_storage_t *input =
      (const vpu_storage_t *)(uintptr_t)0x2000000u;
  const vpu_storage_t *aux =
      (const vpu_storage_t *)(uintptr_t)0x2100000u;
  const vpu_storage_t *cosine =
      (const vpu_storage_t *)(uintptr_t)0x2200000u;
  const vpu_storage_t *sine =
      (const vpu_storage_t *)(uintptr_t)0x2300000u;
  vpu_storage_t *output = (vpu_storage_t *)(uintptr_t)0x2400000u;

  reset_trace();
  vpu_batch_begin();
  assert(publish_count == 1u);
  assert(count_opcode(VPU_OP_C_FENCE) == 0u);

  assert(vpu_rmsnorm_enqueue(
             input, aux, output, elements, 1.0e-5f) == 0u);
  assert(vpu_final_norm_enqueue(
             input, aux, output, elements, 1.0e-5f) == 0u);
  assert(vpu_softmax_enqueue(input, output, elements) == 0u);
  assert(vpu_rope_enqueue(
             input, cosine, sine, output, 2u, VPU_VLEN,
             VPU_ROPE_INTERLEAVED) == 0u);
  assert(vpu_add_enqueue(input, aux, output, elements) == 0u);
  assert(vpu_mul_enqueue(input, aux, output, elements) == 0u);
  assert(vpu_silu_enqueue(input, output, elements) == 0u);
  assert(vpu_relu_enqueue(input, output, elements) == 0u);
  assert(vpu_sigmoid_enqueue(input, output, elements) == 0u);
  assert(vpu_tanh_enqueue(input, output, elements) == 0u);
  assert(vpu_gelu_enqueue(input, output, elements) == 0u);
  assert(vpu_swiglu_enqueue(input, aux, output, elements) == 0u);

  assert(publish_count == 1u);
  assert(count_opcode(VPU_OP_C_FENCE) == 0u);
  assert(vpu_batch_finish() == 0u);
  assert(publish_count == 1u);
  assert(count_opcode(VPU_OP_C_FENCE) == 1u);
}

int main(void) {
  const size_t physical = vpu_stream_physical_batch_capacity();
  const size_t expected_capacity = physical < VPU_NONLINEAR_CHUNK_ROWS
      ? physical : (size_t)VPU_NONLINEAR_CHUNK_ROWS;
  assert(VPU_NONLINEAR_CHUNK_ROWS >= 1u);
  assert(vpu_stream_batch_capacity() == expected_capacity);
  assert(vpu_stream_batch_capacity() <= physical);
  assert(vpu_stream_batch_capacity() <= VPU_NONLINEAR_CHUNK_ROWS);

  test_ping_pong_tail_schedule();
  test_fixed_2048_chunk_schedule();
  test_binary_elementwise_schedule();
  test_resident_reduction_chunk_schedule();
  test_synchronous_auto_boundaries();
  test_multi_enqueue_single_boundary();
  return 0;
}
