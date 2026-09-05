// See LICENSE for license details.

#define _GNU_SOURCE

// Linux-only, model-free FireSim microbenchmark for the three Llama2-7B
// projection shapes. It uses the shared_multi job tiling/planning semantics to
// precompute compact raw RoCC descriptors, then measures only descriptor
// replay and one final drain. It does not link or execute llama.cpp/ggml.

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "include/gemmini_testutils_all.h"

#ifdef BAREMETAL
#error "llama2_matmul_concurrency_microbench is a Linux-only FireSim test"
#endif

#define MICRO_MAX_JOBS 4
#define MICRO_MAX_TENANTS 4
#define MICRO_SHAPE_COUNT 3
#define MICRO_AXIS_COUNT 3
#define MICRO_CAPACITY_MAX_GEMMINIS 3
#define MICRO_CONCURRENT_POLICY_COUNT 5
#define MICRO_M_MIN 8
#define MICRO_M_MAX 512
#define MICRO_M_STEP 8
#define MICRO_L2_BYTES (2u * 1024u * 1024u)
#define MICRO_EVICTION_BYTES (8u * 1024u * 1024u)
#define MICRO_CACHE_LINE_BYTES 64u
#define MICRO_MAX_DESCRIPTOR_BYTES (32u * 1024u * 1024u)
#define MICRO_PAGE_PACKING_MASK 0x2u
#define MICRO_EXPECTED_EPOCHS \
  (MICRO_SHAPE_COUNT * (MICRO_M_MAX / MICRO_M_STEP) * \
   (MICRO_CAPACITY_MAX_GEMMINIS * MICRO_AXIS_COUNT + \
    MICRO_CONCURRENT_POLICY_COUNT))
#define MICRO_MEMBER_COMMANDS 8u

typedef struct {
  const char *name;
  size_t n;
  size_t k;
  int full_m_call_multiplicity;
} micro_shape_t;

typedef struct {
  const char *name;
  int tenant_count;
  unsigned masks[MICRO_MAX_TENANTS];
  int queue_jobs[MICRO_MAX_TENANTS];
} micro_policy_t;

typedef struct {
  const char *name;
  void *data;
  size_t bytes;
} micro_region_t;

typedef struct {
  bool present;
  unsigned mask;
  unsigned members;
  unsigned allocated_quarters;
  unsigned q;
  int queue_jobs;
  gemmini_partition_axis_t axis;
  size_t sp_start;
  size_t sp_end;
  size_t sp_range;
  size_t sp_rows_per_operand_bank;
  size_t acc_start;
  size_t acc_range;
  size_t acc_rows_per_half;
  size_t a_begin[2];
  size_t a_end[2];
  size_t b_begin[2];
  size_t b_end[2];
  size_t acc_begin[2];
  size_t acc_end[2];
} micro_tenant_layout_t;

typedef struct {
  bool present;
  int tenant;
  unsigned mask;
  unsigned members;
  gemmini_partition_axis_t axis;
  int tile_id;
  size_t tile_i;
  size_t tile_j;
  size_t tile_k;
  size_t used_a_rows;
  size_t used_b_rows;
  size_t used_acc_rows;
  size_t steps;
  size_t min_active_members;
  size_t max_active_members;
  size_t active_member_steps[5];
} micro_job_metrics_t;

typedef struct {
  int tenant_count;
  int job_count;
  micro_tenant_layout_t tenants[MICRO_MAX_TENANTS];
  micro_job_metrics_t jobs[MICRO_MAX_JOBS];
  uint64_t tenant_issue_done_cycles[MICRO_MAX_TENANTS];
  uint64_t tenant_issue_done_round[MICRO_MAX_TENANTS];
  uint64_t scheduler_rounds;
  uint64_t issue_cycles;
  uint64_t fence_cycles;
  uint64_t makespan_cycles;
  size_t descriptor_bytes;
  size_t timed_rocc_commands;
} micro_run_metrics_t;

typedef struct {
  uint64_t rs1;
  uint64_t rs2;
} micro_rocc_args_t;

enum {
  MICRO_CMD_SPADDR = 0,
  MICRO_CMD_PARTITION,
  MICRO_CMD_BOUNDS,
  MICRO_CMD_ADDRS_AB,
  MICRO_CMD_ADDRS_DC,
  MICRO_CMD_STRIDES_AB,
  MICRO_CMD_STRIDES_DC,
  MICRO_CMD_PAGE_OFFSETS,
};

typedef struct {
  micro_rocc_args_t command[MICRO_MEMBER_COMMANDS];
} micro_member_descriptor_t;

/*
 * Each variable-size record is a launch pair followed by exactly one member
 * descriptor per set bit in launch.rs1[51:48], in ascending physical-Gemmini
 * order.  Both pieces are 16-byte multiples, so every record stays naturally
 * aligned without padding unused members into the timed descriptor stream.
 */
typedef struct {
  micro_rocc_args_t launch;
} micro_step_descriptor_t;

typedef struct {
  uint8_t *data;
  size_t bytes;
  size_t used_bytes;
  size_t steps;
  size_t timed_rocc_commands;
} micro_replay_schedule_t;

static const micro_shape_t micro_shapes[MICRO_SHAPE_COUNT] = {
  {"attn_qkvo", 4096u, 4096u, 128},
  {"ffn_gate_up", 11008u, 4096u, 62},
  {"ffn_down", 4096u, 11008u, 31},
};

/*
 * Exact g4 best-axis lookup from the prior full-capacity isolated sweep:
 *   sims/firesim/deploy/results-workload/
 *     2026-09-02--05-37-56-gemmini-uniform/gemmini-uniform0/output
 *   SHA-256:
 *     d73996cd91374aac44c7c22e7431952ac2c135cb72c86d235e2b8761116c9a4a
 *
 * For every (shape, M), rows with gemmini_count=4 were compared by the
 * gemmini_run_cycles field.  The minimum was selected, with exact ties
 * resolved in M < N < K order.  Values use the generated axis ABI:
 * 0=M, 1=N, 2=K.  Columns are M=8,16,...,512.
 */
static const uint8_t micro_prior_g4_best_axis
    [MICRO_SHAPE_COUNT][MICRO_M_MAX / MICRO_M_STEP] = {
  {
    0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  },
  {
    0, 0, 0, 0, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 0, 0, 1, 1, 1, 1,
    1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
  },
  {
    0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 2, 0, 2, 2, 2, 0,
    2, 2, 2, 0, 1, 1, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  },
};

static const micro_policy_t
    micro_policies[MICRO_CONCURRENT_POLICY_COUNT] = {
  {"queued_g4", 1, {0xfu, 0u, 0u, 0u}, {4, 0, 0, 0}},
  {"2+2", 2, {0x3u, 0xcu, 0u, 0u}, {2, 2, 0, 0}},
  {"3+1", 2, {0x7u, 0x8u, 0u, 0u}, {3, 1, 0, 0}},
  {"2+1+1", 3, {0x3u, 0x4u, 0x8u, 0u}, {2, 1, 1, 0}},
  {"1+1+1+1", 4, {0x1u, 0x2u, 0x4u, 0x8u}, {1, 1, 1, 1}},
};

static volatile sig_atomic_t micro_abort_requested = 0;
static volatile uint64_t micro_cache_sink = 0;

static void micro_signal_handler(int signal_number) {
  (void)signal_number;
  micro_abort_requested = 1;
}

static inline void micro_cpu_fence(void) {
  asm volatile("fence rw, rw" ::: "memory");
}

static inline uint64_t micro_read_cycles(void) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles) : : "memory");
  return cycles;
}

static const char *micro_axis_name(gemmini_partition_axis_t axis) {
  switch (axis) {
  case GEMMINI_PARTITION_AXIS_M:
    return "M";
  case GEMMINI_PARTITION_AXIS_N:
    return "N";
  case GEMMINI_PARTITION_AXIS_K:
    return "K";
  default:
    return "?";
  }
}

static unsigned micro_popcount4(unsigned mask) {
  unsigned count = 0;
  mask &= 0xfu;
  while (mask != 0u) {
    count += mask & 1u;
    mask >>= 1;
  }
  return count;
}

static bool micro_descriptor_record_bytes(
    size_t active_members, size_t *bytes) {
  if (bytes == NULL || active_members == 0u || active_members > 4u)
    return false;
  const size_t member_bytes =
      active_members * sizeof(micro_member_descriptor_t);
  if (member_bytes > SIZE_MAX - sizeof(micro_step_descriptor_t))
    return false;
  *bytes = sizeof(micro_step_descriptor_t) + member_bytes;
  return true;
}

static bool micro_lookup_prior_g4_axis(
    size_t shape_index, size_t m, gemmini_partition_axis_t *axis) {
  if (axis == NULL || shape_index >= MICRO_SHAPE_COUNT ||
      m < MICRO_M_MIN || m > MICRO_M_MAX || m % MICRO_M_STEP != 0u)
    return false;
  const size_t m_index = m / MICRO_M_STEP - 1u;
  const uint8_t axis_id = micro_prior_g4_best_axis[shape_index][m_index];
  if (axis_id >= MICRO_AXIS_COUNT)
    return false;
  *axis = (gemmini_partition_axis_t)axis_id;
  return true;
}

static bool micro_checked_mul(size_t lhs, size_t rhs, size_t *result) {
  if (result == NULL || (lhs != 0u && rhs > SIZE_MAX / lhs))
    return false;
  *result = lhs * rhs;
  return true;
}

static size_t micro_round_up(size_t value, size_t alignment) {
  if (alignment == 0u || value > SIZE_MAX - (alignment - 1u))
    return 0u;
  return ((value + alignment - 1u) / alignment) * alignment;
}

static bool micro_region_map(
    micro_region_t *region, const char *name, size_t requested_bytes) {
  if (region == NULL || name == NULL || requested_bytes == 0u)
    return false;
  const long page_size_long = sysconf(_SC_PAGESIZE);
  if (page_size_long <= 0)
    return false;
  const size_t page_size = (size_t)page_size_long;
  const size_t bytes = micro_round_up(requested_bytes, page_size);
  if (bytes == 0u)
    return false;

  void *data = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (data == MAP_FAILED) {
    fprintf(stderr, "MICROBENCH mmap failed name=%s bytes=%zu errno=%d\n",
            name, bytes, errno);
    return false;
  }

  // One same-value store per page creates distinct writable resident pages
  // without initializing operand values for the benchmark.
  volatile uint8_t *pages = (volatile uint8_t *)data;
  for (size_t offset = 0; offset < bytes; offset += page_size) {
    const uint8_t value = pages[offset];
    pages[offset] = value;
  }
  if (mlock(data, bytes) != 0) {
    fprintf(stderr,
            "MICROBENCH warning mlock failed name=%s bytes=%zu errno=%d\n",
            name, bytes, errno);
  }

  region->name = name;
  region->data = data;
  region->bytes = bytes;
  return true;
}

static void micro_region_unmap(micro_region_t *region) {
  if (region != NULL && region->data != NULL) {
    (void)munlock(region->data, region->bytes);
    (void)munmap(region->data, region->bytes);
    region->data = NULL;
    region->bytes = 0u;
  }
}

static bool micro_shape_storage(
    size_t m, const micro_shape_t *shape,
    size_t *a_bytes, size_t *b_bytes, size_t *c_bytes) {
  size_t a_elements = 0u;
  size_t c_elements = 0u;
  size_t b_pages = 0u;
  if (shape == NULL || m == 0u || shape->n == 0u || shape->k == 0u ||
      !micro_checked_mul(m, shape->k, &a_elements) ||
      !micro_checked_mul(m, shape->n, &c_elements))
    return false;
  b_pages = gemmini_page_packed_b_page_count(shape->k, shape->n);
  return micro_checked_mul(a_elements, sizeof(elem_t), a_bytes) &&
         micro_checked_mul(b_pages, GEMMINI_PAGE_PACKED_PAGE_BYTES, b_bytes) &&
         micro_checked_mul(c_elements, sizeof(acc_t), c_bytes);
}

static void micro_flush_all_gemmini_tlbs_once(void) {
  gemmini_flush(custom0, 0);
  gemmini_flush(custom1, 0);
  gemmini_flush(custom2, 0);
  gemmini_flush(custom3, 0);
  gemmini_fence();
}

static void micro_evict_l2(const micro_region_t *eviction) {
  volatile const uint8_t *bytes = (volatile const uint8_t *)eviction->data;
  uint64_t sink = micro_cache_sink;
  for (size_t offset = 0; offset < eviction->bytes;
       offset += MICRO_CACHE_LINE_BYTES)
    sink += bytes[offset];
  micro_cache_sink = sink;
}

static void micro_make_descriptors_recent(
    const micro_replay_schedule_t *schedule) {
  volatile const uint8_t *bytes =
      (volatile const uint8_t *)(const void *)schedule->data;
  uint64_t sink = micro_cache_sink;
  for (size_t offset = 0; offset < schedule->bytes;
       offset += MICRO_CACHE_LINE_BYTES)
    sink += bytes[offset];
  micro_cache_sink = sink;
}

static void micro_make_a_recent_round_robin(
    const micro_region_t a_regions[MICRO_MAX_JOBS],
    int job_count, size_t a_bytes) {
  for (size_t offset = 0; offset < a_bytes;
       offset += MICRO_CACHE_LINE_BYTES) {
    for (int job = 0; job < job_count; ++job) {
      volatile uint8_t *bytes = (volatile uint8_t *)a_regions[job].data;
      const uint8_t value = bytes[offset];
      bytes[offset] = value;
    }
  }
  micro_cpu_fence();
}

static bool micro_make_tenant_layout(
    unsigned mask, unsigned q, unsigned allocated_quarters, int queue_jobs,
    gemmini_partition_axis_t axis,
    micro_tenant_layout_t *layout) {
  if (layout == NULL || mask == 0u || (mask & ~0xfu) != 0u ||
      allocated_quarters == 0u || queue_jobs <= 0 || q >= 4u ||
      q + allocated_quarters > 4u)
    return false;
  const unsigned members = micro_popcount4(mask);
  if (members == 0u || q + members > 4u ||
      mask != (((1u << members) - 1u) << q))
    return false;

  memset(layout, 0, sizeof(*layout));
  layout->present = true;
  layout->mask = mask;
  layout->members = members;
  layout->allocated_quarters = allocated_quarters;
  layout->q = q;
  layout->queue_jobs = queue_jobs;
  layout->axis = axis;
  layout->sp_rows_per_operand_bank =
      (size_t)allocated_quarters * BANK_ROWS / 4u;
  layout->sp_start = (size_t)q * BANK_ROWS / 4u;
  layout->sp_end = layout->sp_start;
  layout->sp_range = 4u * layout->sp_rows_per_operand_bank;
  layout->acc_rows_per_half =
      (size_t)allocated_quarters * (ACC_ROWS / 2u) / 4u;
  layout->acc_start = (size_t)q * (ACC_ROWS / 2u) / 4u;
  layout->acc_range = 2u * layout->acc_rows_per_half;

  for (unsigned phase = 0; phase < 2u; ++phase) {
    const size_t sp_half_base = phase * (2u * BANK_ROWS);
    layout->a_begin[phase] = sp_half_base + layout->sp_start;
    layout->a_end[phase] =
        layout->a_begin[phase] + layout->sp_rows_per_operand_bank;
    layout->b_end[phase] =
        sp_half_base + 2u * BANK_ROWS - layout->sp_end;
    layout->b_begin[phase] =
        layout->b_end[phase] - layout->sp_rows_per_operand_bank;
    layout->acc_begin[phase] =
        phase * (ACC_ROWS / 2u) + layout->acc_start;
    layout->acc_end[phase] =
        layout->acc_begin[phase] + layout->acc_rows_per_half;

    const size_t a_bank_begin = sp_half_base;
    const size_t a_bank_end = a_bank_begin + BANK_ROWS;
    const size_t b_bank_begin = a_bank_end;
    const size_t b_bank_end = b_bank_begin + BANK_ROWS;
    const size_t acc_half_begin = phase * (ACC_ROWS / 2u);
    const size_t acc_half_end = acc_half_begin + ACC_ROWS / 2u;
    if (layout->a_begin[phase] < a_bank_begin ||
        layout->a_end[phase] > a_bank_end ||
        layout->b_begin[phase] < b_bank_begin ||
        layout->b_end[phase] > b_bank_end ||
        layout->acc_begin[phase] < acc_half_begin ||
        layout->acc_end[phase] > acc_half_end)
      return false;
  }
  return true;
}

static bool micro_intervals_overlap(
    size_t lhs_begin, size_t lhs_end,
    size_t rhs_begin, size_t rhs_end) {
  return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

static bool micro_validate_layout_set(
    const micro_tenant_layout_t layouts[MICRO_MAX_TENANTS],
    int tenant_count, bool require_full_cover) {
  if (tenant_count <= 0 || tenant_count > MICRO_MAX_TENANTS)
    return false;
  unsigned union_mask = 0u;
  unsigned total_members = 0u;
  unsigned total_allocated_quarters = 0u;
  size_t total_sp_rows = 0u;
  size_t total_acc_rows = 0u;
  for (int tenant = 0; tenant < tenant_count; ++tenant) {
    const micro_tenant_layout_t *layout = &layouts[tenant];
    if (!layout->present || (union_mask & layout->mask) != 0u)
      return false;
    union_mask |= layout->mask;
    total_members += layout->members;
    total_allocated_quarters += layout->allocated_quarters;
    total_sp_rows += layout->sp_rows_per_operand_bank;
    total_acc_rows += layout->acc_rows_per_half;
    for (int prior = 0; prior < tenant; ++prior) {
      for (unsigned phase = 0; phase < 2u; ++phase) {
        if (micro_intervals_overlap(
                layout->a_begin[phase], layout->a_end[phase],
                layouts[prior].a_begin[phase], layouts[prior].a_end[phase]) ||
            micro_intervals_overlap(
                layout->b_begin[phase], layout->b_end[phase],
                layouts[prior].b_begin[phase], layouts[prior].b_end[phase]) ||
            micro_intervals_overlap(
                layout->acc_begin[phase], layout->acc_end[phase],
                layouts[prior].acc_begin[phase], layouts[prior].acc_end[phase]))
          return false;
      }
    }
  }
  if (total_members > 4u || total_allocated_quarters > 4u ||
      total_sp_rows > BANK_ROWS ||
      total_acc_rows > ACC_ROWS / 2u)
    return false;
  return !require_full_cover ||
      (union_mask == 0xfu && total_members == 4u &&
       total_allocated_quarters == 4u &&
       total_sp_rows == BANK_ROWS && total_acc_rows == ACC_ROWS / 2u);
}

static bool micro_configure_job(
    shared_multi_matmul_job_t *job,
    micro_job_metrics_t *metrics,
    const micro_shape_t *shape,
    size_t m,
    int job_index,
    int tenant,
    const micro_tenant_layout_t *layout,
    const micro_region_t a_regions[MICRO_MAX_JOBS],
    const micro_region_t b_regions[MICRO_MAX_JOBS],
    const micro_region_t c_regions[MICRO_MAX_JOBS]) {
  if (job == NULL || metrics == NULL || shape == NULL || layout == NULL ||
      job_index < 0 || job_index >= MICRO_MAX_JOBS)
    return false;
  memset(job, 0, sizeof(*job));
  job->gemmini_list = (int)layout->mask;
  // tile_id is reporting metadata only; replay group IDs come from ACC space.
  job->tile_id = job_index;
  job->sp_addr_start_stack = layout->sp_start;
  job->sp_addr_end_stack = layout->sp_end;
  job->acc_addr_start_stack = layout->acc_start;
  job->sp_addr_range = layout->sp_range;
  job->acc_addr_range = layout->acc_range;
  job->dim_I = m;
  job->dim_J = shape->n;
  job->dim_K = shape->k;
  job->A = (const elem_t *)a_regions[job_index].data;
  job->B = (const elem_t *)b_regions[job_index].data;
  job->D = NULL;
  job->C = c_regions[job_index].data;
  job->stride_A = shape->k;
  job->stride_B = GEMMINI_PAGE_PACKED_STRIDE(shape->n);
  job->stride_D = 0u;
  job->stride_C = shape->n;
  job->A_scale_factor = MVIN_SCALE_IDENTITY;
  job->B_scale_factor = MVIN_SCALE_IDENTITY;
  job->D_scale_factor = MVIN_SCALE_IDENTITY;
  job->act = NO_ACTIVATION;
  job->scale = ACC_SCALE_IDENTITY;
  job->bert_scale = 0;
  job->repeating_bias = false;
  job->a_transpose = false;
  job->b_transpose = false;
  job->full_C = true;
  job->low_D = false;
  job->weightA = 1u;
  job->dataflow = WEIGHT_STATIONARY;
  job->partition_axis = layout->axis;

  const gemmini_tiling_request_t tiling_request = {
    .dim_I = m,
    .dim_J = shape->n,
    .dim_K = shape->k,
    .dim = DIM,
    .gemmini_count = layout->members,
    .sp_addr_range = layout->sp_range,
    .acc_addr_range = layout->acc_range,
    .double_buffered = true,
    .act = NO_ACTIVATION,
  };
  const gemmini_tiling_factors_t tiling =
      gemmini_shared_multi_choose_tiling_axis(
          &tiling_request, layout->axis);
  job->gemmini_num = (int)layout->members;
  if (!shared_multi_matmul_job_set_exact_tiling(
          job, tiling.tile_I, tiling.tile_J, tiling.tile_K) ||
      job->partition_status != SHARED_MULTI_PARTITION_OK ||
      job->tile_I == 0u || job->tile_J == 0u || job->tile_K == 0u ||
      job->sp_addr_range != layout->sp_range ||
      job->acc_addr_range != layout->acc_range ||
      job->sp_addr_A_stacked > layout->sp_rows_per_operand_bank ||
      job->sp_addr_B_stacked > layout->sp_rows_per_operand_bank ||
      job->acc_addr_stacked > layout->acc_rows_per_half)
    return false;

  memset(metrics, 0, sizeof(*metrics));
  metrics->present = true;
  metrics->tenant = tenant;
  metrics->mask = layout->mask;
  metrics->members = layout->members;
  metrics->axis = layout->axis;
  metrics->tile_id = job_index;
  metrics->tile_i = job->tile_I;
  metrics->tile_j = job->tile_J;
  metrics->tile_k = job->tile_K;
  metrics->used_a_rows = job->sp_addr_A_stacked;
  metrics->used_b_rows = job->sp_addr_B_stacked;
  metrics->used_acc_rows = job->acc_addr_stacked;
  metrics->min_active_members = SIZE_MAX;
  return true;
}

static bool micro_initialize_job_state(shared_multi_matmul_job_t *job) {
  if (job == NULL)
    return false;
  job->partition_status =
      shared_multi_partition_axis_status(job, job->partition_axis);
  if (job->partition_status != SHARED_MULTI_PARTITION_OK ||
      !shared_multi_matmul_job_init_state(job))
    return false;
  if (job->no_bias)
    job->D = (void *)1;
  return !job->done && job->partition_status == SHARED_MULTI_PARTITION_OK;
}

/* All jobs in an epoch have identical command configuration.  Configure each
 * physical Gemmini in the epoch union exactly once, outside timed replay. */
static bool micro_configure_hw_once(
    const shared_multi_matmul_job_t jobs[MICRO_MAX_JOBS],
    int job_count, unsigned union_mask) {
  if (jobs == NULL || job_count <= 0 || job_count > MICRO_MAX_JOBS ||
      union_mask == 0u || (union_mask & ~0xfu) != 0u)
    return false;
  const shared_multi_matmul_job_t *base = &jobs[0];
  for (int job = 1; job < job_count; ++job) {
    const shared_multi_matmul_job_t *other = &jobs[job];
    if (other->stride_A != base->stride_A ||
        other->stride_B != base->stride_B ||
        other->stride_C != base->stride_C ||
        other->stride_D != base->stride_D ||
        other->A_scale_factor != base->A_scale_factor ||
        other->B_scale_factor != base->B_scale_factor ||
        other->D_scale_factor != base->D_scale_factor ||
        other->act != base->act || other->scale != base->scale ||
        other->repeating_bias != base->repeating_bias ||
        other->a_transpose != base->a_transpose ||
        other->b_transpose != base->b_transpose ||
        other->low_D != base->low_D ||
        other->dataflow != base->dataflow ||
        other->sizeof_D != base->sizeof_D ||
        other->sizeof_C != base->sizeof_C)
      return false;
  }

  const size_t config_stride_A =
      gemmini_page_packed_a_dma_stride_bytes(base->stride_A);
  const size_t config_stride_B =
      gemmini_page_packed_b_dma_stride_bytes(base->stride_B);
  const size_t config_stride_D = gemmini_page_packed_acc_dma_stride_bytes(
      base->stride_D, base->sizeof_D);
  const size_t config_stride_C = gemmini_page_packed_acc_dma_stride_bytes(
      base->stride_C, base->sizeof_C);

#define MICRO_CONFIGURE_ONE(custom_num)                                      \
  do {                                                                        \
    gemmini_extended_config_ex(                                               \
        custom_num, base->dataflow, base->act & 3, 0, 1,                     \
        base->a_transpose, base->b_transpose);                                \
    gemmini_extended_config_st(                                               \
        custom_num, config_stride_C, base->act & 3, base->scale);             \
    gemmini_extended3_config_ld(                                              \
        custom_num, config_stride_A, base->A_scale_factor, false, 0);         \
    gemmini_extended3_config_ld(                                              \
        custom_num, config_stride_B, base->B_scale_factor, false, 1);         \
    gemmini_extended3_config_ld(                                              \
        custom_num, base->repeating_bias ? 0 : config_stride_D,               \
        base->D_scale_factor, base->low_D, 2);                                \
  } while (0)

  if ((union_mask & 0x1u) != 0u)
    MICRO_CONFIGURE_ONE(custom0);
  if ((union_mask & 0x2u) != 0u)
    MICRO_CONFIGURE_ONE(custom1);
  if ((union_mask & 0x4u) != 0u)
    MICRO_CONFIGURE_ONE(custom2);
  if ((union_mask & 0x8u) != 0u)
    MICRO_CONFIGURE_ONE(custom3);

#undef MICRO_CONFIGURE_ONE
  return true;
}

/* Plan from an initialized copy so instrumentation never enters timed issue. */
static bool micro_collect_job_plan_metrics(
    const shared_multi_matmul_job_t *initialized_job,
    micro_job_metrics_t *metrics) {
  if (initialized_job == NULL || metrics == NULL || initialized_job->done)
    return false;
  shared_multi_matmul_job_t dry_job = *initialized_job;
  metrics->steps = 0u;
  metrics->min_active_members = SIZE_MAX;
  metrics->max_active_members = 0u;
  memset(metrics->active_member_steps, 0,
         sizeof(metrics->active_member_steps));
  while (!dry_job.done) {
    shared_multi_matmul_job_step_t planned_step;
    if (!shared_multi_matmul_job_plan_step(&dry_job, &planned_step) ||
        planned_step.active_gemmini_count == 0u ||
        planned_step.active_gemmini_count > 4u)
      return false;
    if (planned_step.active_gemmini_count < metrics->min_active_members)
      metrics->min_active_members = planned_step.active_gemmini_count;
    if (planned_step.active_gemmini_count > metrics->max_active_members)
      metrics->max_active_members = planned_step.active_gemmini_count;
    ++metrics->active_member_steps[planned_step.active_gemmini_count];
    ++metrics->steps;
    shared_multi_matmul_job_complete_step(&dry_job);
  }
  return dry_job.partition_status == SHARED_MULTI_PARTITION_OK &&
      metrics->steps != 0u && metrics->min_active_members != SIZE_MAX;
}

static bool micro_supported_active_mask(unsigned mask) {
  switch (mask) {
  case 0x1u:
  case 0x2u:
  case 0x3u:
  case 0x4u:
  case 0x7u:
  case 0x8u:
  case 0xcu:
  case 0xfu:
    return true;
  default:
    return false;
  }
}

static void micro_release_schedule(micro_replay_schedule_t *schedule) {
  if (schedule != NULL) {
    free(schedule->data);
    memset(schedule, 0, sizeof(*schedule));
  }
}

static bool micro_append_step_descriptor(
    shared_multi_matmul_job_t *job, micro_replay_schedule_t *schedule) {
  if (job == NULL || schedule == NULL || job->done ||
      job->dataflow != WEIGHT_STATIONARY || !job->no_bias ||
      job->act != NO_ACTIVATION || job->a_transpose || job->b_transpose ||
      gemmini_page_packed_stride_is_packed(job->stride_A) ||
      !gemmini_page_packed_stride_is_packed(job->stride_B) ||
      gemmini_page_packed_stride_is_packed(job->stride_D) ||
      gemmini_page_packed_stride_is_packed(job->stride_C))
    return false;

  shared_multi_matmul_job_step_t step;
  if (!shared_multi_matmul_job_plan_step(job, &step) ||
      step.active_gemmini_count == 0u ||
      step.active_gemmini_count > 4u ||
      micro_popcount4((unsigned)step.group_list) !=
          step.active_gemmini_count ||
      !micro_supported_active_mask((unsigned)step.group_list))
    return false;

  size_t record_bytes = 0u;
  if (!micro_descriptor_record_bytes(
          step.active_gemmini_count, &record_bytes) ||
      schedule->used_bytes > schedule->bytes ||
      record_bytes > schedule->bytes - schedule->used_bytes)
    return false;

  micro_step_descriptor_t *descriptor =
      (micro_step_descriptor_t *)(void *)(
          schedule->data + schedule->used_bytes);
  micro_member_descriptor_t *members =
      (micro_member_descriptor_t *)(void *)(descriptor + 1);

  const size_t plain_stride_A =
      gemmini_page_packed_stride_payload(job->stride_A);
  const size_t plain_stride_B =
      gemmini_page_packed_stride_payload(job->stride_B);
  const size_t plain_stride_D =
      gemmini_page_packed_stride_payload(job->stride_D);
  const size_t plain_stride_C =
      gemmini_page_packed_stride_payload(job->stride_C);
  const size_t t = (size_t)(job->inner_call_counter & 1);
  const size_t local_sp_addr_start = t == 0u
      ? job->sp_addr_start_stack
      : job->sp_addr_start_stack + BANK_NUM * BANK_ROWS / 2u;
  const size_t local_sp_addr_end = t == 0u
      ? BANK_NUM * BANK_ROWS / 2u - job->sp_addr_end_stack
      : BANK_NUM * BANK_ROWS - job->sp_addr_end_stack;
  const size_t local_acc_addr_start = job->lastK_toggle
      ? job->acc_addr_start_stack
      : job->acc_addr_start_stack + ACC_ROWS / 2u;

  /* One ID per physical ACC quarter start.  It stays fixed across all outer-K
   * steps accumulating one C tile and changes only with the real ACC region. */
  if (ACC_ROWS % 8u != 0u || job->acc_addr_range % 2u != 0u)
    return false;
  const size_t acc_group_quantum = ACC_ROWS / 8u;
  const size_t acc_phase_begin = job->lastK_toggle ? 0u : ACC_ROWS / 2u;
  const size_t acc_phase_end = acc_phase_begin + ACC_ROWS / 2u;
  const size_t acc_allocation_rows = job->acc_addr_range / 2u;
  if (acc_group_quantum == 0u || acc_allocation_rows == 0u ||
      local_acc_addr_start % acc_group_quantum != 0u ||
      local_acc_addr_start < acc_phase_begin ||
      local_acc_addr_start > acc_phase_end ||
      acc_allocation_rows > acc_phase_end - local_acc_addr_start)
    return false;
  const size_t group_id = local_acc_addr_start / acc_group_quantum;
  if (group_id >= 8u)
    return false;

  const size_t A_row_offset = step.i0 * job->tile_I;
  const size_t A_col_offset = step.k0 * job->tile_K;
  const size_t B_row_offset = step.k0 * job->tile_K;
  const size_t B_col_offset = step.j0 * job->tile_J;
  const size_t D_row_offset = job->repeating_bias ? 0u : A_row_offset;
  const size_t D_col_offset = B_col_offset;
  const size_t C_row_offset = A_row_offset;
  const size_t C_col_offset = B_col_offset;

  if (!gemmini_loop_bounds_fields_valid(
          step.I, step.J, step.K,
          step.pad_I, step.pad_J, step.pad_K) ||
      !gemmini_page_offset_fields_valid(
          A_row_offset, A_col_offset, B_row_offset, B_col_offset,
          D_row_offset, D_col_offset, C_row_offset, C_col_offset))
    return false;

  const uintptr_t a_outer = (uintptr_t)job->A +
      (step.i0 * job->tile_I * DIM * plain_stride_A +
       step.k0 * job->tile_K * DIM) * sizeof(elem_t);
  const uintptr_t b_outer = (uintptr_t)job->B;
  const uintptr_t c_outer = step.final_k_step
      ? (uintptr_t)job->C +
            (step.i0 * job->tile_I * DIM * plain_stride_C +
             step.j0 * job->tile_J * DIM) * job->sizeof_C
      : UINT64_C(0);

  const size_t whole_partition_extent =
      job->partition_axis == GEMMINI_PARTITION_AXIS_M
          ? step.I
          : (job->partition_axis == GEMMINI_PARTITION_AXIS_N
                 ? step.J
                 : step.K);
  const size_t whole_aux_extent =
      job->partition_axis == GEMMINI_PARTITION_AXIS_M ? step.K : step.I;
  if (!gemmini_shared_partition_page_offsets_valid(
          job->partition_axis,
          whole_partition_extent, 0u, whole_aux_extent, 0u,
          step.I, step.J, step.K,
          false, true, false, false,
          A_row_offset, A_col_offset, B_row_offset, B_col_offset,
          D_row_offset, D_col_offset, C_row_offset, C_col_offset))
    return false;

  for (size_t rank = 0u; rank < step.active_gemmini_count; ++rank) {
    gemmini_partition_plan_t plan;
    if (!gemmini_partition_plan_member(
            job->partition_axis,
            step.I, step.J, step.K,
            step.pad_I, step.pad_J, step.pad_K,
            step.active_gemmini_count, rank, &plan) ||
        !gemmini_shared_partition_page_offsets_valid(
            job->partition_axis,
            plan.partition_extent, plan.partition_offset,
            plan.aux_extent, plan.aux_offset,
            step.I, step.J, step.K,
            false, true, false, false,
            A_row_offset, A_col_offset, B_row_offset, B_col_offset,
            D_row_offset, D_col_offset, C_row_offset, C_col_offset))
      return false;

    size_t member_pad_I = step.pad_I;
    size_t member_pad_J = step.pad_J;
    size_t member_pad_K = step.pad_K;
    size_t A_i_offset = 0u;
    size_t A_k_offset = 0u;
    size_t B_k_offset = 0u;
    size_t B_j_offset = 0u;
    size_t D_i_offset = 0u;
    size_t D_j_offset = 0u;
    size_t C_i_offset = 0u;
    size_t C_j_offset = 0u;
    if (job->partition_axis == GEMMINI_PARTITION_AXIS_M) {
      member_pad_I = plan.partition_pad;
      A_i_offset = plan.partition_offset;
      B_k_offset = plan.aux_offset;
      D_i_offset = plan.partition_offset;
      C_i_offset = plan.partition_offset;
    } else if (job->partition_axis == GEMMINI_PARTITION_AXIS_N) {
      member_pad_J = plan.partition_pad;
      A_i_offset = plan.aux_offset;
      B_j_offset = plan.partition_offset;
      D_j_offset = plan.partition_offset;
      C_j_offset = plan.partition_offset;
    } else if (job->partition_axis == GEMMINI_PARTITION_AXIS_K) {
      member_pad_K = plan.partition_pad;
      A_k_offset = plan.partition_offset;
      B_k_offset = plan.partition_offset;
      D_i_offset = plan.aux_offset;
      C_i_offset = plan.aux_offset;
    } else {
      return false;
    }

    const uintptr_t a_local = a_outer +
        (A_i_offset * plain_stride_A + A_k_offset) * DIM * sizeof(elem_t);
    /* B is page-packed: keep its allocation base and encode outer/member
     * logical coordinates only through page-offset and partition commands. */
    const uintptr_t b_local = b_outer;
    const uintptr_t d_local = UINT64_C(0);
    const uintptr_t c_local = c_outer == 0u
        ? UINT64_C(0)
        : c_outer +
              (C_i_offset * plain_stride_C + C_j_offset) *
                  DIM * job->sizeof_C;
    (void)plain_stride_B;
    (void)plain_stride_D;
    (void)B_k_offset;
    (void)B_j_offset;
    (void)D_i_offset;
    (void)D_j_offset;

    micro_member_descriptor_t *member = &members[rank];
    member->command[MICRO_CMD_SPADDR].rs1 = local_acc_addr_start;
    member->command[MICRO_CMD_SPADDR].rs2 =
        ((uint64_t)local_sp_addr_end << 16) | local_sp_addr_start;
    member->command[MICRO_CMD_PARTITION].rs1 =
        gemmini_shared_partition_pack_rs1(
            job->partition_axis, plan.partition_offset, plan.aux_offset);
    member->command[MICRO_CMD_PARTITION].rs2 =
        gemmini_shared_partition_pack_rs2(
            plan.partition_extent, plan.aux_extent, plan.aux_pad);
    member->command[MICRO_CMD_BOUNDS].rs1 =
        ((uint64_t)member_pad_K << 32) |
        ((uint64_t)member_pad_J << 16) | member_pad_I;
    member->command[MICRO_CMD_BOUNDS].rs2 =
        ((uint64_t)step.K << 32) |
        ((uint64_t)step.J << 16) | step.I;
    member->command[MICRO_CMD_ADDRS_AB].rs1 = a_local;
    member->command[MICRO_CMD_ADDRS_AB].rs2 = b_local;
    member->command[MICRO_CMD_ADDRS_DC].rs1 = d_local;
    member->command[MICRO_CMD_ADDRS_DC].rs2 = c_local;
    member->command[MICRO_CMD_STRIDES_AB].rs1 = job->stride_A;
    member->command[MICRO_CMD_STRIDES_AB].rs2 = job->stride_B;
    member->command[MICRO_CMD_STRIDES_DC].rs1 = job->stride_D;
    member->command[MICRO_CMD_STRIDES_DC].rs2 = job->stride_C;
    member->command[MICRO_CMD_PAGE_OFFSETS].rs1 =
        GEMMINI_PACK_PAGE_BLOCK_OFFSETS(
            A_row_offset, A_col_offset, B_row_offset, B_col_offset);
    member->command[MICRO_CMD_PAGE_OFFSETS].rs2 =
        GEMMINI_PACK_PAGE_BLOCK_OFFSETS(
            D_row_offset, D_col_offset, C_row_offset, C_col_offset);
  }

  const bool ex_accumulate =
      gemmini_partition_ex_accumulate(job->no_bias, step.k0);
  descriptor->launch.rs1 =
      ((uint64_t)(unsigned)step.group_list << 48) |
      ((uint64_t)group_id << 32) |
      ((uint64_t)job->act << 8) |
      ((uint64_t)job->low_D << 2) |
      ((uint64_t)job->full_C << 1) |
      (uint64_t)ex_accumulate;
  descriptor->launch.rs2 =
      ((uint64_t)job->b_transpose << 1) |
      (uint64_t)job->a_transpose;

  schedule->used_bytes += record_bytes;
  ++schedule->steps;
  schedule->timed_rocc_commands +=
      step.active_gemmini_count * MICRO_MEMBER_COMMANDS + 1u;
  shared_multi_matmul_job_complete_step(job);
  return job->partition_status == SHARED_MULTI_PARTITION_OK;
}

static bool micro_prepare_replay_schedule(
    const shared_multi_matmul_job_t initialized_jobs[MICRO_MAX_JOBS],
    int job_count, int tenant_count,
    const int tenant_job_indices[MICRO_MAX_TENANTS][MICRO_MAX_JOBS],
    const int queue_jobs[MICRO_MAX_TENANTS],
    micro_run_metrics_t *result,
    micro_replay_schedule_t *schedule) {
  if (initialized_jobs == NULL || result == NULL || schedule == NULL ||
      job_count <= 0 || job_count > MICRO_MAX_JOBS ||
      tenant_count <= 0 || tenant_count > MICRO_MAX_TENANTS)
    return false;
  memset(schedule, 0, sizeof(*schedule));

  size_t expected_steps = 0u;
  size_t expected_bytes = 0u;
  size_t expected_commands = 0u;
  for (int job = 0; job < job_count; ++job) {
    const micro_job_metrics_t *metrics = &result->jobs[job];
    if (!metrics->present || metrics->steps == 0u ||
        metrics->min_active_members == 0u ||
        metrics->max_active_members > 4u ||
        expected_steps > SIZE_MAX - metrics->steps)
      return false;
    expected_steps += metrics->steps;
    for (size_t active = 1u; active <= 4u; ++active) {
      const size_t count = metrics->active_member_steps[active];
      size_t record_bytes = 0u;
      if (!micro_descriptor_record_bytes(active, &record_bytes) ||
          (count != 0u && record_bytes > SIZE_MAX / count) ||
          expected_bytes > SIZE_MAX - count * record_bytes)
        return false;
      expected_bytes += count * record_bytes;
      const size_t commands = active * MICRO_MEMBER_COMMANDS + 1u;
      if (count != 0u && commands > SIZE_MAX / count)
        return false;
      const size_t job_commands = count * commands;
      if (expected_commands > SIZE_MAX - job_commands)
        return false;
      expected_commands += job_commands;
    }
  }
  if (expected_steps == 0u || expected_bytes == 0u ||
      expected_bytes > MICRO_MAX_DESCRIPTOR_BYTES)
    return false;
  schedule->data = (uint8_t *)malloc(expected_bytes);
  if (schedule->data == NULL)
    return false;
  schedule->bytes = expected_bytes;

  shared_multi_matmul_job_t dry_jobs[MICRO_MAX_JOBS];
  memcpy(dry_jobs, initialized_jobs, sizeof(dry_jobs));
  int tenant_positions[MICRO_MAX_TENANTS] = {0, 0, 0, 0};
  int active_jobs[MICRO_MAX_TENANTS] = {-1, -1, -1, -1};
  for (int tenant = 0; tenant < tenant_count; ++tenant)
    active_jobs[tenant] = tenant_job_indices[tenant][0];

  int live_tenants = tenant_count;
  bool ok = true;
  while (live_tenants != 0 && ok) {
    bool progressed = false;
    ++result->scheduler_rounds;
    for (int tenant = 0; tenant < tenant_count; ++tenant) {
      const int job_index = active_jobs[tenant];
      if (job_index < 0)
        continue;
      progressed = true;
      if (!micro_append_step_descriptor(&dry_jobs[job_index], schedule)) {
        ok = false;
        break;
      }
      if (!dry_jobs[job_index].done)
        continue;
      const int next = ++tenant_positions[tenant];
      if (next >= queue_jobs[tenant]) {
        active_jobs[tenant] = -1;
        --live_tenants;
        /* No timed intermediate rdcycle in lean replay. */
        result->tenant_issue_done_cycles[tenant] = 0u;
        result->tenant_issue_done_round[tenant] =
            result->scheduler_rounds;
      } else {
        active_jobs[tenant] = tenant_job_indices[tenant][next];
      }
    }
    if (!progressed)
      ok = false;
  }

  if (!ok || schedule->steps != expected_steps ||
      schedule->used_bytes != expected_bytes ||
      schedule->timed_rocc_commands != expected_commands) {
    micro_release_schedule(schedule);
    return false;
  }
  result->descriptor_bytes = schedule->bytes;
  result->timed_rocc_commands = schedule->timed_rocc_commands;
  return true;
}

#define MICRO_ISSUE_MEMBER(custom_num, member_ptr)                            \
  do {                                                                        \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_num,                                                           \
        (member_ptr)->command[MICRO_CMD_SPADDR].rs1,                          \
        (member_ptr)->command[MICRO_CMD_SPADDR].rs2,                          \
        k_LOOP_WS_CONFIG_SPADDR);                                             \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_num,                                                           \
        (member_ptr)->command[MICRO_CMD_PARTITION].rs1,                       \
        (member_ptr)->command[MICRO_CMD_PARTITION].rs2,                       \
        k_LOOP_WS_CONFIG_PARTITION_BOUNDS);                                   \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_num,                                                           \
        (member_ptr)->command[MICRO_CMD_BOUNDS].rs1,                          \
        (member_ptr)->command[MICRO_CMD_BOUNDS].rs2,                          \
        k_LOOP_WS_CONFIG_BOUNDS);                                             \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_num,                                                           \
        (member_ptr)->command[MICRO_CMD_ADDRS_AB].rs1,                        \
        (member_ptr)->command[MICRO_CMD_ADDRS_AB].rs2,                        \
        k_LOOP_WS_CONFIG_ADDRS_AB);                                           \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_num,                                                           \
        (member_ptr)->command[MICRO_CMD_ADDRS_DC].rs1,                        \
        (member_ptr)->command[MICRO_CMD_ADDRS_DC].rs2,                        \
        k_LOOP_WS_CONFIG_ADDRS_DC);                                           \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_num,                                                           \
        (member_ptr)->command[MICRO_CMD_STRIDES_AB].rs1,                      \
        (member_ptr)->command[MICRO_CMD_STRIDES_AB].rs2,                      \
        k_LOOP_WS_CONFIG_STRIDES_AB);                                         \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_num,                                                           \
        (member_ptr)->command[MICRO_CMD_STRIDES_DC].rs1,                      \
        (member_ptr)->command[MICRO_CMD_STRIDES_DC].rs2,                      \
        k_LOOP_WS_CONFIG_STRIDES_DC);                                         \
    ROCC_INSTRUCTION_RS1_RS2(                                                 \
        custom_num,                                                           \
        (member_ptr)->command[MICRO_CMD_PAGE_OFFSETS].rs1,                    \
        (member_ptr)->command[MICRO_CMD_PAGE_OFFSETS].rs2,                    \
        k_LOOP_WS_CONFIG_PAGE_OFFSETS);                                       \
  } while (0)

static inline __attribute__((always_inline)) const uint8_t *
micro_issue_step_descriptor(const uint8_t *cursor) {
  const micro_step_descriptor_t *descriptor =
      (const micro_step_descriptor_t *)(const void *)cursor;
  const micro_member_descriptor_t *member =
      (const micro_member_descriptor_t *)(const void *)(descriptor + 1);
  const uint64_t launch_rs1 = descriptor->launch.rs1;
  const uint64_t launch_rs2 = descriptor->launch.rs2;
  const unsigned group_list = (unsigned)((launch_rs1 >> 48) & 0xfu);

  switch (group_list) {
  case 0x1u:
    MICRO_ISSUE_MEMBER(custom0, &member[0]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom0, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 1);
  case 0x2u:
    MICRO_ISSUE_MEMBER(custom1, &member[0]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom1, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 1);
  case 0x3u:
    MICRO_ISSUE_MEMBER(custom0, &member[0]);
    MICRO_ISSUE_MEMBER(custom1, &member[1]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom1, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 2);
  case 0x4u:
    MICRO_ISSUE_MEMBER(custom2, &member[0]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom2, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 1);
  case 0x7u:
    MICRO_ISSUE_MEMBER(custom0, &member[0]);
    MICRO_ISSUE_MEMBER(custom1, &member[1]);
    MICRO_ISSUE_MEMBER(custom2, &member[2]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom2, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 3);
  case 0x8u:
    MICRO_ISSUE_MEMBER(custom3, &member[0]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom3, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 1);
  case 0xcu:
    MICRO_ISSUE_MEMBER(custom2, &member[0]);
    MICRO_ISSUE_MEMBER(custom3, &member[1]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom3, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 2);
  case 0xfu:
    MICRO_ISSUE_MEMBER(custom0, &member[0]);
    MICRO_ISSUE_MEMBER(custom1, &member[1]);
    MICRO_ISSUE_MEMBER(custom2, &member[2]);
    MICRO_ISSUE_MEMBER(custom3, &member[3]);
    ROCC_INSTRUCTION_RS1_RS2(
        custom3, launch_rs1, launch_rs2, k_LOOP_WS);
    return (const uint8_t *)(const void *)(member + 4);
  default:
    __builtin_unreachable();
  }
}

#undef MICRO_ISSUE_MEMBER

static bool micro_run_ready_set(
    const micro_shape_t *shape,
    size_t m,
    int tenant_count,
    const unsigned masks[MICRO_MAX_TENANTS],
    const int queue_jobs[MICRO_MAX_TENANTS],
    const gemmini_partition_axis_t axes[MICRO_MAX_TENANTS],
    bool require_full_cover,
    const micro_region_t a_regions[MICRO_MAX_JOBS],
    const micro_region_t b_regions[MICRO_MAX_JOBS],
    const micro_region_t c_regions[MICRO_MAX_JOBS],
    const micro_region_t *eviction,
    size_t a_bytes,
    micro_run_metrics_t *result) {
  shared_multi_matmul_job_t jobs[MICRO_MAX_JOBS] = {{0}};
  int tenant_job_indices[MICRO_MAX_TENANTS][MICRO_MAX_JOBS] = {{0}};
  unsigned q = 0u;
  int job_count = 0;

  if (result == NULL || tenant_count <= 0 ||
      tenant_count > MICRO_MAX_TENANTS)
    return false;
  memset(result, 0, sizeof(*result));
  result->tenant_count = tenant_count;

  for (int tenant = 0; tenant < tenant_count; ++tenant) {
    const unsigned members = micro_popcount4(masks[tenant]);
    const unsigned allocated_quarters = members;
    if (!micro_make_tenant_layout(
            masks[tenant], q, allocated_quarters, queue_jobs[tenant],
            axes[tenant],
            &result->tenants[tenant]))
      return false;
    q += result->tenants[tenant].allocated_quarters;
    for (int queued = 0; queued < queue_jobs[tenant]; ++queued) {
      if (job_count >= MICRO_MAX_JOBS)
        return false;
      tenant_job_indices[tenant][queued] = job_count;
      if (!micro_configure_job(
              &jobs[job_count], &result->jobs[job_count], shape, m,
              job_count, tenant, &result->tenants[tenant],
              a_regions, b_regions, c_regions))
        return false;
      ++job_count;
    }
  }
  result->job_count = job_count;
  if (!micro_validate_layout_set(
          result->tenants, tenant_count, require_full_cover) ||
      (require_full_cover && job_count != MICRO_MAX_JOBS))
    return false;

  // Initialize only software cursor state here. Hardware configuration is
  // emitted once per physical Gemmini below, never once per logical job.
  for (int job = 0; job < job_count; ++job) {
    if (!micro_initialize_job_state(&jobs[job]) ||
        !micro_collect_job_plan_metrics(&jobs[job], &result->jobs[job]))
      return false;
  }

  micro_replay_schedule_t schedule;
  if (!micro_prepare_replay_schedule(
          jobs, job_count, tenant_count, tenant_job_indices, queue_jobs,
          result, &schedule))
    return false;

  unsigned union_mask = 0u;
  for (int tenant = 0; tenant < tenant_count; ++tenant)
    union_mask |= result->tenants[tenant].mask;
  if (!micro_configure_hw_once(jobs, job_count, union_mask)) {
    micro_release_schedule(&schedule);
    return false;
  }
  gemmini_fence();

  // gemmini_flush is deliberately not used for cache conditioning. B and C
  // are never touched after eviction. Descriptor prepass precedes the final A
  // round-robin pass; descriptors larger than L2 are not claimed fully warm.
  micro_evict_l2(eviction);
  micro_make_descriptors_recent(&schedule);
  micro_make_a_recent_round_robin(a_regions, job_count, a_bytes);

  micro_cpu_fence();
  const uint64_t start = micro_read_cycles();
  const uint8_t *cursor = schedule.data;
  for (size_t step = 0u; step < schedule.steps; ++step)
    cursor = micro_issue_step_descriptor(cursor);
  const uint64_t issue_end = micro_read_cycles();
  // Every retained capacity/concurrent epoch has exactly one final drain.
  gemmini_fence();
  const uint64_t end = micro_read_cycles();
  micro_cpu_fence();

  const bool consumed_exactly =
      cursor == schedule.data + schedule.bytes;
  result->issue_cycles = issue_end - start;
  result->fence_cycles = end - issue_end;
  result->makespan_cycles = end - start;
  micro_release_schedule(&schedule);
  return consumed_exactly;
}

static void micro_write_header(FILE *output) {
  fprintf(output,
          "schema_version,epoch_index,record_kind,shape_id,"
          "full_m_call_multiplicity,M,N,K,policy,calibration_axis,"
          "calibration_axis_id,calibration_gemmini_count,is_best_axis,"
          "tenant_count,job_count,cache_policy,l2_bytes,eviction_bytes,"
          "page_packing_mask,page_packed_a,page_packed_b,page_packed_c,"
          "page_packed_d,distinct_a,distinct_b,distinct_c,a_bytes,b_bytes,"
          "c_bytes,aggregate_a_bytes,aggregate_b_bytes,aggregate_c_bytes");
  for (int tenant = 0; tenant < MICRO_MAX_TENANTS; ++tenant) {
    fprintf(output,
            ",tenant%d_present,tenant%d_mask,tenant%d_members,"
            "tenant%d_allocated_quarters,tenant%d_queue_jobs,tenant%d_axis,"
            "tenant%d_axis_id,tenant%d_q,"
            "tenant%d_sp_start,tenant%d_sp_end,tenant%d_sp_range,"
            "tenant%d_sp_rows_per_operand_bank,tenant%d_acc_start,"
            "tenant%d_acc_range,tenant%d_acc_rows_per_half,"
            "tenant%d_a0_begin,tenant%d_a0_end,tenant%d_b0_begin,"
            "tenant%d_b0_end,tenant%d_a1_begin,tenant%d_a1_end,"
            "tenant%d_b1_begin,tenant%d_b1_end,tenant%d_acc0_begin,"
            "tenant%d_acc0_end,tenant%d_acc1_begin,tenant%d_acc1_end",
            tenant, tenant, tenant, tenant, tenant, tenant, tenant, tenant,
            tenant, tenant, tenant, tenant, tenant, tenant, tenant,
            tenant, tenant, tenant, tenant, tenant, tenant, tenant,
            tenant, tenant, tenant, tenant, tenant);
  }
  for (int job = 0; job < MICRO_MAX_JOBS; ++job) {
    fprintf(output,
            ",job%d_present,job%d_tenant,job%d_mask,job%d_members,"
            "job%d_axis,job%d_axis_id,job%d_tile_id,job%d_tile_i,"
            "job%d_tile_j,job%d_tile_k,job%d_used_a_rows,job%d_used_b_rows,"
            "job%d_used_acc_rows,job%d_steps,job%d_min_active_members,"
            "job%d_max_active_members,job%d_active1_steps,"
            "job%d_active2_steps,job%d_active3_steps,job%d_active4_steps",
            job, job, job, job, job, job, job, job,
            job, job, job, job, job, job, job, job,
            job, job, job, job);
  }
  fprintf(output, ",issue_cycles,fence_cycles,makespan_cycles");
  for (int tenant = 0; tenant < MICRO_MAX_TENANTS; ++tenant) {
    fprintf(output,
            ",tenant%d_issue_done_cycles,tenant%d_issue_done_round",
            tenant, tenant);
  }
  fprintf(output,
          ",scheduler_rounds,execution_path,descriptor_bytes,"
          "timed_rocc_commands,group_id_policy,"
          "tenant_issue_done_cycles_valid\n");
}

static void micro_write_result(
    FILE *output,
    int epoch_index,
    const char *record_kind,
    const micro_shape_t *shape,
    size_t m,
    const char *policy,
    gemmini_partition_axis_t calibration_axis,
    int calibration_count,
    bool is_best_axis,
    size_t a_bytes,
    size_t b_bytes,
    size_t c_bytes,
    const micro_run_metrics_t *result) {
  const bool distinct = result->job_count > 1;
  fprintf(output,
          "4,%d,%s,%s,%d,%zu,%zu,%zu,%s,%s,%d,%d,%d,%d,%d,"
          "B_cold_descriptor_prepass_then_A_recent_round_robin_C_cold,"
          "%u,%u,%u,0,1,0,0,%d,%d,%d,"
          "%zu,%zu,%zu,%zu,%zu,%zu",
          epoch_index, record_kind, shape->name,
          shape->full_m_call_multiplicity, m, shape->n, shape->k, policy,
          calibration_count > 0 ? micro_axis_name(calibration_axis) : "-",
          calibration_count > 0 ? (int)calibration_axis : -1,
          calibration_count, is_best_axis ? 1 : 0,
          result->tenant_count, result->job_count,
          MICRO_L2_BYTES, MICRO_EVICTION_BYTES, MICRO_PAGE_PACKING_MASK,
          distinct ? 1 : 0, distinct ? 1 : 0, distinct ? 1 : 0,
          a_bytes, b_bytes, c_bytes,
          a_bytes * (size_t)result->job_count,
          b_bytes * (size_t)result->job_count,
          c_bytes * (size_t)result->job_count);

  for (int tenant = 0; tenant < MICRO_MAX_TENANTS; ++tenant) {
    const micro_tenant_layout_t *layout = &result->tenants[tenant];
    fprintf(output,
            ",%d,0x%x,%u,%u,%d,%s,%d,%u,%zu,%zu,%zu,%zu,%zu,%zu,%zu,"
            "%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu",
            layout->present ? 1 : 0,
            layout->mask,
            layout->members,
            layout->allocated_quarters,
            layout->queue_jobs,
            layout->present ? micro_axis_name(layout->axis) : "-",
            layout->present ? (int)layout->axis : -1,
            layout->q,
            layout->sp_start,
            layout->sp_end,
            layout->sp_range,
            layout->sp_rows_per_operand_bank,
            layout->acc_start,
            layout->acc_range,
            layout->acc_rows_per_half,
            layout->a_begin[0], layout->a_end[0],
            layout->b_begin[0], layout->b_end[0],
            layout->a_begin[1], layout->a_end[1],
            layout->b_begin[1], layout->b_end[1],
            layout->acc_begin[0], layout->acc_end[0],
            layout->acc_begin[1], layout->acc_end[1]);
  }
  for (int job = 0; job < MICRO_MAX_JOBS; ++job) {
    const micro_job_metrics_t *metrics = &result->jobs[job];
    fprintf(output,
            ",%d,%d,0x%x,%u,%s,%d,%d,%zu,%zu,%zu,%zu,%zu,%zu,%zu,"
            "%zu,%zu,%zu,%zu,%zu,%zu",
            metrics->present ? 1 : 0,
            metrics->present ? metrics->tenant : -1,
            metrics->mask,
            metrics->members,
            metrics->present ? micro_axis_name(metrics->axis) : "-",
            metrics->present ? (int)metrics->axis : -1,
            metrics->present ? metrics->tile_id : -1,
            metrics->tile_i,
            metrics->tile_j,
            metrics->tile_k,
            metrics->used_a_rows,
            metrics->used_b_rows,
            metrics->used_acc_rows,
            metrics->steps,
            metrics->present ? metrics->min_active_members : 0u,
            metrics->max_active_members,
            metrics->active_member_steps[1],
            metrics->active_member_steps[2],
            metrics->active_member_steps[3],
            metrics->active_member_steps[4]);
  }
  fprintf(output, ",%llu,%llu,%llu",
          (unsigned long long)result->issue_cycles,
          (unsigned long long)result->fence_cycles,
          (unsigned long long)result->makespan_cycles);
  for (int tenant = 0; tenant < MICRO_MAX_TENANTS; ++tenant) {
    fprintf(output, ",%llu,%llu",
            (unsigned long long)result->tenant_issue_done_cycles[tenant],
            (unsigned long long)result->tenant_issue_done_round[tenant]);
  }
  fprintf(output,
          ",%llu,precomputed_compact_rocc_replay,%zu,%zu,"
          "acc_local_quarter_start,0\n",
          (unsigned long long)result->scheduler_rounds,
          result->descriptor_bytes,
          result->timed_rocc_commands);
  fflush(output);
}

static bool micro_map_all_regions(
    micro_region_t a_regions[MICRO_MAX_JOBS],
    micro_region_t b_regions[MICRO_MAX_JOBS],
    micro_region_t c_regions[MICRO_MAX_JOBS],
    micro_region_t *eviction,
    size_t max_a_bytes,
    size_t max_b_bytes,
    size_t max_c_bytes) {
  static const char *a_names[] = {"A0", "A1", "A2", "A3"};
  static const char *b_names[] = {"B0", "B1", "B2", "B3"};
  static const char *c_names[] = {"C0", "C1", "C2", "C3"};
  for (int job = 0; job < MICRO_MAX_JOBS; ++job) {
    if (!micro_region_map(&a_regions[job], a_names[job], max_a_bytes) ||
        !micro_region_map(&b_regions[job], b_names[job], max_b_bytes) ||
        !micro_region_map(&c_regions[job], c_names[job], max_c_bytes))
      return false;
  }
  return micro_region_map(
      eviction, "L2_eviction", MICRO_EVICTION_BYTES);
}

int main(int argc, char **argv) {
  const char *output_path = argc > 1 ? argv[1] : "/output";
  if (argc > 2) {
    fprintf(stderr, "usage: %s [output.csv]\n", argv[0]);
    return 2;
  }
  if (DIM != 8 || BANK_NUM != 4 || BANK_ROWS != 4096 || ACC_ROWS != 8192 ||
      ACC_ROWS % 8 != 0 || sizeof(micro_rocc_args_t) != 16u ||
      sizeof(micro_member_descriptor_t) != 128u ||
      sizeof(micro_step_descriptor_t) != 16u) {
    fprintf(stderr,
            "MICROBENCH incompatible geometry DIM=%d banks=%d bank_rows=%d "
            "acc_rows=%d\n",
            DIM, BANK_NUM, BANK_ROWS, ACC_ROWS);
    return 2;
  }
  if (gemmini_partition_generated_axis_abi_version() !=
      GEMMINI_SHARED_PARTITION_ENCODING_VERSION) {
    fprintf(stderr, "MICROBENCH shared M/N/K partition ABI unavailable\n");
    return 2;
  }

  signal(SIGINT, micro_signal_handler);
  signal(SIGTERM, micro_signal_handler);

  size_t max_a_bytes = 0u;
  size_t max_b_bytes = 0u;
  size_t max_c_bytes = 0u;
  for (size_t shape_index = 0; shape_index < MICRO_SHAPE_COUNT;
       ++shape_index) {
    size_t a_bytes = 0u;
    size_t b_bytes = 0u;
    size_t c_bytes = 0u;
    if (!micro_shape_storage(
            MICRO_M_MAX, &micro_shapes[shape_index],
            &a_bytes, &b_bytes, &c_bytes)) {
      fprintf(stderr, "MICROBENCH storage overflow\n");
      return 2;
    }
    if (a_bytes > max_a_bytes)
      max_a_bytes = a_bytes;
    if (b_bytes > max_b_bytes)
      max_b_bytes = b_bytes;
    if (c_bytes > max_c_bytes)
      max_c_bytes = c_bytes;
  }

  micro_region_t a_regions[MICRO_MAX_JOBS] = {{0}};
  micro_region_t b_regions[MICRO_MAX_JOBS] = {{0}};
  micro_region_t c_regions[MICRO_MAX_JOBS] = {{0}};
  micro_region_t eviction = {0};
  if (!micro_map_all_regions(
          a_regions, b_regions, c_regions, &eviction,
          max_a_bytes, max_b_bytes, max_c_bytes)) {
    fprintf(stderr, "MICROBENCH failed to map resident buffers\n");
    return 2;
  }

  FILE *output = fopen(output_path, "w");
  if (output == NULL) {
    fprintf(stderr, "MICROBENCH failed to open %s errno=%d\n",
            output_path, errno);
    return 2;
  }
  micro_write_header(output);
  fflush(output);

  printf("LLAMA2-MATMUL-MICROBENCH-BEGIN schema=4 expected_epochs=%d "
         "M=8..512/8 shapes=3 "
         "capacity_calibration=MNKxg1..g3 allocated_capacity=u/4 "
         "g4_axis=prior_exact_sweep_sha256_d73996cd91374aac44c7c22e7431952ac2c135cb72c86d235e2b8761116c9a4a "
         "concurrent_policies=5 concurrent_ready_set=4_distinct_jobs "
         "execution=precomputed_compact_rocc_replay "
         "descriptor_epoch_limit_bytes=%u "
         "concurrent_step_interleave=precomputed_tenant_round_robin "
         "all_epoch_final_fences=1 serial_g4=excluded "
         "group_id=acc_local_quarter_start "
         "page_packing=A0B1C0D0 "
         "cache=B_cold_descriptor_prepass_then_A_recent_rr_C_cold "
         "cpu_gold=0 data_fill=0 output=%s\n",
         MICRO_EXPECTED_EPOCHS, MICRO_MAX_DESCRIPTOR_BYTES, output_path);
  printf("LLAMA2-MATMUL-MICROBENCH-SPAD bank0=A0 bank1=B0 "
         "bank2=A1 bank3=B1 partition=within_each_bank "
         "ACC=pingpong_halves_each_partitioned_into_contiguous_q_quarters\n");
  printf("LLAMA2-MATMUL-MICROBENCH-TIMING "
         "tenant_issue_done_cycles=unmeasured_zero "
         "scheduler_round=one_precomputed_concurrent_tenant_pass "
         "timed_path=descriptor_dispatch_rocc_issue_loop_control_final_fence\n");
  fflush(stdout);

  // TLB maintenance is one-time setup only; it does not define L2 state.
  micro_flush_all_gemmini_tlbs_once();

  int epoch_index = 0;
  bool ok = true;
  for (size_t shape_index = 0;
       shape_index < MICRO_SHAPE_COUNT && ok && !micro_abort_requested;
       ++shape_index) {
    const micro_shape_t *shape = &micro_shapes[shape_index];
    for (size_t m = MICRO_M_MIN;
         m <= MICRO_M_MAX && ok && !micro_abort_requested;
         m += MICRO_M_STEP) {
      size_t a_bytes = 0u;
      size_t b_bytes = 0u;
      size_t c_bytes = 0u;
      if (!micro_shape_storage(
              m, shape, &a_bytes, &b_bytes, &c_bytes)) {
        ok = false;
        break;
      }

      gemmini_partition_axis_t best_axes[5] = {
        GEMMINI_PARTITION_AXIS_M,
        GEMMINI_PARTITION_AXIS_M,
        GEMMINI_PARTITION_AXIS_M,
        GEMMINI_PARTITION_AXIS_M,
        GEMMINI_PARTITION_AXIS_M,
      };
      if (!micro_lookup_prior_g4_axis(shape_index, m, &best_axes[4])) {
        fprintf(stderr,
                "MICROBENCH prior g4 axis lookup failed shape=%s M=%zu\n",
                shape->name, m);
        ok = false;
        break;
      }

      for (int count = 1;
           count <= MICRO_CAPACITY_MAX_GEMMINIS && ok; ++count) {
        micro_run_metrics_t calibration[MICRO_AXIS_COUNT];
        uint64_t best_cycles = UINT64_MAX;
        int best_axis_index = 0;
        for (int axis_id = 0; axis_id < MICRO_AXIS_COUNT; ++axis_id) {
          const unsigned masks[MICRO_MAX_TENANTS] = {
            (1u << count) - 1u, 0u, 0u, 0u};
          const int queue_jobs[MICRO_MAX_TENANTS] = {1, 0, 0, 0};
          const gemmini_partition_axis_t axes[MICRO_MAX_TENANTS] = {
            (gemmini_partition_axis_t)axis_id,
            GEMMINI_PARTITION_AXIS_M,
            GEMMINI_PARTITION_AXIS_M,
            GEMMINI_PARTITION_AXIS_M,
          };
          if (!micro_run_ready_set(
                  shape, m, 1, masks, queue_jobs, axes, false,
                  a_regions, b_regions, c_regions, &eviction,
                  a_bytes, &calibration[axis_id])) {
            fprintf(stderr,
                    "MICROBENCH calibration failed shape=%s M=%zu count=%d "
                    "axis=%s\n",
                    shape->name, m, count,
                    micro_axis_name((gemmini_partition_axis_t)axis_id));
            ok = false;
            break;
          }
          if (calibration[axis_id].makespan_cycles < best_cycles ||
              (calibration[axis_id].makespan_cycles == best_cycles &&
               axis_id < best_axis_index)) {
            best_cycles = calibration[axis_id].makespan_cycles;
            best_axis_index = axis_id;
          }
        }
        if (!ok)
          break;
        best_axes[count] = (gemmini_partition_axis_t)best_axis_index;
        for (int axis_id = 0; axis_id < MICRO_AXIS_COUNT; ++axis_id) {
          micro_write_result(
              output, epoch_index++, "capacity_calibration", shape, m,
              "capacity_isolated",
              (gemmini_partition_axis_t)axis_id, count,
              axis_id == best_axis_index,
              a_bytes, b_bytes, c_bytes, &calibration[axis_id]);
        }
      }

      for (size_t policy_index = 0;
           policy_index < sizeof(micro_policies) / sizeof(micro_policies[0]) &&
           ok;
           ++policy_index) {
        const micro_policy_t *policy = &micro_policies[policy_index];
        gemmini_partition_axis_t axes[MICRO_MAX_TENANTS] = {
          GEMMINI_PARTITION_AXIS_M,
          GEMMINI_PARTITION_AXIS_M,
          GEMMINI_PARTITION_AXIS_M,
          GEMMINI_PARTITION_AXIS_M,
        };
        for (int tenant = 0; tenant < policy->tenant_count; ++tenant)
          axes[tenant] = best_axes[micro_popcount4(policy->masks[tenant])];

        micro_run_metrics_t result;
        if (!micro_run_ready_set(
                shape, m, policy->tenant_count,
                policy->masks, policy->queue_jobs, axes, true,
                a_regions, b_regions, c_regions, &eviction,
                a_bytes, &result)) {
          fprintf(stderr,
                  "MICROBENCH policy failed shape=%s M=%zu policy=%s\n",
                  shape->name, m, policy->name);
          ok = false;
          break;
        }
        micro_write_result(
            output, epoch_index++, "policy", shape, m, policy->name,
            GEMMINI_PARTITION_AXIS_M, 0, false,
            a_bytes, b_bytes, c_bytes, &result);
      }

      printf("LLAMA2-MATMUL-MICROBENCH-PROGRESS epochs=%d/%d shape=%s M=%zu\n",
             epoch_index, MICRO_EXPECTED_EPOCHS, shape->name, m);
      fflush(stdout);
    }
  }

  fflush(output);
  (void)fsync(fileno(output));
  fclose(output);
  sync();

  for (int job = 0; job < MICRO_MAX_JOBS; ++job) {
    micro_region_unmap(&a_regions[job]);
    micro_region_unmap(&b_regions[job]);
    micro_region_unmap(&c_regions[job]);
  }
  micro_region_unmap(&eviction);

  if (micro_abort_requested) {
    printf("LLAMA2-MATMUL-MICROBENCH-ABORT epochs=%d/%d output=%s\n",
           epoch_index, MICRO_EXPECTED_EPOCHS, output_path);
    return 130;
  }
  if (!ok || epoch_index != MICRO_EXPECTED_EPOCHS) {
    printf("LLAMA2-MATMUL-MICROBENCH-FAIL epochs=%d/%d output=%s\n",
           epoch_index, MICRO_EXPECTED_EPOCHS, output_path);
    return 1;
  }
  printf("LLAMA2-MATMUL-MICROBENCH-PASS epochs=%d output=%s cache_sink=%llu\n",
         epoch_index, output_path,
         (unsigned long long)micro_cache_sink);
  return 0;
}
