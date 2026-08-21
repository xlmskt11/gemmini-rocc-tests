include $(abs_top_srcdir)/Makefrag

tests = \
	template \
	template_all \
	template_op0 \
	template_all2_2 \
	template_all2 \
	template_32x32 \
	template_32x32_2 \
	template_all2_multi1 \
	template_all2_multi1_conv \
	template_32x32_conv \
	template_all2_multi1_dynamic \
	template_all2_multi1_conv_dynamic \
	one_baremetal \
	two_baremetal \
	static_tiled_conv_test \
	static_tiled_matmul_test \
	static_tiled_matmul_2group_test \
	static_tiled_matmul_transpose_test \
	static_tiled_matmul_test_profile \
	three_baremetal \
	static_tiled_matmul_test_2 \
	llama_shape_test0 \
	llama_shape_test1 \
	llama_shape_test2

tests_baremetal = $(tests:=-baremetal)

# VPU programs require the custom0/funct64 RTL model. They are built with the
# ordinary bareMetalC programs, but remain outside the Spike run list below.
vpu_tests = vpu_dma vpu_nonlinear vpu_interleave vpu_mask_rearrange \
	vpu_strided_dma
vpu_tests_baremetal = $(vpu_tests:=-baremetal)

# One operation per binary keeps long scalar-libm RTL measurements selectable.
# The implementations share a compile-time benchmark driver, while each file
# remains an independently runnable simulation workload.
vpu_perf_tests = \
	rmsnorm_vs_CPU_perf final_norm_vs_CPU_perf \
	relu_vs_CPU_perf sigmoid_vs_CPU_perf tanh_vs_CPU_perf \
	gelu_vs_CPU_perf silu_vs_CPU_perf swiglu_vs_CPU_perf \
	softmax_vs_CPU_perf rope_vs_CPU_perf permute_vs_CPU_perf
vpu_perf_tests_baremetal = $(vpu_perf_tests:=-baremetal)

# This program targets the BF16-Gemmini/FP32-VPU fusion configuration.  Keep
# it separate from the standalone VPU list: neither Spike nor the legacy
# INT8+VPU configurations model the shared VSRAM matrix ports/group gate.
fusion_vpu_tests = vpu_flashattention
fusion_vpu_tests_baremetal = $(fusion_vpu_tests:=-baremetal)
fusion_vpu_perf_tests = vpu_flashattention_vs_CPU_perf
fusion_vpu_perf_tests_baremetal = $(fusion_vpu_perf_tests:=-baremetal)

ifeq ($(findstring spike,$(RUNNER)),spike)
# Currently don't support conv or conv-with-pool on spike
runs_baremetal = $(addsuffix .run,$(filter-out conv-baremetal conv_with_pool-baremetal,$(tests_baremetal)))
else
# Don't run very long benchmarks for RTL sim
runs_baremetal = $(addsuffix .run,$(filter-out tiled_matmul_cpu-baremetal tiled_matmul_option-baremetal,$(tests_baremetal)))
endif

ifeq ($(BAREMETAL_ONLY),1)
	tests_linux =
	tests_pk =
	vpu_tests_linux =
	vpu_tests_pk =
	vpu_perf_tests_linux =
	vpu_perf_tests_pk =
	fusion_vpu_tests_linux =
	fusion_vpu_tests_pk =
	fusion_vpu_perf_tests_linux =
	fusion_vpu_perf_tests_pk =
else
	tests_linux = $(tests:=-linux)
	tests_pk = $(tests:=-pk)
	vpu_tests_linux = $(vpu_tests:=-linux)
	vpu_tests_pk = $(vpu_tests:=-pk)
	vpu_perf_tests_linux = $(vpu_perf_tests:=-linux)
	vpu_perf_tests_pk = $(vpu_perf_tests:=-pk)
	fusion_vpu_tests_linux = $(fusion_vpu_tests:=-linux)
	fusion_vpu_tests_pk = $(fusion_vpu_tests:=-pk)
	fusion_vpu_perf_tests_linux = $(fusion_vpu_perf_tests:=-linux)
	fusion_vpu_perf_tests_pk = $(fusion_vpu_perf_tests:=-pk)
endif

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_testutils.h $(abs_top_srcdir)/include/gemmini_testutils_all.h $(abs_top_srcdir)/include/gemmini_all.h $(abs_top_srcdir)/include/gemmini_tiling.h
# Workload dimensions and benchmark options belong to their C source/header.
# Keep this Makefile limited to build flags, targets, and dependencies.
VPU_PARAMS_HEADER = $(abs_top_srcdir)/include/vpu_params.h
VPU_HEADERS = $(abs_top_srcdir)/include/vpu.h $(VPU_PARAMS_HEADER) $(abs_top_srcdir)/include/vpu_kernels.h $(abs_top_srcdir)/include/vpu_testutils.h
VPU_PERF_HEADERS = $(VPU_HEADERS) $(abs_top_srcdir)/bareMetalC/vpu_vs_cpu_perf_common.h
FUSION_VPU_HEADERS = $(VPU_HEADERS) $(abs_top_srcdir)/include/vpu_flashattention_kernel.h $(abs_top_srcdir)/include/gemmini_page_packed.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_tiling.h

CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-fno-tree-loop-distribute-patterns \
	-march=rv64gc -Wa,-march=rv64gc \
	-lm \
	-lgcc \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-DID_STRING=$(ID_STRING) \
	-DPRINT_TILE=0 \

CFLAGS_PK := \
	$(CFLAGS) \
	-static \
	-DBAREMETAL=1 \

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

# A single generated vpu_params.h describes the most recently elaborated VPU.
# Only add fusion-only programs to the default build when that header says the
# matrix bridge/grouped dependency hardware is present.
FUSION_VPU_ACTIVE := $(shell \
	grep -q '^\#define VPU_GROUPED_COMMANDS 1u' $(VPU_PARAMS_HEADER) 2>/dev/null && \
	grep -q '^\#define VPU_SHARED_DEPS 1u' $(VPU_PARAMS_HEADER) 2>/dev/null && \
	grep -Eq '^\#define VPU_MATRIX_PORTS [1-9][0-9]*u' $(VPU_PARAMS_HEADER) 2>/dev/null && \
	echo 1 || echo 0)
ifeq ($(FUSION_VPU_ACTIVE),1)
	default_fusion_tests = $(fusion_vpu_tests_baremetal) \
		$(fusion_vpu_tests_linux) $(fusion_vpu_tests_pk) \
		$(fusion_vpu_perf_tests_baremetal) \
		$(fusion_vpu_perf_tests_linux) $(fusion_vpu_perf_tests_pk)
else
	default_fusion_tests =
endif

all: $(tests_baremetal) $(tests_linux) $(tests_pk) \
	$(vpu_tests_baremetal) $(vpu_tests_linux) $(vpu_tests_pk) \
	$(vpu_perf_tests_baremetal) $(vpu_perf_tests_linux) \
	$(vpu_perf_tests_pk) $(default_fusion_tests)

.PHONY: vpu-baremetal vpu-linux vpu-pk \
	vpu-perf-baremetal vpu-perf-linux vpu-perf-pk \
	fusion-vpu-baremetal fusion-vpu-linux fusion-vpu-pk \
	fusion-vpu-perf-baremetal fusion-vpu-perf-linux fusion-vpu-perf-pk
vpu-baremetal: $(vpu_tests_baremetal)
vpu-linux: $(vpu_tests_linux)
vpu-pk: $(vpu_tests_pk)
vpu-perf-baremetal: $(vpu_perf_tests_baremetal)
vpu-perf-linux: $(vpu_perf_tests_linux)
vpu-perf-pk: $(vpu_perf_tests_pk)
fusion-vpu-baremetal: $(fusion_vpu_tests_baremetal)
fusion-vpu-linux: $(fusion_vpu_tests_linux)
fusion-vpu-pk: $(fusion_vpu_tests_pk)
fusion-vpu-perf-baremetal: $(fusion_vpu_perf_tests_baremetal)
fusion-vpu-perf-linux: $(fusion_vpu_perf_tests_linux)
fusion-vpu-perf-pk: $(fusion_vpu_perf_tests_pk)

vpath %.c $(src_dir)

%-baremetal: %.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

%-linux: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@ $(LIBS)

%-pk: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS_PK) $< $(LFLAGS) -o $@ $(LIBS)

vpu_test_targets = $(vpu_tests_baremetal) $(vpu_tests_linux) $(vpu_tests_pk)
vpu_perf_targets = $(vpu_perf_tests_baremetal) $(vpu_perf_tests_linux) \
	$(vpu_perf_tests_pk)
fusion_vpu_targets = $(fusion_vpu_tests_baremetal) \
	$(fusion_vpu_tests_linux) $(fusion_vpu_tests_pk)
fusion_vpu_perf_targets = $(fusion_vpu_perf_tests_baremetal) \
	$(fusion_vpu_perf_tests_linux) $(fusion_vpu_perf_tests_pk)

$(vpu_test_targets) $(vpu_perf_targets) $(fusion_vpu_targets) \
$(fusion_vpu_perf_targets): LIBS += -lm -lc -lgcc
$(vpu_tests_baremetal): CFLAGS_BAREMETAL += -fno-fast-math
$(vpu_tests_linux): CFLAGS += -fno-fast-math
$(vpu_tests_pk): CFLAGS_PK += -fno-fast-math
$(vpu_test_targets): $(VPU_HEADERS)

$(vpu_perf_tests_baremetal): CFLAGS_BAREMETAL += -fno-fast-math
$(vpu_perf_tests_linux): CFLAGS += -fno-fast-math
$(vpu_perf_tests_pk): CFLAGS_PK += -fno-fast-math
$(vpu_perf_targets): $(VPU_PERF_HEADERS)

$(fusion_vpu_tests_baremetal) $(fusion_vpu_perf_tests_baremetal): CFLAGS_BAREMETAL += -fno-fast-math \
	-I$(abs_top_srcdir)/include
$(fusion_vpu_tests_linux) $(fusion_vpu_perf_tests_linux): CFLAGS += -fno-fast-math \
	-I$(abs_top_srcdir)/include
$(fusion_vpu_tests_pk) $(fusion_vpu_perf_tests_pk): CFLAGS_PK += -fno-fast-math \
	-I$(abs_top_srcdir)/include
$(fusion_vpu_targets) $(fusion_vpu_perf_targets): $(FUSION_VPU_HEADERS)

run-baremetal: $(runs_baremetal)

%-baremetal.run: %-baremetal
	$(RUNNER)$(abs_top_srcdir)/build/bareMetalC/$^

# Clean every possible output even when this invocation has no Linux compiler
# and therefore sets BAREMETAL_ONLY=1.
junk += $(tests_baremetal) $(tests:=-linux) $(tests:=-pk) \
	$(vpu_tests_baremetal) $(vpu_tests:=-linux) $(vpu_tests:=-pk) \
	$(vpu_perf_tests_baremetal) $(vpu_perf_tests:=-linux) $(vpu_perf_tests:=-pk) \
	$(fusion_vpu_tests_baremetal) $(fusion_vpu_tests:=-linux) $(fusion_vpu_tests:=-pk) \
	$(fusion_vpu_perf_tests_baremetal) $(fusion_vpu_perf_tests:=-linux) $(fusion_vpu_perf_tests:=-pk)
