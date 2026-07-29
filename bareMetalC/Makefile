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

# VPU programs require the custom0/funct64 RTL model and are intentionally not
# added to the default Spike-backed test list.
vpu_tests = vpu_dma vpu_nonlinear vpu_interleave vpu_mask_rearrange \
	vpu_strided_dma
vpu_tests_baremetal = $(vpu_tests:=-baremetal)
vpu_tests_pk = $(vpu_tests:=-pk)

ifeq ($(findstring spike,$(RUNNER)),spike)
# Currently don't support conv or conv-with-pool on spike
runs_baremetal = $(addsuffix .run,$(filter-out conv-baremetal conv_with_pool-baremetal,$(tests_baremetal)))
else
# Don't run very long benchmarks for RTL sim
runs_baremetal = $(addsuffix .run,$(filter-out tiled_matmul_cpu-baremetal tiled_matmul_option-baremetal,$(tests_baremetal)))
endif

ifdef BAREMETAL_ONLY
	tests_linux =
	tests_pk =
else
	tests_linux = $(tests:=-linux)
	tests_pk = $(tests:=-pk)
endif

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_testutils.h $(abs_top_srcdir)/include/gemmini_testutils_all.h $(abs_top_srcdir)/include/gemmini_all.h
VPU_GENERATED_HEADER ?= vpu_params_generated.h
VPU_GENERATED_HEADER_DEP = $(if $(filter /%,$(VPU_GENERATED_HEADER)),$(VPU_GENERATED_HEADER),$(abs_top_srcdir)/include/$(VPU_GENERATED_HEADER))
VPU_HEADERS = $(abs_top_srcdir)/include/vpu.h $(abs_top_srcdir)/include/vpu_params.h $(abs_top_srcdir)/include/vpu_kernels.h $(abs_top_srcdir)/include/vpu_testutils.h $(VPU_GENERATED_HEADER_DEP)

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

all: $(tests_baremetal) $(tests_linux) $(tests_pk)

.PHONY: vpu-baremetal vpu-pk
vpu-baremetal: $(vpu_tests_baremetal)
vpu-pk: $(vpu_tests_pk)

vpath %.c $(src_dir)

%-baremetal: %.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS)

%-linux: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) $< $(LFLAGS) -o $@

%-pk: %.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS_PK) $< $(LFLAGS) -o $@ $(LIBS)

$(vpu_tests_baremetal) $(vpu_tests_pk): LIBS += -lm -lc -lgcc
$(vpu_tests_baremetal): CFLAGS_BAREMETAL += -fno-fast-math -DVPU_PARAMS_GENERATED_HEADER='"$(VPU_GENERATED_HEADER)"'
$(vpu_tests_pk): CFLAGS_PK += -fno-fast-math -DVPU_PARAMS_GENERATED_HEADER='"$(VPU_GENERATED_HEADER)"'
$(vpu_tests_baremetal) $(vpu_tests_pk): $(VPU_HEADERS)

run-baremetal: $(runs_baremetal)

%-baremetal.run: %-baremetal
	$(RUNNER)$(abs_top_srcdir)/build/bareMetalC/$^

junk += $(tests_baremetal) $(tests_linux) $(tests_pk) $(vpu_tests_baremetal) $(vpu_tests_pk)
