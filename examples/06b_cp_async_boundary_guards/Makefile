# ----------------------------------------------
# PTX Tensor Core Example 06b (cp.async Boundary Guards)
# ----------------------------------------------

# Toolchain
PYTHON ?= python3
NVCC ?= nvcc
NVCCFLAGS ?= -arch=sm_89

# Project paths
CONFIG := example.json
SCRIPTS_DIR := ../../scripts
SRC_DIR := src
BINARY := tc_mma_ptx_cp_async_bounds
SOURCES := $(SRC_DIR)/main.cu $(SRC_DIR)/kernel.cu
INPUT_DIR := inputs
OUTPUT_DIR := outputs
INPUT_A := $(INPUT_DIR)/A.txt
INPUT_B := $(INPUT_DIR)/B.txt

# ANSI colors for readable logs
RESET := \033[0m
BOLD := \033[1m
BLUE := \033[1;34m
GREEN := \033[1;32m
YELLOW := \033[1;33m
CYAN := \033[1;36m

# Log helpers
log_info = @printf "%b\n" "$(BLUE)[INFO]$(RESET) $(1)"
log_ok = @printf "%b\n" "$(GREEN)[OK]$(RESET) $(1)"
log_warn = @printf "%b\n" "$(YELLOW)[WARN]$(RESET) $(1)"
log_step = @printf "%b\n" "$(CYAN)=>$(RESET) $(1)"

.PHONY: all build gen-input prep-input run compare clean help

all: ## Build, generate inputs, run kernel, and compare results
	$(call log_step,Building example binary)
	@$(MAKE) build
	$(call log_step,Generating input matrices)
	@$(MAKE) gen-input
	$(call log_step,Running GPU kernel)
	@$(MAKE) run
	$(call log_step,Comparing GPU and CPU outputs)
	@$(MAKE) compare
	$(call log_ok,Completed full workflow)

build: $(BINARY) ## Compile this example

$(BINARY): $(SOURCES)
	$(call log_info,Compiling $(BINARY) with $(NVCCFLAGS))
	$(NVCC) $(NVCCFLAGS) $(SOURCES) -o $(BINARY)
	$(call log_ok,Built $(BINARY))

gen-input: ## Generate A/B input matrices from $(CONFIG)
	$(call log_info,Generating inputs using $(CONFIG))
	$(PYTHON) $(SCRIPTS_DIR)/generate_input.py --config $(CONFIG) --example-root .
	$(call log_ok,Input files ready in $(INPUT_DIR)/)

prep-input: ## Generate inputs only if missing
	@test -f $(INPUT_A) && test -f $(INPUT_B) || $(MAKE) gen-input

run: build prep-input ## Run the compiled binary
	$(call log_info,Launching $(BINARY))
	./$(BINARY) --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR)
	$(call log_ok,GPU output written to $(OUTPUT_DIR)/)

compare: ## Compare GPU output with CPU reference
	$(call log_info,Running comparison script)
	$(PYTHON) $(SCRIPTS_DIR)/compare_gpu_output.py --config $(CONFIG) --example-root .
	$(call log_ok,Comparison completed)

clean: ## Remove binary and generated input/output files
	$(call log_warn,Removing generated files)
	rm -rf $(BINARY) $(INPUT_DIR) $(OUTPUT_DIR)
	$(call log_ok,Workspace cleaned)

help: ## Show this help message
	@printf "%b\n" "$(BOLD)Usage: make <target>$(RESET)"
	@printf "%b\n" ""
	@printf "%b\n" "$(CYAN)Targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  \033[1;36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
