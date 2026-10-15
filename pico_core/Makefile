.DEFAULT_GOAL := all

# Compiler and flags
CFLAGS = $(CFLAGS_COMMON)

.PHONY: all clean FORCE

BIN_DIR            ?= ../bin
PICO_CORE_SRC      := $(wildcard *.c)

PICO_CORE_OBJ_DIR       ?= ../obj/pico_core
PICO_CORE_OBJS          := $(patsubst %.c, $(PICO_CORE_OBJ_DIR)/%.o, $(PICO_CORE_SRC))
PICO_CORE_DEPS          := $(patsubst %.c, $(PICO_CORE_OBJ_DIR)/%.d, $(PICO_CORE_SRC))
PICO_CORE_CONFIG        := $(PICO_CORE_OBJ_DIR)/.build-config
LIB_DIR                 ?= ../lib
LIBPICO                 := $(LIB_DIR)/libpico.a
PICO_CORE               := $(BIN_DIR)/pico_core

PICO_CORE_OBJ_DIR_CUDA  ?= ../obj/pico_core_cuda
PICO_CORE_OBJS_CUDA     := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_CUDA)/%.o, $(PICO_CORE_SRC))
PICO_CORE_DEPS_CUDA     := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_CUDA)/%.d, $(PICO_CORE_SRC))
PICO_CORE_CONFIG_CUDA   := $(PICO_CORE_OBJ_DIR_CUDA)/.build-config
LIBPICO_CUDA            := $(LIB_DIR)/libpico_cuda.a
PICO_CORE_CUDA          := $(BIN_DIR)/pico_core_cuda
ifeq ($(GPU_NATIVE_SUPPORT),1)
CFLAGS_CUDA += -DGPU_NATIVE_SUPPORT
endif

PICO_CORE_OBJ_DIR_NCCL  ?= ../obj/pico_core_nccl
PICO_CORE_OBJS_NCCL     := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_NCCL)/%.o, $(PICO_CORE_SRC))
PICO_CORE_DEPS_NCCL     := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_NCCL)/%.d, $(PICO_CORE_SRC))
PICO_CORE_CONFIG_NCCL   := $(PICO_CORE_OBJ_DIR_NCCL)/.build-config
LIBPICO_NCCL            := $(LIB_DIR)/libpico_nccl.a
PICO_CORE_NCCL          := $(BIN_DIR)/pico_core_nccl

PICO_CORE_OBJ_DIR_UCC   ?= ../obj/pico_core_ucc
PICO_CORE_OBJS_UCC      := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_UCC)/%.o, $(PICO_CORE_SRC))
PICO_CORE_DEPS_UCC      := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_UCC)/%.d, $(PICO_CORE_SRC))
PICO_CORE_CONFIG_UCC    := $(PICO_CORE_OBJ_DIR_UCC)/.build-config
LIBPICO_UCC             := $(LIB_DIR)/libpico.a
PICO_CORE_UCC           := $(BIN_DIR)/pico_core_ucc

PICO_CORE_OBJ_DIR_UCC_CUDA   ?= ../obj/pico_core_ucc_cuda
PICO_CORE_OBJS_UCC_CUDA      := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_UCC_CUDA)/%.o, $(PICO_CORE_SRC))
PICO_CORE_DEPS_UCC_CUDA      := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_UCC_CUDA)/%.d, $(PICO_CORE_SRC))
PICO_CORE_CONFIG_UCC_CUDA    := $(PICO_CORE_OBJ_DIR_UCC_CUDA)/.build-config
LIBPICO_UCC_CUDA             := $(LIB_DIR)/libpico.a
PICO_CORE_UCC_CUDA           := $(BIN_DIR)/pico_core_ucc_cuda

NCCL_LDLIBS ?= $(LDLIBS)
UCC_LDLIBS ?= $(LDLIBS)

ifeq ($(DEBUG),1)
	CFLAGS := $(filter-out -O%,$(CFLAGS)) -O0
endif

FORCE:

$(PICO_CORE_CONFIG): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' "PICOCC=$(PICOCC)" "CFLAGS=$(CFLAGS)" > "$@.tmp"
	@if cmp -s "$@.tmp" "$@"; then rm -f "$@.tmp"; else mv "$@.tmp" "$@"; fi

$(PICO_CORE_CONFIG_CUDA): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' "PICOCC=$(PICOCC)" "CFLAGS=$(CFLAGS)" \
		"CFLAGS_CUDA=$(CFLAGS_CUDA)" > "$@.tmp"
	@if cmp -s "$@.tmp" "$@"; then rm -f "$@.tmp"; else mv "$@.tmp" "$@"; fi

$(PICO_CORE_CONFIG_NCCL): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' "PICOCC=$(PICOCC)" "CFLAGS=$(CFLAGS)" \
		"NCCL_FLAG=$(NCCL_FLAG)" "NCCL_LDLIBS=$(NCCL_LDLIBS)" > "$@.tmp"
	@if cmp -s "$@.tmp" "$@"; then rm -f "$@.tmp"; else mv "$@.tmp" "$@"; fi

$(PICO_CORE_CONFIG_UCC): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' "PICOCC=$(PICOCC)" "CFLAGS=$(CFLAGS)" \
		"UCC_FLAG=$(UCC_FLAG)" "UCC_LDLIBS=$(UCC_LDLIBS)" > "$@.tmp"
	@if cmp -s "$@.tmp" "$@"; then rm -f "$@.tmp"; else mv "$@.tmp" "$@"; fi

$(PICO_CORE_CONFIG_UCC_CUDA): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' "PICOCC=$(PICOCC)" "CFLAGS=$(CFLAGS)" \
		"UCC_FLAG=$(UCC_FLAG)" "UCC_LDLIBS=$(UCC_LDLIBS)" > "$@.tmp"
	@if cmp -s "$@.tmp" "$@"; then rm -f "$@.tmp"; else mv "$@.tmp" "$@"; fi

$(PICO_CORE_OBJS): $(PICO_CORE_CONFIG)
$(PICO_CORE_OBJS_CUDA): $(PICO_CORE_CONFIG_CUDA)
$(PICO_CORE_OBJS_NCCL): $(PICO_CORE_CONFIG_NCCL)
$(PICO_CORE_OBJS_UCC): $(PICO_CORE_CONFIG_UCC)
$(PICO_CORE_OBJS_UCC_CUDA): $(PICO_CORE_CONFIG_UCC_CUDA)

ifeq ($(PICO_MPI_CUDA_AWARE),1)
all: $(PICO_CORE_CUDA) $(PICO_CORE)
else
all: $(PICO_CORE)
endif

ifeq ($(PICO_NCCL),1)
all: $(PICO_CORE_NCCL)
endif

ifeq ($(strip $(PICO_UCC))-$(strip $(PICO_MPI_CUDA_AWARE)), 1-1)
all: $(PICO_CORE_UCC_CUDA) $(PICO_CORE_UCC)
else ifeq ($(PICO_UCC), 1)
all: $(PICO_CORE_UCC)
endif

# Build the executable
$(PICO_CORE): $(PICO_CORE_OBJS) $(LIBPICO)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking pico_core: $< -> $@$(NC)"
	$(PICOCC) $(CFLAGS) -o $@ $(PICO_CORE_OBJS) $(LIBPICO) -lm

# Compile source files
$(PICO_CORE_OBJ_DIR)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) -c $< -o $@

-include $(PICO_CORE_DEPS)

$(PICO_CORE_CUDA): $(PICO_CORE_OBJS_CUDA) $(LIBPICO_CUDA)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking pico_core_cuda: $< -> $@$(NC)"
	$(PICOCC) $(CFLAGS) -o $@ $(PICO_CORE_OBJS_CUDA) $(LIBPICO_CUDA) -lcudart -lstdc++ -lm

$(PICO_CORE_OBJ_DIR_CUDA)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling CUDA $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) $(CFLAGS_CUDA) -DPICO_MPI_CUDA_AWARE -lpico_cuda -lcudart -c $< -o $@

-include $(PICO_CORE_DEPS_CUDA)

$(PICO_CORE_NCCL): $(PICO_CORE_OBJS_NCCL) $(LIBPICO_NCCL)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking pico_core_nccl: $< -> $@$(NC)"
	$(PICOCC) $(CFLAGS) $(NCCL_LDLIBS) $(NCCL_FLAG) -o $@ $(PICO_CORE_OBJS_NCCL) $(LIBPICO_NCCL) -lcudart -lm

$(PICO_CORE_OBJ_DIR_NCCL)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR_NCCL)
	@echo -e "$(YELLOW)[CC] Compiling NCCL $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) $(NCCL_LDLIBS) $(NCCL_FLAG) -DPICO_NCCL -lcudart -c $< -o $@

-include $(PICO_CORE_DEPS_NCCL)

# Build the executable (UCC)
$(PICO_CORE_UCC): $(PICO_CORE_OBJS_UCC) $(LIBPICO_UCC)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking pico_core_ucc: $< -> $@$(NC)"
	$(PICOCC) $(CFLAGS) $(UCC_LDLIBS) $(UCC_FLAG) -o $@ $(PICO_CORE_OBJS_UCC) $(LIBPICO_UCC) -lm

# Compile source files (UCC)
$(PICO_CORE_OBJ_DIR_UCC)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR_UCC)
	@echo -e "$(YELLOW)[CC] Compiling UCC $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) $(UCC_LDLIBS) $(UCC_FLAG) -DPICO_UCC -c $< -o $@

-include $(PICO_CORE_DEPS_UCC)

# Build the executable (UCC CUDA)
$(PICO_CORE_UCC_CUDA): $(PICO_CORE_OBJS_UCC_CUDA) $(LIBPICO_UCC_CUDA)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking pico_core_ucc_cuda: $< -> $@$(NC)"
	$(PICOCC) $(CFLAGS) $(UCC_LDLIBS) $(UCC_FLAG) -o $@ $(PICO_CORE_OBJS_UCC_CUDA) $(LIBPICO_UCC_CUDA) -lcudart -lm

# Compile source files (UCC CUDA)
$(PICO_CORE_OBJ_DIR_UCC_CUDA)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR_UCC_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling UCC CUDA $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) $(UCC_LDLIBS) $(UCC_FLAG) -DPICO_UCC -DPICO_MPI_CUDA_AWARE -c $< -o $@

-include $(PICO_CORE_DEPS_UCC_CUDA)

clean:
	@echo -e "[CLEAN] Removing pico_core object files and binary..."
	@rm -rf $(PICO_CORE_OBJ_DIR) $(PICO_CORE) $(PICO_CORE_OBJ_DIR_CUDA) $(PICO_CORE_CUDA) $(PICO_CORE_OBJ_DIR_NCCL) $(PICO_CORE_NCCL) $(PICO_CORE_OBJ_DIR_UCC) $(PICO_CORE_UCC) $(PICO_CORE_OBJ_DIR_UCC_CUDA) $(PICO_CORE_UCC_CUDA)
