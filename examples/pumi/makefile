# Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Use the MFEM build directory
MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../..
MFEM_INSTALL_DIR ?= ../../mfem
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/examples/pumi/,)
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

# All PUMI examples require MPI
SEQ_EXAMPLES =
PAR_EXAMPLES = ex1 ex1p ex2 ex6p
ifeq ($(MFEM_USE_MPI),NO)
   EXAMPLES = $(SEQ_EXAMPLES)
else
   EXAMPLES = $(PAR_EXAMPLES) $(SEQ_EXAMPLES)
endif

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: $(SRC)%.cpp $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $< -o $@ $(MFEM_LIBS)

all: $(EXAMPLES)

ifeq ($(MFEM_USE_PUMI),NO)
$(EXAMPLES):
	$(error MFEM is not configured with PUMI)
endif

MFEM_TESTS = EXAMPLES
include $(MFEM_TEST_MK)

ifneq (,$(filter test%,$(MAKECMDGOALS)))
   ifeq (,$(wildcard ../../data/pumi))
      $(info PUMI data directory not found. The PUMI tests will be SKIPPED.)
      mfem-test = printf "   $(3) [$(2) $(1) ... ]: "; $(PRINT_SKIP)
   endif
endif

# Testing: Parallel vs. serial runs
RUN_MPI_NP = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP)
RUN_MPI = $(RUN_MPI_NP) $(MFEM_MPI_NP)
SERIAL_NAME := Serial PUMI example
PARALLEL_NAME := Parallel PUMI example
%-test-par: %
	@$(call mfem-test,$<, $(RUN_MPI), $(PARALLEL_NAME))
%-test-seq: %
	@$(call mfem-test,$<,, $(SERIAL_NAME))

# Testing: Example-specific execution options:
ex1-test-par: ex1
	@$(call mfem-test,$<, $(RUN_MPI_NP) 1, $(PARALLEL_NAME))
ex1p-test-par: ex1p
	@$(call mfem-test,$<, $(RUN_MPI_NP) 8, $(PARALLEL_NAME))
ex2-test-par: ex2
	@$(call mfem-test,$<, $(RUN_MPI_NP) 1, $(PARALLEL_NAME))
ex6p-test-par: ex6p
	@$(call mfem-test,$<, $(RUN_MPI_NP) 8, $(PARALLEL_NAME))

# Testing: "test" target and mfem-test* variables are defined in config/test.mk

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(SEQ_EXAMPLES) $(PAR_EXAMPLES)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -f refined.mesh sol.gf mesh.* sol.* displaced.mesh
