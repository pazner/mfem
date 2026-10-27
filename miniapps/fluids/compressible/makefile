# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
# This file is part of MFEM. See LICENSE for details.
MFEM_DIR ?= ../../..
MFEM_BUILD_DIR ?= ../../..
MFEM_INSTALL_DIR ?= ../../../mfem
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),$(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))
MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)
.DEFAULT_GOAL := all
AGGLOM_DIR = ../../dg-agglomeration
AGGLOM_SRC = $(wildcard $(AGGLOM_DIR)/*.cpp)
# Build the objects locally (flattened) instead of inside dg-agglomeration/
AGGLOM_OBJ = $(notdir $(AGGLOM_SRC:.cpp=.o))

COMMON_OBJ = physics.o problems.o solver.o $(AGGLOM_OBJ)

vpath %.cpp $(AGGLOM_DIR)
vpath %.hpp $(AGGLOM_DIR)
ifeq ($(MFEM_USE_MPI),YES)
MINIAPPS = compressible test_compressible
endif
.PHONY: all clean clean-build clean-exec check
all: $(MINIAPPS)
compressible: compressible.o $(COMMON_OBJ) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(filter %.o,$^) -o $@ $(MFEM_LIBS)
test_compressible: test_compressible.o $(COMMON_OBJ) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(filter %.o,$^) -o $@ $(MFEM_LIBS)
%.o: %.cpp flow.hpp $(wildcard $(AGGLOM_DIR)/*.hpp) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@
$(MFEM_LIB_FILE):
	$(error Build MFEM first with make in the repository root)
PAR_TEST_EXECS = compressible test_compressible
MFEM_TESTS = TEST_EXECS
include $(MFEM_TEST_MK)
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
compressible-test-par: compressible
	@$(call mfem-test,$<, $(RUN_MPI), Compressible, -n 3 -o 1 -tf 0.01 -p manufactured -mu 0.01, YES)
test_compressible-test-par: test_compressible
	@$(call mfem-test,$<, $(RUN_MPI), Compressible Jacobians,, YES)
check: all
	$(RUN_MPI) ./test_compressible
	$(RUN_MPI) ./compressible -n 3 -o 1 -tf .01 -p manufactured -mu .01
clean: clean-build clean-exec
clean-build:
	rm -f *.o compressible test_compressible
clean-exec:
	rm -rf ParaView
