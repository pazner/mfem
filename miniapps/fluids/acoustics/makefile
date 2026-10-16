# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
# This file is part of MFEM. See LICENSE for details.
MFEM_DIR ?= ../../..
MFEM_BUILD_DIR ?= ../../..
MFEM_INSTALL_DIR ?= ../../../mfem
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),$(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))
MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)
.DEFAULT_GOAL := all
COMMON_OBJ = physics.o solver.o
ifeq ($(MFEM_USE_MPI),YES)
MINIAPPS = acoustics test_acoustics
endif
.PHONY: all clean clean-build clean-exec check
all: $(MINIAPPS)
acoustics: acoustics.o $(COMMON_OBJ) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(filter %.o,$^) -o $@ $(MFEM_LIBS)
test_acoustics: test_acoustics.o $(COMMON_OBJ) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(filter %.o,$^) -o $@ $(MFEM_LIBS)
%.o: %.cpp acoustics.hpp $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@
$(MFEM_LIB_FILE):
	$(error Build MFEM first with make in the repository root)
PAR_TEST_EXECS = acoustics test_acoustics
MFEM_TESTS = TEST_EXECS
include $(MFEM_TEST_MK)
RUN_MPI = $(MFEM_MPIEXEC) $(MFEM_MPIEXEC_NP) $(MFEM_MPI_NP)
acoustics-test-par: acoustics
	@$(call mfem-test,$<, $(RUN_MPI), Acoustics, -n 3 -o 1 -tf 0.01, YES)
test_acoustics-test-par: test_acoustics
	@$(call mfem-test,$<, $(RUN_MPI), Acoustics operators,, YES)
check: all
	$(RUN_MPI) ./test_acoustics
	$(RUN_MPI) ./acoustics -n 3 -o 1 -tf .01
	$(RUN_MPI) ./acoustics -n 3 -o 1 --pc-test -dt .1
clean: clean-build clean-exec
clean-build:
	rm -f *.o acoustics test_acoustics
clean-exec:
	rm -rf ParaView
