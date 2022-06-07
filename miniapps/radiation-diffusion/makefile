# Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
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
SRC = $(if $(MFEM_DIR:../..=),$(MFEM_DIR)/miniapps/radiation-diffusion/,)
CONFIG_MK = $(MFEM_BUILD_DIR)/config/config.mk
# Use the MFEM install directory
# MFEM_INSTALL_DIR = ../../mfem
# CONFIG_MK = $(MFEM_INSTALL_DIR)/share/mfem/config.mk

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

RAD_DIFF_HEADERS = radiation_diffusion.hpp linear_solver.hpp mms.hpp \
						 nonlinear_iteration.hpp energy_integrator.hpp
RAD_DIFF_SRC = main.cpp radiation_diffusion.cpp linear_solver.cpp mms.cpp \
					nonlinear_iteration.cpp energy_integrator.cpp
RAD_DIFF_OBJ = $(RAD_DIFF_SRC:.cpp=.o)

MINIAPPS = radiation_diffusion

.SUFFIXES:
.SUFFIXES: .o .cpp .mk
.PHONY: all clean clean-build clean-exec
.PRECIOUS: %.o

all: $(MINIAPPS)

# Remove built-in rules
%: %.cpp
%.o: %.cpp

$(MINIAPPS):%: $(RAD_DIFF_OBJ)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(RAD_DIFF_OBJ) -o $@ $(MFEM_LIBS)

%.o: $(SRC)%.cpp $(RAD_DIFF_HEADERS) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

# Generate an error message if the MFEM library is not built and exit
$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean: clean-build clean-exec

clean-build:
	rm -f *.o *~ $(MINIAPPS)
	rm -rf *.dSYM *.TVD.*breakpoints

clean-exec:
	@rm -rf mesh.* sol.* ParaView
