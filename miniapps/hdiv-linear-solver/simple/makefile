# Build against the existing MFEM makefile configuration.
MFEM_DIR ?= ../../..
MFEM_BUILD_DIR ?= $(MFEM_DIR)
MFEM_INSTALL_DIR ?= $(MFEM_DIR)/mfem
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

SOURCES = hb_simple.cpp hb.cpp fast_diag.cpp broken_hdiv.cpp
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = hb.hpp fast_diag.hpp broken_hdiv.hpp

.PHONY: all clean
all: hb_simple

ifeq ($(MFEM_USE_MPI),NO)
$(error hb_simple requires MFEM built with MPI)
endif
ifeq ($(MFEM_USE_LAPACK),NO)
$(error hb_simple requires MFEM built with LAPACK)
endif

hb_simple: $(OBJECTS)
	$(MFEM_CXX) $(MFEM_LINK_FLAGS) $(OBJECTS) -o $@ $(MFEM_LIBS)

%.o: %.cpp $(HEADERS) $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) -c $< -o $@

$(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -f $(OBJECTS) hb_simple
