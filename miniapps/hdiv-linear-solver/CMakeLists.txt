# Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

if (MFEM_USE_MPI)
  add_mfem_miniapp(hb
    MAIN hb.cpp
    EXTRA_SOURCES fast_diag.cpp broken_hdiv.cpp discrete_divergence.cpp
    EXTRA_HEADERS hb.hpp fast_diag.hpp broken_hdiv.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(darcy
    MAIN darcy.cpp
    EXTRA_SOURCES change_basis.cpp discrete_divergence.cpp hdiv_linear_solver.cpp
    EXTRA_HEADERS change_basis.hpp discrete_divergence.hpp hdiv_linear_solver.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(grad_div
    MAIN grad_div.cpp
    EXTRA_SOURCES change_basis.cpp discrete_divergence.cpp hdiv_linear_solver.cpp
    EXTRA_HEADERS change_basis.hpp discrete_divergence.hpp hdiv_linear_solver.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(crooked_pipe
    MAIN crooked_pipe.cpp
    EXTRA_SOURCES change_basis.cpp discrete_divergence.cpp hdiv_linear_solver.cpp
    EXTRA_HEADERS change_basis.hpp discrete_divergence.hpp hdiv_linear_solver.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(spe10
    MAIN spe10.cpp
    EXTRA_SOURCES change_basis.cpp discrete_divergence.cpp hdiv_linear_solver.cpp
    EXTRA_HEADERS change_basis.hpp discrete_divergence.hpp hdiv_linear_solver.hpp
                  generic_coefficient.hpp spe10.hpp
    LIBRARIES mfem)

  add_mfem_miniapp(spe10_hb
    MAIN spe10_hb.cpp
    EXTRA_SOURCES fast_diag.cpp broken_hdiv.cpp discrete_divergence.cpp
    EXTRA_HEADERS hb.hpp fast_diag.hpp broken_hdiv.hpp generic_coefficient.hpp
                  spe10.hpp
    LIBRARIES mfem)

  # Add the corresponding tests to the "test" target
  if (MFEM_ENABLE_TESTING)
    add_test(NAME darcy_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:darcy> -m ../../data/fichera-q2.mesh
      ${MPIEXEC_POSTFLAGS})

    add_test(NAME grad_div_np${MFEM_MPI_NP}
      COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MFEM_MPI_NP}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:grad_div> -sp -ams -lor -hb -m ../../data/fichera-q2.mesh -rp 0
      ${MPIEXEC_POSTFLAGS})
  endif()
endif()
