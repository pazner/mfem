// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MG_AGGLOM_HPP
#define MFEM_MG_AGGLOM_HPP

#include "mfem.hpp"
#include <vector>

#ifndef MFEM_USE_MPI
#error "mg_agglom requires MFEM built with MPI (MFEM_USE_MPI=YES)"
#endif

namespace mfem
{

/** Rank-local agglomeration hierarchy.

    Each MPI rank agglomerates only its own elements, so no macro element ever
    crosses a processor boundary. All indices in the returned structure are
    rank-local.

    E[i] describes level i (ordered coarsest to finest). For a macro element j
    on level i-1, E[i][j] lists the indices of the level-i entities it contains.
    On the finest level, E[num_levels-1][j] lists the (rank-local) mesh element
    indices of aggregate j. Every rank returns exactly num_levels levels. */
std::vector<std::vector<std::vector<int>>> Agglomerate(ParMesh &mesh,
                                                       int ncoarse,
                                                       int num_levels);

class AgglomerationMultigrid : public Multigrid
{
   friend class TruncatedMultigrid;
public:
   // NOTE: As in the serial version, this constructor is declared but its
   // implementation is not provided (it was commented out in the original).
   AgglomerationMultigrid(ParFiniteElementSpace &fes, HypreParMatrix &Af,
                          int ncoarse, int num_levels, int smoother_choice,
                          bool paraview_vis);

   const HypreParMatrix &GetFinestProlongation() const;
};

class TruncatedMultigrid : public Multigrid
{
public:
   TruncatedMultigrid(const AgglomerationMultigrid &other);
};

class SmoothedAggregationGMG : public Multigrid
{
   /// hypre column partitionings of the coarse levels. Kept alive for the
   /// lifetime of the hierarchy, since (depending on the hypre version) the
   /// matrices may reference these arrays rather than copy them. Sized once in
   /// the constructor and never resized, so the data pointers stay valid.
   std::vector<Array<HYPRE_BigInt>> coarse_starts;

public:
   enum class SmootherType { BlockJacobi, BlockGaussSeidel, BlockILU };
   SmoothedAggregationGMG(ParFiniteElementSpace &fes, HypreParMatrix &Af,
                          int ncoarse, int num_levels, bool paraview_vis, SmootherType smoother_type);
};

} // namespace mfem

#endif
