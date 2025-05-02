// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef PAR_MG_HPP
#define PAR_MG_HPP

#include "mfem.hpp"
#include "voxel_mesh.hpp"

namespace mfem
{

struct CoarseToFineCommunication
{
   struct CoarseToFineIndex
   {
      int coarse_element_index;
      int pmat_index;
   };
   int rank; // rank owning the fine elements
   std::vector<CoarseToFineIndex> coarse_to_fine;

   CoarseToFineCommunication() = default;
   CoarseToFineCommunication(int rank_) : rank(rank_) { }
};

struct FineToCoarseCommunication
{
   struct FineToCoarseIndex
   {
      int fine_element_index;
      int pmat_index;
   };
   int rank; // rank owning the coarse elements
   std::vector<FineToCoarseIndex> fine_to_coarse;

   FineToCoarseCommunication() = default;
   FineToCoarseCommunication(int rank_) : rank(rank_) { }
};

struct ParVoxelMapping
{
   Array<ParentIndex> local_parents;
   Array<int> local_parent_offsets;

   std::vector<CoarseToFineCommunication> coarse_to_fine;
   std::vector<FineToCoarseCommunication> fine_to_coarse;
};

std::vector<ParVoxelMapping> CreateParVoxelMappings(
   const int nranks,
   const int dim,
   const Array<ParentIndex> &parents,
   const Array<int> &parent_offsets,
   const Array<int> &fine_partitioning,
   const Array<int> &coarse_partitioning);

class ParVoxelProlongation : public Operator
{
   const ParFiniteElementSpace &coarse_fes;
   const ParFiniteElementSpace &fine_fes;
   const Array<int> &coarse_ess_dofs;
   const Array<int> &fine_ess_dofs;
   const ParVoxelMapping &mapping;
   DenseTensor local_P;
   DenseTensor local_R;

   int ndof_per_el;

   mutable std::vector<std::vector<double>> c2f_buffers;
   mutable std::vector<std::vector<double>> f2c_buffers;

   mutable Vector u_coarse_lvec;
   mutable Vector u_fine_lvec;

public:
   ParVoxelProlongation(
      const ParFiniteElementSpace &coarse_fes_,
      const Array<int> &coarse_ess_dofs_,
      const ParFiniteElementSpace &fine_fes_,
      const Array<int> &fine_ess_dofs_,
      const ParVoxelMapping &mapping_);

   void Mult(const Vector &u_coarse, Vector &u_fine) const override;
   void MultTranspose(const Vector &u_fine, Vector &u_coarse) const override;
   void Coarsen(const Vector &u_fine, Vector &u_coarse) const;
};

template <typename T> using vec_unique_ptr = std::vector<std::unique_ptr<T>>;

enum class ProblemType
{
   Poisson,
   Elasticity,
   VectorPoisson
};

class ParVoxelMultigrid : public MultigridBase
{
private:
   vec_unique_ptr<ParMesh> meshes;
   vec_unique_ptr<ParFiniteElementSpace> spaces;
   vec_unique_ptr<ParBilinearForm> forms;
   vec_unique_ptr<Array<int>> ess_dofs;
   vec_unique_ptr<ParVoxelProlongation> prolongations;
   vec_unique_ptr<ParVoxelMapping> mappings;
   std::unique_ptr<H1_FECollection> fec;

   ConstantCoefficient lambda{3846.0};
   ConstantCoefficient mu{5769.0};

   IntegrationRules irs{0, Quadrature1D::GaussLobatto};

   virtual const Operator* GetProlongationAtLevel(int level) const override
   {
      return prolongations[level].get();
   }

public:
   ParVoxelMultigrid(const std::string &dir, int order = 1,
                     ProblemType pt = ProblemType::Poisson,
                     const std::vector<int> &ess_bdr_attrs = {});

   ParFiniteElementSpace &GetFineSpace() { return *spaces.back(); }
   Operator &GetFineOperator() { return *operators.Last(); }
   ParBilinearForm &GetFineForm() { return *forms.back(); }
   Array<int> &GetFineEssDofs() { return *ess_dofs.back(); }
   void FormFineLinearSystem(
      Vector& x, Vector& b, OperatorHandle& A, Vector& X, Vector& B);
};

}

#endif
