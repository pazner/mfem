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

#ifndef MG_HPP
#define MG_HPP

#include "mfem.hpp"
#include "image_mesh.hpp"
#include "voxel_mesh.hpp"

namespace mfem
{

struct ParentIndex
{
   int element_index;
   int pmat_index;
};

class VoxelProlongation : public Operator
{
   const FiniteElementSpace &coarse_fes;
   const FiniteElementSpace &fine_fes;
   Array<ParentIndex> parents;
   Array<int> parent_offsets;
   const Array<int> &coarse_ess_dofs;
   const Array<int> &fine_ess_dofs;
   DenseTensor local_P;
   DenseTensor local_R;
public:
   VoxelProlongation(
      const FiniteElementSpace &coarse_fes_,
      const Array<int> &coarse_ess_dofs_,
      const FiniteElementSpace &fine_fes_,
      const Array<int> &fine_ess_dofs_);

   void Mult(const Vector &u_coarse, Vector &u_fine) const override;
   void MultTranspose(const Vector &u_fine, Vector &u_coarse) const override;
   void Coarsen(const Vector &u_fine, Vector &u_coarse) const;
};

template <typename T> using vec_unique_ptr = std::vector<std::unique_ptr<T>>;

class VoxelMultigrid : public MultigridBase
{
public:
   vec_unique_ptr<VoxelMesh> meshes;
   vec_unique_ptr<FiniteElementSpace> spaces;
   vec_unique_ptr<BilinearForm> forms;
   vec_unique_ptr<Array<int>> ess_dofs;
   vec_unique_ptr<VoxelProlongation> prolongations;
   int nlevels;

   virtual const Operator* GetProlongationAtLevel(int level) const override
   {
      return prolongations[level].get();
   }

public:
   VoxelMultigrid(const VoxelMesh &&fine_mesh, FiniteElementCollection &fec);

   FiniteElementSpace &GetFineSpace() { return *spaces.back(); }
   Operator &GetFineOperator() { return *operators.Last(); }
   BilinearForm &GetFineForm() { return *forms.back(); }
   void FormFineLinearSystem(
      Vector& x, Vector& b, OperatorHandle& A, Vector& X, Vector& B);
};

class ImageProlongation : public Operator
{
   const FiniteElementSpace &coarse_fes;
   const FiniteElementSpace &fine_fes;
   Array<ParentIndex> parents;
   Array<int> parent_offsets;
   const Array<int> &coarse_ess_dofs;
   const Array<int> &fine_ess_dofs;
   DenseTensor local_P;
   DenseTensor local_R;
public:
   ImageProlongation(
      const FiniteElementSpace &coarse_fes_,
      const Array<int> &coarse_ess_dofs_,
      const FiniteElementSpace &fine_fes_,
      const Array<int> &fine_ess_dofs_);

   void Mult(const Vector &u_coarse, Vector &u_fine) const override;

   void MultTranspose(const Vector &u_fine, Vector &u_coarse) const override;

   void Coarsen(const Vector &u_fine, Vector &u_coarse) const;
};

class ImageMultigrid : public MultigridBase
{
public:
   std::vector<std::unique_ptr<ImageMesh>> meshes;
   std::vector<std::unique_ptr<FiniteElementSpace>> spaces;
   std::vector<std::unique_ptr<BilinearForm>> forms;
   std::vector<std::unique_ptr<Array<int>>> ess_dofs;
   std::vector<std::unique_ptr<ImageProlongation>> prolongations;
   int nlevels;

   virtual const Operator* GetProlongationAtLevel(int level) const override
   {
      return prolongations[level].get();
   }

public:
   ImageMultigrid(const ImageMesh &&fine_mesh, FiniteElementCollection &fec);

   FiniteElementSpace &GetFineSpace() { return *spaces.back(); }
   Operator &GetFineOperator() { return *operators.Last(); }
   BilinearForm &GetFineForm() { return *forms.back(); }
   void FormFineLinearSystem(
      Vector& x, Vector& b, OperatorHandle& A, Vector& X, Vector& B);
};

}

#endif
