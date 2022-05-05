// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "discrete_divergence.hpp"
#include "general/forall.hpp"

namespace mfem
{

void DiscreteDivergence::FormDivergenceMatrix(OperatorHandle &D_op)
{
   const Mesh &mesh = *fes_rt.GetMesh();
   const int dim = mesh.Dimension();
   const int order = fes_rt.GetMaxElementOrder();

   const int n_rt = fes_rt.GetTrueVSize();
   const int n_l2 = fes_l2.GetTrueVSize();

   SparseMatrix D;
   D.OverrideSize(n_l2, n_rt);

   D.GetMemoryI().New(n_l2 + 1);
   // Each row always has two nonzeros
   const int nnz = n_l2*2*dim;
   auto I = D.HostWriteI();
   for (int i = 0; i < n_l2 + 1; ++i)
   {
      I[i] = 2*dim*i;
   }

   // element2face is a mapping of size (2*dim, nvol_per_el) such that with a
   // macro element, subelement i (in lexicographic ordering) has faces (also
   // in lexicographic order) given by the entries (j, i).

   const int o = order;
   const int op1 = order + 1;

   Array<int> element2face;
   element2face.SetSize(2*dim*pow(o, dim));

   MFEM_VERIFY(dim == 2, "Not yet 3D.");

   for (int iy = 0; iy < o; ++iy)
   {
      for (int ix = 0; ix < o; ++ix)
      {
         int ivol = ix + iy*o;
         element2face[0 + 4*ivol] = -1 - (ix + iy*op1);
         element2face[1 + 4*ivol] = ix+1 + iy*op1;
         element2face[2 + 4*ivol] = -1 - (ix + iy*o + o*op1);
         element2face[3 + 4*ivol] = ix + (iy+1)*o + o*op1;
      }
   }

   const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   const auto *R_rt = dynamic_cast<const ElementRestriction*>(
                         fes_rt.GetElementRestriction(ordering));

   const int nel_ho = mesh.GetNE();
   const int nface_per_el = dim*order*(order+1);
   const int nvol_per_el = pow(order, dim);

   const auto gather_rt = Reshape(R_rt->gather_map.Read(), nface_per_el, nel_ho);

   const auto e2f = Reshape(element2face.Read(), 2*dim, nvol_per_el);

   // Fill J and data
   D.GetMemoryJ().New(nnz);
   D.GetMemoryData().New(nnz);

   auto J = D.HostWriteJ();
   auto V = D.HostWriteData();

   // Loop over L2 DOFs
   for (int i = 0; i < n_l2; ++i)
   {
      const int i_loc = i%nvol_per_el;
      const int i_el = i/nvol_per_el;

      for (int k = 0; k < 2*dim; ++k)
      {
         const int sjv_loc = e2f(k, i_loc);
         const int jv_loc = (sjv_loc >= 0) ? sjv_loc : -1 - sjv_loc;
         const int sgn1 = (sjv_loc >= 0) ? 1 : -1;
         MFEM_VERIFY(k < nface_per_el, "");
         const int sj = gather_rt(jv_loc, i_el);
         const int j = (sj >= 0) ? sj : -1 - sj;
         MFEM_VERIFY(j >= 0 && j < n_rt, "");
         const int sgn2 = (sj >= 0) ? 1 : -1;

         J[k + 2*dim*i] = j;
         V[k + 2*dim*i] = sgn1*sgn2;
      }
   }
}

DiscreteDivergence::DiscreteDivergence(
   FiniteElementSpace &fes_rt_, FiniteElementSpace &fes_l2_)
   : fes_rt(fes_rt_), fes_l2(fes_l2_)
{
   FormDivergenceMatrix();
}

} // namespace mfem
