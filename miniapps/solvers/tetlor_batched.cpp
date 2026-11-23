// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "tetlor.hpp"
#include "general/scan.hpp"
#include "linalg/kernels.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>

namespace tetlor
{
namespace
{

// Only the coarse-entity DOF layout is needed for conformity, boundary DOFs
// and MPI prolongation. Integration uses linear elements on the reference patch.
class PatchCollection : public FiniteElementCollection
{
   int counts[Geometry::NumGeom] = {};
   Array<int> identity;

public:
   PatchCollection(int q, const std::vector<BarycentricLabel> &labels)
      : FiniteElementCollection(q)
   {
      int per_entity[16] = {};
      for (const auto &label : labels)
      {
         int mask = 0;
         for (int i = 0; i < 4; ++i) { if (label[i]) { mask |= 1 << i; } }
         ++per_entity[mask];
      }
      counts[Geometry::POINT] = per_entity[1];
      counts[Geometry::SEGMENT] = per_entity[3];
      counts[Geometry::TRIANGLE] = per_entity[7];
      counts[Geometry::TETRAHEDRON] = per_entity[15];
      identity.SetSize(*std::max_element(per_entity, per_entity+16));
      std::iota(identity.begin(), identity.end(), 0);
   }
   int GetContType() const override { return CONTINUOUS; }
   int DofForGeometry(Geometry::Type g) const override { return counts[g]; }
   // Interior nodes of each entity are already in canonical vertex order.
   const int *DofOrderForOrientation(Geometry::Type, int) const override
   { return identity.GetData(); }
   const FiniteElement *FiniteElementForGeometry(Geometry::Type) const override
   { return nullptr; }
};

// Map reference nodes to the vertex/edge/face/interior blocks returned by
// GetElementDofs. Within each block, sort the exact barycentric labels in
// increasing coarse vertex ID order. Adjacent patches then agree on the trace.
Array<int> NodePermutation(const std::vector<BarycentricLabel> &labels,
                           const int *native_to_patch,
                           const std::array<int,4> &sorted)
{
   using Tet = Geometry::Constants<Geometry::TETRAHEDRON>;
   int entity[16] = {};
   for (int i = 0; i < 4; ++i) { entity[1 << i] = i; }
   for (int i = 0; i < 6; ++i)
   {
      entity[(1 << Tet::Edges[i][0]) | (1 << Tet::Edges[i][1])] = 4+i;
   }
   for (int i = 0; i < 4; ++i)
   {
      const int *f = Tet::FaceVert[i];
      entity[(1 << f[0]) | (1 << f[1]) | (1 << f[2])] = 10+i;
   }
   entity[15] = 14;

   const int n = labels.size();
   std::vector<std::pair<int,BarycentricLabel>> keys(n);
   for (int j = 0; j < n; ++j)
   {
      int mask = 0;
      for (int i = 0; i < 4; ++i)
      {
         const auto weight = labels[j][native_to_patch[sorted[i]]];
         keys[j].second[i] = weight;
         if (weight) { mask |= 1 << sorted[i]; }
      }
      keys[j].first = entity[mask];
   }
   Array<int> order(n), inverse(n);
   std::iota(order.begin(), order.end(), 0);
   std::sort(order.begin(), order.end(),
   [&](int i, int j) { return keys[i] < keys[j]; });
   for (int i = 0; i < n; ++i) { inverse[order[i]] = i; }
   return inverse;
}

// Tabulate the six independent diffusion tensors on just one patch. Using
// MFEM's child integrator preserves its quadrature on Duffy wedges, pyramids
// and hexes as well as Isaac tetrahedra.
void ReferenceStencil(Mesh &patch, Array<int> &rows, Array<int> &columns,
                      DenseMatrix &stencil)
{
   H1_FECollection linear(1);
   FiniteElementSpace low(&patch, &linear);
   std::vector<std::map<int,std::array<real_t,6>>> graph(patch.GetNV());
   DenseMatrix unit(3), child;
   Array<int> nodes;
   int k = 0;
   for (int a = 0; a < 3; ++a)
   {
      for (int b = a; b < 3; ++b, ++k)
      {
         unit = 0.0;
         unit(a,b) = unit(b,a) = 1.0;
         MatrixConstantCoefficient coefficient(unit);
         DiffusionIntegrator diffusion(coefficient);
         for (int s = 0; s < patch.GetNE(); ++s)
         {
            low.GetElementDofs(s, nodes);
            diffusion.AssembleElementMatrix(*low.GetFE(s),
                                            *low.GetElementTransformation(s), child);
            for (int i = 0; i < nodes.Size(); ++i)
            {
               for (int j = 0; j < nodes.Size(); ++j)
               {
                  graph[nodes[i]][nodes[j]][k] += child(i,j);
               }
            }
         }
      }
   }
   rows.SetSize(patch.GetNV()+1);
   rows[0] = 0;
   for (int i = 0; i < patch.GetNV(); ++i)
   {
      rows[i+1] = rows[i]+graph[i].size();
   }
   columns.SetSize(rows.Last());
   stencil.SetSize(6, columns.Size());
   for (int i = 0; i < patch.GetNV(); ++i)
   {
      int j = rows[i];
      for (const auto &entry : graph[i])
      {
         columns[j] = entry.first;
         std::copy_n(entry.second.data(), 6, stencil.GetColumn(j++));
      }
   }
}

void ComputeMetrics(const DenseMatrix &jacobians, DenseMatrix &metrics,
                    Vector &volumes)
{
   // Upper triangle of |det J| J^{-1} J^{-T}, in the ReferenceStencil order.
   const auto J = jacobians.Read();
   metrics.SetSize(6, jacobians.Width());
   volumes.SetSize(jacobians.Width());
   auto volume = volumes.Write();
   auto G = metrics.Write();
   mfem::forall(jacobians.Width(), [=] MFEM_HOST_DEVICE (int e)
   {
      real_t inv[9];
      kernels::CalcInverse<3>(J+9*e, inv);
      const real_t det = std::abs(kernels::Det<3>(J+9*e));
      volume[e] = det/6;
      int k = 0;
      for (int a = 0; a < 3; ++a)
      {
         for (int b = a; b < 3; ++b)
         {
            G[6*e+k++] = det*(inv[a]*inv[b] + inv[a+3]*inv[b+3] + inv[a+6]*inv[b+6]);
         }
      }
   });
}

MFEM_HOST_DEVICE inline real_t Contract(const real_t *g, const real_t *s)
{
   real_t value = 0;
   for (int i = 0; i < 6; ++i) { value += g[i]*s[i]; }
   return value;
}

} // namespace

BatchedLOR::BatchedLOR(Mesh &mesh, int q, const std::string &grid_type)
{
   const int ne = mesh.GetNE();
   MFEM_VERIFY(mesh.Conforming() && mesh.Dimension() == 3 &&
               mesh.SpaceDimension() == 3 && ne > 0 && mesh.GetNumGeometries(3) == 1 &&
               mesh.GetElementGeometry(0) == Geometry::TETRAHEDRON,
               "Expected a conforming affine tetrahedral mesh.");
   MFEM_VERIFY(mesh.HasBoundaryElements(), "Expected a Dirichlet boundary.");
   MFEM_VERIFY(mesh.CheckElementOrientation(false) == 0,
               "Inverted coarse tetrahedron.");
   MFEM_VERIFY(q > 0 && ne <= std::numeric_limits<int>::max()/q/q/q,
               "Invalid LOR order or patch count.");
   patch = MakeReferenceLORMesh(q, grid_type, labels);
   const int nl = patch.GetNV();
   MFEM_VERIFY(ne <= std::numeric_limits<int>::max()/nl, "Too many patch nodes.");
   collection = std::make_unique<PatchCollection>(q, labels);
#ifdef MFEM_USE_MPI
   if (auto *pm = dynamic_cast<ParMesh*>(&mesh))
   {
      space = std::make_unique<ParFiniteElementSpace>(pm, collection.get());
   }
   else
#endif
   {
      space = std::make_unique<FiniteElementSpace>(&mesh, collection.get());
   }

   dofs.SetSize(ne*nl);
   vertex_permutations.SetSize(ne, 4);
   DenseMatrix jacobians(9, ne);
   std::map<std::array<int,4>,Array<int>> permutations;
   Array<int> vertices, element_dofs;
   for (int e = 0; e < ne; ++e)
   {
      MFEM_VERIFY(mesh.GetElementTransformation(e)->OrderJ() == 0,
                  "Expected affine tetrahedra.");
      const int *native = mesh.GetElement(e)->GetVertices();
      // ParMesh partitioning preserves the relative serial vertex ordering.
      GetPatchVertices(mesh, e, grid_type == "duffy", vertices);
      int *native_to_patch = vertex_permutations[e];
      for (int i = 0; i < 4; ++i) { native_to_patch[i] = vertices.Find(native[i]); }
      std::array<int,4> sorted = {0,1,2,3};
      std::sort(sorted.begin(), sorted.end(),
      [&](int i, int j) { return native[i] < native[j]; });
      auto &perm = permutations[sorted];
      if (!perm.Size()) { perm = NodePermutation(labels, native_to_patch, sorted); }
      space->GetElementDofs(e, element_dofs);
      MFEM_VERIFY(element_dofs.Size() == nl, "Inconsistent patch node layout.");
      for (int j = 0; j < nl; ++j) { dofs[e*nl+j] = element_dofs[perm[j]]; }
      for (int j = 0; j < 3; ++j)
      {
         for (int i = 0; i < 3; ++i)
         {
            jacobians(i+3*j,e) = mesh.GetVertex(vertices[j+1])[i] -
                                 mesh.GetVertex(vertices[0])[i];
         }
      }
   }
   ComputeMetrics(jacobians, metrics, volumes);
   ReferenceStencil(patch, rows, columns, stencil);
}

void BatchedLOR::ApplyPatchStiffness(int e, const DenseMatrix &Q,
                                     DenseMatrix &QtK) const
{
   const real_t identity[6] = {1,0,0,1,0,1};
   ApplyPatchStiffness(e < 0 ? identity : metrics.HostRead()+6*e,Q,QtK);
}

void BatchedLOR::ApplyPatchStiffness(const real_t g[6], const DenseMatrix &Q,
                                     DenseMatrix &QtK) const
{
   const auto I = rows.HostRead(), J = columns.HostRead();
   const auto S = stencil.HostRead();
   QtK.SetSize(Q.Width(), Q.Height());
   QtK = 0.0;
   for (int i = 0; i < Q.Height(); ++i)
   {
      for (int k = I[i]; k < I[i+1]; ++k)
      {
         const real_t value = Contract(g, S+6*k);
         for (int j = 0; j < Q.Width(); ++j) { QtK(j,i) += value*Q(J[k],j); }
      }
   }
}

void BatchedLOR::GetPatchStiffness(int e, DenseMatrix &K) const
{
   GetPatchStiffness(metrics.HostRead() + 6*e, K);
}

void BatchedLOR::GetPatchStiffness(const real_t g[6], DenseMatrix &K) const
{
   const auto I = rows.HostRead(), J = columns.HostRead();
   const auto S = stencil.HostRead();
   K.SetSize(patch.GetNV());
   K = 0.0;
   for (int i = 0; i < K.Height(); ++i)
   {
      for (int k = I[i]; k < I[i+1]; ++k) { K(i,J[k]) = Contract(g, S+6*k); }
   }
}

void BatchedLOR::AssembleLocal(SparseMatrix &matrix) const
{
   const int n = space->GetVSize(), nl = patch.GetNV(), ne = space->GetNE();
   MFEM_VERIFY(ne <= std::numeric_limits<int>::max()/columns.Size(),
               "Local patch stencil entries exceed the CSR index range.");
   // Gather all patch rows touching each local DOF. Their combined lengths
   // bound the CSR row before merging columns shared by adjacent patches.
   Table incidence;
   Transpose(dofs, incidence, n);
   const auto owners = incidence.ReadI(), occurrences = incidence.ReadJ();
   const auto I = rows.Read(), J = columns.Read(), nodes = dofs.Read();
   const auto S = stencil.Read(), G = metrics.Read();
   Array<int> offsets(n+1);
   auto offset = offsets.Write();
   mfem::forall(n+1, [=] MFEM_HOST_DEVICE (int i)
   {
      int count = 0;
      if (i < n)
      {
         for (int t = owners[i]; t < owners[i+1]; ++t)
         {
            const int r = occurrences[t]%nl;
            count += I[r+1]-I[r];
         }
      }
      offset[i] = count;
   });
   ExclusiveScan(true, offset, offset, n+1, 0);

   const int capacity = ne*columns.Size();
   Array<int> work_columns(capacity);
   Vector work_values(capacity);
   auto cols = work_columns.Write();
   auto vals = work_values.Write();
   matrix.OverrideSize(n,n);
   matrix.GetMemoryI().New(n+1);
   auto row = matrix.WriteI();
   mfem::forall(n+1, [=] MFEM_HOST_DEVICE (int i)
   {
      if (i == n) { row[i] = 0; return; }
      const int begin = offset[i];
      int end = begin;
      const bool shared = owners[i+1]-owners[i] > 1;
      for (int t = owners[i]; t < owners[i+1]; ++t)
      {
         const int e = occurrences[t]/nl, r = occurrences[t]%nl;
         for (int j = I[r]; j < I[r+1]; ++j)
         {
            const int col = nodes[e*nl+J[j]];
            const real_t value = Contract(G+6*e, S+6*j);
            int k = end;
            if (shared)
            {
               for (k = begin; k < end && cols[k] != col; ++k) { }
            }
            if (k == end) { cols[end] = col; vals[end++] = value; }
            else { vals[k] += value; }
         }
      }
      row[i] = end-begin;
   });
   ExclusiveScan(true, row, row, n+1, 0);
   Array<int> last;
   last.MakeRef(matrix.GetMemoryI(), n, 1);
   const int nnz = last.HostRead()[0];
   matrix.GetMemoryJ().New(nnz);
   matrix.GetMemoryData().New(nnz);
   auto out_col = matrix.WriteJ();
   auto out_val = matrix.WriteData();
   mfem::forall(n, [=] MFEM_HOST_DEVICE (int i)
   {
      for (int k = row[i]; k < row[i+1]; ++k)
      {
         const int j = offset[i]+(k-row[i]);
         out_col[k] = cols[j];
         out_val[k] = vals[j];
      }
   });
}

void BatchedLOR::FormSystemMatrix(const Array<int> &ess_dofs,
                                  OperatorHandle &A) const
{
   auto local = std::make_unique<SparseMatrix>();
   AssembleLocal(*local);
#ifdef MFEM_USE_MPI
   if (auto *ps = dynamic_cast<ParFiniteElementSpace*>(space.get()))
   {
      ParBilinearForm(ps).ParallelRAP(*local, A, true);
      A.As<HypreParMatrix>()->EliminateBC(ess_dofs, Operator::DIAG_ONE);
      return;
   }
#endif
   local->EliminateBC(ess_dofs, Operator::DIAG_ONE);
   A.Reset(local.release());
}

} // namespace tetlor
