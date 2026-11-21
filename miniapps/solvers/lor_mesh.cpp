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

#include "lor_mesh.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

namespace tetlor
{
using namespace mfem;

// Isaac's recursive simplex nodes (doi:10.1137/20M1321802), starting from
// Gauss-Lobatto points on a segment. Restriction to a face uses the same rule.
static void IsaacNode(const int *alpha, int n, real_t *b)
{
   const int q = std::accumulate(alpha, alpha+n, 0);
   const real_t *gll = poly1d.ClosedPoints(q);
   if (n == 2)
   {
      b[0] = gll[alpha[0]];
      b[1] = gll[alpha[1]];
      return;
   }
   std::fill_n(b, n, 0.0);
   real_t sum = 0.0;
   for (int i = 0; i < n; i++)
   {
      if (alpha[i] == q) { continue; }
      int face[3];
      real_t x[3], weight = gll[q-alpha[i]];
      for (int j = 0; j < n-1; j++) { face[j] = alpha[j+(j>=i)]; }
      IsaacNode(face, n-1, x);
      for (int j = 0; j < n-1; j++) { b[j+(j>=i)] += weight*x[j]; }
      sum += weight;
   }
   for (int i = 0; i < n; i++) { b[i] /= sum; }
}

// Both refinements have q^3 children per macroelement. The low-order space
// is ordinary H1 of order one on this mesh, including on the collapsed cells.
Mesh MakeLORMesh(const Mesh &mesh, int q, const std::string &grid_type)
{
   MFEM_VERIFY(mesh.Dimension() == 3 && mesh.SpaceDimension() == 3 &&
               mesh.Conforming() && mesh.GetNE() && mesh.GetNumGeometries(3) == 1 &&
               mesh.GetElementGeometry(0) == Geometry::TETRAHEDRON,
               "Expected a conforming tetrahedral mesh.");
   IsoparametricTransformation T;
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementTransformation(e, &T);
      MFEM_VERIFY(T.OrderJ() == 0, "Expected affine tetrahedra.");
   }
   MFEM_VERIFY(mesh.GetNBE(), "Expected exterior Dirichlet boundary elements.");
   MFEM_VERIFY(q >= 1, "Expected positive LOR order.");
   MFEM_VERIFY(grid_type == "isaac" || grid_type == "duffy",
               "Expected isaac or duffy.");
   MFEM_VERIFY(mesh.GetNE() <= std::numeric_limits<int>::max()/q/q/q,
               "The LOR mesh exceeds the MFEM element index range.");
   const bool duffy = grid_type == "duffy";
   GeometryRefiner refiner;
   auto &rg = *refiner.Refine(duffy ? Geometry::CUBE : Geometry::TETRAHEDRON, q);
   const int n = rg.RefPts.GetNPoints();
   std::vector<std::array<real_t,4>> bary(n);
   std::vector<std::array<long long,4>> label(n);
   const real_t *gll = poly1d.ClosedPoints(q);
   for (int l = 0; l < n; l++)
   {
      const auto &ip = rg.RefPts.IntPoint(l);
      const int i = std::lround(q*ip.x), j = std::lround(q*ip.y);
      const int k = std::lround(q*ip.z);
      if (duffy)
      {
         const real_t a = gll[i], b = gll[j], c = gll[k];
         bary[l] = {{(1-a)*(1-b)*(1-c), a*(1-b)*(1-c), b*(1-c), c}};
         label[l] = {{
               1LL*(q-i)*(q-j)*(q-k), 1LL*i*(q-j)*(q-k),
               1LL*q*j*(q-k), 1LL*q*q*k
            }
         };
      }
      else
      {
         // Freudenthal chains in (lambda_3, lambda_2+lambda_3, 1-lambda_0).
         const int alpha[] = {q-i-j-k,i,k,j};
         IsaacNode(alpha, 4, bary[l].data());
         std::copy_n(alpha, 4, label[l].begin());
      }
   }

   Mesh mesh_lor(3, 0, mesh.GetNE()*q*q*q);
   // Integer barycentric labels on the uniform lattice identify shared nodes
   // exactly, including the coincident nodes on collapsed tensor fibers.
   using Key = std::array<std::pair<int,long long>,4>;
   std::map<Key,int> vertex;
   Array<int> v, ids(n);
   for (int e = 0; e < mesh.GetNE(); e++)
   {
      mesh.GetElementVertices(e, v);
      if (duffy)
      {
         // A global vertex order gives the same collapse apex on every shared
         // face. Reversing the first edge preserves the grid and fixes parity.
         int inversions = 0;
         for (int i = 0; i < 4; i++)
         {
            for (int j = 0; j < i; j++) { inversions += v[j] > v[i]; }
         }
         v.Sort();
         if (inversions%2) { std::swap(v[0], v[1]); }
      }
      for (int l = 0; l < n; l++)
      {
         Key key{};
         for (int d = 0; d < 4; d++)
         {
            if (label[l][d]) { key[d] = {v[d],label[l][d]}; }
         }
         std::sort(key.begin(), key.end());
         auto entry = vertex.emplace(key, mesh_lor.GetNV());
         ids[l] = entry.first->second;
         if (entry.second)
         {
            real_t x[3] = {};
            for (int d = 0; d < 4; d++)
            {
               for (int s = 0; s < 3; s++) { x[s] += bary[l][d]*mesh.GetVertex(v[d])[s]; }
            }
            mesh_lor.AddVertex(x);
         }
      }

      const int nv = duffy ? 8 : 4;
      for (int l = 0; l < rg.RefGeoms.Size(); l += nv)
      {
         int t[8];
         for (int d = 0; d < nv; d++) { t[d] = ids[rg.RefGeoms[l+d]]; }
         if (!duffy) { mesh_lor.AddTet(t[1], t[0], t[2], t[3]); }
         else if (t[2] == t[3] && t[4] == t[5]) { mesh_lor.AddTet(t[0],t[1],t[2],t[4]); }
         else if (t[4] == t[5]) { mesh_lor.AddPyramid(t); }
         else if (t[2] == t[3]) { mesh_lor.AddWedge(t[0],t[1],t[2],t[4],t[5],t[6]); }
         else { mesh_lor.AddHex(t); }
      }
   }
   mesh_lor.FinalizeMesh(0, false);
   MFEM_VERIFY(mesh_lor.CheckElementOrientation(false) == 0,
               "Inverted LOR element.");
   MFEM_VERIFY(mesh_lor.GetNBE() == mesh.GetNBE()*q*q, "Incompatible LOR faces.");
   return mesh_lor;
}

} // namespace tetlor
