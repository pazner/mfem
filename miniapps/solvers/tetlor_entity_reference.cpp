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

#include "tetlor_entity.hpp"
#include <algorithm>
#include <cmath>

namespace tetlor
{
namespace entity
{
namespace
{

// A four-bit mask names a reference entity. List each open entity in
// barycentric order and its boundary in the same relative subentity order
// on every copy. This also handles the asymmetric Duffy face layout.
struct Layout
{
   Array<int> open[16], boundary[16];

   explicit Layout(const std::vector<BarycentricLabel> &labels)
   {
      for (int i = 0; i < int(labels.size()); ++i)
      {
         int mask = 0;
         for (int v = 0; v < 4; ++v) { if (labels[i][v]) { mask |= 1 << v; } }
         open[mask].Append(i);
      }
      for (int e = 1; e < 16; ++e)
      {
         auto &ids = open[e];
         if (ids.Size())
         {
            std::sort(ids.begin(),ids.end(),
            [&](int i, int j) { return labels[i] < labels[j]; });
         }
         for (int f = 1; f < e; ++f)
         { if ((f & e) == f) { boundary[e].Append(open[f]); } }
      }
   }

   // DOFs supported on entities containing e, restricted to the skeleton.
   Array<int> Incident(int e) const
   {
      Array<int> ids;
      for (int f = 1; f < 15; ++f)
      { if ((f & e) == e && f != e) { ids.Append(open[f]); } }
      return ids;
   }
};

Array<int> Join(Array<int> a, const Array<int> &b)
{
   a.Append(b);
   return a;
}

// Minimum-energy values on free for prescribed values on fixed. All other
// entries vanish. The columns give the individual extension profiles.
DenseMatrix Lift(const DenseMatrix &S, const Array<int> &free,
                 const Array<int> &fixed)
{
   DenseMatrix A, X;
   S.GetSubMatrix(free,fixed,X);
   if (free.Size())
   {
      S.GetSubMatrix(free,A);
      X *= -1.0;
      DenseMatrixInverse(A,true).Mult(X);
   }
   return X;
}

struct TraceRule
{
   DenseMatrix energy, lift;

   TraceRule(const Layout &layout, int e)
      : energy(layout.open[e].Size()), lift(energy.Height(),layout.boundary[e].Size())
   { energy = 0.0; lift = 0.0; }
};

constexpr int edges[] = {3,5,6,9,10,12};
constexpr int faces[] = {7,11,13,14};

// Add the trace of an extension, in common row/column order on every copy.
void AddProfile(const DenseMatrix &profile, const Array<int> &free,
                const Layout &layout, int to, int from, real_t weight,
                DenseMatrix &lift)
{
   for (int j = 0; j < layout.open[from].Size(); ++j)
   {
      const int col = layout.boundary[to].Find(layout.open[from][j]);
      for (int i = 0; i < layout.open[to].Size(); ++i)
      { lift(i,col) += weight*profile(free.Find(layout.open[to][i]),j); }
   }
}

// Constrain vertex extensions to vanish on the opposite face. For an open
// edge, both faces missing an endpoint have zero trace. Average the aligned
// reference contributions to get just one rule for each entity dimension.
void IncidentRules(const Layout &d, const DenseMatrix &S,
                   TraceRule &edge, TraceRule &face)
{
   for (int v : {1,2,4,8})
   {
      auto free = d.Incident(v);
      DenseMatrix profile = Lift(S,free,d.open[v]);
      for (int e : edges)
      {
         if (e & v) { AddProfile(profile,free,d,e,v,1.0/6,edge.lift); }
      }
      for (int f : faces)
      {
         if (f & v) { AddProfile(profile,free,d,f,v,1.0/4,face.lift); }
      }
   }

   for (int e : edges)
   {
      const auto free = d.Incident(e);
      DenseMatrix profile = Lift(S,free,d.open[e]);
      edge.energy.Add(1.0/6,Condense(S,d.open[e],free));
      for (int f : faces)
      {
         if ((e & f) != e) { continue; }
         AddProfile(profile,free,d,f,e,1.0/4,face.lift);
         // For the open-face energy hold the edge fixed and minimize over
         // the other incident face.
         const int other = e | (15 ^ f);
         face.energy.Add(1.0/12,Condense(S,d.open[f],d.open[other]));
      }
   }

   // Edge profiles act on the increment left after the vertex lift.
   for (int e : {3,5,6})
   {
      for (int v : {1,2,4})
      {
         if (!(e & v)) { continue; }
         const int col = d.boundary[7].Find(d.open[v][0]);
         const int endpoint = d.boundary[e].Find(d.open[v][0]);
         for (int k = 0; k < d.open[e].Size(); ++k)
         {
            const int kcol = d.boundary[7].Find(d.open[e][k]);
            for (int i = 0; i < face.lift.Height(); ++i)
            { face.lift(i,col) -= face.lift(i,kcol)*edge.lift(k,endpoint); }
         }
      }
   }
}

struct Templates
{
   TraceRule edge, face;

   Templates(const Layout &d, const DenseMatrix &K, int p)
      : edge(d,3), face(d,7)
   {
      DenseMatrix S(K.Height());
      S = 0.0;
      S.SetSubMatrix(d.boundary[15],Condense(K,d.boundary[15],d.open[15]));
      if (p >= 3) { IncidentRules(d,S,edge,face); }
      else
      {
         // At p=2 only the edge rule is needed. Use the harmonic endpoint
         // lift of the mean unconstrained edge Schur form.
         DenseMatrix mean(d.open[3].Size()+2);
         mean = 0.0;
         for (int e : edges)
         {
            auto keep = Join(d.open[e],d.boundary[e]);
            Array<int> elim;
            for (int k : d.boundary[15]) { if (keep.Find(k)<0) { elim.Append(k); } }
            mean.Add(1.0/6,Condense(S,keep,elim));
         }
         Array<int> interior(edge.energy.Height()), ends(2);
         for (int i = 0; i < interior.Size(); ++i) { interior[i] = i; }
         ends[0] = interior.Size();
         ends[1] = ends[0]+1;
         mean.GetSubMatrix(interior,edge.energy);
         edge.lift = Lift(mean,interior,ends);
      }
   }
};

// Complete the rows on an open entity with R_I = U (x_I - L x_B) + H R_B.
void Complete(DenseMatrix &R, int e, const Layout &low, const Layout &high,
              const DenseMatrix &U, const DenseMatrix &L, const DenseMatrix &H)
{
   DenseMatrix trace, block(U.Height(),L.Width());
   R.GetSubMatrix(high.boundary[e],low.boundary[e],trace);
   Mult(H,trace,block);
   AddMult_a(-1.0,U,L,block);
   R.SetSubMatrix(high.open[e],low.boundary[e],block);
   R.SetSubMatrix(high.open[e],low.open[e],U);
}

} // namespace

void ReferenceMatrix(BatchedLOR &lor, const FiniteElement &bern, DenseMatrix &R)
{
   const int p = bern.GetOrder(), nh = bern.GetDof(),
             nl = lor.ReferencePatch().GetNV();
   Layout low(lor.PatchLabels()), high(BernsteinLabels(bern));
   R.SetSize(nh,nl);
   R = 0.0;
   for (int v : {1,2,4,8}) { R(high.open[v][0],low.open[v][0]) = 1.0; }
   if (p == 1) { return; }

   // The regular tetrahedron has edges (1,1,0), (1,0,1), (0,1,1).
   // Its pulled-back diffusion metric is 2 I - 11^t/2.
   const real_t metric[6] = {1.5,-0.5,-0.5,1.5,-0.5,1.5};
   DenseMatrix A,H,G(3);
   lor.GetPatchStiffness(metric,A);
   G = -0.5;
   for (int i = 0; i < 3; ++i) { G(i,i) = 1.5; }
   MatrixConstantCoefficient coefficient(G);
   IsoparametricTransformation T;
   T.SetIdentityTransformation(Geometry::TETRAHEDRON);
   DiffusionIntegrator(coefficient).AssembleElementMatrix(bern,T,H);

   DenseMatrix sampling;
   SamplePatchBasis(lor.ReferencePatch(),bern,sampling);
   Templates source(low,A,p), target(high,H,p);
   auto trace = [&](int e, TraceRule &a, TraceRule &h)
   {
      DenseMatrix J;
      sampling.GetSubMatrix(low.open[e],high.open[e],J);
      return TracePolar(a.energy,h.energy,J,a.lift,h.lift);
   };
   DenseMatrix E = trace(3,source.edge,target.edge);
   for (int e : edges) { Complete(R,e,low,high,E,source.edge.lift,target.edge.lift); }
   if (p >= 3)
   {
      DenseMatrix F = trace(7,source.face,target.face);
      for (int f : faces) { Complete(R,f,low,high,F,source.face.lift,target.face.lift); }
   }
   if (p >= 4)
   {
      auto ld = Join(low.open[15],low.boundary[15]), hd = Join(high.open[15],
                                                               high.boundary[15]);
      DenseMatrix a,h,J,from_low,from_high,trace_values,volume(high.open[15].Size(),
                                                               ld.Size());
      A.GetSubMatrix(ld,a);
      H.GetSubMatrix(hd,h);
      sampling.GetSubMatrix(low.open[15],high.open[15],J);
      PolarBlocks(a,h,J,from_low,from_high);
      R.GetSubMatrix(high.boundary[15],ld,trace_values);
      Mult(from_high,trace_values,volume);
      volume += from_low;
      R.SetSubMatrix(high.open[15],ld,volume);
   }
}

} // namespace entity
} // namespace tetlor
