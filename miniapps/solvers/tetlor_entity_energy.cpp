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
#include <cmath>
#include <algorithm>
#include <limits>

namespace tetlor
{
namespace entity
{

DenseMatrix Condense(const DenseMatrix &K, const Array<int> &keep,
                     const Array<int> &eliminate)
{
   DenseMatrix S;
   K.GetSubMatrix(keep, S);
   if (eliminate.Size())
   {
      DenseMatrix A, B;
      K.GetSubMatrix(eliminate, A);
      K.GetSubMatrix(eliminate, keep, B);
      CholeskyFactors L(A.Data());
      MFEM_VERIFY(L.Factor(A.Height()), "Singular eliminated energy block.");
      L.LSolve(A.Height(), B.Width(), B.Data());
      AddMult_a_AtB(-1.0, B, B, S);
   }
   return S;
}

// Return the thin polar factor U V^t of a tall, full-column-rank matrix.
static DenseMatrix Polar(DenseMatrix &A)
{
   const int m = A.Height(), n = A.Width();
   DenseMatrix polar(m,n);
#ifdef MFEM_USE_LAPACK
   DenseMatrixSVD svd(A, 'S', 'S');
   svd.Eval(A);
   Mult(svd.LeftSingularvectors(), svd.RightSingularvectors(), polar);
#else
   // One-sided Jacobi SVD, keeping the right rotations. Orthogonalizing the
   // columns directly avoids squaring the condition number of the sampling map.
   DenseMatrix V(n);
   V.Diag(1.0,n);
   const real_t tol = 8*std::numeric_limits<real_t>::epsilon();
   for (int sweep = 0; ; ++sweep)
   {
      bool converged = true;
      for (int j = 1; j < n; ++j)
      {
         for (int i = 0; i < j; ++i)
         {
            Vector a, b, vi, vj;
            A.GetColumnReference(i,a); A.GetColumnReference(j,b);
            const real_t aa = a*a, bb = b*b, ab = a*b;
            if (std::abs(ab) <= tol*std::sqrt(aa)*std::sqrt(bb)) { continue; }
            converged = false;
            const real_t tau = (bb-aa)/(2*ab);
            const real_t t = std::copysign(1.0,tau)/(std::abs(tau)+std::hypot(1.0,tau));
            const real_t c = 1/std::hypot(1.0,t), s = c*t;
            auto rotate = [=](Vector &x, Vector &y)
            {
               for (int k = 0; k < x.Size(); ++k)
               {
                  const real_t u = x(k), v = y(k);
                  x(k) = c*u-s*v;
                  y(k) = s*u+c*v;
               }
            };
            rotate(a,b);
            V.GetColumnReference(i,vi); V.GetColumnReference(j,vj);
            rotate(vi,vj);
         }
      }
      if (converged) { break; }
      MFEM_VERIFY(sweep < 100, "Entity sampling SVD did not converge.");
   }
   for (int j = 0; j < n; ++j)
   {
      Vector a;
      A.GetColumnReference(j,a);
      const real_t sigma = a.Norml2();
      MFEM_VERIFY(sigma > 0.0, "Rank deficient entity sampling.");
      a /= sigma;
   }
   MultABt(A,V,polar);
#endif
   return polar;
}

struct HarmonicExtension
{
   DenseMatrix factor, lift;
   CholeskyFactors L;

   HarmonicExtension(const DenseMatrix &S, int n)
      : factor(n), lift(n,S.Width()-n), L(factor.Data())
   {
      factor.CopyMN(S,n,n,0,0);
      MFEM_VERIFY(L.Factor(n), "Singular interior energy on entity.");
      if (lift.Width())
      {
         lift.CopyMN(S,n,lift.Width(),0,n);
         lift *= -1.0;
         L.Solve(n,lift.Width(),lift.Data());
      }
   }
};

// Rotate the polar isometry in the plane of the two whitened constant
// residuals, then scale that direction to reproduce the constant exactly.
static void AlignConstant(DenseMatrix &U, Vector &w, Vector &v)
{
   const real_t nw = w.Norml2(), nv = v.Norml2();
   w /= nw;
   v /= nv;
   Vector z(w.Size()), tangent(w), a(v.Size()), b(v.Size());
   U.Mult(v,z);
   const real_t c = std::clamp(z*w,real_t(-1),real_t(1));
   tangent.Add(-c,z);
   const real_t s = tangent.Norml2();
   U.MultTranspose(z,a);
   if (s > 32*std::numeric_limits<real_t>::epsilon())
   {
      tangent /= s;
      U.MultTranspose(tangent,b);
      AddMult_a_VWt(c-1,z,a,U);
      AddMult_a_VWt(s,tangent,a,U);
      AddMult_a_VWt(-s,z,b,U);
      AddMult_a_VWt(c-1,tangent,b,U);
   }
   else if (c < 0) { AddMult_a_VWt(-2,z,a,U); }
   AddMult_a_VWt(nv/nw-1,w,v,U);
}

static DenseMatrix BubbleMap(const HarmonicExtension &source,
                             const HarmonicExtension &target,
                             const DenseMatrix &sampling,
                             Vector *w = nullptr, Vector *v = nullptr)
{
   const int nl = sampling.Height(), nh = sampling.Width();
   // Whiten the sampling J as L_low^t J L_high^{-t}.
   DenseMatrix J(sampling), Jt;
   source.L.UMult(nl,nh,J.Data());
   Jt.Transpose(J);
   target.L.LSolve(nh,nl,Jt.Data());
   J.Transpose(Jt);
   DenseMatrix U = Polar(J), R;
   if (w)
   {
      source.L.UMult(nl,1,w->GetData());
      target.L.UMult(nh,1,v->GetData());
      AlignConstant(U,*w,*v);
   }
   source.L.LMult(nl,nh,U.Data());
   R.Transpose(U);
   target.L.USolve(nh,nl,R.Data());

   return R;
}

DenseMatrix TracePolar(const DenseMatrix &low, const DenseMatrix &high,
                       const DenseMatrix &sampling, DenseMatrix &low_lift,
                       DenseMatrix &high_lift)
{
   HarmonicExtension source(low,low.Height()), target(high,high.Height());
   auto residual = [](const DenseMatrix &lift)
   {
      Vector ones(lift.Width()), r(lift.Height());
      ones = 1.0;
      r = 1.0;
      lift.AddMult(ones,r,-1.0);
      return r;
   };
   Vector w = residual(low_lift), v = residual(high_lift);
   if (std::min(w.Normlinf(),v.Normlinf()) < 1e-12)
   {
      // A vanishing residual cannot be paired with a nonzero one.
      Vector a(low_lift.Width()), b(high_lift.Width());
      a = 1.0/a.Size();
      b = 1.0/b.Size();
      AddMult_a_VWt(1,w,a,low_lift);
      AddMult_a_VWt(1,v,b,high_lift);
      return BubbleMap(source,target,sampling);
   }
   return BubbleMap(source,target,sampling,&w,&v);
}

void PolarBlocks(const DenseMatrix &low, const DenseMatrix &high,
                 const DenseMatrix &sampling, DenseMatrix &from_low,
                 DenseMatrix &from_boundary)
{
   const int nl = sampling.Height(), nh = sampling.Width();
   HarmonicExtension source(low,nl), target(high,nh);
   DenseMatrix R = BubbleMap(source,target,sampling);
   // u_high,I = R (u_low,I - lift_low u_low,B) + lift_high u_high,B.
   from_low.SetSize(nh,low.Width());
   from_low.CopyMN(R,0,0);
   if (source.lift.Width())
   {
      DenseMatrix correction(nh,source.lift.Width());
      Mult(R,source.lift,correction);
      correction *= -1.0;
      from_low.CopyMN(correction,0,nl);
   }
   from_boundary = target.lift;
}

} // namespace entity
} // namespace tetlor
