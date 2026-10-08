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

#include "fast_diag.hpp"
#include <algorithm>

namespace mfem
{

/// Evaluate coeff (one if null) at the points of the rule it is integrated on,
/// in element-contiguous order, and return that rule.
static const IntegrationRule &EvalCoefficient(
   Mesh &mesh, Coefficient *coeff, const IntegrationRule &default_rule,
   Vector &values)
{
   if (const IntegrationRule *rule = GetCoefficientRule(coeff))
   {
      const auto &qf =
         static_cast<QuadratureFunctionCoefficient*>(coeff)->GetQuadFunction();
      MFEM_VERIFY(qf.GetVDim() == 1, "Expected a scalar quadrature function.");
      values = qf;
      return *rule;
   }
   QuadratureSpace qs(mesh, default_rule);
   QuadratureFunction qf(qs);
   if (coeff) { coeff->Project(qf); }
   else { qf = 1.0; }
   values = qf;
   return default_rule;
}

FastDiagonalizationSolver::FastDiagonalizationSolver(
   FiniteElementSpace &fes, const Array<int> &ess_dofs,
   Coefficient *alpha, Coefficient *beta)
   : Solver(0), dim(fes.GetMesh()->Dimension()), p(fes.GetMaxElementOrder()),
     nd(fes.GetTypicalFE()->GetDof()), ne(fes.GetNE()),
     masks(ne, dim), mass_inverse(dim, ne), div_scale(ne), work(nd), scratch(nd)
{
#ifndef MFEM_USE_LAPACK
   MFEM_ABORT("FastDiagonalizationSolver requires MFEM_USE_LAPACK.");
#else
   const auto *fec = dynamic_cast<const RT_FECollection*>(fes.FEColl());
   MFEM_VERIFY(fec && fec->GetClosedBasisType() == BasisType::GaussLobatto &&
               fec->GetOpenBasisType() == BasisType::IntegratedGLL,
               "Expected GaussLobatto/IntegratedGLL RT elements.");
   MFEM_VERIFY((dim == 2 || dim == 3) && fes.GetVDim() == 1 &&
               fes.GetMesh()->SpaceDimension() == dim && !fes.IsVariableOrder(),
               "Expected a uniform-order 2D or 3D RT space.");
   const auto geom = Geometry::TensorProductGeometry(dim);
   MFEM_VERIFY(fes.GetTypicalFE()->GetGeomType() == geom,
               "Expected quadrilateral or hexahedral elements.");
   dof_map = dynamic_cast<const TensorBasisElement&>(
                *fes.GetTypicalFE()).GetDofMap();
   height = width = ne*nd;

   // Identify complete constrained faces. Interior/partial-face constraints
   // do not admit this tensor factorization.
   Array<int> constrained(height);
   constrained = 0;
   for (int d : ess_dofs)
   {
      MFEM_VERIFY(d >= 0 && d < height, "Invalid essential broken DOF.");
      constrained[d] = 1;
   }
   masks = 0;
   const int nc = nd/dim;
   for (int e = 0; e < ne; ++e)
   {
      MFEM_VERIFY(fes.GetFE(e)->GetGeomType() == geom,
                  "Expected quadrilateral or hexahedral elements.");
      int stride = 1;
      for (int c = 0; c < dim; ++c)
      {
         int count[2] = {0, 0};
         for (int j = 0; j < nc; ++j)
         {
            if (!constrained[e*nd+UnsignIndex(dof_map[c*nc+j])]) { continue; }
            const int k = (j/stride)%(p+1);
            MFEM_VERIFY(k == 0 || k == p, "Cannot constrain interior DOFs.");
            ++count[k == p];
         }
         for (int side = 0; side < 2; ++side)
         {
            MFEM_VERIFY(count[side] == 0 || count[side] == nc/(p+1),
                        "Essential DOFs must cover complete element faces.");
            if (count[side]) { masks(e,c) |= 1 << side; }
         }
         stride *= p;
      }
   }

   DenseMatrix Mc(p+1), Mo(p);
   Mc = 0.0;
   Mo = 0.0;
   Vector closed(p+1), deriv(p+1), open(p);
   const auto &cb = poly1d.GetBasis(p, BasisType::GaussLobatto);
   auto &ob = poly1d.GetBasis(p-1, BasisType::IntegratedGLL);
   ob.ScaleIntegrated(false);
   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, 2*p);
   for (int q = 0; q < ir.Size(); ++q)
   {
      const auto &ip = ir.IntPoint(q);
      cb.Eval(ip.x, closed, deriv);
      ob.EvalIntegrated(deriv, open);
      AddMult_a_VVt(ip.weight, closed, Mc);
      AddMult_a_VVt(ip.weight, open, Mo);
   }
   DenseMatrix Lo(Mo);
   CholeskyFactors lo(Lo.Data());
   MFEM_VERIFY(lo.Factor(p), "Invalid open mass matrix.");
   for (int mask = 0; mask < 4; ++mask)
   {
      Basis &b = basis[mask];
      b.closed.SetSize(p+1);
      b.closed = 0.0;
      b.sigma.SetSize(p);
      b.sigma = 0.0;
      const int first = mask & 1;
      const int n = p+1-first-((mask >> 1) & 1);
      b.open.Diag(1.0, p);
      if (n > 0)
      {
         DenseMatrix Lc(n), F(p,n), Ft, Qc;
         for (int j = 0; j < n; ++j)
         {
            for (int i = 0; i < n; ++i) { Lc(i,j) = Mc(i+first,j+first); }
            for (int i = 0; i < p; ++i)
            { F(i,j) = (j+first == i+1) - (j+first == i); }
         }
         CholeskyFactors lc(Lc.Data());
         MFEM_VERIFY(lc.Factor(n), "Invalid closed mass matrix.");
         // F = Lo^T E Lc^{-T}; full singular vectors retain all null modes.
         lo.UMult(p, n, F.Data());
         Ft.Transpose(F);
         lc.LSolve(n, p, Ft.Data());
         F.Transpose(Ft);
         DenseMatrixSVD svd(F, 'A', 'A');
         svd.Eval(F);
         b.open = svd.LeftSingularvectors();
         Qc.Transpose(svd.RightSingularvectors());
         lc.USolve(n, n, Qc.Data());
         for (int j = 0; j < n; ++j)
         {
            for (int i = 0; i < n; ++i) { b.closed(i+first,j) = Qc(i,j); }
         }
         for (int i = 0; i < std::min(p,n); ++i)
         { b.sigma(i) = svd.Singularvalue(i); }
      }
      lo.USolve(p, p, b.open.Data());
   }

   // Average the diagonal of alpha J^T J/det J and beta/det J using
   // reference quadrature weights. Off-diagonal mass couplings are omitted.
   // Quadrature function coefficients are averaged on their own rules.
   Mesh &mesh = *fes.GetMesh();
   const FiniteElement &fe = *fes.GetTypicalFE();
   const IntegrationRule &default_rule = MassIntegrator::GetRule(
                                            fe, fe, *mesh.GetTypicalElementTransformation());
   Vector alpha_values, beta_values;
   const IntegrationRule &alpha_rule = EvalCoefficient(mesh, alpha, default_rule,
                                                       alpha_values);
   const IntegrationRule &beta_rule = EvalCoefficient(mesh, beta, default_rule,
                                                      beta_values);
   const int flags = GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS;
   const auto *alpha_factors = mesh.GetGeometricFactors(alpha_rule, flags);
   const auto *beta_factors = mesh.GetGeometricFactors(beta_rule, flags);
   const int nqa = alpha_rule.GetNPoints(), nqb = beta_rule.GetNPoints();
   const auto J = Reshape(alpha_factors->J.HostRead(), nqa, dim, dim, ne);
   const auto det_a = Reshape(alpha_factors->detJ.HostRead(), nqa, ne);
   const auto det_b = Reshape(beta_factors->detJ.HostRead(), nqb, ne);
   const auto a_val = Reshape(alpha_values.HostRead(), nqa, ne);
   const auto b_val = Reshape(beta_values.HostRead(), nqb, ne);
   const real_t alpha_weight_sum = alpha_rule.GetWeights().Sum();
   const real_t beta_weight_sum = beta_rule.GetWeights().Sum();
   for (int e = 0; e < ne; ++e)
   {
      real_t mass[3] = {}, div = 0.0;
      for (int q = 0; q < nqa; ++q)
      {
         MFEM_VERIFY(det_a(q,e) > 0.0, "Nonpositive element Jacobian.");
         const real_t w = alpha_rule.IntPoint(q).weight*a_val(q,e)/det_a(q,e);
         for (int c = 0; c < dim; ++c)
         {
            for (int a = 0; a < dim; ++a)
            { mass[c] += w*J(q,a,c,e)*J(q,a,c,e); }
         }
      }
      for (int q = 0; q < nqb; ++q)
      {
         MFEM_VERIFY(det_b(q,e) > 0.0, "Nonpositive element Jacobian.");
         div += beta_rule.IntPoint(q).weight*b_val(q,e)/det_b(q,e);
      }
      MFEM_VERIFY(div >= 0.0, "Invalid coefficients.");
      for (int c = 0; c < dim; ++c)
      {
         MFEM_VERIFY(mass[c] > 0.0, "Invalid coefficients.");
         mass_inverse(c,e) = alpha_weight_sum/mass[c];
      }
      div_scale(e) = div/beta_weight_sum;
   }
#endif
}

void FastDiagonalizationSolver::Transform(int e, bool transpose) const
{
   const int nc = nd/dim;
   for (int c = 0; c < dim; ++c)
   {
      int stride = 1;
      for (int a = 0; a < dim; ++a)
      {
         const auto &b = basis[masks(e,a)];
         const DenseMatrix &Q = a == c ? b.closed : b.open;
         const int n = Q.Height();
         for (int outer = 0; outer < nc; outer += n*stride)
         {
            for (int inner = 0; inner < stride; ++inner)
            {
               for (int i = 0; i < n; ++i)
               {
                  real_t sum = 0.0;
                  for (int j = 0; j < n; ++j)
                  {
                     sum += (transpose ? Q(j,i) : Q(i,j))*
                            work(c*nc+outer+j*stride+inner);
                  }
                  scratch(c*nc+outer+i*stride+inner) = sum;
               }
            }
         }
         std::copy(scratch.GetData()+c*nc, scratch.GetData()+(c+1)*nc,
                   work.GetData()+c*nc);
         stride *= n;
      }
   }
}

void FastDiagonalizationSolver::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == height, "Invalid vector size.");
   y = x; // Also preserves identity equations and permits in-place application.
   real_t *output = y.HostReadWrite();
   const int nc = nd/dim;
   const int nvol = nc*p/(p+1);
   for (int e = 0; e < ne; ++e)
   {
      for (int j = 0; j < nd; ++j)
      {
         const int d = dof_map[j];
         work(j) = (d < 0 ? -1.0 : 1.0)*output[e*nd+UnsignIndex(d)];
      }
      Transform(e, true);
      for (int c = 0; c < dim; ++c)
      {
         for (int j = 0; j < nc; ++j) { work(c*nc+j) *= mass_inverse(c,e); }
      }
      // Sherman-Morrison for diag(m_c) + beta/det(J) s s^T per scalar mode.
      for (int k = 0; k < nvol; ++k)
      {
         int index[3], stride = 1;
         real_t s[3], dot = 0.0, norm = 0.0;
         for (int c = 0; c < dim; ++c)
         {
            const int mode = (k/stride)%p;
            index[c] = c*nc + k + (k/(stride*p))*stride;
            s[c] = basis[masks(e,c)].sigma(mode);
            dot += s[c]*work(index[c]);
            norm += s[c]*s[c]*mass_inverse(c,e);
            stride *= p;
         }
         const real_t correction = div_scale(e)*dot/(1.0+div_scale(e)*norm);
         for (int c = 0; c < dim; ++c)
         { work(index[c]) -= mass_inverse(c,e)*s[c]*correction; }
      }
      Transform(e, false);
      for (int j = 0; j < nd; ++j)
      {
         const int c = j/nc, stride = c == 0 ? 1 : (c == 1 ? p : p*p);
         const int k = ((j%nc)/stride)%(p+1), mask = masks(e,c);
         if ((k == 0 && (mask & 1)) || (k == p && (mask & 2))) { continue; }
         const int d = dof_map[j];
         output[e*nd+UnsignIndex(d)] = (d < 0 ? -1.0 : 1.0)*work(j);
      }
   }
}

} // namespace mfem
