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
     masks(ne*dim), mass_inverse(dim, ne), div_scale(ne)
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
   MFEM_VERIFY(p+1 <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D,
               "Order exceeds the H(div) limit HDIV_MAX_D1D.");
   const auto geom = Geometry::TensorProductGeometry(dim);
   MFEM_VERIFY(fes.GetTypicalFE()->GetGeomType() == geom,
               "Expected quadrilateral or hexahedral elements.");
   dof_map = dynamic_cast<const TensorBasisElement&>(
                *fes.GetTypicalFE()).GetDofMap();
   // The source may be valid only on the device; dof_map is used on the host.
   dof_map.HostRead();
   height = width = ne*nd;

   // Identify complete constrained faces. Interior/partial-face constraints
   // do not admit this tensor factorization.
   Array<int> constrained(height);
   constrained = 0;
   // The essential DOF list may already have been used on the device.
   const int *h_ess = ess_dofs.HostRead();
   for (int k = 0; k < ess_dofs.Size(); ++k)
   {
      const int d = h_ess[k];
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
            if (count[side]) { masks[e*dim+c] |= 1 << side; }
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
   struct Basis
   {
      // Pad constrained closed spaces with zeros to retain the RT layout.
      DenseMatrix closed, open;
      Vector sigma;
   } bases[4]; // Bits select constrained left/right endpoints.
   for (int mask = 0; mask < 4; ++mask)
   {
      Basis &b = bases[mask];
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

   // Pack the bases for device access.
   const int nb = (p+1)*(p+1) + p*p + p;
   basis.SetSize(4*nb);
   real_t *h_basis = basis.HostWrite();
   for (int mask = 0; mask < 4; ++mask)
   {
      real_t *data = h_basis + mask*nb;
      const Basis &b = bases[mask];
      std::copy(b.closed.Data(), b.closed.Data() + (p+1)*(p+1), data);
      std::copy(b.open.Data(), b.open.Data() + p*p, data + (p+1)*(p+1));
      std::copy(b.sigma.GetData(), b.sigma.GetData() + p,
                data + (p+1)*(p+1) + p*p);
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
   // Sum the point weights directly: the cached GetWeights() array may be
   // valid only on the device.
   auto weight_sum = [](const IntegrationRule &rule)
   {
      real_t sum = 0.0;
      for (int q = 0; q < rule.GetNPoints(); ++q) { sum += rule.IntPoint(q).weight; }
      return sum;
   };
   const real_t alpha_weight_sum = weight_sum(alpha_rule);
   const real_t beta_weight_sum = weight_sum(beta_rule);
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

/// Apply the 1D transforms selected by the element's direction masks
/// (transposed if @a transpose) in each direction to all components of the
/// lexicographic element vector u, using v as scratch. The pointers are
/// swapped so that u holds the result.
MFEM_HOST_DEVICE static inline void Transform(
   int dim, int p, int nd, const int *mask, const real_t *basis, int nb,
   bool transpose, real_t *&u, real_t *&v)
{
   const int nc = nd/dim;
   for (int a = 0; a < dim; ++a)
   {
      MFEM_FOREACH_THREAD(j, x, nd)
      {
         // Component c has size p+1 in direction c and p in the others.
         const int c = j/nc, jc = j%nc;
         int stride = 1;
         for (int b = 0; b < a; ++b) { stride *= b == c ? p+1 : p; }
         const int n = a == c ? p+1 : p;
         const int i = (jc/stride)%n;
         const int base = c*nc + jc - i*stride;
         const real_t *Q = basis + mask[a]*nb + (a == c ? 0 : (p+1)*(p+1));
         real_t sum = 0.0;
         for (int k = 0; k < n; ++k)
         {
            sum += (transpose ? Q[k+i*n] : Q[i+k*n])*u[base+k*stride];
         }
         v[j] = sum;
      }
      MFEM_SYNC_THREAD;
      real_t *t = u; u = v; v = t;
   }
}

void FastDiagonalizationSolver::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == height, "Invalid vector size.");
   y = x; // Also preserves identity equations and permits in-place application.
   const int dim = this->dim, p = this->p, nd = this->nd;
   const int nc = nd/dim;
   const int nvol = nc*p/(p+1);
   const int nb = (p+1)*(p+1) + p*p + p;
   const int sigma_offset = (p+1)*(p+1) + p*p;
   const int *map = dof_map.Read();
   const int *d_masks = masks.Read();
   const real_t *d_basis = basis.Read();
   const real_t *m_inv = mass_inverse.Read();
   const real_t *d_div = div_scale.Read();
   real_t *Y = y.ReadWrite();
   mfem::forall_2D(ne, std::min(nd, 256), 1, [=] MFEM_HOST_DEVICE (int e)
   {
      // Size for 3D elements at the H(div) limit of this compilation pass.
      constexpr int MD1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MAX_ND = 3*MD1D*(MD1D-1)*(MD1D-1);
      MFEM_SHARED real_t s_u[MAX_ND], s_v[MAX_ND];
      real_t *u = s_u, *v = s_v;
      const int *mask = d_masks + e*dim;
      const real_t *mi = m_inv + e*dim;
      real_t *ye = Y + e*nd;

      MFEM_FOREACH_THREAD(j, x, nd)
      {
         const int d = map[j];
         u[j] = (d < 0 ? -1.0 : 1.0)*ye[UnsignIndex(d)];
      }
      MFEM_SYNC_THREAD;
      Transform(dim, p, nd, mask, d_basis, nb, true, u, v);
      MFEM_FOREACH_THREAD(j, x, nd) { u[j] *= mi[j/nc]; }
      MFEM_SYNC_THREAD;
      // Sherman-Morrison for diag(m_c) + beta/det(J) s s^T per scalar mode.
      MFEM_FOREACH_THREAD(k, x, nvol)
      {
         int index[3], stride = 1;
         real_t s[3], dot = 0.0, norm = 0.0;
         for (int c = 0; c < dim; ++c)
         {
            const int mode = (k/stride)%p;
            index[c] = c*nc + k + (k/(stride*p))*stride;
            s[c] = d_basis[mask[c]*nb + sigma_offset + mode];
            dot += s[c]*u[index[c]];
            norm += s[c]*s[c]*mi[c];
            stride *= p;
         }
         const real_t correction = d_div[e]*dot/(1.0+d_div[e]*norm);
         for (int c = 0; c < dim; ++c) { u[index[c]] -= mi[c]*s[c]*correction; }
      }
      MFEM_SYNC_THREAD;
      Transform(dim, p, nd, mask, d_basis, nb, false, u, v);
      MFEM_FOREACH_THREAD(j, x, nd)
      {
         const int c = j/nc, stride = c == 0 ? 1 : (c == 1 ? p : p*p);
         const int k = ((j%nc)/stride)%(p+1), m = mask[c];
         if ((k == 0 && (m & 1)) || (k == p && (m & 2))) { continue; }
         const int d = map[j];
         ye[UnsignIndex(d)] = (d < 0 ? -1.0 : 1.0)*u[j];
      }
   });
}

} // namespace mfem
