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

/// Average alpha diag(J^T J)/det J and beta/det J over each element.
/// The omitted off-diagonal terms make this an approximation on skew cells.
static void AverageElementGeometry(
   int dim, int ne, const IntegrationRule &rule,
   const GeometricFactors &geometry, const Vector &alpha, const Vector &beta,
   DenseMatrix &mass_inverse, Vector &div_scale)
{
   constexpr int NT = 64;
   const int nq = rule.GetNPoints();
   real_t weight_sum = 0.0;
   for (int q = 0; q < nq; ++q) { weight_sum += rule.IntPoint(q).weight; }
   const real_t *weights = rule.GetWeights().Read();
   const auto J = Reshape(geometry.J.Read(), nq, dim, dim, ne);
   const auto det = Reshape(geometry.detJ.Read(), nq, ne);
   const auto A = Reshape(alpha.Read(), nq, ne);
   const auto B = Reshape(beta.Read(), nq, ne);
   auto inverse = Reshape(mass_inverse.Write(), dim, ne);
   real_t *scale = div_scale.Write();
   Array<int> errors(1);
   errors = 0;
   int *err = errors.ReadWrite();

   mfem::forall_2D(ne, NT, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      MFEM_SHARED real_t partial[4][NT];
      MFEM_FOREACH_THREAD(t,x,NT)
      {
         real_t mass[3] = {0.0, 0.0, 0.0}, div = 0.0;
         for (int q = t; q < nq; q += NT)
         {
            if (!(det(q,e) > 0.0 && A(q,e) > 0.0 && B(q,e) > 0.0))
            { AtomicAdd(err[0], 1); }
            const real_t w = weights[q]/det(q,e);
            for (int c = 0; c < dim; ++c)
            {
               for (int a = 0; a < dim; ++a)
               { mass[c] += w*A(q,e)*J(q,a,c,e)*J(q,a,c,e); }
            }
            div += w*B(q,e);
         }
         for (int c = 0; c < 3; ++c) { partial[c][t] = mass[c]; }
         partial[3][t] = div;
      }
      MFEM_SYNC_THREAD;
      if (MFEM_THREAD_ID(x) == 0)
      {
         for (int c = 0; c < dim; ++c)
         {
            real_t mass = 0.0;
            for (int t = 0; t < NT; ++t) { mass += partial[c][t]; }
            inverse(c,e) = weight_sum/mass;
         }
         real_t div = 0.0;
         for (int t = 0; t < NT; ++t) { div += partial[3][t]; }
         scale[e] = div/weight_sum;
      }
   });
   MFEM_VERIFY(errors.HostRead()[0] == 0,
               "Expected positive Jacobians and positive scalar coefficients.");
}

/// Zero the n entries of a, on the device.
static void ZeroDevice(Array<int> &a)
{
   int *d_a = a.Write();
   mfem::forall(a.Size(), [=] MFEM_HOST_DEVICE (int i) { d_a[i] = 0; });
}

void GetConstrainedFaceMasks(const FiniteElementSpace &fes,
                             const Array<int> &ess_dofs, Array<int> &masks)
{
   const int dim = fes.GetMesh()->Dimension();
   const int p = fes.GetMaxElementOrder();
   const int nd = fes.GetTypicalFE()->GetDof();
   const int ne = fes.GetNE();
   const int height = ne*nd;
   masks.SetSize(ne*dim);
   ZeroDevice(masks);
   const int n_ess = ess_dofs.Size();
   if (n_ess == 0) { return; }

   // Mark the essential DOFs, then count them on the two end faces of each
   // element and direction. Interior/partial-face constraints do not admit
   // the tensor-product structure; they are counted in errors[1] and [2].
   Array<int> constrained(height), errors(3);
   ZeroDevice(constrained);
   ZeroDevice(errors);
   {
      const int *ess = ess_dofs.Read();
      int *marker = constrained.ReadWrite();
      int *err = errors.ReadWrite();
      mfem::forall(n_ess, [=] MFEM_HOST_DEVICE (int k)
      {
         const int d = ess[k];
         if (d < 0 || d >= height) { AtomicAdd(err[0], 1); }
         else { marker[d] = 1; }
      });
   }
   {
      const int nc = nd/dim;
      const int *marker = constrained.Read();
      int *m = masks.ReadWrite();
      int *err = errors.ReadWrite();
      mfem::forall(ne*dim, [=] MFEM_HOST_DEVICE (int i)
      {
         const int e = i/dim, c = i%dim;
         int stride = 1;
         for (int b = 0; b < c; ++b) { stride *= p; }
         int count[2] = {0, 0};
         bool interior = false;
         for (int j = 0; j < nc; ++j)
         {
            if (!marker[e*nd+c*nc+j]) { continue; }
            const int k = (j/stride)%(p+1);
            if (k == 0) { ++count[0]; }
            else if (k == p) { ++count[1]; }
            else { interior = true; }
         }
         if (interior) { AtomicAdd(err[1], 1); }
         int mask = 0;
         for (int side = 0; side < 2; ++side)
         {
            if (count[side] != 0 && count[side] != nc/(p+1)) { AtomicAdd(err[2], 1); }
            if (count[side]) { mask |= 1 << side; }
         }
         m[i] = mask;
      });
   }
   const int *h_err = errors.HostRead();
   MFEM_VERIFY(h_err[0] == 0, "Invalid essential broken DOF.");
   MFEM_VERIFY(h_err[1] == 0, "Cannot constrain interior DOFs.");
   MFEM_VERIFY(h_err[2] == 0,
               "Essential DOFs must cover complete element faces.");
}

FastDiagonalizationSolver::FastDiagonalizationSolver(
   FiniteElementSpace &fes, const Array<int> &ess_dofs,
   Coefficient &alpha, Coefficient &beta)
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
   height = width = ne*nd;

   MFEM_VERIFY(fes.GetMesh()->GetNumGeometries(dim) <= 1,
               "Expected quadrilateral or hexahedral elements only.");
   GetConstrainedFaceMasks(fes, ess_dofs, masks);

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

   // Use the same quadrature as the PA operator.
   Mesh &mesh = *fes.GetMesh();
   const FiniteElement &fe = *fes.GetTypicalFE();
   const IntegrationRule &rule = MassIntegrator::GetRule(
                                    fe, fe, *mesh.GetTypicalElementTransformation());
   QuadratureSpace qs(mesh, rule);
   CoefficientVector alpha_values(alpha, qs, CoefficientStorage::FULL);
   CoefficientVector beta_values(beta, qs, CoefficientStorage::FULL);
   const auto *geometry = mesh.GetGeometricFactors(
                             rule, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);
   AverageElementGeometry(dim, ne, rule, *geometry, alpha_values, beta_values,
                          mass_inverse, div_scale);
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

void FastDiagonalizationSolver::SetOperator(const Operator &)
{
   MFEM_ABORT("FastDiagonalizationSolver must be reconstructed.");
}

void FastDiagonalizationSolver::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == height && y.Size() == height, "Invalid vector size.");
   const int dim = this->dim, p = this->p, nd = this->nd;
   const int nc = nd/dim;
   const int nvol = nc*p/(p+1);
   const int nb = (p+1)*(p+1) + p*p + p;
   const int sigma_offset = (p+1)*(p+1) + p*p;
   const int *d_masks = masks.Read();
   const real_t *d_basis = basis.Read();
   const real_t *m_inv = mass_inverse.Read();
   const real_t *d_div = div_scale.Read();
   // Every entry of y is written; constrained DOFs keep identity equations.
   // In-place application is supported: each element reads x before writing.
   const real_t *X = x.Read();
   real_t *Y = y.Write();
   mfem::forall_2D(ne, std::min(nd, 256), 1, [=] MFEM_HOST_DEVICE (int e)
   {
      // Size for 3D elements at the H(div) limit of this compilation pass.
      constexpr int MD1D = DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MAX_ND = 3*MD1D*(MD1D-1)*(MD1D-1);
      MFEM_SHARED real_t s_u[MAX_ND], s_v[MAX_ND];
      real_t *u = s_u, *v = s_v;
      const int *mask = d_masks + e*dim;
      const real_t *mi = m_inv + e*dim;
      const real_t *xe = X + e*nd;
      real_t *ye = Y + e*nd;

      MFEM_FOREACH_THREAD(j, x, nd) { u[j] = xe[j]; }
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
         const bool constrained = (k == 0 && (m & 1)) || (k == p && (m & 2));
         ye[j] = constrained ? xe[j] : u[j];
      }
   });
}

} // namespace mfem
