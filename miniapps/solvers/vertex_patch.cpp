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

#include "vertex_patch.hpp"
#if defined(MFEM_USE_MPI) && defined(MFEM_USE_LAPACK) && defined(MFEM_USE_METIS) && defined(MFEM_USE_METIS_5)
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>
#include <metis.h>

namespace mfem
{
namespace vertex_patch
{
namespace
{
int Power(int n, int d) { int v = 1; while (d--) { v *= n; } return v; }
void Coordinates(int i, int n, int dim, int *a)
{ for (int d = 0; d < dim; ++d) { a[d] = i % n; i /= n; } }
int Index(const int *a, int n, int dim)
{ int i = 0; for (int d = dim - 1; d >= 0; --d) { i = i * n + a[d]; } return i; }
bool Interior(int i, int n, int dim)
{
   int a[3]; Coordinates(i, n, dim, a);
   for (int d = 0; d < dim; ++d) { if (a[d] == 0 || a[d] == n - 1) { return false; } }
   return true;
}

// Rectangular tensor contractions with contiguous inner loops. Read directly
// from x and write the final contraction to y, without endpoint buffer copies.
template <int DIM, bool TRANSPOSE>
void TensorKernel(const Vector &x, Vector &y,
                  const std::array<const DenseMatrix*, 3> &b,
                  Vector &u, Vector &v)
{
   int sizes[DIM];
   for (int d = 0; d < DIM; ++d)
   { sizes[d] = TRANSPOSE ? b[d]->Height() : b[d]->Width(); }
   const real_t *input = x.GetData();
   for (int axis = 0; axis < DIM; ++axis)
   {
      const DenseMatrix &m = *b[axis];
      int before = 1, after = 1;
      for (int d = 0; d < axis; ++d) { before *= sizes[d]; }
      for (int d = axis + 1; d < DIM; ++d) { after *= sizes[d]; }
      const int ni = sizes[axis], no = TRANSPOSE ? m.Width() : m.Height();
      Vector &output = axis == DIM - 1 ? y : (axis == 0 ? u : v);
      output.SetSize(before * no * after);
      real_t *out = output.GetData();
      for (int k = 0; k < after; ++k)
         for (int j = 0; j < no; ++j)
         {
            real_t *dest = out + before * (j + no * k);
            if (axis == 0)
            {
               real_t value = 0.;
               for (int l = 0; l < ni; ++l)
               { value += (TRANSPOSE ? m(l, j) : m(j, l)) * input[l + ni * k]; }
               dest[0] = value;
            }
            else
            {
               std::fill(dest, dest + before, 0.);
               for (int l = 0; l < ni; ++l)
               {
                  const real_t factor = TRANSPOSE ? m(l, j) : m(j, l);
                  const real_t *src = input + before * (l + ni * k);
                  for (int i = 0; i < before; ++i) { dest[i] += factor * src[i]; }
               }
            }
         }
      input = out;
      sizes[axis] = no;
   }
}
void Tensor(const Vector &x, Vector &y,
            const std::array<const DenseMatrix*, 3> &b, int dim,
            bool transpose, Vector &u, Vector &v)
{
   if (dim == 2)
   {
      if (transpose) { TensorKernel<2, true>(x, y, b, u, v); }
      else { TensorKernel<2, false>(x, y, b, u, v); }
   }
   else
   {
      MFEM_VERIFY(dim == 3, "Expected a 2D or 3D tensor contraction");
      if (transpose) { TensorKernel<3, true>(x, y, b, u, v); }
      else { TensorKernel<3, false>(x, y, b, u, v); }
   }
}
using Rows = std::vector<std::unordered_map<int, real_t>>;
std::unique_ptr<SparseMatrix> CSR(const Rows &rows, int width = -1)
{
   const int n = int(rows.size());
   int *i = new int[n + 1]; i[0] = 0;
   for (int r = 0; r < n; ++r) { i[r + 1] = i[r] + int(rows[r].size()); }
   int *j = new int[i[n]];
   real_t *a = new real_t[i[n]];
   for (int r = 0; r < n; ++r)
   {
      std::vector<std::pair<int, real_t>> entries(rows[r].begin(), rows[r].end());
      std::sort(entries.begin(), entries.end());
      int k = i[r];
      for (const auto &entry : entries) { j[k] = entry.first; a[k++] = entry.second; }
   }
   return std::unique_ptr<SparseMatrix>(new SparseMatrix(i, j, a, n,
                                                         width < 0 ? n : width));
}
// Build symbolic rows once, then accumulate numerical values in contiguous CSR.
struct SparsePattern
{
   std::vector<int> offsets, columns;
   SparsePattern() = default;
   explicit SparsePattern(std::vector<std::vector<int>> rows)
   {
      offsets.push_back(0);
      for (auto &row : rows)
      {
         std::sort(row.begin(), row.end());
         row.erase(std::unique(row.begin(), row.end()), row.end());
         columns.insert(columns.end(), row.begin(), row.end());
         offsets.push_back(int(columns.size()));
      }
   }
   explicit SparsePattern(const SparseMatrix &a)
      : offsets(a.GetI(), a.GetI() + a.Height() + 1),
        columns(a.GetJ(), a.GetJ() + a.NumNonZeroElems()) { }
   int Slot(int row, int col) const
   {
      auto begin = columns.begin() + offsets[row];
      auto end = columns.begin() + offsets[row + 1];
      auto it = std::lower_bound(begin, end, col);
      MFEM_VERIFY(it != end && *it == col, "Missing symbolic matrix entry");
      return int(it - columns.begin());
   }
   std::unique_ptr<SparseMatrix> NewMatrix() const
   {
      const int height = int(offsets.size()) - 1;
      int *i = new int[offsets.size()], *j = new int[columns.size()];
      std::copy(offsets.begin(), offsets.end(), i);
      std::copy(columns.begin(), columns.end(), j);
      return std::unique_ptr<SparseMatrix>(new SparseMatrix(
                                              i, j, nullptr, height, height,
                                              true, true, true));
   }
   bool Matches(const SparseMatrix &a) const
   {
      return int(offsets.size()) == a.Height() + 1 &&
             int(columns.size()) == a.NumNonZeroElems() &&
             std::equal(offsets.begin(), offsets.end(), a.GetI()) &&
             std::equal(columns.begin(), columns.end(), a.GetJ());
   }
};

struct CondensationPlan
{
   struct Update { int slot, left, right, diagonal; };
   SparsePattern source, schur;
   std::vector<int> boundary, diagonals, interior_offdiagonals;
   std::vector<Update> updates;
   CondensationPlan() = default;
   CondensationPlan(const SparsePattern &pattern, int n, int dim) : source(pattern)
   {
      const int nd = int(source.offsets.size()) - 1;
      std::vector<int> index(nd, -1);
      for (int i = 0; i < nd; ++i) if (!Interior(i, n, dim))
         { index[i] = int(boundary.size()); boundary.push_back(i); }
      std::vector<std::vector<int>> rows(boundary.size());
      const auto &ai = source.offsets, &aj = source.columns;
      for (int i = 0; i < nd; ++i)
      {
         if (index[i] >= 0)
         {
            for (int k = ai[i]; k < ai[i + 1]; ++k) if (index[aj[k]] >= 0)
               { rows[index[i]].push_back(index[aj[k]]); }
         }
         else
         {
            for (int k = ai[i]; k < ai[i + 1]; ++k) if (index[aj[k]] >= 0)
                  for (int l = ai[i]; l < ai[i + 1]; ++l) if (index[aj[l]] >= 0)
                     { rows[index[aj[k]]].push_back(index[aj[l]]); }
         }
      }
      schur = SparsePattern(std::move(rows));
      // Preserve the original row/update ordering, including symbolic zeros.
      for (int i = 0; i < nd; ++i)
      {
         if (index[i] >= 0)
         {
            for (int k = ai[i]; k < ai[i + 1]; ++k) if (index[aj[k]] >= 0)
               { updates.push_back({schur.Slot(index[i], index[aj[k]]), k, -1, -1}); }
         }
         else
         {
            int diagonal = -1;
            for (int k = ai[i]; k < ai[i + 1]; ++k)
            {
               if (aj[k] == i) { diagonal = k; }
               else if (index[aj[k]] < 0) { interior_offdiagonals.push_back(k); }
            }
            MFEM_VERIFY(diagonal >= 0, "Missing interior diagonal");
            diagonals.push_back(diagonal);
            for (int k = ai[i]; k < ai[i + 1]; ++k) if (index[aj[k]] >= 0)
                  for (int l = ai[i]; l < ai[i + 1]; ++l) if (index[aj[l]] >= 0)
                     { updates.push_back({schur.Slot(index[aj[k]], index[aj[l]]), k, l, diagonal}); }
         }
      }
   }
   std::unique_ptr<SparseMatrix> Apply(const SparseMatrix &a) const
   {
      const real_t *av = a.GetData();
      for (int k : diagonals)
      { MFEM_VERIFY(av[k] > 0., "Nonpositive interior diagonal"); }
      for (int k : interior_offdiagonals)
      { MFEM_VERIFY(av[k] == 0., "Interior block is not diagonal"); }
      auto result = schur.NewMatrix();
      real_t *values = result->GetData();
      for (const auto &u : updates)
      {
         if (u.right < 0) { values[u.slot] += av[u.left]; }
         else { values[u.slot] -= av[u.left] * av[u.right] / av[u.diagonal]; }
      }
      return result;
   }
};
}

Basis::Basis(int degree) : p(degree), S(p + 1), inverse(p + 1), G(p + 1),
   derivative(p, p + 1), eigenvalues(p - 1), parity(p + 1)
{
   MFEM_VERIFY(p >= 1, "FDM degree must be positive");
   H1_SegmentElement fe(p);
   // Work in lexicographic GLL order on MFEM's interval [0,1].
   const auto &map = fe.GetDofMap();
   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, 2 * p + 2);
   DenseMatrix mass(p + 1), stiffness(p + 1), ds(p + 1, 1);
   Vector shape(p + 1);
   mass = 0.; stiffness = 0.;
   for (int q = 0; q < ir.GetNPoints(); ++q)
   {
      const auto &ip = ir.IntPoint(q);
      fe.CalcShape(ip, shape); fe.CalcDShape(ip, ds);
      for (int i = 0; i <= p; ++i) for (int j = 0; j <= p; ++j)
         {
            mass(i, j) += ip.weight * shape[map[i]] * shape[map[j]];
            stiffness(i, j) += ip.weight * ds(map[i], 0) * ds(map[j], 0);
         }
   }
   S = 0.; S(0, 0) = S(p, p) = 1.; parity = 1;
   if (p > 1)
   {
      DenseMatrix a(p - 1), b(p - 1), eigenvectors;
      for (int i = 1; i < p; ++i) for (int j = 1; j < p; ++j)
         { a(i - 1, j - 1) = stiffness(i, j); b(i - 1, j - 1) = mass(i, j); }
      a.Eigenvalues(b, eigenvalues, eigenvectors);
      for (int j = 1; j < p; ++j)
      {
         real_t reflection = 0;
         for (int i = 1; i < p; ++i)
         { reflection += eigenvectors(i - 1, j - 1) * eigenvectors(p - i - 1, j - 1); }
         parity[j] = reflection >= 0 ? 1 : -1;
         for (int i = 1; i < p; ++i)
         { S(i, j) = 0.5 * (eigenvectors(i - 1, j - 1) + parity[j] * eigenvectors(p - i - 1, j - 1)); }
         real_t norm = 0;
         for (int i = 1; i < p; ++i) for (int k = 1; k < p; ++k)
            { norm += S(i, j) * mass(i, k) * S(k, j); }
         const real_t scale = (S(1, j) < 0 ? -1. : 1.) / std::sqrt(norm);
         for (int i = 1; i < p; ++i) { S(i, j) *= scale; }
      }
      for (int e : {0, p}) for (int j = 1; j < p; ++j)
         {
            real_t projection = 0;
            for (int i = 1; i < p; ++i) { projection += S(i, j) * mass(i, e); }
            for (int i = 1; i < p; ++i) { S(i, e) -= S(i, j) * projection; }
         }
   }
   DenseMatrixInverse sinv(S); sinv.GetInverseMatrix(inverse);
   DenseMatrix tmp(p + 1), transformed(p + 1);
   Mult(mass, S, tmp); MultAtB(S, tmp, transformed);
   real_t error = 0;
   for (int i = 1; i < p; ++i) for (int j = 0; j <= p; ++j)
      { error = std::max(error, std::abs(transformed(i, j) - (i == j ? 1. : 0.))); }
   Mult(stiffness, S, tmp); MultAtB(S, tmp, transformed);
   for (int i = 1; i < p; ++i) for (int j = 1; j < p; ++j)
      {
         error = std::max(error, std::abs(transformed(i,
                                                      j) - (i == j ? eigenvalues[i - 1] : 0.)) /
                          std::sqrt(eigenvalues[i - 1] * eigenvalues[j - 1]));
      }
   MFEM_VERIFY(error < 1.e-9, "FDM reference orthogonality failed: " << error);
   derivative = 0.; derivative(0, 0) = -1.; derivative(0, p) = 1.;
   for (int j = 1; j < p; ++j)
   {
      derivative(j, j) = std::sqrt(eigenvalues[j - 1]);
      for (int e : {0, p})
      { derivative(j, e) = transformed(j, e) / std::sqrt(eigenvalues[j - 1]); }
   }
   Mult(mass, S, tmp); MultAtB(S, tmp, transformed);
   // Symmetric square root of the endpoint mass block. Its inverse maps
   // the two endpoint basis functions to an orthonormal broken basis.
   const real_t a = 0.5 * (transformed(0, 0) + transformed(p, p));
   const real_t b = transformed(0, p);
   const real_t plus = std::sqrt(a + b), minus = std::sqrt(a - b);
   G = 0.;
   for (int i = 1; i < p; ++i) { G(i, i) = 1.; }
   G(0, 0) = G(p, p) = 0.5 * (plus + minus);
   G(0, p) = G(p, 0) = 0.5 * (plus - minus);
}

std::shared_ptr<const Basis> Basis::Get(int degree)
{
   static std::map<int, std::shared_ptr<const Basis>> cache;
   auto &entry = cache[degree];
   if (!entry) { entry = std::make_shared<Basis>(degree); }
   return entry;
}

BasisChange::BasisChange(ParFiniteElementSpace &space,
                         std::shared_ptr<const Basis> reference_basis)
   : Operator(space.GetTrueVSize()), fes(space), basis(reference_basis),
     dim(space.GetMesh()->Dimension()), nd(Power(basis->p + 1, dim))
{
   const int n = basis->p + 1, ne = fes.GetNE();
   const auto *restriction = dynamic_cast<const ElementRestriction*>(
                                fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC));
   MFEM_VERIFY(restriction, "Expected conforming H1 element restriction");
   nodal = restriction->GatherMap();
   modal.SetSize(ne * nd); signs.SetSize(ne * nd);
   nodal_mask.SetSize(ne * nd); modal_mask.SetSize(ne * nd);
   Array<int> nseen(fes.GetVSize()), mseen(fes.GetVSize()); nseen = 0; mseen = 0;
   Array<HYPRE_BigInt> vertices;
   fes.GetParMesh()->GetGlobalVertexIndices(vertices);
   Array<int> ev;
   for (int e = 0; e < ne; ++e)
   {
      fes.GetMesh()->GetElementVertices(e, ev);
      const IntegrationRule *vr = Geometries.GetVertices(
                                     fes.GetMesh()->GetElementGeometry(e));
      HYPRE_BigInt corners[8];
      for (int j = 0; j < ev.Size(); ++j)
      {
         const auto &v = vr->IntPoint(j);
         const int bit = int(v.x) + 2 * int(v.y) + (dim == 3 ? 4 * int(v.z) : 0);
         corners[bit] = vertices[ev[j]];
      }
      for (int i = 0; i < nd; ++i)
      {
         int a[3], slot[3]; Coordinates(i, n, dim, a);
         int origin = -1;
         for (int v = 0; v < (1 << dim); ++v)
         {
            bool belongs = true;
            for (int d = 0; d < dim; ++d)
            { if ((a[d] == 0 || a[d] == n - 1) && ((v >> d) & 1) != (a[d] == n - 1)) { belongs = false; } }
            if (belongs && (origin < 0 || corners[v] < corners[origin])) { origin = v; }
         }
         int sign = 1;
         for (int d = 0; d < dim; ++d)
         {
            slot[d] = a[d];
            if (a[d] > 0 && a[d] < n - 1 && ((origin >> d) & 1))
            { slot[d] = n - 1 - a[d]; sign *= basis->parity[a[d]]; }
         }
         const int k = e * nd + i;
         modal[k] = nodal[e * nd + Index(slot, n, dim)]; signs[k] = sign;
         nodal_mask[k] = !nseen[nodal[k]]; nseen[nodal[k]] = 1;
         modal_mask[k] = !mseen[modal[k]]; mseen[modal[k]] = 1;
      }
   }
}

void BasisChange::Apply(const Vector &x, Vector &y, bool invert,
                        bool transpose) const
{
   const auto &in = invert ? nodal : modal;
   const auto &out = invert ? modal : nodal;
   const auto &mask = invert ? modal_mask : nodal_mask;
   const DenseMatrix &s = invert ? basis->inverse : basis->S;
   const std::array<const DenseMatrix*, 3> matrices = {{&s, &s, &s}};
   local_in.SetSize(fes.GetVSize()); local_out.SetSize(fes.GetVSize());
   if (transpose) { fes.GetRestrictionMatrix()->MultTranspose(x, local_in); }
   else { fes.GetProlongationMatrix()->Mult(x, local_in); }
   local_out = 0.; element.SetSize(nd);
   for (int e = 0; e < fes.GetNE(); ++e)
   {
      for (int i = 0; i < nd; ++i)
      {
         const int k = e * nd + i;
         element[i] = transpose ? local_in[out[k]] * mask[k] * (invert ? signs[k] : 1)
                      : local_in[in[k]] * (invert ? 1 : signs[k]);
      }
      Tensor(element, result, matrices, dim, transpose, scratch1, scratch2);
      for (int i = 0; i < nd; ++i)
      {
         const int k = e * nd + i;
         if (transpose) { local_out[in[k]] += result[i] * (invert ? 1 : signs[k]); }
         else if (mask[k]) { local_out[out[k]] = result[i] * (invert ? signs[k] : 1); }
      }
   }
   y.SetSize(height);
   if (transpose) { fes.GetProlongationMatrix()->MultTranspose(local_out, y); }
   else { fes.GetRestrictionMatrix()->Mult(local_out, y); }
}

struct Basis::AssemblyData
{
   struct Contribution { int row, column, slot; real_t weight; };
   struct Stencil
   {
      std::vector<int> offsets;
      std::vector<Contribution> entries;
   };
   DenseMatrix values, deriv;
   IntegrationRule quadrature;
   SparsePattern pattern;
   CondensationPlan condensation;
   std::array<Stencil, 3> stencils;

   AssemblyData(const Basis &basis, int dim)
   {
      const int p = basis.p, n = p + 1, nd = Power(n, dim);
      IntegrationRules rules(0, Quadrature1D::GaussLobatto);
      const IntegrationRule &ir = rules.Get(Geometry::SEGMENT, 3 * p);
      const int nq = ir.GetNPoints();
      H1_SegmentElement fe(p);
      const auto &map = fe.GetDofMap();
      DenseMatrix gi(n), broken_coeff(n);
      values.SetSize(n, nq); deriv.SetSize(p, nq);
      DenseMatrixInverse inverse_g(basis.G); inverse_g.GetInverseMatrix(gi);
      Mult(basis.S, gi, broken_coeff);
      Vector shape(n); DenseMatrix dshape(n, 1);
      for (int q = 0; q < nq; ++q)
      {
         fe.CalcShape(ir.IntPoint(q), shape); fe.CalcDShape(ir.IntPoint(q), dshape);
         for (int i = 0; i < n; ++i)
         {
            real_t v = 0;
            for (int j = 0; j < n; ++j) { v += shape[map[j]] * broken_coeff(j, i); }
            values(i, q) = v * v;
         }
         deriv(0, q) = 1.;
         for (int i = 1; i < p; ++i)
         {
            real_t v = 0;
            for (int j = 0; j < n; ++j) { v += dshape(map[j], 0) * basis.S(j, i); }
            deriv(i, q) = v * v / basis.eigenvalues[i - 1];
         }
      }

      quadrature.SetSize(Power(nq, dim));
      for (int q = 0; q < quadrature.GetNPoints(); ++q)
      {
         int a[3] = {0, 0, 0}; Coordinates(q, nq, dim, a);
         auto &ip = quadrature.IntPoint(q);
         ip.Init(q);
         ip.Set3(ir.IntPoint(a[0]).x, ir.IntPoint(a[1]).x,
                 dim == 3 ? ir.IntPoint(a[2]).x : 0.);
         ip.weight = 1.;
         for (int d = 0; d < dim; ++d) { ip.weight *= ir.IntPoint(a[d]).weight; }
      }
      std::vector<std::vector<int>> rows(nd);
      for (int axis = 0; axis < dim; ++axis)
      {
         auto &stencil = stencils[axis];
         stencil.offsets.push_back(0);
         for (int r = 0; r < p * Power(n, dim - 1); ++r)
         {
            int t = r;
            std::vector<std::pair<int, real_t>> entries(1, {0, 1.});
            int stride = 1;
            for (int d = 0; d < dim; ++d)
            {
               const int count = d == axis ? p : n, row = t % count; t /= count;
               const DenseMatrix &factor = d == axis ? basis.derivative : basis.G;
               std::vector<std::pair<int, real_t>> next;
               for (int col = 0; col < n; ++col) if (factor(row, col) != 0.)
                     for (const auto &entry : entries)
                     { next.emplace_back(entry.first + stride * col, entry.second * factor(row, col)); }
               entries.swap(next); stride *= n;
            }

            for (const auto &a : entries) for (const auto &b : entries)
               {
                  rows[a.first].push_back(b.first);
                  stencil.entries.push_back({a.first, b.first, -1, a.second * b.second});
               }
            stencil.offsets.push_back(int(stencil.entries.size()));
         }
      }
      pattern = SparsePattern(std::move(rows));
      for (int axis = 0; axis < dim; ++axis)
         for (auto &entry : stencils[axis].entries)
         { entry.slot = pattern.Slot(entry.row, entry.column); }
      condensation = CondensationPlan(pattern, n, dim);
   }
};

const Basis::AssemblyData &Basis::GetAssemblyData(int dimension) const
{
   MFEM_VERIFY(dimension == 2 ||
               dimension == 3, "Expected quadrilaterals/hexahedra");
   auto &data = assembly[dimension - 2];
   if (!data) { data = std::make_shared<AssemblyData>(*this, dimension); }
   return *data;
}

std::unique_ptr<SparseMatrix> AssembleElement(
   const Basis &basis, ElementTransformation &trans, Coefficient &coefficient)
{
   const int dim = trans.GetDimension();
   const auto &data = basis.GetAssemblyData(dim);
   std::array<Vector, 3> metric;
   for (int d = 0; d < dim; ++d) { metric[d].SetSize(data.quadrature.GetNPoints()); }
   for (int q = 0; q < data.quadrature.GetNPoints(); ++q)
   {
      const auto &ip = data.quadrature.IntPoint(q);
      trans.SetIntPoint(&ip);
      const real_t alpha = coefficient.Eval(trans, ip);
      MFEM_VERIFY(alpha > 0. && std::isfinite(alpha) && trans.Jacobian().Det() > 0.,
                  "Expected positive diffusion and a positively oriented element");
      const DenseMatrix &inv = trans.InverseJacobian();
      for (int d = 0; d < dim; ++d)
      {
         real_t norm = 0.;
         for (int j = 0; j < dim; ++j) { norm += inv(d, j) * inv(d, j); }
         metric[d][q] = ip.weight * trans.Weight() * alpha * norm;
      }
   }
   auto matrix = data.pattern.NewMatrix();
   real_t *av = matrix->GetData();
   Vector diagonal, u, v;
   for (int axis = 0; axis < dim; ++axis)
   {
      std::array<const DenseMatrix*, 3> b = {{&data.values, &data.values, &data.values}};
      b[axis] = &data.deriv;
      Tensor(metric[axis], diagonal, b, dim, false, u, v);
      const auto &stencil = data.stencils[axis];
      for (int r = 0; r < diagonal.Size(); ++r)
      {
         MFEM_VERIFY(diagonal[r] > 0., "Nonpositive auxiliary mass diagonal");
         for (int k = stencil.offsets[r]; k < stencil.offsets[r + 1]; ++k)
         {
            const auto &entry = stencil.entries[k];
            av[entry.slot] += diagonal[r] * entry.weight;
         }
      }
   }
   return matrix;
}

namespace
{
using Global = HYPRE_BigInt;
constexpr int patch_message_tag = 17;
struct Message : VarMessage<patch_message_tag>
{
   void Encode(int) override { }
   void Decode(int) override { }
};
using Messages = std::map<int, Message>;
template <typename T> void Pack(std::string &s, const T &x)
{ s.append(reinterpret_cast<const char*>(&x), sizeof(T)); }
template <typename T> T Unpack(const std::string &s, size_t &pos)
{
   MFEM_VERIFY(pos + sizeof(T) <= s.size(), "Invalid patch message");
   T x; std::memcpy(&x, s.data() + pos, sizeof(T)); pos += sizeof(T); return x;
}
Messages Exchange(Messages &send, MPI_Comm comm)
{
   Messages recv;
   for (auto &entry : send) { recv[entry.first]; }
   Message::IsendAll(send, comm);
   // Receive exactly once from each neighbor. RecvAll probes MPI_ANY_SOURCE
   // and can consume a fast neighbor's next exchange before a slow neighbor's
   // current exchange, since the setup phases reuse the same tag.
   for (auto &entry : recv)
   {
      MPI_Status status;
      MPI_Probe(entry.first, patch_message_tag, comm, &status);
      int size;
      MPI_Get_count(&status, MPI_BYTE, &size);
      entry.second.Recv(entry.first, size, comm);
   }
   Message::WaitAllSent(send);
   // Keep the neighbor keys for the next exchange, but release sent storage.
   for (auto &entry : send) { std::string().swap(entry.second.data); }
   return recv;
}
struct ElementData
{
   std::unique_ptr<SparseMatrix> matrix;
   std::vector<int> lex, dofs, signs;
};

// Injection of trace coefficients into the full modal space; the transpose
// extracts the trace residual. Reuse MFEM's ownership/reduction operators.
class TraceInjection : public Operator
{
   ParFiniteElementSpace &full, &trace;
   Array<int> map;
   mutable Vector f, t;
public:
   TraceInjection(ParFiniteElementSpace &fs, ParFiniteElementSpace &ts,
                  const Array<int> &m)
      : Operator(fs.GetTrueVSize(), ts.GetTrueVSize()), full(fs), trace(ts), map(m) { }
   void Mult(const Vector &x, Vector &y) const override
   {
      t.SetSize(trace.GetVSize()); f.SetSize(full.GetVSize());
      trace.GetProlongationMatrix()->Mult(x, t); f = 0.;
      for (int i = 0; i < map.Size(); ++i) if (map[i] >= 0) { f[i] = t[map[i]]; }
      y.SetSize(height); full.GetRestrictionMatrix()->Mult(f, y);
   }
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      t.SetSize(trace.GetVSize()); f.SetSize(full.GetVSize());
      full.GetRestrictionMatrix()->MultTranspose(x, f); t = 0.;
      for (int i = 0; i < map.Size(); ++i) if (map[i] >= 0) { t[map[i]] += f[i]; }
      y.SetSize(width); trace.GetProlongationMatrix()->MultTranspose(t, y);
   }
};

struct PatchAssembly
{
   std::map<Global, int> index;
   std::vector<Global> global;
   std::vector<bool> interior;
   Rows rows;
   int Insert(Global g, bool cell)
   {
      const auto it = index.find(g);
      if (it != index.end()) { return it->second; }
      const int i = int(global.size()); index[g] = i;
      global.push_back(g); interior.push_back(cell); rows.emplace_back(); return i;
   }
};
void ReceivePatches(const std::string &s,
                    std::map<Global, PatchAssembly> &patches)
{
   size_t pos = 0;
   while (pos < s.size())
   {
      const Global vertex = Unpack<Global>(s, pos);
      auto &patch = patches[vertex];
      const int n = Unpack<int>(s, pos);
      std::vector<int> local(n);
      for (int i = 0; i < n; ++i)
      {
         Global g = Unpack<Global>(s, pos); int interior = Unpack<int>(s, pos);
         local[i] = patch.Insert(g, interior != 0);
      }
      const int nz = Unpack<int>(s, pos);
      for (int k = 0; k < nz; ++k)
      {
         const int i = Unpack<int>(s, pos), j = Unpack<int>(s, pos);
         const real_t value = Unpack<real_t>(s, pos);
         patch.rows[local[i]][local[j]] += value;
      }
   }
}
Array<int> PatchOrdering(const Rows &pattern, const std::vector<bool> &interior)
{
   std::vector<int> interface, inverse(pattern.size(), -1);
   Array<int> permutation(int(pattern.size())); int first = 0;
   for (int i = 0; i < int(pattern.size()); ++i)
   {
      if (interior[i]) { permutation[first++] = i; }
      else { inverse[i] = int(interface.size()); interface.push_back(i); }
   }
   idx_t n = idx_t(interface.size());
   if (n > 0)
   {
      std::vector<idx_t> offsets(n + 1), columns;
      for (int i = 0; i < n; ++i)
      {
         std::vector<idx_t> row;
         for (const auto &entry : pattern[interface[i]])
         { if (inverse[entry.first] >= 0 && entry.first != interface[i]) { row.push_back(inverse[entry.first]); } }
         std::sort(row.begin(), row.end());
         columns.insert(columns.end(), row.begin(), row.end());
         offsets[i + 1] = idx_t(columns.size());
      }
      std::vector<idx_t> perm(n), iperm(n);
      idx_t options[METIS_NOPTIONS]; METIS_SetDefaultOptions(options);
      options[METIS_OPTION_NUMBERING] = 0; options[METIS_OPTION_SEED] = 0;
      if (n > 1 && !columns.empty())
      {
         MFEM_VERIFY(METIS_NodeND(&n, offsets.data(), columns.data(), nullptr, options,
                                  perm.data(), iperm.data()) == METIS_OK, "METIS ordering failed");
      }
      else { for (int i = 0; i < n; ++i) { perm[i] = i; } }
      for (int i = 0; i < n; ++i) { permutation[first + i] = interface[perm[i]]; }
   }
   return permutation;
}

class PatchSolver : public Solver
{
   struct Patch
   {
      std::unique_ptr<IncompleteCholeskySolver> factor;
      std::vector<int> indices;
      mutable Vector rhs, solution;
   };
   struct Peer
   {
      std::vector<int> requested, supplied;
      mutable std::vector<real_t> receive, send;
   };
   MPI_Comm comm;
   int rank;
   Global offset;
   std::vector<Patch> patches;
   std::vector<int> local_patches, remote_patches;
   std::map<int, Peer> peers;
   std::vector<int> local_indices;
   mutable Vector extended, correction;
   mutable std::vector<MPI_Request> requests;
   real_t damping = 1.;
   void ApplyPatch(const Patch &patch) const
   {
      for (int i = 0; i < int(patch.indices.size()); ++i)
      { patch.rhs[i] = extended[patch.indices[i]]; }
      patch.factor->Mult(patch.rhs, patch.solution);
      for (int i = 0; i < int(patch.indices.size()); ++i)
      { correction[patch.indices[i]] += patch.solution[i]; }
   }
public:
   long long factor_nonzeros = 0, patch_unknowns = 0, communication_values = 0;
   long long NumPatches() const { return patches.size(); }
   real_t GetDamping() const { return damping; }
   void SetDamping(real_t value) { damping = value; }
   PatchSolver(ParFiniteElementSpace &fes,
               const std::vector<ElementData> &elements,
               const Array<int> &ess_bdr, int p, const Options &options)
      : Solver(fes.GetTrueVSize()), offset(fes.GetMyTDofOffset())
   {
      MPI_Comm_dup(fes.GetComm(), &comm); MPI_Comm_rank(comm, &rank);
      ParMesh &mesh = *fes.GetParMesh(); const int dim = mesh.Dimension(), n = p + 1;
      const auto &gt = mesh.gtopo;
      Messages send;
      for (int i = 1; i < gt.GetNumNeighbors(); ++i) { send[gt.GetNeighborRank(i)]; }
      // Only neighboring ownership ranges are needed, even at vertex-only contacts.
      for (auto &entry : send)
      { Pack(entry.second.data, offset); Pack(entry.second.data, offset + Global(height)); }
      auto ranges_received = Exchange(send, comm);
      std::map<int, std::pair<Global, Global>> ranges;
      ranges[rank] = {offset, offset + height};
      for (auto &entry : ranges_received)
      {
         size_t pos = 0; Global begin = Unpack<Global>(entry.second.data, pos);
         Global end = Unpack<Global>(entry.second.data, pos); ranges[entry.first] = {begin, end};
      }
      ranges_received.clear();
      Array<Global> vertices; mesh.GetGlobalVertexIndices(vertices);
      Array<int> owner(mesh.GetNV()); owner = rank;
      for (int g = 1; g < gt.NGroups(); ++g)
      {
         int lowest = rank;
         for (int k = 0; k < gt.GetGroupSize(g); ++k)
         { lowest = std::min(lowest, gt.GetNeighborRank(gt.GetGroup(g)[k])); }
         for (int k = 0; k < mesh.GroupNVertices(g); ++k) { owner[mesh.GroupVertex(g, k)] = lowest; }
      }
      Array<int> essential; fes.GetEssentialVDofs(ess_bdr, essential);
      Array<int> ev;
      std::string local_records;
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         const auto &el = elements[e];
         const int *ai = el.matrix->GetI(), *aj = el.matrix->GetJ();
         const real_t *av = el.matrix->GetData();
         mesh.GetElementVertices(e, ev);
         const auto *vr = Geometries.GetVertices(mesh.GetElementGeometry(e));
         for (int vertex = 0; vertex < ev.Size(); ++vertex)
         {
            const auto &ip = vr->IntPoint(vertex);
            const int corner[3] = {int(ip.x)*p, int(ip.y)*p, int(ip.z)*p};
            std::vector<int> selected, inverse(el.lex.size(), -1);
            for (int i = 0; i < int(el.lex.size()); ++i)
            {
               int a[3]; Coordinates(el.lex[i], n, dim, a);
               bool inside = !essential[el.dofs[i]];
               for (int d = 0; d < dim; ++d)
               { if ((a[d] == 0 || a[d] == p) && a[d] != corner[d]) { inside = false; } }
               if (inside) { inverse[i] = int(selected.size()); selected.push_back(i); }
            }
            if (selected.empty()) { continue; }
            const int root = owner[ev[vertex]];
            std::string &buffer = root == rank ? local_records : send[root].data;
            Pack(buffer, vertices[ev[vertex]]); Pack(buffer, int(selected.size()));
            for (int i : selected)
            { Pack(buffer, fes.GetGlobalTDofNumber(el.dofs[i])); Pack(buffer, int(Interior(el.lex[i], n, dim))); }
            int nz = 0;
            for (int i : selected) for (int k = ai[i]; k < ai[i + 1]; ++k)
               { if (inverse[aj[k]] >= 0) { ++nz; } }
            Pack(buffer, nz);
            for (int i : selected) for (int k = ai[i]; k < ai[i + 1]; ++k)
               {
                  const int j = aj[k];
                  if (inverse[j] >= 0)
                  { Pack(buffer, inverse[i]); Pack(buffer, inverse[j]); Pack(buffer, av[k]*el.signs[i]*el.signs[j]); }
               }
         }
      }
      auto received = Exchange(send, comm);
      std::map<Global, PatchAssembly> assembled;
      ReceivePatches(local_records, assembled);
      std::string().swap(local_records);
      for (auto &entry : received)
      {
         ReceivePatches(entry.second.data, assembled);
         std::string().swap(entry.second.data);
      }
      received.clear();
      std::map<Global, int> all_dofs;
      for (const auto &entry : assembled) for (Global g : entry.second.global) { all_dofs.emplace(g, 0); }
      int count = 0;
      for (auto &entry : all_dofs) { entry.second = count++; }
      extended.SetSize(count); correction.SetSize(count);
      local_indices.resize(count, -1);
      for (const auto &entry : all_dofs)
      {
         if (entry.first >= offset && entry.first < offset + height)
         { local_indices[entry.second] = int(entry.first - offset); }
         else
         {
            int owner_rank = -1;
            for (const auto &r : ranges)
            { if (entry.first >= r.second.first && entry.first < r.second.second) { owner_rank = r.first; break; } }
            MFEM_VERIFY(owner_rank >= 0, "Patch DOF owner is not a vertex neighbor");
            peers[owner_rank].requested.push_back(entry.second);
            Pack(send[owner_rank].data, entry.first);
         }
      }
      auto incoming_requests = Exchange(send, comm);
      for (const auto &entry : incoming_requests)
      {
         auto &peer = peers[entry.first]; size_t pos = 0;
         while (pos < entry.second.data.size())
         {
            const Global g = Unpack<Global>(entry.second.data, pos);
            MFEM_VERIFY(g >= offset && g < offset + height, "Incorrect patch DOF owner");
            peer.supplied.push_back(int(g - offset));
         }
      }
      incoming_requests.clear();
      for (auto &entry : peers)
      {
         auto &peer = entry.second;
         peer.receive.resize(peer.requested.size());
         peer.send.resize(peer.supplied.size());
         communication_values += 2 * peer.supplied.size();
      }
      requests.resize(2 * peers.size());
      for (auto it = assembled.begin(); it != assembled.end(); )
      {
         auto &a = it->second;
         auto matrix = CSR(a.rows);
         // Keep precisely the fill induced by eliminating cell interiors.
         // Condensed patches already have this graph; do not copy it again.
         Rows filled_pattern;
         std::unique_ptr<SparseMatrix> graph;
         if (!options.static_condensation)
         {
            filled_pattern = a.rows;
            for (int i = 0; i < int(a.rows.size()); ++i) if (a.interior[i])
                  for (const auto &j : a.rows[i]) if (!a.interior[j.first])
                        for (const auto &k : a.rows[i]) if (!a.interior[k.first])
                           { filled_pattern[j.first].emplace(k.first, 0.); }
            graph = CSR(filled_pattern);
         }
         Array<int> ordering = PatchOrdering(
                                  options.static_condensation ? a.rows : filled_pattern, a.interior);
         Patch patch; patch.factor.reset(new IncompleteCholeskySolver);
         const int failed = patch.factor->Factor(*matrix, graph ? *graph : *matrix,
                                                 ordering,
                                                 options.diagonal_shift);
         MFEM_VERIFY(failed < 0, "ICC nonpositive pivot " << failed <<
                     " in vertex patch "
                     << it->first << " on rank " << rank <<
                     "; inspect mesh/coefficient or set --diagonal-shift");
         for (Global g : a.global) { patch.indices.push_back(all_dofs.at(g)); }
         patch.rhs.SetSize(int(a.global.size()));
         patch.solution.SetSize(int(a.global.size()));
         factor_nonzeros += patch.factor->NumNonZeroElems();
         patch_unknowns += a.global.size();
         bool local = true;
         for (int i : patch.indices) { if (local_indices[i] < 0) { local = false; break; } }
         (local ? local_patches : remote_patches).push_back(int(patches.size()));
         patches.push_back(std::move(patch));
         // The factor and DOF indices now own everything needed by this patch.
         it = assembled.erase(it);
      }
   }
   ~PatchSolver() override { MPI_Comm_free(&comm); }
   void SetOperator(const Operator &) override { MFEM_ABORT("Reconstruct PatchSolver"); }
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void Mult(const Vector &x, Vector &y) const override
   {
      int k = 0;
      for (auto &entry : peers)
      {
         auto &peer = entry.second;
         for (int i = 0; i < int(peer.supplied.size()); ++i) { peer.send[i] = x[peer.supplied[i]]; }
         MPI_Irecv(peer.receive.data(), int(peer.receive.size()),
                   MPITypeMap<real_t>::mpi_type,
                   entry.first, 1, comm, &requests[k++]);
         MPI_Isend(peer.send.data(), int(peer.send.size()), MPITypeMap<real_t>::mpi_type,
                   entry.first, 1, comm, &requests[k++]);
      }
      for (int i = 0; i < extended.Size(); ++i)
      { if (local_indices[i] >= 0) { extended[i] = x[local_indices[i]]; } }
      correction = 0.;
      // These patches need no incoming values and can run during the gather.
      for (int i : local_patches) { ApplyPatch(patches[i]); }
      MPI_Waitall(k, requests.data(), MPI_STATUSES_IGNORE);
      for (const auto &entry : peers)
      {
         const auto &peer = entry.second;
         for (int i = 0; i < int(peer.requested.size()); ++i) { extended[peer.requested[i]] = peer.receive[i]; }
      }
      for (int i : remote_patches) { ApplyPatch(patches[i]); }
      k = 0;
      for (auto &entry : peers)
      {
         auto &peer = entry.second;
         for (int i = 0; i < int(peer.requested.size()); ++i) { peer.receive[i] = correction[peer.requested[i]]; }
         MPI_Irecv(peer.send.data(), int(peer.send.size()), MPITypeMap<real_t>::mpi_type,
                   entry.first, 2, comm, &requests[k++]);
         MPI_Isend(peer.receive.data(), int(peer.receive.size()),
                   MPITypeMap<real_t>::mpi_type,
                   entry.first, 2, comm, &requests[k++]);
      }
      y.SetSize(height); y = 0.;
      for (int i = 0; i < correction.Size(); ++i)
      { if (local_indices[i] >= 0) { y[local_indices[i]] += correction[i]; } }
      MPI_Waitall(k, requests.data(), MPI_STATUSES_IGNORE);
      for (const auto &entry : peers)
      {
         const auto &peer = entry.second;
         for (int i = 0; i < int(peer.supplied.size()); ++i) { y[peer.supplied[i]] += peer.send[i]; }
      }
      y *= damping;
   }
};

class CoarseTransfer : public Operator
{
   const Operator &interpolation;
   const BasisChange &basis;
   const TraceInjection *trace;
   Array<int> fine_essential, coarse_essential;
   mutable Vector a, b, c;
public:
   CoarseTransfer(const Operator &p, const BasisChange &s, const TraceInjection *t,
                  const Array<int> &fine, const Array<int> &coarse)
      : Operator(t ? t->Width() : s.Height(), p.Width()), interpolation(p), basis(s),
        trace(t),
        fine_essential(fine), coarse_essential(coarse) { }
   void Mult(const Vector &x, Vector &y) const override
   {
      a = x; a.SetSubVector(coarse_essential, 0.);
      b.SetSize(basis.Height()); interpolation.Mult(a, b); basis.InverseMult(b, c);
      if (trace) { trace->MultTranspose(c, y); }
      else { y = c; }
      y.SetSubVector(fine_essential, 0.);
   }
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      a = x; a.SetSubVector(fine_essential, 0.);
      if (trace) { trace->Mult(a, b); }
      else { b = a; }
      basis.InverseMultTranspose(b, c); y.SetSize(width);
      interpolation.MultTranspose(c, y);
      y.SetSubVector(coarse_essential, 0.);
   }
};
} // namespace

std::unique_ptr<SparseMatrix> CondenseElement(
   const SparseMatrix &matrix, int degree, int dimension,
   std::vector<int> &boundary)
{
   MFEM_VERIFY(degree >= 1 && (dimension == 2 || dimension == 3) &&
               matrix.Height() == Power(degree + 1, dimension) &&
               matrix.Width() == matrix.Height(), "Invalid FDM element dimensions");
   boundary.clear();
   const auto &cached = Basis::Get(degree)->GetAssemblyData(
                           dimension).condensation;
   if (cached.source.Matches(matrix))
   {
      boundary = cached.boundary;
      return cached.Apply(matrix);
   }
   // The public API also accepts matrices with a different sparse graph.
   const CondensationPlan plan(SparsePattern(matrix), degree + 1, dimension);
   boundary = plan.boundary;
   return plan.Apply(matrix);
}

struct Preconditioner::Impl
{
   ParFiniteElementSpace &fes;
   MPI_Comm comm;
   Options options;
   Array<int> essential, active_essential;
   std::shared_ptr<const Basis> reference;
   std::unique_ptr<BasisChange> basis;
   std::unique_ptr<H1_FECollection> coarse_collection;
   std::unique_ptr<FiniteElementCollection> trace_collection;
   std::unique_ptr<ParFiniteElementSpace> coarse_space, trace_space;
   std::unique_ptr<ParBilinearForm> coarse_form;
   std::unique_ptr<HypreParMatrix> coarse_matrix, auxiliary, coupling;
   std::unique_ptr<HypreBoomerAMG> coarse_solver;
   std::unique_ptr<TraceInjection> trace;
   std::unique_ptr<PRefinementTransferOperator> interpolation;
   std::unique_ptr<CoarseTransfer> transfer;
   std::unique_ptr<PatchSolver> patches;
   std::unique_ptr<Multigrid> multigrid;
   Vector inverse_interior;
   mutable Vector rhs, interior, reduced, trace_solution, solution, temp;
   long long auxiliary_nonzeros = 0;
   int degree;

   Impl(ParFiniteElementSpace &space, Coefficient &coefficient,
        const Array<int> &ess_bdr, const Options &opts)
      : fes(space), comm(space.GetComm()), options(opts),
        degree(space.GetMaxElementOrder())
   {
      ParMesh &mesh = *space.GetParMesh(); const int dim = mesh.Dimension();
      const auto *collection = dynamic_cast<const H1_FECollection*>(space.FEColl());
      MFEM_VERIFY((dim == 2 || dim == 3) && mesh.SpaceDimension() == dim &&
                  !space.Nonconforming() && !space.IsVariableOrder() &&
                  space.GetVDim() == 1 && collection &&
                  collection->GetBasisType() == BasisType::GaussLobatto,
                  "Vertex patches require conforming scalar GLL H1 with uniform degree in 2D/3D");
      MFEM_VERIFY(options.damping >= 0. && std::isfinite(options.damping) &&
                  options.diagonal_shift >= 0. && std::isfinite(options.diagonal_shift) &&
                  options.power_iterations > 0, "Invalid vertex-patch options");
      int wrong_geometry = 0;
      for (int e = 0; e < mesh.GetNE(); ++e)
      { if (mesh.GetElementGeometry(e) != TensorBasisElement::GetTensorProductGeometry(dim)) { wrong_geometry = 1; } }
      MPI_Allreduce(MPI_IN_PLACE, &wrong_geometry, 1, MPI_INT, MPI_MAX, comm);
      MFEM_VERIFY(!wrong_geometry, "Only quadrilaterals/hexahedra are supported");
      fes.GetEssentialTrueDofs(ess_bdr, essential);
      long long nessential = essential.Size();
      MPI_Allreduce(MPI_IN_PLACE, &nessential, 1, MPI_LONG_LONG, MPI_SUM, comm);
      MFEM_VERIFY(nessential > 0,
                  "Pure Neumann problems require nullspace handling and are not supported");
      coarse_collection.reset(new H1_FECollection(1, dim));
      coarse_space.reset(new ParFiniteElementSpace(&mesh, coarse_collection.get()));
      Array<int> coarse_essential;
      coarse_space->GetEssentialTrueDofs(ess_bdr, coarse_essential);
      coarse_form.reset(new ParBilinearForm(coarse_space.get()));
      coarse_form->AddDomainIntegrator(new DiffusionIntegrator(coefficient));
      coarse_form->Assemble();
      coarse_matrix.reset(new HypreParMatrix);
      coarse_form->FormSystemMatrix(coarse_essential, *coarse_matrix);
      coarse_solver.reset(new HypreBoomerAMG(*coarse_matrix));
      coarse_solver->SetPrintLevel(0); coarse_solver->SetMaxIter(1);
      coarse_solver->SetTol(0.); coarse_solver->SetRelaxType(6);
      if (degree == 1) { return; }
      reference = Basis::Get(degree);
      basis.reset(new BasisChange(fes, reference));
      const int n = degree + 1, nd = Power(n, dim);
      ParFiniteElementSpace *active = &fes;
      Array<int> full_to_trace(fes.GetVSize()); full_to_trace = -1;
      if (options.static_condensation)
      {
         trace_collection.reset(fes.FEColl()->GetTraceCollection());
         trace_space.reset(new ParFiniteElementSpace(&mesh, trace_collection.get()));
         Array<int> fd, td;
         for (int e = 0; e < mesh.GetNE(); ++e)
         {
            fes.GetElementDofs(e, fd); trace_space->GetElementDofs(e, td);
            for (int i = 0; i < td.Size(); ++i) { full_to_trace[fd[i]] = td[i]; }
         }
         trace.reset(new TraceInjection(fes, *trace_space, full_to_trace));
         active = trace_space.get();
      }
      active->GetEssentialTrueDofs(ess_bdr, active_essential);
      Rows assembled(active->GetVSize());
      Rows cross(options.static_condensation ? active->GetVSize() : 0);
      Vector local_inverse(fes.GetVSize()); local_inverse = 0.;
      std::vector<ElementData> elements(mesh.GetNE());
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         auto matrix = AssembleElement(*reference, *mesh.GetElementTransformation(e),
                                       coefficient);
         auto &el = elements[e];
         if (options.static_condensation)
         {
            const int *ai = matrix->GetI(), *aj = matrix->GetJ();
            const real_t *av = matrix->GetData();
            for (int i = 0; i < nd; ++i) if (Interior(i, n, dim))
               {
                  const int col = basis->ModalMap()[e * nd + i],
                            sign = basis->Signs()[e * nd + i];
                  for (int k = ai[i]; k < ai[i + 1]; ++k)
                  {
                     const int j = aj[k];
                     if (j == i) { local_inverse[col] = 1. / av[k]; }
                     else if (!Interior(j, n, dim))
                     {
                        const int row = full_to_trace[basis->ModalMap()[e * nd + j]];
                        cross[row][col] += av[k] * sign * basis->Signs()[e * nd + j];
                     }
                  }
               }
            el.matrix = CondenseElement(*matrix, degree, dim, el.lex);
         }
         else
         {
            el.matrix = std::move(matrix); el.lex.resize(nd);
            for (int i = 0; i < nd; ++i) { el.lex[i] = i; }
         }
         for (int i : el.lex)
         {
            int dof = basis->ModalMap()[e * nd + i];
            if (options.static_condensation) { dof = full_to_trace[dof]; }
            MFEM_VERIFY(dof >= 0, "Invalid trace mapping");
            el.dofs.push_back(dof); el.signs.push_back(basis->Signs()[e * nd + i]);
         }
         const int *ai = el.matrix->GetI(), *aj = el.matrix->GetJ();
         const real_t *av = el.matrix->GetData();
         for (int i = 0; i < int(el.lex.size()); ++i)
            for (int k = ai[i]; k < ai[i + 1]; ++k)
            { assembled[el.dofs[i]][el.dofs[aj[k]]] += av[k] * el.signs[i] * el.signs[aj[k]]; }
      }
      auto local_matrix = CSR(assembled);
      Rows().swap(assembled);
      ParBilinearForm form(active);
      auxiliary.reset(form.ParallelAssemble(local_matrix.get()));
      local_matrix.reset();
      delete auxiliary->EliminateRowsCols(active_essential);
      auxiliary_nonzeros = auxiliary->NNZ();
      if (options.static_condensation)
      {
         auto local_cross = CSR(cross, fes.GetVSize());
         Rows().swap(cross);
         ParMixedBilinearForm mixed(&fes, active);
         coupling.reset(mixed.ParallelAssemble(local_cross.get()));
         inverse_interior.SetSize(fes.GetTrueVSize());
         fes.GetRestrictionMatrix()->Mult(local_inverse, inverse_interior);
      }
      patches.reset(new PatchSolver(*active, elements, ess_bdr, degree, options));
      long long constrained = active_essential.Size();
      MPI_Allreduce(MPI_IN_PLACE, &constrained, 1, MPI_LONG_LONG, MPI_SUM, comm);
      const bool no_free_trace = (active->GlobalTrueVSize() == constrained);
      real_t damping = no_free_trace ? 1. : options.damping;
      if (damping == 0.)
      {
         ProductOperator product(patches.get(), auxiliary.get(), false, false);
         PowerMethod power(comm); Vector eigenvector(active->GetTrueVSize());
         const real_t largest = power.EstimateLargestEigenvalue(
                                   product, eigenvector, options.power_iterations, 1.e-6, 12345);
         MFEM_VERIFY(largest > 0. &&
                     std::isfinite(largest), "Invalid smoother eigenvalue estimate");
         damping = 1. / (1.2 * largest);
      }
      patches->SetDamping(damping);
      interpolation.reset(new PRefinementTransferOperator(*coarse_space, fes, true));
      transfer.reset(new CoarseTransfer(*interpolation->GetTrueTransferOperator(),
                                        *basis, trace.get(), active_essential, coarse_essential));
      Array<Operator*> operators(2), prolongations(1);
      Array<Solver*> smoothers(2);
      Array<bool> own(2), own_p(1); own = false; own_p = false;
      operators[0] = coarse_matrix.get(); operators[1] = auxiliary.get();
      smoothers[0] = coarse_solver.get(); smoothers[1] = patches.get();
      prolongations[0] = transfer.get();
      multigrid.reset(new Multigrid(operators, smoothers, prolongations, own, own,
                                    own_p));
      multigrid->SetCycleType(Multigrid::CycleType::VCYCLE, 1, 1);
   }
   void Mult(const Vector &x, Vector &y) const
   {
      if (degree == 1) { y.SetSize(x.Size()); coarse_solver->Mult(x, y); return; }
      temp = x; temp.SetSubVector(essential, 0.);
      basis->MultTranspose(temp, rhs);
      if (options.static_condensation)
      {
         interior.SetSize(rhs.Size());
         for (int i = 0; i < rhs.Size(); ++i) { interior[i] = inverse_interior[i] * rhs[i]; }
         trace->MultTranspose(rhs, reduced);
         temp.SetSize(reduced.Size()); coupling->Mult(interior, temp); reduced -= temp;
         reduced.SetSubVector(active_essential, 0.);
         trace_solution.SetSize(reduced.Size());
         multigrid->Mult(reduced, trace_solution);
         trace->Mult(trace_solution, solution);
         temp.SetSize(rhs.Size()); coupling->MultTranspose(trace_solution, temp);
         for (int i = 0; i < rhs.Size(); ++i)
         { solution[i] += interior[i] - inverse_interior[i] * temp[i]; }
      }
      else
      {
         rhs.SetSubVector(essential, 0.);
         solution.SetSize(rhs.Size()); multigrid->Mult(rhs, solution);
      }
      basis->Mult(solution, y);
      for (int i : essential) { y[i] = x[i]; }
   }
};

Preconditioner::Preconditioner(ParFiniteElementSpace &space,
                               Coefficient &coefficient,
                               const Array<int> &ess_bdr, const Options &options)
   : Solver(space.GetTrueVSize()), impl(new Impl(space, coefficient, ess_bdr,
                                                 options)) { }
Preconditioner::~Preconditioner() = default;
void Preconditioner::Mult(const Vector &x, Vector &y) const { impl->Mult(x, y); }
void Preconditioner::PrintStatistics(std::ostream &os) const
{
   long long stats[4] = {0, 0, 0, 0};
   if (impl->patches)
   {
      stats[0] = impl->patches->NumPatches();
      stats[1] = impl->patches->patch_unknowns;
      stats[2] = impl->patches->factor_nonzeros;
      stats[3] = impl->patches->communication_values;
   }
   MPI_Allreduce(MPI_IN_PLACE, stats, 4, MPI_LONG_LONG, MPI_SUM, impl->comm);
   int rank; MPI_Comm_rank(impl->comm, &rank);
   if (rank == 0)
   {
      os << "Vertex patches: " << stats[0] << ", patch unknowns: " << stats[1]
         << ", factor nonzeros: " << stats[2] << '\n'
         << "Auxiliary nonzeros: " << impl->auxiliary_nonzeros << '\n'
         << "Factor arrays including solve workspace (bytes): "
         << (stats[2] + stats[1])*(sizeof(real_t) + sizeof(int)) + stats[0]*sizeof(
            int) << '\n'
         << "Patch exchange values per smoothing step (gather + scatter): " << stats[3]
         << '\n'
         << "Smoother damping: " << (impl->patches ? impl->patches->GetDamping() : 1.) <<
         std::endl;
   }
}

} // namespace vertex_patch
} // namespace mfem
#endif
