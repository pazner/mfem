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

#include "vertex_patch_solver.hpp"
#include "../../general/forall.hpp"
#if defined(MFEM_USE_MPI) && defined(MFEM_USE_LAPACK) && defined(MFEM_USE_METIS) && defined(MFEM_USE_METIS_5)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <array>
#include <limits>
#include <tuple>
#ifdef MFEM_USE_CUDA
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#endif
#include "../../linalg/kernels.hpp"
#include <metis.h>

namespace mfem
{
namespace vertex_patch
{
namespace
{
#include "vertex_patch_kernels.hpp"

MFEM_HOST_DEVICE int Power(int n, int d) { int v = 1; while (d--) { v *= n; } return v; }
MFEM_HOST_DEVICE void Coordinates(int i, int n, int dim, int *a)
{ for (int d = 0; d < dim; ++d) { a[d] = i % n; i /= n; } }
MFEM_HOST_DEVICE int Index(const int *a, int n, int dim)
{ int i = 0; for (int d = dim - 1; d >= 0; --d) { i = i * n + a[d]; } return i; }
MFEM_HOST_DEVICE bool Interior(int i, int n, int dim)
{
   int a[3]; Coordinates(i, n, dim, a);
   for (int d = 0; d < dim; ++d) { if (a[d] == 0 || a[d] == n - 1) { return false; } }
   return true;
}

// Rectangular tensor contraction, with the first coordinate varying fastest.
void Tensor(const Vector &x, Vector &y,
            const std::vector<const DenseMatrix*> &b,
            bool transpose, Vector &u, Vector &v)
{
   const int dim = b.size();
   std::vector<int> sizes(dim);
   for (int d = 0; d < dim; ++d)
   { sizes[d] = transpose ? b[d]->Height() : b[d]->Width(); }
   u = x;
   for (int axis = 0; axis < dim; ++axis)
   {
      const DenseMatrix &m = *b[axis];
      int before = 1, after = 1;
      for (int d = 0; d < axis; ++d) { before *= sizes[d]; }
      for (int d = axis + 1; d < dim; ++d) { after *= sizes[d]; }
      const int ni = sizes[axis], no = transpose ? m.Width() : m.Height();
      v.SetSize(before * no * after);
      for (int k = 0; k < after; ++k)
      {
         DenseMatrix input(u.GetData() + before * ni * k, before, ni);
         DenseMatrix output(v.GetData() + before * no * k, before, no);
         if (transpose) { Mult(input, m, output); }
         else { MultABt(input, m, output); }
      }
      u.Swap(v);
      sizes[axis] = no;
   }
   y = u;
}
}

Basis::Basis(int degree) : p(degree), S(p + 1), inverse(p + 1), G(p + 1),
   derivative(p, p + 1), eigenvalues(p - 1), parity(p + 1)
{
   MFEM_VERIFY(p >= 1, "FDM degree must be positive");
   H1_SegmentElement fe(p);
   // Work in lexicographic GLL order on MFEM's interval [0,1].
   const auto &map = fe.GetDofMap();
   const IntegrationRule &ir = IntRules.Get(Geometry::SEGMENT, 2 * p + 2);
   IsoparametricTransformation identity;
   identity.SetIdentityTransformation(Geometry::SEGMENT);
   DenseMatrix nodal_mass, nodal_stiffness, mass, stiffness;
   MassIntegrator(&ir).AssembleElementMatrix(fe, identity, nodal_mass);
   DiffusionIntegrator(&ir).AssembleElementMatrix(fe, identity, nodal_stiffness);
   nodal_mass.GetSubMatrix(map, mass);
   nodal_stiffness.GetSubMatrix(map, stiffness);
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
{ BuildMaps(); }

void BasisChange::BuildMaps()
{
   BuildTrueDofMap(fes,true_dofs);
   const int n = basis->p + 1, ne = fes.GetNE();
   const auto *restriction = dynamic_cast<const ElementRestriction*>(
                                fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC));
   MFEM_VERIFY(restriction, "Expected conforming H1 element restriction");
   nodal = restriction->GatherMap();
   modal.SetSize(ne*nd); signs.SetSize(ne*nd); mask.SetSize(ne*nd);
   Array<int> first(fes.GetVSize()); first = std::numeric_limits<int>::max();
   Array<HYPRE_BigInt> vertices, corners(ne*(1<<dim));
   fes.GetParMesh()->GetGlobalVertexIndices(vertices);
   Array<int> ev;
   for (int e = 0; e < ne; ++e)
   {
      fes.GetMesh()->GetElementVertices(e,ev);
      const auto *vr = Geometries.GetVertices(fes.GetMesh()->GetElementGeometry(e));
      for (int j = 0; j < ev.Size(); ++j)
      {
         const auto &v = vr->IntPoint(j);
         const int bit = int(v.x)+2*int(v.y)+(dim == 3 ? 4*int(v.z) : 0);
         corners[e*(1<<dim)+bit] = vertices[ev[j]];
      }
   }
   const int *N = nodal.Read(), *parity = basis->parity.Read();
   int *F = first.ReadWrite(), *M = modal.Write(), *S = signs.Write(),
        *K = mask.Write();
   const auto *V = corners.Read(); const int dimension = dim, dofs = nd;
   mfem::forall(ne*nd, [=] MFEM_HOST_DEVICE (int k) { AtomicMinIndex(F[N[k]],k); });
   mfem::forall(ne*nd, [=] MFEM_HOST_DEVICE (int k)
   {
      const int e = k/dofs, i = k%dofs;
      int a[3], slot[3]; Coordinates(i,n,dimension,a);
      int origin = -1;
      for (int v = 0; v < (1<<dimension); ++v)
      {
         bool belongs = true;
         for (int d = 0; d < dimension; ++d)
         { if ((a[d] == 0 || a[d] == n-1) && ((v>>d)&1) != (a[d] == n-1)) { belongs = false; } }
         if (belongs && (origin < 0 ||
                         V[e*(1<<dimension)+v] < V[e*(1<<dimension)+origin])) { origin=v; }
      }
      int sign = 1;
      for (int d = 0; d < dimension; ++d)
      {
         slot[d] = a[d];
         if (a[d] > 0 && a[d] < n-1 && ((origin>>d)&1))
         { slot[d] = n-1-a[d]; sign *= parity[a[d]]; }
      }
      M[k] = N[e*dofs+Index(slot,n,dimension)]; S[k] = sign;
      K[k] = F[N[k]] == k;
   });

}

void BasisChange::Apply(const Vector &x, Vector &y, bool invert,
                        bool transpose) const
{
   const auto &in = invert ? nodal : modal;
   const auto &out = invert ? modal : nodal;
   const DenseMatrix &s = invert ? basis->inverse : basis->S;
   local_in.SetSize(fes.GetVSize()); local_out.SetSize(fes.GetVSize());
   local_in.UseDevice(true); local_out.UseDevice(true);
   if (transpose) { RestrictTrueDofs(true_dofs,x,local_in,true); }
   else { fes.GetProlongationMatrix()->Mult(x, local_in); }
   local_out = 0.;
   const int count = fes.GetNE()*nd, n = basis->p+1, dofs = nd;
   element.SetSize(count); result.SetSize(count);
   element.UseDevice(true); result.UseDevice(true);
   const int *im = in.Read(), *om = out.Read(), *sg = signs.Read(),
              *mk = mask.Read();
   const real_t *li = local_in.Read();
   real_t *el = element.Write();
   mfem::forall(count, [=] MFEM_HOST_DEVICE (int k)
   {
      el[k] = transpose ? li[om[k]]*mk[k]*(invert ? sg[k] : 1)
              : li[im[k]]*(invert ? 1 : sg[k]);
   });
   const real_t *S = s.Read();
   int stride = 1;
   for (int axis = 0; axis < dim; ++axis)
   {
      const real_t *u = element.Read(); real_t *v = result.Write();
      const int step = stride;
      mfem::forall(count, [=] MFEM_HOST_DEVICE (int k)
      {
         const int i = (k % dofs / step) % n;
         real_t value = 0.;
         for (int j = 0; j < n; ++j)
         { value += u[k+(j-i)*step]*S[transpose ? j+n*i : i+n*j]; }
         v[k] = value;
      });
      element.Swap(result); stride *= n;
   }
   const real_t *r = element.Read(); real_t *lo = local_out.ReadWrite();
   mfem::forall(count, [=] MFEM_HOST_DEVICE (int k)
   {
      if (transpose) { AtomicAdd(lo[im[k]], r[k]*(invert ? 1 : sg[k])); }
      else if (mk[k]) { lo[om[k]] = r[k]*(invert ? sg[k] : 1); }
   });
   y.SetSize(height);
   if (transpose) { fes.GetProlongationMatrix()->MultTranspose(local_out, y); }
   else { RestrictTrueDofs(true_dofs,local_out,y,false); }
}

std::unique_ptr<SparseMatrix> AssembleElement(
   const Basis &basis, ElementTransformation &trans, Coefficient &coefficient)
{
   const int p = basis.p, n = p + 1, dim = trans.GetDimension();
   basis.S.HostRead(); basis.G.HostRead(); basis.derivative.HostRead();
   basis.eigenvalues.HostRead();
   IntegrationRules rules(0, Quadrature1D::GaussLobatto);
   const IntegrationRule &ir = rules.Get(Geometry::SEGMENT, 3 * p);
   const int nq = ir.GetNPoints();
   H1_SegmentElement fe(p);
   const auto &map = fe.GetDofMap();
   DenseMatrix gi(n), broken_coeff(n), values(n, nq), deriv(p, nq);
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
   const IntegrationRule &volume_rule = rules.Get(
                                           TensorBasisElement::GetTensorProductGeometry(dim), 3 * p);
   std::vector<Vector> metric(dim, Vector(volume_rule.GetNPoints()));
   for (int q = 0; q < metric[0].Size(); ++q)
   {
      const IntegrationPoint &ip = volume_rule.IntPoint(q);
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
   SparseMatrix mass_factor(n), derivative_factor(p, n);
   for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
      {
         if (basis.G(i, j) != 0.) { mass_factor.Add(i, j, basis.G(i, j)); }
         if (i < p && basis.derivative(i, j) != 0.)
         { derivative_factor.Add(i, j, basis.derivative(i, j)); }
      }
   mass_factor.Finalize(); derivative_factor.Finalize();
   std::unique_ptr<SparseMatrix> matrix;
   Vector diagonal, u, v;
   for (int axis = 0; axis < dim; ++axis)
   {
      std::vector<const DenseMatrix*> b(dim, &values); b[axis] = &deriv;
      Tensor(metric[axis], diagonal, b, false, u, v);
      for (int i = 0; i < diagonal.Size(); ++i)
      { MFEM_VERIFY(diagonal[i] > 0., "Nonpositive auxiliary mass diagonal"); }
      // B = G x ... x derivative x ... x G, first coordinate fastest.
      auto tensor = std::make_unique<SparseMatrix>(
                       axis == 0 ? derivative_factor : mass_factor);
      for (int d = 1; d < dim; ++d)
      { tensor.reset(OuterProduct(d == axis ? derivative_factor : mass_factor, *tensor)); }
      std::unique_ptr<SparseMatrix> contribution(Mult_AtDA(*tensor, diagonal));
      if (matrix) { matrix.reset(Add(*matrix, *contribution)); }
      else { matrix = std::move(contribution); }
   }
   // Sparse products and sums retain structural zeros for the ICC pattern.
   matrix->SortColumnIndices();
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
   std::vector<int> lex, dofs, signs;
};

std::unique_ptr<SparseMatrix> Condense(const SparseMatrix &a, int n, int dim,
                                       std::vector<int> &boundary)
{
   const int nd = a.Height();
   std::vector<int> index(nd, -1);
   for (int i = 0; i < nd; ++i) if (!Interior(i, n, dim))
      { index[i] = boundary.size(); boundary.push_back(i); }
   auto matrix = std::make_unique<SparseMatrix>(boundary.size());
   const int *ai = a.HostReadI(), *aj = a.HostReadJ();
   const real_t *av = a.HostReadData();
   for (int i = 0; i < nd; ++i)
   {
      if (index[i] >= 0)
      {
         for (int k = ai[i]; k < ai[i + 1]; ++k) if (index[aj[k]] >= 0)
            { matrix->Add(index[i], index[aj[k]], av[k]); }
      }
      else
      {
         real_t diagonal = 0.;
         for (int k = ai[i]; k < ai[i + 1]; ++k)
         {
            if (aj[k] == i) { diagonal = av[k]; }
            else { MFEM_VERIFY(index[aj[k]] >= 0 || av[k] == 0., "Interior block is not diagonal"); }
         }
         MFEM_VERIFY(diagonal > 0., "Nonpositive interior diagonal");
         for (int k = ai[i]; k < ai[i + 1]; ++k) if (index[aj[k]] >= 0)
               for (int l = ai[i]; l < ai[i + 1]; ++l) if (index[aj[l]] >= 0)
                  { matrix->Add(index[aj[k]], index[aj[l]], -av[k] * av[l] / diagonal); }
      }
   }
   // Keep structural zeros for the prescribed ICC sparsity pattern.
   matrix->Finalize(0); matrix->SortColumnIndices();
   return matrix;
}

// Conforming scalar H1 trace and volume DOFs have the same entity owner.
// Build the injection directly between owned true DOFs, without communication.
std::unique_ptr<SparseMatrix> BuildTraceInjection(
   ParFiniteElementSpace &full, ParFiniteElementSpace &trace,
   const Array<int> &full_to_trace)
{
   auto injection = std::make_unique<SparseMatrix>(full.GetTrueVSize(),
                                                   trace.GetTrueVSize());
   for (int i = 0; i < full_to_trace.Size(); ++i)
   {
      if (full_to_trace[i] < 0) { continue; }
      const int row = full.GetLocalTDofNumber(i);
      const int col = trace.GetLocalTDofNumber(full_to_trace[i]);
      MFEM_VERIFY((row >= 0) == (col >= 0), "Inconsistent trace DOF ownership");
      if (row >= 0) { injection->Set(row, col, 1.); }
   }
   injection->Finalize();
   return injection;
}

struct PatchAssembly
{
   std::map<Global, int> index;
   std::vector<Global> global;
   std::vector<bool> interior;
   std::unique_ptr<SparseMatrix> matrix;
   std::vector<int> element_maps, element_signs, element_sources;
   int Insert(Global g, bool cell)
   {
      const auto it = index.find(g);
      if (it != index.end()) { return it->second; }
      const int i = global.size(); index[g] = i;
      global.push_back(g); interior.push_back(cell); return i;
   }
};
void ReceivePatches(const std::string &s,
                    std::map<Global, PatchAssembly> &patches, bool assemble,
                    int nd, int base = 0)
{
   size_t pos = sizeof(int);
   while (pos < s.size())
   {
      const Global vertex = Unpack<Global>(s,pos);
      auto &patch = patches[vertex];
      const int n = Unpack<int>(s,pos), source = Unpack<int>(s,pos);
      const int start = patch.element_maps.size();
      if (assemble)
      {
         patch.element_maps.resize(start+nd,-1); patch.element_signs.resize(start+nd,0);
         patch.element_sources.push_back(base+source);
      }
      for (int i = 0; i < n; ++i)
      {
         const Global g = Unpack<Global>(s,pos);
         const int interior = Unpack<int>(s,pos), local = Unpack<int>(s,pos),
                   sign = Unpack<int>(s,pos);
         MFEM_VERIFY(local >= 0 && local < nd,"Invalid patch element index");
         if (assemble)
         { patch.element_maps[start+local] = patch.index.at(g); patch.element_signs[start+local] = sign; }
         else { patch.Insert(g,interior != 0); }
      }
   }
}
Array<int> PatchOrdering(const SparseMatrix &pattern,
                         const std::vector<bool> &interior)
{
   pattern.HostReadI(); pattern.HostReadJ();
   std::vector<int> interface, inverse(pattern.Height(), -1);
   Array<int> permutation(pattern.Height()); int first = 0;
   for (int i = 0; i < pattern.Height(); ++i)
   {
      if (interior[i]) { permutation[first++] = i; }
      else { inverse[i] = interface.size(); interface.push_back(i); }
   }
   idx_t n = idx_t(interface.size());
   if (n > 0)
   {
      std::vector<idx_t> offsets(n + 1), columns;
      for (int i = 0; i < n; ++i)
      {
         const int *cols = pattern.GetRowColumns(interface[i]);
         for (int k = 0; k < pattern.RowSize(interface[i]); ++k)
         {
            if (inverse[cols[k]] >= 0 && cols[k] != interface[i])
            { columns.push_back(inverse[cols[k]]); }
         }
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

} // namespace

class Preconditioner::PatchSolver : public Solver
{
   struct Patch
   {
      int factor;
      Array<int> indices;
   };
   struct Peer
   {
      Array<int> requested, supplied;
      mutable Vector receive, send;
   };
   MPI_Comm comm;
   int rank;
   Global offset;
   std::vector<Patch> patches;
   int local_patch_count = 0;
   std::map<int, Peer> peers;
   Array<int> local_indices, patch_order;
   PatchFactors factors;
   mutable Vector extended, correction;
   mutable std::vector<MPI_Request> requests;
   real_t damping = 1.;
   void ApplyPatches(int begin, int end) const
   {
      factors.Apply(extended, correction, patch_order, begin, end);
   }
public:
   std::map<Global, PatchAssembly> AssemblePatches(
      ParFiniteElementSpace &fes, const std::vector<ElementData> &elements,
      const Array<int> &ess_bdr, int p, Messages &send, const Vector &element_values,
      const SparseMatrix &pattern, const Array<int> &entry_rows)
   {
      ParMesh &mesh = *fes.GetParMesh();
      const int dim = mesh.Dimension(), n = p + 1;
      const auto &gt = mesh.gtopo;
      Array<Global> vertices; mesh.GetGlobalVertexIndices(vertices);
      Array<int> owner(mesh.GetNV()); owner = rank;
      for (int g = 1; g < gt.NGroups(); ++g)
      {
         for (int k = 0; k < mesh.GroupNVertices(g); ++k)
         { owner[mesh.GroupVertex(g, k)] = gt.GetGroupMasterRank(g); }
      }
      Array<int> essential; fes.GetEssentialVDofs(ess_bdr, essential);
      Array<int> ev;
      std::string local_records;
      std::map<int,std::vector<int>> sources;
      std::map<int,std::map<int,int>> slots;
      const int nnz = pattern.NumNonZeroElems(), nd = pattern.Height();
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         const auto &el = elements[e];
         mesh.GetElementVertices(e, ev);
         const auto *vr = Geometries.GetVertices(mesh.GetElementGeometry(e));
         for (int vertex = 0; vertex < ev.Size(); ++vertex)
         {
            const auto &ip = vr->IntPoint(vertex);
            const int corner[3] = {int(ip.x)*p, int(ip.y)*p, int(ip.z)*p};
            std::vector<int> selected;
            for (size_t i = 0; i < el.lex.size(); ++i)
            {
               int a[3]; Coordinates(el.lex[i], n, dim, a);
               bool inside = !essential[el.dofs[i]];
               for (int d = 0; d < dim; ++d)
               { if ((a[d] == 0 || a[d] == p) && a[d] != corner[d]) { inside = false; } }
               if (inside) { selected.push_back(i); }
            }
            if (selected.empty()) { continue; }
            const int root = owner[ev[vertex]];
            std::string &buffer = root == rank ? local_records : send[root].data;
            auto slot = slots[root].find(e);
            if (slot == slots[root].end())
            {
               slot = slots[root].emplace(e,sources[root].size()).first;
               sources[root].push_back(e);
            }
            Pack(buffer,vertices[ev[vertex]]); Pack(buffer,int(selected.size()));
            Pack(buffer,slot->second*nnz);
            for (int i : selected)
            {
               Pack(buffer,fes.GetGlobalTDofNumber(el.dofs[i]));
               Pack(buffer,int(Interior(el.lex[i],n,dim))); Pack(buffer,i);
               Pack(buffer,el.signs[i]);
            }
         }
      }
      // Metadata is exchanged separately from numerical values. Only message
      // buffers are staged when MPI cannot consume device pointers.
      for (auto &entry : send)
      {
         std::string header; Pack(header,int(sources[entry.first].size())*nnz);
         entry.second.data.insert(0,header);
      }
      std::string header; Pack(header,int(sources[rank].size())*nnz);
      local_records.insert(0,header);
      auto received = Exchange(send, comm);
      received[rank].data = std::move(local_records);
      std::map<int,int> bases;
      std::map<int,Vector> incoming, outgoing;
      int total = 0;
      for (const auto &entry : received)
      {
         size_t pos = 0; const int count = Unpack<int>(entry.second.data,pos);
         bases[entry.first] = total; total += count;
         incoming[entry.first].SetSize(count); incoming[entry.first].UseDevice(true);
      }
      const bool gpu_mpi = Device::GetGPUAwareMPI();
      std::vector<MPI_Request> req;
      const real_t *V = element_values.Read();
      for (const auto &entry : received)
      {
         const int peer = entry.first;
         Array<int> src(sources[peer].data(),sources[peer].size());
         Vector &buffer = peer == rank ? incoming[peer] : outgoing[peer];
         buffer.SetSize(src.Size()*nnz); buffer.UseDevice(true);
         const int *S = src.Read(); real_t *B = buffer.Write();
         mfem::forall(buffer.Size(), [=] MFEM_HOST_DEVICE (int k) { B[k] = V[S[k/nnz]*nnz+k%nnz]; });
         if (peer == rank) { continue; }
         auto &in = incoming[peer]; MPI_Request r;
         MPI_Irecv(gpu_mpi ? in.Write() : in.HostWrite(),in.Size(),
                   MPITypeMap<real_t>::mpi_type,peer,3,comm,&r); req.push_back(r);
         if (gpu_mpi) { MFEM_STREAM_SYNC; }
         MPI_Isend(gpu_mpi ? buffer.Read() : buffer.HostRead(),buffer.Size(),
                   MPITypeMap<real_t>::mpi_type,peer,3,comm,&r); req.push_back(r);
      }
      MPI_Waitall(req.size(),req.data(),MPI_STATUSES_IGNORE);
      Vector numerical(total); numerical.UseDevice(true);
      real_t *N = numerical.Write();
      for (const auto &entry : incoming)
      {
         const int base = bases[entry.first]; const real_t *B = entry.second.Read();
         mfem::forall(entry.second.Size(), [=] MFEM_HOST_DEVICE (int k) { N[base+k] = B[k]; });
      }
      std::map<Global, PatchAssembly> assembled;
      for (const auto &entry : received) { ReceivePatches(entry.second.data,assembled,false,nd); }
      for (const auto &entry : received)
      { ReceivePatches(entry.second.data,assembled,true,nd,bases[entry.first]); }
      const int *ER = entry_rows.Read(), *EC = pattern.ReadJ();
      for (auto &entry : assembled)
      {
         auto &a = entry.second;
         Array<int> map(a.element_maps.data(),a.element_maps.size());
         Array<int> sign(a.element_signs.data(),a.element_signs.size());
         Array<int> src(a.element_sources.data(),a.element_sources.size());
         const int count = src.Size()*nnz;
         Array<int> rows(count), cols(count), indices(count); Vector weights(count);
         weights.UseDevice(true);
         const int *M = map.Read(), *G = sign.Read(), *B = src.Read();
         int *R = rows.Write(), *C = cols.Write(), *S = indices.Write();
         real_t *W = weights.Write();
         mfem::forall(count, [=] MFEM_HOST_DEVICE (int k)
         {
            const int e = k/nnz, j = k%nnz, i0 = e*nd+ER[j], i1 = e*nd+EC[j];
            const int row = M[i0], col = M[i1];
            R[k] = row < 0 || col < 0 ? -1 : row; C[k] = col;
            S[k] = B[e]+j; W[k] = G[i0]*G[i1];
         });
         SparseAccumulator accumulator(a.global.size(),a.global.size());
         accumulator.Assemble(numerical,rows,cols,indices,weights);
         a.matrix = std::move(accumulator.matrix);
      }
      return assembled;
   }
   std::map<Global, int> BuildExchange(
      const std::map<Global, PatchAssembly> &assembled, Messages &send)
   {
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
      std::map<Global, int> all_dofs;
      for (const auto &entry : assembled) for (Global g : entry.second.global) { all_dofs.emplace(g, 0); }
      int count = 0;
      for (auto &entry : all_dofs) { entry.second = count++; }
      extended.SetSize(count); correction.SetSize(count);
      extended.UseDevice(true); correction.UseDevice(true);
      local_indices.SetSize(count); local_indices = -1;
      for (const auto &entry : all_dofs)
      {
         if (entry.first >= offset && entry.first < offset + height)
         { local_indices[entry.second] = entry.first - offset; }
         else
         {
            int owner_rank = -1;
            for (const auto &r : ranges)
            { if (entry.first >= r.second.first && entry.first < r.second.second) { owner_rank = r.first; break; } }
            MFEM_VERIFY(owner_rank >= 0, "Patch DOF owner is not a vertex neighbor");
            peers[owner_rank].requested.Append(entry.second);
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
            peer.supplied.Append(g - offset);
         }
      }
      incoming_requests.clear();
      for (auto &entry : peers)
      {
         auto &peer = entry.second;
         peer.receive.SetSize(peer.requested.Size());
         peer.send.SetSize(peer.supplied.Size());
         peer.receive.UseDevice(true); peer.send.UseDevice(true);
         communication_values += 2 * peer.supplied.Size();
      }
      requests.resize(2 * peers.size());
      return all_dofs;
   }
   void FactorPatches(std::map<Global, PatchAssembly> &assembled,
                      const std::map<Global, int> &all_dofs, real_t diagonal_shift)
   {
      for (auto it = assembled.begin(); it != assembled.end(); )
      {
         auto &a = it->second;
         auto matrix = std::move(a.matrix);
         Array<int> ordering = PatchOrdering(*matrix, a.interior);
         Patch patch;
         for (Global g : a.global) { patch.indices.Append(all_dofs.at(g)); }
         patch.factor = factors.Add(*matrix, ordering, patch.indices, it->first);
         patch_unknowns += a.global.size();
         patches.push_back(std::move(patch));
         // The factor and DOF indices now own everything needed by this patch.
         it = assembled.erase(it);
      }
      // Keep patches needing only owned residual DOFs ahead of remote patches.
      const auto first_remote = std::stable_partition(
                                   patches.begin(), patches.end(), [this](const Patch &patch)
      {
         return std::all_of(patch.indices.begin(), patch.indices.end(),
         [this](int i) { return local_indices[i] >= 0; });
      });
      local_patch_count = first_remote - patches.begin();
      patch_order.SetSize(patches.size());
      for (int i = 0; i < patch_order.Size(); ++i) { patch_order[i] = patches[i].factor; }
      factor_nonzeros = factors.Nonzeros();
      factors.Factor(diagonal_shift, rank);
   }
public:
   long long factor_nonzeros = 0, patch_unknowns = 0, communication_values = 0;
   long long NumPatches() const { return patches.size(); }
   long long StorageBytes() const { return factors.StorageBytes()+sizeof(int)*patch_order.Size(); }
   real_t GetDamping() const { return damping; }
   void SetDamping(real_t value) { damping = value; }
   PatchSolver(ParFiniteElementSpace &fes,
               const std::vector<ElementData> &elements,
               const Array<int> &ess_bdr, int p, const Options &options,
               const Vector &element_values,
               const SparseMatrix &pattern, const Array<int> &entry_rows)
      : Solver(fes.GetTrueVSize()), offset(fes.GetMyTDofOffset())
   {
      MPI_Comm_dup(fes.GetComm(), &comm); MPI_Comm_rank(comm, &rank);
      const auto &gt = fes.GetParMesh()->gtopo;
      Messages send;
      for (int i = 1; i < gt.GetNumNeighbors(); ++i) { send[gt.GetNeighborRank(i)]; }
      auto assembled = AssemblePatches(fes, elements, ess_bdr, p, send,
                                       element_values, pattern, entry_rows);
      auto all_dofs = BuildExchange(assembled, send);
      FactorPatches(assembled, all_dofs, options.diagonal_shift);
   }
   ~PatchSolver() override { MPI_Comm_free(&comm); }
   void SetOperator(const Operator &) override { MFEM_ABORT("Reconstruct PatchSolver"); }
   void MultTranspose(const Vector &x, Vector &y) const override { Mult(x, y); }
   void Mult(const Vector &x, Vector &y) const override
   {
      const bool gpu_mpi = Device::GetGPUAwareMPI();
      int k = 0;
      for (auto &entry : peers)
      {
         auto &peer = entry.second;
         x.GetSubVector(peer.supplied, peer.send);
         MPI_Irecv(gpu_mpi ? peer.receive.Write() : peer.receive.HostWrite(),
                   peer.receive.Size(),
                   MPITypeMap<real_t>::mpi_type,
                   entry.first, 1, comm, &requests[k++]);
         if (gpu_mpi) { MFEM_STREAM_SYNC; }
         MPI_Isend(gpu_mpi ? peer.send.Read() : peer.send.HostRead(), peer.send.Size(),
                   MPITypeMap<real_t>::mpi_type,
                   entry.first, 1, comm, &requests[k++]);
      }
      const int *indices = local_indices.Read();
      const real_t *X = x.Read(); real_t *E = extended.ReadWrite();
      mfem::forall(extended.Size(), [=] MFEM_HOST_DEVICE (int i)
      { if (indices[i] >= 0) { E[i] = X[indices[i]]; } });
      correction = 0.;
      // Solve local patches while remote residuals are in flight.
      ApplyPatches(0, local_patch_count);
      MPI_Waitall(k, requests.data(), MPI_STATUSES_IGNORE);
      for (const auto &entry : peers)
      {
         const auto &peer = entry.second;
         extended.SetSubVector(peer.requested, peer.receive);
      }
      ApplyPatches(local_patch_count, patches.size());
      k = 0;
      for (auto &entry : peers)
      {
         auto &peer = entry.second;
         correction.GetSubVector(peer.requested, peer.receive);
         MPI_Irecv(gpu_mpi ? peer.send.Write() : peer.send.HostWrite(), peer.send.Size(),
                   MPITypeMap<real_t>::mpi_type,
                   entry.first, 2, comm, &requests[k++]);
         if (gpu_mpi) { MFEM_STREAM_SYNC; }
         MPI_Isend(gpu_mpi ? peer.receive.Read() : peer.receive.HostRead(),
                   peer.receive.Size(),
                   MPITypeMap<real_t>::mpi_type,
                   entry.first, 2, comm, &requests[k++]);
      }
      y.SetSize(height); y.UseDevice(true); y = 0.;
      const real_t *C = correction.Read(); real_t *Y = y.ReadWrite();
      mfem::forall(correction.Size(), [=] MFEM_HOST_DEVICE (int i)
      { if (indices[i] >= 0) { Y[indices[i]] += C[i]; } });
      MPI_Waitall(k, requests.data(), MPI_STATUSES_IGNORE);
      for (const auto &entry : peers)
      {
         const auto &peer = entry.second;
         y.AddElementVector(peer.supplied, peer.send);
      }
      y *= damping;
   }
};

class Preconditioner::CoarseTransfer : public Operator
{
   const Operator &interpolation;
   const BasisChange &basis;
   const Operator *trace;
   Array<int> fine_essential, coarse_essential;
   mutable Vector a, b, c;
public:
   CoarseTransfer(const Operator &p, const BasisChange &s, const Operator *t,
                  const Array<int> &fine, const Array<int> &coarse)
      : Operator(t ? t->Width() : s.Height(), p.Width()), interpolation(p), basis(s),
        trace(t),
        fine_essential(fine), coarse_essential(coarse)
   { a.UseDevice(true); b.UseDevice(true); c.UseDevice(true); }
   void Mult(const Vector &x, Vector &y) const override
   {
      a = x; a.SetSubVector(coarse_essential, 0.);
      b.SetSize(basis.Height()); interpolation.Mult(a, b); basis.InverseMult(b, c);
      if (trace) { y.SetSize(height); trace->MultTranspose(c, y); }
      else { y = c; }
      y.SetSubVector(fine_essential, 0.);
   }
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      a = x; a.SetSubVector(fine_essential, 0.);
      if (trace) { b.SetSize(trace->Height()); trace->Mult(a, b); }
      else { b = a; }
      basis.InverseMultTranspose(b, c); y.SetSize(width);
      interpolation.MultTranspose(c, y);
      y.SetSubVector(coarse_essential, 0.);
   }
};

std::unique_ptr<SparseMatrix> CondenseElement(
   const SparseMatrix &matrix, int degree, int dimension,
   std::vector<int> &boundary)
{
   MFEM_VERIFY(degree >= 1 && (dimension == 2 || dimension == 3) &&
               matrix.Height() == Power(degree + 1, dimension) &&
               matrix.Width() == matrix.Height(), "Invalid FDM element dimensions");
   boundary.clear();
   return Condense(matrix, degree + 1, dimension, boundary);
}

#include "vertex_patch_assembly.hpp"

Preconditioner::Preconditioner(ParFiniteElementSpace &space,
                               Coefficient &coefficient,
                               const Array<int> &ess_bdr, const Options &opts)
   : Solver(space.GetTrueVSize()), fes(space), comm(space.GetComm()),
     options(opts),
     degree(space.GetMaxElementOrder())
{
   ParMesh &mesh = *space.GetParMesh(); const int dim = mesh.Dimension();
   for (Vector *v :
        {
           &rhs, &interior, &reduced, &trace_solution, &solution, &temp,
           &inverse_interior
        }) { v->UseDevice(true); }
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
   StopWatch stage; stage.Start();
   Array<int> coarse_essential;
   BuildCoarse(coefficient, ess_bdr, coarse_essential);
   MFEM_STREAM_SYNC; stage.Stop(); setup_timings[0] = stage.RealTime();
   if (degree == 1) { return; }
   BuildAuxiliary(coefficient, ess_bdr);
   stage.Clear(); stage.Start();
   BuildMultigrid(coarse_essential);
   MFEM_STREAM_SYNC; stage.Stop(); setup_timings[5] = stage.RealTime();
}

void Preconditioner::BuildCoarse(Coefficient &coefficient,
                                 const Array<int> &ess_bdr,
                                 Array<int> &coarse_essential)
{
   ParMesh &mesh = *fes.GetParMesh();
   const int dim = mesh.Dimension();
   coarse_collection.reset(new H1_FECollection(1, dim));
   coarse_space.reset(new ParFiniteElementSpace(&mesh, coarse_collection.get()));
   coarse_space->GetEssentialTrueDofs(ess_bdr, coarse_essential);
   coarse_form.reset(new ParBilinearForm(coarse_space.get()));
   coarse_form->SetAssemblyLevel(AssemblyLevel::FULL);
   coarse_form->AddDomainIntegrator(new DiffusionIntegrator(coefficient));
   coarse_form->Assemble();
   coarse_matrix.reset(new HypreParMatrix);
   coarse_form->FormSystemMatrix(coarse_essential, coarse_handle);
   coarse_matrix->MakeRef(*coarse_handle.As<HypreParMatrix>());
   coarse_solver.reset(new HypreBoomerAMG(*coarse_matrix));
   coarse_solver->SetPrintLevel(0); coarse_solver->SetMaxIter(1);
   coarse_solver->SetTol(0.);
   coarse_solver->SetRelaxType(Device::Allows(Backend::DEVICE_MASK) ? 18 : 6);
   Vector zero(coarse_space->GetTrueVSize()), probe(zero.Size());
   zero.UseDevice(true); probe.UseDevice(true); zero = 0.;
   coarse_solver->Mult(zero,probe);
}

void Preconditioner::BuildAuxiliary(Coefficient &coefficient,
                                    const Array<int> &ess_bdr)
{
   ParMesh &mesh = *fes.GetParMesh();
   const int dim = mesh.Dimension();
   auto reference = Basis::Get(degree);
   basis.reset(new BasisChange(fes, reference));
   basis->ModalMap().HostRead(); basis->Signs().HostRead();
   const int nd = Power(degree+1, dim);
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
      trace = BuildTraceInjection(fes, *trace_space, full_to_trace);
      active = trace_space.get();
   }
   active->GetEssentialTrueDofs(ess_bdr, active_essential);
   StopWatch stage; stage.Start();
   auto batch_owner = ElementBatch::Get(*reference, dim,
                                        options.static_condensation);
   auto &batch = *batch_owner;
   stage.Stop(); setup_timings[1] = stage.RealTime(); stage.Clear(); stage.Start();
   batch.Assemble(mesh, coefficient, options.static_condensation);
   MFEM_STREAM_SYNC; stage.Stop(); setup_timings[2] = stage.RealTime();
   stage.Clear(); stage.Start();
   SparseAccumulator local(active->GetVSize(), active->GetVSize());
   SparseAccumulator cross(active->GetVSize(), fes.GetVSize());
   Vector local_inverse(fes.GetVSize()); local_inverse.UseDevice(true);
   local_inverse = 0.;
   const int ne = mesh.GetNE(), active_nd = batch.pattern->Height(),
             an = batch.active_nz;
   const int *modal = basis->ModalMap().Read(), *signs = basis->Signs().Read();
   const int *lex = batch.lex_map.Read(), *trace_map = full_to_trace.Read();
   const bool condensed = options.static_condensation;
   Array<int> element_dofs(ne*active_nd), element_signs(ne*active_nd);
   int *ED = element_dofs.Write(), *ES = element_signs.Write();
   mfem::forall(ne*active_nd, [=] MFEM_HOST_DEVICE (int k)
   {
      const int e = k/active_nd, i = lex[k%active_nd], dof = modal[e*nd+i];
      ED[k] = condensed ? trace_map[dof] : dof; ES[k] = signs[e*nd+i];
   });
   Array<int> rows(ne*an), cols(ne*an), sources(ne*an); Vector weights(ne*an);
   weights.UseDevice(true);
   int *R = rows.Write(), *C = cols.Write(), *S = sources.Write();
   real_t *W = weights.Write();
   const int *ER = batch.entry_rows.Read(), *EC = batch.pattern->ReadJ();
   mfem::forall(ne*an, [=] MFEM_HOST_DEVICE (int k)
   {
      const int e = k/an, a = k%an, i = e*active_nd+ER[a], j = e*active_nd+EC[a];
      R[k] = ED[i]; C[k] = ED[j]; S[k] = k; W[k] = ES[i]*ES[j];
   });
   local.Assemble(batch.active_values,rows,cols,sources,weights);
   // MPI ownership metadata is built once from the device-generated maps.
   const int *dh = element_dofs.HostRead(), *sh = element_signs.HostRead();
   std::vector<ElementData> elements(ne);
   for (int e = 0; e < ne; ++e)
   {
      auto &el = elements[e]; el.lex = batch.lex;
      el.dofs.assign(dh+e*active_nd,dh+(e+1)*active_nd);
      el.signs.assign(sh+e*active_nd,sh+(e+1)*active_nd);
   }
   ParBilinearForm form(active);
   auxiliary.reset(form.ParallelAssemble(local.matrix.get()));
   delete auxiliary->EliminateRowsCols(active_essential);
   auxiliary_nonzeros = auxiliary->NNZ();
   if (options.static_condensation)
   {
      const int cn = batch.cross_sources.Size(), fn = batch.nz;
      rows.SetSize(ne*cn); cols.SetSize(ne*cn); sources.SetSize(ne*cn);
      weights.SetSize(ne*cn);
      R = rows.Write(); C = cols.Write(); S = sources.Write(); W = weights.Write();
      const int *CR = batch.cross_rows.Read(), *CC = batch.cross_cols.Read(),
                 *CS = batch.cross_sources.Read();
      mfem::forall(ne*cn, [=] MFEM_HOST_DEVICE (int k)
      {
         const int e = k/cn, a = k%cn, i = e*nd+CR[a], j = e*nd+CC[a];
         R[k] = trace_map[modal[i]]; C[k] = modal[j]; S[k] = e*fn+CS[a];
         W[k] = signs[i]*signs[j];
      });
      cross.Assemble(batch.full_values,rows,cols,sources,weights);
      ParMixedBilinearForm mixed(&fes, active);
      coupling.reset(mixed.ParallelAssemble(cross.matrix.get()));
      const int count = batch.interior_sources.Size();
      const int *I = batch.interior_dofs.Read(), *IS = batch.interior_sources.Read();
      const real_t *V = batch.full_values.Read();
      real_t *D = local_inverse.ReadWrite();
      mfem::forall(ne*count, [=] MFEM_HOST_DEVICE (int k)
      { const int e = k/count, a = k%count; D[modal[e*nd+I[a]]] = 1./V[e*fn+IS[a]]; });
      inverse_interior.SetSize(fes.GetTrueVSize());
      fes.GetRestrictionMatrix()->Mult(local_inverse, inverse_interior);
   }
   MFEM_STREAM_SYNC; stage.Stop(); setup_timings[3] = stage.RealTime();
   stage.Clear(); stage.Start();
   patches.reset(new PatchSolver(*active, elements, ess_bdr, degree, options,
                                 batch.active_values, *batch.pattern, batch.entry_rows));
   MFEM_STREAM_SYNC; stage.Stop(); setup_timings[4] = stage.RealTime();
   batch.full_values.Destroy(); batch.active_values.Destroy();
}

void Preconditioner::BuildMultigrid(const Array<int> &coarse_essential)
{
   ParFiniteElementSpace *active = trace_space ? trace_space.get() : &fes;
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
   interpolation.reset(new LinearCoarseTransfer(*coarse_space, fes));
   transfer.reset(new CoarseTransfer(*interpolation,
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

void Preconditioner::Mult(const Vector &x, Vector &y) const
{
   if (degree == 1) { y.SetSize(x.Size()); coarse_solver->Mult(x, y); return; }
   temp = x; temp.SetSubVector(essential, 0.);
   basis->MultTranspose(temp, rhs);
   if (options.static_condensation)
   {
      interior = rhs;
      interior *= inverse_interior;
      reduced.SetSize(trace->Width()); trace->MultTranspose(rhs, reduced);
      temp.SetSize(reduced.Size()); coupling->Mult(interior, temp); reduced -= temp;
      reduced.SetSubVector(active_essential, 0.);
      trace_solution.SetSize(reduced.Size());
      multigrid->Mult(reduced, trace_solution);
      solution.SetSize(rhs.Size()); trace->Mult(trace_solution, solution);
      temp.SetSize(rhs.Size()); coupling->MultTranspose(trace_solution, temp);
      temp *= inverse_interior;
      interior -= temp;
      solution += interior;
   }
   else
   {
      rhs.SetSubVector(essential, 0.);
      solution.SetSize(rhs.Size()); multigrid->Mult(rhs, solution);
   }
   basis->Mult(solution, y);
   const int *ess = essential.Read();
   const real_t *X = x.Read(); real_t *Y = y.ReadWrite();
   mfem::forall(essential.Size(), [=] MFEM_HOST_DEVICE (int k)
   { const int i = ess[k]; Y[i] = X[i]; });
}

Preconditioner::~Preconditioner() = default;

void Preconditioner::PrintStatistics(std::ostream &os) const
{
   long long stats[5] = {0, 0, 0, 0, 0};
   real_t times[6];
   MPI_Allreduce(setup_timings,times,6,MPITypeMap<real_t>::mpi_type,MPI_MAX,comm);
   if (patches)
   {
      stats[0] = patches->NumPatches();
      stats[1] = patches->patch_unknowns;
      stats[2] = patches->factor_nonzeros;
      stats[3] = patches->communication_values;
      stats[4] = patches->StorageBytes();
   }
   MPI_Allreduce(MPI_IN_PLACE, stats, 5, MPI_LONG_LONG, MPI_SUM, comm);
   int rank; MPI_Comm_rank(comm, &rank);
   if (rank == 0)
   {
      os << "Vertex patches: " << stats[0] << ", patch unknowns: " << stats[1]
         << ", factor nonzeros: " << stats[2] << '\n'
         << "Auxiliary nonzeros: " << auxiliary_nonzeros << '\n'
         << "Factor arrays including solve workspace (bytes): "
         << stats[4] << '\n'
         << "Patch exchange values per smoothing step (gather + scatter): " << stats[3]
         << '\n'
         << "Setup stages (MPI max seconds): coarse=" << times[0]
         << ", reference=" << times[1] << ", elements=" << times[2]
         << ", global assembly=" << times[3] << ", patches=" << times[4]
         << ", damping/transfer=" << times[5] << '\n'
         << "Smoother damping: " << (patches ? patches->GetDamping() : 1.) <<
         std::endl;
   }
}

} // namespace vertex_patch
} // namespace mfem
#endif
