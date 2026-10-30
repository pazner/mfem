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
#if defined(MFEM_USE_MPI) && defined(MFEM_USE_LAPACK) && defined(MFEM_USE_METIS) && defined(MFEM_USE_METIS_5)
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
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
{
   const int n = basis->p + 1, ne = fes.GetNE();
   const auto *restriction = dynamic_cast<const ElementRestriction*>(
                                fes.GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC));
   MFEM_VERIFY(restriction, "Expected conforming H1 element restriction");
   nodal = restriction->GatherMap();
   modal.SetSize(ne * nd); signs.SetSize(ne * nd);
   mask.SetSize(ne * nd);
   Array<int> seen(fes.GetVSize()); seen = 0;
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
         // Both orderings use the same first incident element of this entity.
         mask[k] = !seen[nodal[k]]; seen[nodal[k]] = 1;
      }
   }
}

void BasisChange::Apply(const Vector &x, Vector &y, bool invert,
                        bool transpose) const
{
   const auto &in = invert ? nodal : modal;
   const auto &out = invert ? modal : nodal;
   const DenseMatrix &s = invert ? basis->inverse : basis->S;
   std::vector<const DenseMatrix*> matrices(dim, &s);
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
      Tensor(element, result, matrices, transpose, scratch1, scratch2);
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

std::unique_ptr<SparseMatrix> AssembleElement(
   const Basis &basis, ElementTransformation &trans, Coefficient &coefficient)
{
   const int p = basis.p, n = p + 1, dim = trans.GetDimension();
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
   std::unique_ptr<SparseMatrix> matrix;
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
   const int *ai = a.GetI(), *aj = a.GetJ(); const real_t *av = a.GetData();
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
   int Insert(Global g, bool cell)
   {
      const auto it = index.find(g);
      if (it != index.end()) { return it->second; }
      const int i = global.size(); index[g] = i;
      global.push_back(g); interior.push_back(cell); return i;
   }
};
void ReceivePatches(const std::string &s,
                    std::map<Global, PatchAssembly> &patches, bool assemble)
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
         local[i] = assemble ? patch.index.at(g) : patch.Insert(g, interior != 0);
      }
      const int nz = Unpack<int>(s, pos);
      if (!assemble)
      {
         const size_t record_size = 2 * sizeof(int) + sizeof(real_t);
         MFEM_VERIFY(nz >= 0 && size_t(nz) <= (s.size() - pos) / record_size,
                     "Invalid patch matrix message");
         pos += size_t(nz) * record_size;
         continue;
      }
      for (int k = 0; k < nz; ++k)
      {
         const int i = Unpack<int>(s, pos), j = Unpack<int>(s, pos);
         const real_t value = Unpack<real_t>(s, pos);
         patch.matrix->Add(local[i], local[j], value);
      }
   }
}
Array<int> PatchOrdering(const SparseMatrix &pattern,
                         const std::vector<bool> &interior)
{
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
      std::unique_ptr<IncompleteCholeskySolver> factor;
      Array<int> indices;
      mutable Vector rhs, solution;
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
   std::vector<int> local_indices;
   mutable Vector extended, correction;
   mutable std::vector<MPI_Request> requests;
   real_t damping = 1.;
   void ApplyPatches(int begin, int end) const
   {
      for (int k = begin; k < end; ++k)
      {
         const auto &patch = patches[k];
         extended.GetSubVector(patch.indices, patch.rhs);
         patch.factor->Mult(patch.rhs, patch.solution);
         correction.AddElementVector(patch.indices, patch.solution);
      }
   }
   std::map<Global, PatchAssembly> AssemblePatches(
      ParFiniteElementSpace &fes, const std::vector<ElementData> &elements,
      const Array<int> &ess_bdr, int p, Messages &send)
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
            for (size_t i = 0; i < el.lex.size(); ++i)
            {
               int a[3]; Coordinates(el.lex[i], n, dim, a);
               bool inside = !essential[el.dofs[i]];
               for (int d = 0; d < dim; ++d)
               { if ((a[d] == 0 || a[d] == p) && a[d] != corner[d]) { inside = false; } }
               if (inside) { inverse[i] = selected.size(); selected.push_back(i); }
            }
            if (selected.empty()) { continue; }
            const int root = owner[ev[vertex]];
            std::string &buffer = root == rank ? local_records : send[root].data;
            // Counts and interior flags use int in the serialized format.
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
      // Establish complete patch DOF numbering before allocating the matrices.
      ReceivePatches(local_records, assembled, false);
      for (const auto &entry : received)
      { ReceivePatches(entry.second.data, assembled, false); }
      for (auto &entry : assembled)
      { entry.second.matrix = std::make_unique<SparseMatrix>(entry.second.global.size()); }
      ReceivePatches(local_records, assembled, true);
      std::string().swap(local_records);
      for (auto &entry : received)
      {
         ReceivePatches(entry.second.data, assembled, true);
         std::string().swap(entry.second.data);
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
      local_indices.resize(count, -1);
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
         // Structural zeros carry the ICC fill in the numerical matrix itself.
         // Only interface rows change, so the interior rows can be read in LIL form.
         Array<int> cols;
         Vector values;
         for (int i = 0; i < matrix->Height(); ++i)
         {
            if (!a.interior[i]) { continue; }
            matrix->GetRow(i, cols, values);
            for (int j : cols)
            {
               if (a.interior[j]) { continue; }
               for (int k : cols)
               {
                  if (!a.interior[k]) { matrix->Add(j, k, 0.); }
               }
            }
         }
         matrix->Finalize(0); matrix->SortColumnIndices();
         Array<int> ordering = PatchOrdering(*matrix, a.interior);
         Patch patch; patch.factor = std::make_unique<IncompleteCholeskySolver>();
         const int failed = patch.factor->Factor(*matrix, *matrix, ordering,
                                                 diagonal_shift);
         MFEM_VERIFY(failed < 0, "ICC nonpositive pivot " << failed <<
                     " in vertex patch "
                     << it->first << " on rank " << rank <<
                     "; inspect mesh/coefficient or set --diagonal-shift");
         for (Global g : a.global) { patch.indices.Append(all_dofs.at(g)); }
         patch.rhs.SetSize(a.global.size());
         patch.solution.SetSize(a.global.size());
         factor_nonzeros += patch.factor->NumNonZeroElems();
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
      const auto &gt = fes.GetParMesh()->gtopo;
      Messages send;
      for (int i = 1; i < gt.GetNumNeighbors(); ++i) { send[gt.GetNeighborRank(i)]; }
      auto assembled = AssemblePatches(fes, elements, ess_bdr, p, send);
      auto all_dofs = BuildExchange(assembled, send);
      FactorPatches(assembled, all_dofs, options.diagonal_shift);
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
         x.GetSubVector(peer.supplied, peer.send);
         MPI_Irecv(peer.receive.GetData(), peer.receive.Size(),
                   MPITypeMap<real_t>::mpi_type,
                   entry.first, 1, comm, &requests[k++]);
         MPI_Isend(peer.send.GetData(), peer.send.Size(), MPITypeMap<real_t>::mpi_type,
                   entry.first, 1, comm, &requests[k++]);
      }
      for (int i = 0; i < extended.Size(); ++i)
      { if (local_indices[i] >= 0) { extended[i] = x[local_indices[i]]; } }
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
         MPI_Irecv(peer.send.GetData(), peer.send.Size(), MPITypeMap<real_t>::mpi_type,
                   entry.first, 2, comm, &requests[k++]);
         MPI_Isend(peer.receive.GetData(), peer.receive.Size(),
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
        fine_essential(fine), coarse_essential(coarse) { }
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

Preconditioner::Preconditioner(ParFiniteElementSpace &space,
                               Coefficient &coefficient,
                               const Array<int> &ess_bdr, const Options &opts)
   : Solver(space.GetTrueVSize()), fes(space), comm(space.GetComm()),
     options(opts),
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
   Array<int> coarse_essential;
   BuildCoarse(coefficient, ess_bdr, coarse_essential);
   if (degree == 1) { return; }
   BuildAuxiliary(coefficient, ess_bdr);
   BuildMultigrid(coarse_essential);
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
   coarse_form->AddDomainIntegrator(new DiffusionIntegrator(coefficient));
   coarse_form->Assemble();
   coarse_matrix.reset(new HypreParMatrix);
   coarse_form->FormSystemMatrix(coarse_essential, *coarse_matrix);
   coarse_solver.reset(new HypreBoomerAMG(*coarse_matrix));
   coarse_solver->SetPrintLevel(0); coarse_solver->SetMaxIter(1);
   coarse_solver->SetTol(0.); coarse_solver->SetRelaxType(6);
}

void Preconditioner::BuildAuxiliary(Coefficient &coefficient,
                                    const Array<int> &ess_bdr)
{
   ParMesh &mesh = *fes.GetParMesh();
   const int dim = mesh.Dimension();
   auto reference = Basis::Get(degree);
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
      trace = BuildTraceInjection(fes, *trace_space, full_to_trace);
      active = trace_space.get();
   }
   active->GetEssentialTrueDofs(ess_bdr, active_essential);
   auto local_matrix = std::make_unique<SparseMatrix>(active->GetVSize());
   std::unique_ptr<SparseMatrix> local_cross;
   if (options.static_condensation)
   { local_cross = std::make_unique<SparseMatrix>(active->GetVSize(), fes.GetVSize()); }
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
                     local_cross->Add(row, col, av[k] * sign * basis->Signs()[e * nd + j]);
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
      for (size_t i = 0; i < el.lex.size(); ++i)
         for (int k = ai[i]; k < ai[i + 1]; ++k)
         { local_matrix->Add(el.dofs[i], el.dofs[aj[k]], av[k] * el.signs[i] * el.signs[aj[k]]); }
   }
   local_matrix->Finalize(0); local_matrix->SortColumnIndices();
   ParBilinearForm form(active);
   auxiliary.reset(form.ParallelAssemble(local_matrix.get()));
   local_matrix.reset();
   delete auxiliary->EliminateRowsCols(active_essential);
   auxiliary_nonzeros = auxiliary->NNZ();
   if (options.static_condensation)
   {
      local_cross->Finalize(0); local_cross->SortColumnIndices();
      ParMixedBilinearForm mixed(&fes, active);
      coupling.reset(mixed.ParallelAssemble(local_cross.get()));
      local_cross.reset();
      inverse_interior.SetSize(fes.GetTrueVSize());
      fes.GetRestrictionMatrix()->Mult(local_inverse, inverse_interior);
   }
   patches.reset(new PatchSolver(*active, elements, ess_bdr, degree, options));
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
   for (int i : essential) { y[i] = x[i]; }
}

Preconditioner::~Preconditioner() = default;

void Preconditioner::PrintStatistics(std::ostream &os) const
{
   long long stats[4] = {0, 0, 0, 0};
   if (patches)
   {
      stats[0] = patches->NumPatches();
      stats[1] = patches->patch_unknowns;
      stats[2] = patches->factor_nonzeros;
      stats[3] = patches->communication_values;
   }
   MPI_Allreduce(MPI_IN_PLACE, stats, 4, MPI_LONG_LONG, MPI_SUM, comm);
   int rank; MPI_Comm_rank(comm, &rank);
   if (rank == 0)
   {
      os << "Vertex patches: " << stats[0] << ", patch unknowns: " << stats[1]
         << ", factor nonzeros: " << stats[2] << '\n'
         << "Auxiliary nonzeros: " << auxiliary_nonzeros << '\n'
         << "Factor arrays including solve workspace (bytes): "
         << (stats[2] + stats[1])*(sizeof(real_t) + sizeof(int)) + stats[0]*sizeof(
            int) << '\n'
         << "Patch exchange values per smoothing step (gather + scatter): " << stats[3]
         << '\n'
         << "Smoother damping: " << (patches ? patches->GetDamping() : 1.) <<
         std::endl;
   }
}

} // namespace vertex_patch
} // namespace mfem
#endif
