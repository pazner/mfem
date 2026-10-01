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

// Compile with: make hbp (requires MFEM_USE_MPI=YES)
//
// Sample runs: mpirun -np 4 hbp -m ../data/inline-quad.mesh -o 3 -verify
//              mpirun -np 2 hbp -m ../data/beam-hex.mesh -o 2 -verify
//              mpirun -np 4 hbp -m ../data/star-q3.mesh -o 3 -r 1
//
// Parallel version of hb.cpp: solve -grad(div u) + u = f using RT_{p-1}
// elements in integrated GLL coordinates. Prescribe the normal component of
// a manufactured solution on all physical boundary faces. Only conforming
// quadrilateral/hexahedral volume meshes are supported.
//
// MINRES solves the uncondensed hybridized saddle-point system
//
//       [ A_hat  C^T ] [ u_hat  ] = [ f_hat ]
//       [ C       0  ] [ lambda ]   [   0   ].
//
// A_hat and its inverse use dense high-order element matrices and Cholesky
// factorizations. Each multiplier has one owner; C and C^T communicate across
// partition boundaries. The multiplier preconditioner is J^T B_lor J, where
// B_lor is ONE fixed symmetric BoomerAMG V-cycle on the full hybridized RT0
// LOR system, including the multipliers internal to each macroelement.
// Constraints use sums of outward integrated fluxes, so the low-energy LOR
// multiplier is represented by a constant vector suitable for scalar AMG.
// High-order global assembly is used only by the optional verification path.

#include "mfem.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

using namespace mfem;
using namespace std;

#ifdef MFEM_USE_MPI

namespace
{

int Index(int d) { return d < 0 ? -1-d : d; }
real_t Sign(int d) { return d < 0 ? -1.0 : 1.0; }

real_t GlobalNorm(MPI_Comm comm, const Vector &x)
{
   return sqrt(InnerProduct(comm, x, x));
}

void ExactSolution(const Vector &x, Vector &u)
{
   u = 0.0;
   u(0) = cos(M_PI*x(0))*sin(M_PI*x(1));
   u(1) = cos(M_PI*x(1))*sin(M_PI*x(0));
}

void RightHandSide(const Vector &x, Vector &f)
{
   ExactSolution(x, f);
   f *= 1.0 + 2.0*M_PI*M_PI;
}

// Keep full rank offsets for mapping neighbor-owned multiplier indices. Pass
// the appropriate two-entry view to hypre when it uses assumed partitioning.
struct Partition
{
   Array<HYPRE_BigInt> starts;
   int rank;

   void Init(MPI_Comm comm, int local_size)
   {
      int size;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);
      Array<int> counts(size);
      MPI_Allgather(&local_size, 1, MPI_INT, counts.GetData(), 1, MPI_INT, comm);
      starts.SetSize(size+1);
      starts[0] = 0;
      for (int r = 0; r < size; ++r) { starts[r+1] = starts[r]+counts[r]; }
   }

   HYPRE_BigInt GlobalSize() const { return starts.Last(); }
   HYPRE_BigInt Begin() const { return starts[rank]; }
   int LocalSize() const { return int(starts[rank+1]-starts[rank]); }
   HYPRE_BigInt *Data()
   {
      return starts.GetData() + (HYPRE_AssumedPartitionCheck() ? rank : 0);
   }
};

// The nine-argument constructor copies CSR data and both partitions. Column
// indices are global; hypre constructs the communication maps for remote DOFs.
unique_ptr<HypreParMatrix> ParallelMatrix(
   MPI_Comm comm, Partition &rows, Partition &cols, const Array<int> &I,
   const Array<HYPRE_BigInt> &J, const Vector &data)
{
   return unique_ptr<HypreParMatrix>(new HypreParMatrix(
                                        comm, rows.LocalSize(), rows.GlobalSize(),
                                        cols.GlobalSize(), I.GetData(), J.GetData(),
                                        data.GetData(), rows.Data(), cols.Data()));
}

struct BrokenSpace
{
   ParFiniteElementSpace &fes;
   Partition flux_partition, trace_partition;
   Array<int> offsets, dofs, owners, multiplicity, first, trace_local, trace_owner;
   Array<HYPRE_BigInt> trace_global;
   vector<Array<int>> local;
   unique_ptr<HypreParMatrix> Ct;
   unique_ptr<TransposeOperator> C;

   BrokenSpace(ParFiniteElementSpace &fes_, const Array<int> &essential)
      : fes(fes_), offsets(fes.GetNE()+1), multiplicity(fes.GetVSize()),
        first(fes.GetVSize()), trace_local(fes.GetVSize()),
        trace_owner(fes.GetVSize()), trace_global(fes.GetVSize()),
        local(fes.GetNE())
   {
      const MPI_Comm comm = fes.GetComm();
      int rank;
      MPI_Comm_rank(comm, &rank);
      multiplicity = 0;
      first = -1;
      trace_local = -1;
      trace_owner = -1;
      trace_global = -1;
      offsets[0] = 0;
      Array<int> edofs, balance(fes.GetVSize());
      balance = 0;
      for (int e = 0; e < fes.GetNE(); ++e)
      {
         fes.GetElementDofs(e, edofs);
         for (int j = 0; j < edofs.Size(); ++j)
         {
            const int d = Index(edofs[j]);
            if (essential[d]) { continue; }
            MFEM_VERIFY(first[d] < 0 || owners[first[d]] != e,
                        "Self-periodic elements are not supported.");
            if (first[d] < 0) { first[d] = dofs.Size(); }
            ++multiplicity[d];
            balance[d] += int(Sign(edofs[j]));
            local[e].Append(j);
            dofs.Append(edofs[j]);
            owners.Append(e);
         }
         offsets[e+1] = dofs.Size();
      }
      // Shared DOFs have one copy on each of two ranks. Reduce and broadcast
      // their counts; never count replicated entries in Krylov inner products.
      fes.GroupComm().Reduce<int>(multiplicity, GroupCommunicator::Sum);
      fes.GroupComm().Bcast(multiplicity);
      fes.GroupComm().Reduce<int>(balance, GroupCommunicator::Sum);
      fes.GroupComm().Bcast(balance);
      int ntrace = 0;
      for (int d = 0; d < fes.GetVSize(); ++d)
      {
         MFEM_VERIFY(multiplicity[d] <= 2, "Expected a conforming RT space.");
         if (multiplicity[d] != 2) { continue; }
         // Opposite signs in the globally oriented RT space mean that the
         // native element coordinates are outward fluxes. C has two +1s.
         MFEM_VERIFY(balance[d] == 0, "Inconsistent outward flux orientations.");
         if (fes.GetLocalTDofNumber(d) >= 0)
         {
            trace_local[d] = ntrace++;
            trace_owner[d] = rank;
         }
      }
      trace_partition.Init(comm, ntrace);
      flux_partition.Init(comm, dofs.Size());
      // Communicate an int owner and an int local index, preserving BigInt
      // global indices without relying on floating-point representations.
      fes.GroupComm().Bcast(trace_local);
      fes.GroupComm().Bcast(trace_owner);
      for (int d = 0; d < fes.GetVSize(); ++d)
      {
         if (trace_local[d] >= 0)
         {
            trace_global[d] = trace_partition.starts[trace_owner[d]]+trace_local[d];
         }
      }
      Array<int> I(dofs.Size()+1);
      Array<HYPRE_BigInt> J;
      I[0] = 0;
      for (int i = 0; i < dofs.Size(); ++i)
      {
         const HYPRE_BigInt row = trace_global[Index(dofs[i])];
         if (row >= 0) { J.Append(row); }
         I[i+1] = J.Size();
      }
      Vector data(J.Size());
      data = 1.0;
      Ct = ParallelMatrix(comm, flux_partition, trace_partition, I, J, data);
      C.reset(new TransposeOperator(Ct.get()));
   }

   void Prolongate(const Vector &x, Vector &broken) const
   {
      Vector conforming(fes.GetVSize());
      fes.Dof_TrueDof_Matrix()->Mult(x, conforming);
      broken.SetSize(dofs.Size());
      for (int i = 0; i < dofs.Size(); ++i)
      {
         broken(i) = Sign(dofs[i])*conforming(Index(dofs[i]));
      }
   }

   void Assemble(const Vector &broken, Vector &x) const
   {
      Vector conforming(fes.GetVSize());
      conforming = 0.0;
      for (int i = 0; i < dofs.Size(); ++i)
      {
         conforming(Index(dofs[i])) += Sign(dofs[i])*broken(i);
      }
      x.SetSize(fes.GetTrueVSize());
      fes.Dof_TrueDof_Matrix()->MultTranspose(conforming, x);
   }

   void Average(const Vector &broken, Vector &x) const
   {
      Assemble(broken, x);
      for (int d = 0; d < multiplicity.Size(); ++d)
      {
         const int tdof = fes.GetLocalTDofNumber(d);
         if (tdof >= 0 && multiplicity[d]) { x(tdof) /= multiplicity[d]; }
      }
   }
};

class ElementOperator : public Operator
{
   struct Block
   {
      DenseMatrix matrix;
      unique_ptr<DenseMatrixInverse> inverse;
   };
   const BrokenSpace &space;
   vector<unique_ptr<Block>> blocks;

public:
   ElementOperator(BrokenSpace &space_, const Vector *lift = nullptr,
                   VectorCoefficient *f = nullptr, Vector *rhs = nullptr)
      : Operator(space_.dofs.Size()), space(space_), blocks(space.fes.GetNE())
   {
      ParFiniteElementSpace &fes = space.fes;
      ConstantCoefficient one(1.0);
      DivDivIntegrator divdiv(one);
      VectorFEMassIntegrator mass(one);
      DenseMatrix full, mass_matrix;
      Vector load, el_lift, correction;
      Array<int> edofs;
      if (rhs) { rhs->SetSize(height); *rhs = 0.0; }
      for (int e = 0; e < fes.GetNE(); ++e)
      {
         const int n = space.local[e].Size();
         if (n == 0) { continue; }
         const FiniteElement &fe = *fes.GetFE(e);
         ElementTransformation &T = *fes.GetElementTransformation(e);
         divdiv.AssembleElementMatrix(fe, T, full);
         mass.AssembleElementMatrix(fe, T, mass_matrix);
         full += mass_matrix;
         blocks[e].reset(new Block);
         DenseMatrix &A = blocks[e]->matrix;
         A.SetSize(n);
         for (int j = 0; j < n; ++j)
         {
            for (int i = 0; i < n; ++i)
            {
               A(i,j) = full(space.local[e][i], space.local[e][j]);
            }
         }
         blocks[e]->inverse.reset(new DenseMatrixInverse(A, true));
         if (rhs)
         {
            VectorFEDomainLFIntegrator lf(*f);
            lf.AssembleRHSElementVect(fe, T, load);
            fes.GetElementDofs(e, edofs);
            lift->GetSubVector(edofs, el_lift);
            correction.SetSize(edofs.Size());
            full.Mult(el_lift, correction);
            load -= correction;
            for (int i = 0; i < n; ++i)
            {
               (*rhs)(space.offsets[e]+i) = load(space.local[e][i]);
            }
         }
      }
   }

   void Mult(const Vector &x, Vector &y) const override { Apply(x, y, false); }

   void Apply(const Vector &x, Vector &y, bool inverse) const
   {
      y.SetSize(height);
      for (int e = 0; e < static_cast<int>(blocks.size()); ++e)
      {
         if (!blocks[e]) { continue; }
         const int n = space.local[e].Size();
         Vector xe(const_cast<real_t*>(x.GetData())+space.offsets[e], n);
         Vector ye(y.GetData()+space.offsets[e], n);
         if (inverse) { blocks[e]->inverse->Mult(xe, ye); }
         else { blocks[e]->matrix.Mult(xe, ye); }
      }
   }

   // Only called for RT0 LOR subcells. RAP sums contributions on multiplier
   // owners and assembles both the on-rank and cross-rank couplings in H_lor.
   unique_ptr<HypreParMatrix> HybridizedMatrix(BrokenSpace &lo) const
   {
      SparseMatrix inverse(height);
      DenseMatrix el;
      for (int e = 0; e < static_cast<int>(blocks.size()); ++e)
      {
         if (!blocks[e]) { continue; }
         blocks[e]->inverse->GetInverseMatrix(el);
         for (int j = 0; j < el.Width(); ++j)
         {
            for (int i = 0; i < el.Height(); ++i)
            {
               inverse.Add(space.offsets[e]+i, space.offsets[e]+j, el(i,j));
            }
         }
      }
      inverse.Finalize();
      Partition &part = lo.flux_partition;
      HypreParMatrix parallel_inverse(lo.fes.GetComm(), part.GlobalSize(),
                                      part.Data(), &inverse);
      unique_ptr<HypreParMatrix> H(RAP(&parallel_inverse, lo.Ct.get()));
      H->CopyRowStarts();
      H->CopyColStarts();
      return H;
   }
};

class ElementInverse : public Solver
{
   const ElementOperator &A;
public:
   ElementInverse(const ElementOperator &A_) : Solver(A_.Height()), A(A_) { }
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed operator."); }
   void Mult(const Vector &x, Vector &y) const override { A.Apply(x, y, true); }
};

// Expose the LOCAL signed IH permutation, rather than the owned true-DOF
// permutation returned by GetDofPermutation(). Native LOR parallel numbering
// is needed for the independent refined-face multiplier ownership below.
class LocalLORDiscretization : public ParLORDiscretization
{
public:
   LocalLORDiscretization(ParFiniteElementSpace &fes) : ParLORDiscretization(fes) { }
   void GetLocalPermutation(Array<int> &perm) const
   {
      ConstructLocalDofPermutation(perm);
   }
};

// Find the corresponding high-order macro-broken flux copy, with the same
// parent element, and its sign relative to a native outward subcell flux.
int MacroCopy(const BrokenSpace &ho, const BrokenSpace &lo, int i,
              const Array<int> &perm, const CoarseFineTransformations &cf,
              real_t &sign)
{
   const int ld = lo.dofs[i], hd = perm[Index(ld)];
   const int parent = cf.embeddings[lo.owners[i]].parent;
   for (int j = ho.offsets[parent]; j < ho.offsets[parent+1]; ++j)
   {
      if (Index(ho.dofs[j]) == Index(hd))
      {
         sign = Sign(ld)*Sign(hd)*Sign(ho.dofs[j]);
         return j;
      }
   }
   MFEM_ABORT("Missing macro-broken LOR flux coordinate.");
   return -1;
}

unique_ptr<HypreParMatrix> MultiplierInjection(
   BrokenSpace &ho, BrokenSpace &lo, const Array<int> &perm,
   const CoarseFineTransformations &cf)
{
   const int nrows = lo.trace_partition.LocalSize();
   Array<int> I(nrows+1);
   Array<HYPRE_BigInt> columns(nrows);
   Vector values(nrows);
   columns = -1;
   for (int d = 0; d < lo.trace_global.Size(); ++d)
   {
      if (lo.trace_owner[d] != lo.trace_partition.rank) { continue; }
      const HYPRE_BigInt col = ho.trace_global[Index(perm[d])];
      if (col < 0) { continue; } // Internal subcell multiplier: zero injection.
      real_t sign;
      MacroCopy(ho, lo, lo.first[d], perm, cf, sign);
      const int row = lo.trace_local[d];
      columns[row] = col;
      values(row) = sign;
   }
   Array<HYPRE_BigInt> J;
   vector<real_t> entries;
   I[0] = 0;
   for (int r = 0; r < nrows; ++r)
   {
      if (columns[r] >= 0)
      {
         J.Append(columns[r]);
         entries.push_back(values(r));
      }
      I[r+1] = J.Size();
   }
   Vector data(int(entries.size()));
   for (int i = 0; i < data.Size(); ++i) { data(i) = entries[i]; }
   unique_ptr<HypreParMatrix> injection = ParallelMatrix(
                                             ho.fes.GetComm(), lo.trace_partition,
                                             ho.trace_partition, I, J, data);
   Vector ones(ho.trace_partition.LocalSize()), refined(nrows), count(ones.Size());
   ones = 1.0;
   injection->Mult(ones, refined);
   injection->MultTranspose(refined, count);
   for (int i = 0; i < count.Size(); ++i)
   {
      MFEM_VERIFY(count(i) == 1.0, "Expected one LOR face per IH macro-face DOF.");
   }
   return injection;
}

class MultiplierInverse : public Solver
{
   const HypreParMatrix &J;
   unique_ptr<HypreBoomerAMG> amg;
   mutable Vector rhs, solution;
public:
   MultiplierInverse(const HypreParMatrix *H, const HypreParMatrix &J_)
      : Solver(J_.Width()), J(J_), rhs(J_.Height()), solution(J_.Height())
   {
      if (H)
      {
         amg.reset(new HypreBoomerAMG(*H));
         amg->SetPrintLevel(0);
         amg->SetMaxIter(1);
         amg->SetTol(0.0);
         amg->SetCycleType(1);
         amg->SetRelaxType(18); // l1-Jacobi, identical pre/post relaxation.
         amg->SetCycleNumSweeps(1, 1);
         HYPRE_BoomerAMGSetCycleRelaxType(*amg, 9, 3); // Direct coarse solve.
         amg->iterative_mode = false;
         rhs = 0.0;
         solution = 0.0;
         amg->Mult(rhs, solution); // Build the hierarchy during setup.
      }
   }
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed operator."); }
   void Mult(const Vector &x, Vector &y) const override
   {
      J.Mult(x, rhs);
      solution = 0.0;
      if (amg) { amg->Mult(rhs, solution); } // All ranks participate, even empty ones.
      J.MultTranspose(solution, y);
   }
};

real_t CheckTransfer(const BrokenSpace &ho, const BrokenSpace &lo,
                     const Array<int> &perm, const CoarseFineTransformations &cf,
                     const HypreParMatrix &J)
{
   Vector u(ho.dofs.Size()), q(lo.dofs.Size());
   for (int i = 0; i < u.Size(); ++i)
   {
      u(i) = sin(0.7*(ho.flux_partition.Begin()+i+1));
   }
   for (int i = 0; i < q.Size(); ++i)
   {
      real_t sign;
      const int j = MacroCopy(ho, lo, i, perm, cf, sign);
      q(i) = sign*u(j);
   }
   Vector jump(ho.C->Height()), expected(lo.C->Height()), actual(lo.C->Height());
   ho.C->Mult(u, jump);
   J.Mult(jump, expected);
   lo.C->Mult(q, actual);
   actual -= expected;
   const MPI_Comm comm = ho.fes.GetComm();
   return GlobalNorm(comm, actual)/max(real_t(1.0), GlobalNorm(comm, expected));
}

bool CheckPreconditioner(MPI_Comm comm, const Partition &part,
                         const MultiplierInverse &B)
{
   Vector v(B.Width()), w(v.Size()), bv(v.Size()), bw(v.Size()), sum(v.Size());
   for (int i = 0; i < v.Size(); ++i)
   {
      v(i) = sin(0.37*(part.Begin()+i+1));
      w(i) = cos(0.53*(part.Begin()+i+1));
   }
   B.Mult(v, bv);
   B.Mult(w, bw);
   const real_t vBw = InnerProduct(comm, v, bw);
   const real_t wBv = InnerProduct(comm, w, bv);
   const real_t symmetry = abs(vBw-wBv)/max(real_t(1.0), max(abs(vBw), abs(wBv)));
   add(v, w, sum);
   Vector bsum(v.Size());
   B.Mult(sum, bsum);
   bsum -= bv;
   bsum -= bw;
   const real_t linearity = GlobalNorm(comm, bsum)/
                            max(real_t(1.0), GlobalNorm(comm, bv)+GlobalNorm(comm, bw));
   const real_t energy = InnerProduct(comm, v, bv);
   int rank;
   MPI_Comm_rank(comm, &rank);
   if (!rank)
   {
      cout << "Multiplier preconditioner symmetry error: " << symmetry
           << "\nMultiplier preconditioner linearity error: " << linearity
           << "\nMultiplier preconditioner test energy: " << energy << '\n';
   }
   return isfinite(symmetry) && symmetry <= 1e-10 &&
          isfinite(linearity) && linearity <= 1e-12 &&
          isfinite(energy) && (part.GlobalSize() == 0 || energy > 0.0);
}

} // namespace

#endif // MFEM_USE_MPI

int main(int argc, char *argv[])
{
#ifndef MFEM_USE_MPI
   cerr << "This example requires MFEM_USE_MPI=YES.\n";
   return 1;
#else
   Mpi::Init(argc, argv);
   Hypre::Init();
   const MPI_Comm comm = MPI_COMM_WORLD;
   const int rank = Mpi::WorldRank();
   const int nranks = Mpi::WorldSize();
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 2, refinements = 0, max_iter = 1000;
   real_t tolerance = 1e-10;
   bool verify = false, visualization = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Conforming quad/hex mesh file.");
   args.AddOption(&order, "-o", "--order", "Order p >= 1 (RT_{p-1}).");
   args.AddOption(&refinements, "-r", "--refine",
                  "Uniform refinements before partitioning.");
   args.AddOption(&max_iter, "-mi", "--max-iterations",
                  "Maximum total MINRES iterations.");
   args.AddOption(&tolerance, "-tol", "--tolerance",
                  "Relative residual tolerance.");
   args.AddOption(&verify, "-verify", "--verify", "-no-verify", "--no-verify",
                  "Check transfers, AMG symmetry/linearity, and a conforming CG solve.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable GLVis visualization.");
   args.Parse();
   if (!args.Good()) { if (!rank) { args.PrintUsage(cout); } return 1; }
   if (!rank) { args.PrintOptions(cout); }
   if (order < 1 || refinements < 0 || max_iter < 1 ||
       !isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0)
   {
      if (!rank)
      {
         cerr << "Require order >= 1, refine >= 0, max-iterations >= 1, and 0 < tol < 1.\n";
      }
      return 1;
   }

   unique_ptr<Mesh> serial_mesh(new Mesh(mesh_file, 1, 1));
   const int dim = serial_mesh->Dimension();
   if ((dim != 2 && dim != 3) || serial_mesh->SpaceDimension() != dim ||
       serial_mesh->Nonconforming() || serial_mesh->GetNE() == 0)
   {
      if (!rank) { cerr << "A conforming volume mesh in dimension 2 or 3 is required.\n"; }
      return 1;
   }
   const Geometry::Type geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   for (int e = 0; e < serial_mesh->GetNE(); ++e)
   {
      if (serial_mesh->GetElementBaseGeometry(e) != geometry)
      {
         if (!rank) { cerr << "Only quadrilateral or hexahedral elements are supported.\n"; }
         return 1;
      }
   }
   for (int r = 0; r < refinements; ++r) { serial_mesh->UniformRefinement(); }
   const int ne = serial_mesh->GetNE();
   if (nranks > ne)
   {
      if (!rank) { cerr << "Use at least as many macroelements as MPI ranks; increase -r.\n"; }
      return 1;
   }
   ParMesh mesh(comm, *serial_mesh);
   serial_mesh.reset();
   RT_FECollection fec(order-1, dim, BasisType::GaussLobatto,
                       BasisType::IntegratedGLL);
   ParFiniteElementSpace fes(&mesh, &fec);
   Array<int> ess_tdofs, essential(fes.GetVSize());
   essential = 0;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialVDofs(ess_bdr, essential);
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs);
   }
   VectorFunctionCoefficient exact(dim, ExactSolution), force(dim, RightHandSide);
   ParGridFunction lift(&fes);
   lift.ProjectCoefficient(exact);
   for (int d = 0; d < lift.Size(); ++d)
   {
      if (!essential[d]) { lift(d) = 0.0; }
   }

   StopWatch timer;
   timer.Start();
   BrokenSpace ho(fes, essential);
   Vector load;
   ElementOperator A(ho, &lift, &force, &load);
   ElementInverse local_inverse(A);
   LocalLORDiscretization lor(fes);
   ParFiniteElementSpace &mapped_lor_fes = lor.GetParFESpace();
   Array<int> perm;
   lor.GetLocalPermutation(perm);
   // LORDiscretization copies the high-order prolongation to its FE space.
   // Use the same refined mesh and local numbering, with native RT0 parallel
   // prolongation and ownership for the full refined multiplier system.
   ParFiniteElementSpace lor_fes(mapped_lor_fes.GetParMesh(),
                                 mapped_lor_fes.FEColl());
   Array<int> lor_essential(lor_fes.GetVSize());
   for (int d = 0; d < perm.Size(); ++d)
   {
      lor_essential[d] = essential[Index(perm[d])];
   }
   BrokenSpace lo(lor_fes, lor_essential);
   ElementOperator A_lor(lo);
   unique_ptr<HypreParMatrix> H_lor;
   if (lo.trace_partition.GlobalSize()) { H_lor = A_lor.HybridizedMatrix(lo); }
   const CoarseFineTransformations &cf =
      lor_fes.GetMesh()->GetRefinementTransforms();
   unique_ptr<HypreParMatrix> J = MultiplierInjection(ho, lo, perm, cf);
   MultiplierInverse multiplier_inverse(H_lor.get(), *J);
   timer.Stop();
   real_t local_time = timer.RealTime(), setup_time;
   MPI_Allreduce(&local_time, &setup_time, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
   // GlobalTrueVSize is collective: call it on every rank before printing.
   const HYPRE_BigInt conforming_size = fes.GlobalTrueVSize();
   const HYPRE_BigInt local_lor_elements = lor_fes.GetNE();
   HYPRE_BigInt lor_elements;
   MPI_Allreduce(&local_lor_elements, &lor_elements, 1, HYPRE_MPI_BIG_INT,
                 MPI_SUM, comm);
   if (!rank)
   {
      cout << "MPI ranks: " << nranks
           << "\nMacro elements: " << ne
           << "\nConforming flux DOFs: " << conforming_size
           << "\nFree broken flux DOFs: " << ho.flux_partition.GlobalSize()
           << "\nMacro-interface multipliers: " << ho.trace_partition.GlobalSize()
           << "\nLOR elements: " << lor_elements
           << "\nFull LOR multipliers: " << lo.trace_partition.GlobalSize()
           << "\nSetup time (maximum over ranks): " << setup_time << " seconds\n";
   }

   // The GLOBAL block structure must agree on all ranks, including ranks with
   // zero owned multipliers or zero free flux DOFs.
   const int nblocks = ho.trace_partition.GlobalSize() ? 2 : 1;
   Array<int> offsets(nblocks+1);
   offsets[0] = 0;
   offsets[1] = A.Height();
   if (nblocks == 2) { offsets[2] = A.Height()+ho.C->Height(); }
   BlockOperator K(offsets);
   BlockDiagonalPreconditioner B(offsets);
   K.SetDiagonalBlock(0, &A);
   B.SetDiagonalBlock(0, &local_inverse);
   if (nblocks == 2)
   {
      K.SetBlock(1, 0, ho.C.get());
      K.SetBlock(0, 1, ho.Ct.get());
      B.SetDiagonalBlock(1, &multiplier_inverse);
   }
   BlockVector rhs(offsets), solution(offsets), residual(offsets);
   rhs = 0.0;
   rhs.GetBlock(0) = load;
   solution = 0.0;
   MINRESSolver minres(comm);
   minres.SetOperator(K);
   minres.SetPreconditioner(B);
   minres.SetRelTol(tolerance);
   minres.SetAbsTol(0.0);
   minres.SetPrintLevel(rank ? -1 : 1);
   const real_t scale = max(real_t(1.0), GlobalNorm(comm, load));
   Vector correction, assembled_load, broken, product, conforming_residual;
   ho.Assemble(load, assembled_load);
   const real_t conforming_scale = max(real_t(1.0), GlobalNorm(comm,
                                                               assembled_load));
   real_t flux_residual = 0.0, jump_residual = 0.0, conf_residual = 0.0;
   bool success = false;
   int iterations = 0;
   timer.Clear();
   timer.Start();
   while (true)
   {
      if (ho.flux_partition.GlobalSize())
      {
         minres.SetMaxIter(max_iter-iterations);
         minres.Mult(rhs, solution);
         iterations += minres.GetNumIterations();
      }
      K.Mult(solution, residual);
      residual -= rhs;
      flux_residual = GlobalNorm(comm, residual.GetBlock(0))/scale;
      jump_residual = nblocks == 2 ? GlobalNorm(comm,
                                                residual.GetBlock(1))/scale : 0.0;
      ho.Average(solution.GetBlock(0), correction);
      ho.Prolongate(correction, broken);
      A.Mult(broken, product);
      ho.Assemble(product, conforming_residual);
      conforming_residual -= assembled_load;
      conf_residual = GlobalNorm(comm, conforming_residual)/conforming_scale;
      success = isfinite(flux_residual) && flux_residual <= tolerance &&
                isfinite(jump_residual) && jump_residual <= tolerance &&
                isfinite(conf_residual) && conf_residual <= tolerance;
      if (success || iterations >= max_iter || !minres.GetConverged() ||
          minres.GetNumIterations() == 0 ||
          !isfinite(flux_residual+jump_residual+conf_residual)) { break; }
      break;
      // Check the recovered conforming problem as well as the saddle-point
      // equations. Averaging can amplify a remaining interface discontinuity.
      if (!rank) { cout << "Refining the iterate to meet the true residual tolerance.\n"; }
      minres.iterative_mode = true;
      minres.SetRelTol(min(real_t(0.1), real_t(0.1)*tolerance/
                           max(flux_residual, max(jump_residual, conf_residual))));
   }
   timer.Stop();
   local_time = timer.RealTime();
   real_t solve_time;
   MPI_Allreduce(&local_time, &solve_time, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
   if (!rank)
   {
      cout << "Total MINRES iterations: " << iterations
           << "\nSolve time (maximum over ranks): " << solve_time << " seconds"
           << "\nTrue flux residual (scaled): " << flux_residual
           << "\nContinuity residual (scaled): " << jump_residual
           << "\nConforming residual (scaled): " << conf_residual << '\n';
   }

   if (verify)
   {
      const real_t transfer_error = CheckTransfer(ho, lo, perm, cf, *J);
      if (!rank) { cout << "Two-level constraint transfer error: " << transfer_error << '\n'; }
      const bool preconditioner_ok = CheckPreconditioner(comm, ho.trace_partition,
                                                         multiplier_inverse);
      success = success && isfinite(transfer_error) && transfer_error <= 1e-12 &&
                preconditioner_ok;
      // Independent conforming assembly and Jacobi-preconditioned CG check;
      // no SuiteSparse or parallel sparse direct solver is required.
      ConstantCoefficient one(1.0);
      ParBilinearForm reference(&fes);
      reference.AddDomainIntegrator(new DivDivIntegrator(one));
      reference.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      reference.Assemble();
      ParLinearForm reference_load(&fes);
      reference_load.AddDomainIntegrator(new VectorFEDomainLFIntegrator(force));
      reference_load.Assemble();
      ParGridFunction reference_solution(&fes);
      reference_solution = lift;
      OperatorPtr reference_matrix;
      Vector b, x;
      reference.FormLinearSystem(ess_tdofs, reference_solution, reference_load,
                                 reference_matrix, x, b);
      HypreDiagScale jacobi(*reference_matrix.As<HypreParMatrix>());
      CGSolver cg(comm);
      cg.SetOperator(*reference_matrix);
      cg.SetPreconditioner(jacobi);
      cg.SetRelTol(min(real_t(1e-13), tolerance*real_t(0.01)));
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(20000);
      cg.SetPrintLevel(-1);
      cg.Mult(b, x);
      Vector difference;
      reference.RecoverFEMSolution(x, reference_load, reference_solution);
      reference_solution -= lift;
      reference_solution.GetTrueDofs(difference);
      const real_t norm = max(real_t(1.0), GlobalNorm(comm, difference));
      difference -= correction;
      const real_t error = GlobalNorm(comm, difference)/norm;
      if (!rank)
      {
         cout << "Conforming reference CG iterations: " << cg.GetNumIterations()
              << "\nConforming reference-solve difference (scaled): " << error << '\n';
      }
      success = success && cg.GetConverged() && isfinite(error) &&
                error <= max(real_t(100.0)*tolerance, real_t(1e-12));
   }

   ParGridFunction u(&fes), recovered(&fes);
   recovered.SetFromTrueDofs(correction);
   u = lift;
   u += recovered;
   const real_t l2_error = u.ComputeL2Error(exact);
   if (!rank) { cout << "L2 error: " << l2_error << '\n'; }
   if (!success)
   {
      if (!rank) { cerr << "Solve failed the convergence or verification checks.\n"; }
      return 2;
   }
   ostringstream suffix;
   suffix << '.' << setfill('0') << setw(6) << rank;
   ofstream mesh_out("hbp.mesh"+suffix.str()), solution_out("hbp.gf"+suffix.str());
   mesh_out.precision(16);
   solution_out.precision(16);
   mesh.Print(mesh_out);
   u.Save(solution_out);
   if (visualization)
   {
      socketstream socket("localhost", 19916);
      socket.precision(8);
      socket << "parallel " << nranks << ' ' << rank << '\n'
             << "solution\n" << mesh << u << flush;
   }
   return 0;
#endif
}
