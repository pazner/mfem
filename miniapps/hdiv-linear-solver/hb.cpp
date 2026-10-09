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

// Solve the hybridized system with MINRES.
// Sample run: mpirun -np 2 hb -m ../../data/inline-quad.mesh -o 2

#include "mfem.hpp"
#include "hb.hpp"
#include "fast_diag.hpp"
#include <cmath>
#include <iostream>
#include <string>

using namespace mfem;
using namespace std;

/// Multiplier Schur complement C A^{-1} C^T, with identity on inactive
/// multipliers. Essential broken DOFs are eliminated from A.
class MultiplierSchurComplement : public Operator
{
   const Operator &A_inverse;
   const Operator &Ct;
   const Array<int> &ess_dofs;
   const Array<int> &inactive_dofs;
   mutable Vector z1, z2;

public:
   MultiplierSchurComplement(const Operator &A_inverse_, const Operator &Ct_,
                             const Array<int> &ess_dofs_,
                             const Array<int> &inactive_dofs_)
      : Operator(Ct_.Width()), A_inverse(A_inverse_), Ct(Ct_),
        ess_dofs(ess_dofs_), inactive_dofs(inactive_dofs_),
        z1(Ct_.Height()), z2(Ct_.Height())
   {
      z1.UseDevice(true);
      z2.UseDevice(true);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      Ct.Mult(x, z1);
      z1.SetSubVector(ess_dofs, 0.0);
      A_inverse.Mult(z1, z2);
      z2.SetSubVector(ess_dofs, 0.0);
      Ct.MultTranspose(z2, y);
      CopyEntries(inactive_dofs, x, y);
   }
};

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

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../../data/inline-quad.mesh";
   const char *device_config = "cpu";
   int order = 1;
   int ser_ref = 0;
   int par_ref = 0;
   bool direct_solver = false;
   bool fast_diag = false;
   bool use_cg = false;
   string trace_type = "graph";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&ser_ref, "-rs", "--refine-serial", "Serial mesh refinements.");
   args.AddOption(&par_ref, "-rp", "--refine-parallel",
                  "Parallel mesh refinements.");
   args.AddOption(&trace_type, "-t", "--trace",
                  "Trace solver: graph, hb, or lor.");
   args.AddOption(&direct_solver, "-ds", "--direct", "-a",
                  "--amg", "Direct solver or AMG for HB system.");
   args.AddOption(&fast_diag, "-fd", "--fast-diag", "-no-fd", "--no-fast-diag",
                  "Use fast diagonalization and partial assembly on Cartesian "
                  "elements.");
   args.AddOption(&use_cg, "-cg", "--cg", "-no-cg", "--no-cg",
                  "Solve the multiplier Schur complement with CG (inner CG "
                  "for A_hat) instead of MINRES on the full system.");
   args.ParseCheck();
   MFEM_VERIFY(trace_type == "graph" || trace_type == "hb" || trace_type == "lor",
               "Trace solver must be graph, hb, or lor.");
   MFEM_VERIFY(!direct_solver || Mpi::WorldSize() == 1,
               "The direct solver requires one MPI rank.");

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = [&]()
   {
      Mesh serial_mesh(mesh_file);
      for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
      return ParMesh(MPI_COMM_WORLD, serial_mesh);
   }();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();
   RT_FECollection fec(order - 1, dim, BasisType::GaussLobatto,
                       BasisType::IntegratedGLL);
   ParFiniteElementSpace fes_rt(&mesh, &fec);
   const HYPRE_BigInt n_global_dofs = fes_rt.GlobalTrueVSize();
   if (Mpi::Root()) { cout << "Global RT DOFs: " << n_global_dofs << '\n'; }

   VectorFunctionCoefficient exact_coeff(dim, ExactSolution);
   VectorFunctionCoefficient rhs_coeff(dim, RightHandSide);

   Array<int> ess_tdofs;
   fes_rt.GetBoundaryTrueDofs(ess_tdofs);

   ParGridFunction x(&fes_rt);
   x.ProjectCoefficient(exact_coeff);

   ParLinearForm b(&fes_rt);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(rhs_coeff));
   b.UseFastAssembly(true);
   b.Assemble();
   Vector f(fes_rt.GetTrueVSize());
   b.ParallelAssemble(f);

   ParBilinearForm a(&fes_rt);
   // Mass (alpha) and div (beta) coefficients: piecewise constant on the
   // crooked pipe mesh (attribute 1 is the pipe), unit otherwise.
   unique_ptr<Coefficient> alpha, beta;
   const string mesh_name(mesh_file);
   const string crooked_pipe = "CrookedPipeMesh_1.mesh";
   if (mesh_name.size() >= crooked_pipe.size() &&
       mesh_name.compare(mesh_name.size() - crooked_pipe.size(),
                         crooked_pipe.size(), crooked_pipe) == 0)
   {
      if (Mpi::Root()) { cout << "Using crooked pipe coefficients.\n"; }
      Vector alpha_vec(mesh.attributes.Max()), beta_vec(mesh.attributes.Max());
      alpha_vec = 0.2;
      alpha_vec(0) = 2000.0;
      beta_vec = 1.64098370E+00;
      beta_vec(0) = 1.88046595E-03;
      alpha.reset(new PWConstCoefficient(alpha_vec));
      beta.reset(new PWConstCoefficient(beta_vec));
   }
   a.AddDomainIntegrator(beta ? new DivDivIntegrator(*beta) :
                         new DivDivIntegrator);
   a.AddDomainIntegrator(alpha ? new VectorFEMassIntegrator(*alpha) :
                         new VectorFEMassIntegrator);

   BrokenSpace fes_hat(fes_rt);
   Array<int> offsets({0, fes_hat.GetNumBrokenDofs(),
                       fes_hat.GetNumBrokenDofs() + fes_hat.GetNumTraceDofs()});
   BlockVector rhs(offsets), solution(offsets);
   rhs.UseDevice(true);
   solution.UseDevice(true);
   rhs = 0.0;
   solution = 0.0;

   tic_toc.Restart();

   fes_hat.AverageTranspose(f, rhs.GetBlock(0));
   fes_hat.LocalProlongate(x, solution.GetBlock(0));

   unique_ptr<Operator> A_hat;
   if (fast_diag)
   {
      A_hat.reset(new BrokenElementPAOperator(a));
   }
   else
   {
      auto assembled = make_unique<SparseMatrix>(offsets[1]);
      DenseMatrix element_matrix, element_matrix_lex;
      const Array<int> &dof_map = GetDofMap(fes_rt);
      const int nd = fes_hat.GetNumDofsPerElem();
      Array<int> element_dofs(nd);
      for (int e = 0; e < fes_rt.GetNE(); ++e)
      {
         a.ComputeElementMatrix(e, element_matrix);
         NativeToLexicographic(dof_map, element_matrix, element_matrix_lex);
         for (int i = 0; i < nd; ++i) { element_dofs[i] = e*nd+i; }
         assembled->AddSubMatrix(element_dofs, element_dofs, element_matrix_lex);
      }
      assembled->Finalize();
      A_hat = std::move(assembled);
   }

   auto stype = direct_solver ? DIRECT : AMG;
   unique_ptr<Solver> trace_inverse;
   if (trace_type == "graph")
   {
      trace_inverse.reset(new GraphPreconditioner(fes_hat, ess_tdofs,
                                                  alpha.get(), beta.get(),
                                                  stype));
   }
   else
   {
      const auto amode = trace_type == "lor" ?
                         TraceMultiplierSolver::LOR : TraceMultiplierSolver::ACTUAL;
      trace_inverse.reset(new TraceMultiplierSolver(
                             a, ess_tdofs, amode, stype));
   }
   HypreParMatrix &Ct = [&]() -> HypreParMatrix &
   {
      if (trace_type == "graph") { return fes_hat.GetCt(); }
      return static_cast<TraceMultiplierSolver&>(*trace_inverse)
      .GetConstraintTranspose();
   }();
   TransposeOperator C(Ct);

   // Identity on inactive multipliers (explicit zeros elsewhere).
   Vector inactive_marker(fes_hat.GetNumTraceDofs());
   inactive_marker.UseDevice(true);
   inactive_marker = 0.0;
   inactive_marker.SetSubVector(fes_hat.GetInactiveTraceDofs(), 1.0);
   auto inactive_matrix = DiagonalSparseMatrix(inactive_marker);
   SparseMatrix &inactive = *inactive_matrix;
   BlockOperator K(offsets);
   K.SetBlock(0, 0, A_hat.get());
   K.SetBlock(0, 1, &Ct);
   K.SetBlock(1, 0, &C);
   K.SetBlock(1, 1, &inactive);

   Array<int> ess_broken_dofs;
   fes_hat.GetBrokenEssentialDofs(ess_tdofs, ess_broken_dofs);
   ConstrainedOperator constrained_K(&K, ess_broken_dofs);
   constrained_K.EliminateRHS(solution, rhs);
   unique_ptr<Solver> element_inverse;
   if (fast_diag)
   {
      element_inverse.reset(new FastDiagonalizationSolver(
                               fes_rt, ess_broken_dofs, alpha.get(), beta.get()));
   }
   else { element_inverse.reset(new BrokenElementSolver(a, ess_broken_dofs)); }
   if (use_cg)
   {
      // A_hat is block diagonal and local to each rank, so the inner solver
      // does not need a communicator.
      ConstrainedOperator constrained_A(A_hat.get(), ess_broken_dofs);
      CGSolver A_inverse;
      // Set the operator first: the element inverse rejects SetOperator.
      A_inverse.SetOperator(constrained_A);
      A_inverse.SetPreconditioner(*element_inverse);
      A_inverse.SetRelTol(1e-12);
      A_inverse.SetAbsTol(0.0);
      A_inverse.SetMaxIter(1000);
      A_inverse.iterative_mode = false;

      const Array<int> &inactive_dofs = fes_hat.GetInactiveTraceDofs();
      Vector &g = rhs.GetBlock(0), &h = rhs.GetBlock(1);
      Vector &u = solution.GetBlock(0), &lambda = solution.GetBlock(1);

      // Multiplier right-hand side C A^{-1} g - h.
      Vector z(g.Size()), r(h.Size());
      z.UseDevice(true);
      r.UseDevice(true);
      A_inverse.Mult(g, z);
      z.SetSubVector(ess_broken_dofs, 0.0);
      Ct.MultTranspose(z, r);
      r -= h;
      CopyEntries(inactive_dofs, h, r);

      MultiplierSchurComplement S(A_inverse, Ct, ess_broken_dofs, inactive_dofs);
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetPreconditioner(*trace_inverse);
      cg.SetOperator(S);
      cg.SetRelTol(1e-12);
      cg.SetAbsTol(0.0);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(1);
      cg.Mult(r, lambda);

      // Recover u = A^{-1} (g - C^T lambda); g retains the essential values.
      Ct.Mult(lambda, z);
      z.SetSubVector(ess_broken_dofs, 0.0);
      z.Neg();
      z += g;
      A_inverse.Mult(z, u);
   }
   else
   {
      BlockDiagonalPreconditioner preconditioner(offsets);
      preconditioner.SetDiagonalBlock(0, element_inverse.get());
      preconditioner.SetDiagonalBlock(1, trace_inverse.get());
      MINRESSolver minres(MPI_COMM_WORLD);
      minres.SetPreconditioner(preconditioner);
      minres.SetOperator(constrained_K);
      minres.SetRelTol(1e-12);
      minres.SetAbsTol(0.0);
      minres.SetMaxIter(1000);
      minres.SetPrintLevel(3);
      minres.Mult(rhs, solution);
   }

   if (Mpi::Root())
   {
      cout << "Elapsed: " << tic_toc.RealTime() << '\n';
   }

   return 0;
}
