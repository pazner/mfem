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
#include "broken_hdiv.hpp"
#include <cmath>
#include <iostream>
#include <string>

using namespace mfem;
using namespace std;

/// Multiplier Schur complement C A^{-1} C^T, with identity on inactive
/// multipliers. A is constrained (identity on essential broken DOFs), and C^T
/// has no entries in their rows (see VerifyConstraintStructure), so the
/// essential DOFs do not contribute.
class MultiplierSchurComplement : public Operator
{
   const Operator &A_inverse;
   const Operator &Ct;
   const Array<int> &inactive_dofs;
   mutable Vector z1, z2;

public:
   MultiplierSchurComplement(const Operator &A_inverse_, const Operator &Ct_,
                             const Array<int> &inactive_dofs_)
      : Operator(Ct_.Width()), A_inverse(A_inverse_), Ct(Ct_),
        inactive_dofs(inactive_dofs_),
        z1(Ct_.Height()), z2(Ct_.Height())
   {
      z1.UseDevice(true);
      z2.UseDevice(true);
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      Ct.Mult(x, z1);
      A_inverse.Mult(z1, z2);
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
#if defined(HYPRE_USING_GPU) && MFEM_HYPRE_VERSION >= 22100
   // Use hypre's SpMV kernels rather than the vendor library: cuSPARSE was much
   // slower for some of the AMG interpolation matrices.
   HYPRE_SetSpMVUseVendor(0);
#endif

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
   const string mesh_name(mesh_file);
   const string crooked_pipe_mesh = "CrookedPipeMesh_1.mesh";
   const bool crooked_pipe = mesh_name.size() >= crooked_pipe_mesh.size() &&
                             mesh_name.compare(
                                mesh_name.size() - crooked_pipe_mesh.size(),
                                crooked_pipe_mesh.size(), crooked_pipe_mesh) == 0;
   RT_FECollection fec(order - 1, dim, BasisType::GaussLobatto,
                       BasisType::IntegratedGLL);
   ParFiniteElementSpace fes_rt(&mesh, &fec);
   const HYPRE_BigInt n_global_dofs = fes_rt.GlobalTrueVSize();
   if (Mpi::Root()) { cout << "Global RT DOFs: " << n_global_dofs << '\n'; }

   VectorFunctionCoefficient exact_coeff(dim, ExactSolution);
   VectorFunctionCoefficient manufactured_rhs(dim, RightHandSide);
   // As in crooked_pipe.cpp, the crooked pipe uses the constant source (1,...,1).
   Vector ones(dim);
   ones = 1.0;
   VectorConstantCoefficient constant_rhs(ones);
   VectorCoefficient &rhs_coeff = crooked_pipe ?
                                  static_cast<VectorCoefficient&>(constant_rhs) :
                                  manufactured_rhs;

   // As in crooked_pipe.cpp, the crooked pipe has natural boundary conditions
   // (no essential DOFs), so no boundary values are needed. Otherwise, the
   // essential boundary values come from the exact solution.
   Array<int> ess_tdofs;
   ParGridFunction x(&fes_rt);
   if (crooked_pipe) { x = 0.0; }
   else
   {
      fes_rt.GetBoundaryTrueDofs(ess_tdofs);
      x.ProjectCoefficient(exact_coeff);
   }

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
   if (crooked_pipe)
   {
      if (Mpi::Root())
      {
         cout << "Using crooked pipe coefficients and natural boundary "
              "conditions.\n";
      }
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

   Array<int> ess_broken_dofs;
   fes_hat.GetBrokenEssentialDofs(ess_tdofs, ess_broken_dofs);

   unique_ptr<Operator> A_hat;
   if (fast_diag)
   {
      A_hat.reset(new BrokenHdivOperator(fes_rt, alpha.get(), beta.get()));
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

   // Eliminate the essential flux values u_e from the right-hand side, as
   // ConstrainedOperator::EliminateRHS: g -= A u_e and g_e = u_e. The
   // multiplier block is unchanged, since C has no entries in the essential
   // columns (checked below).
   {
      Vector &u = solution.GetBlock(0), &g = rhs.GetBlock(0);
      Vector u_e(u.Size()), A_u_e(u.Size()), values;
      u_e.UseDevice(true);
      A_u_e.UseDevice(true);
      u.GetSubVector(ess_broken_dofs, values);
      u_e = 0.0;
      u_e.SetSubVector(ess_broken_dofs, values);
      A_hat->Mult(u_e, A_u_e);
      g -= A_u_e;
      g.SetSubVector(ess_broken_dofs, values);
   }
   // Constrain A_hat: identity rows and columns for the essential DOFs.
   if (fast_diag)
   {
      static_cast<BrokenHdivOperator&>(*A_hat).SetEssentialDofs(
         fes_rt, ess_broken_dofs);
   }
   else
   {
      auto &A = static_cast<SparseMatrix&>(*A_hat);
      // The matrix was applied on the device above; eliminate on the host.
      A.HostReadI();
      A.HostReadJ();
      A.HostReadWriteData();
      const int *h_ess = ess_broken_dofs.HostRead();
      for (int k = 0; k < ess_broken_dofs.Size(); ++k)
      {
         A.EliminateRowCol(h_ess[k], Operator::DIAG_ONE);
      }
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
   const Array<int> &inactive_dofs = fes_hat.GetInactiveTraceDofs();
   VerifyConstraintStructure(Ct, ess_broken_dofs, inactive_dofs);

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
      CGSolver A_inverse;
      // Set the operator first: the element inverse rejects SetOperator.
      A_inverse.SetOperator(*A_hat);
      A_inverse.SetPreconditioner(*element_inverse);
      A_inverse.SetRelTol(1e-12);
      A_inverse.SetAbsTol(0.0);
      A_inverse.SetMaxIter(1000);
      A_inverse.iterative_mode = false;

      Vector &g = rhs.GetBlock(0), &h = rhs.GetBlock(1);
      Vector &u = solution.GetBlock(0), &lambda = solution.GetBlock(1);

      // Multiplier right-hand side C A^{-1} g - h.
      Vector z(g.Size()), r(h.Size());
      z.UseDevice(true);
      r.UseDevice(true);
      A_inverse.Mult(g, z);
      Ct.MultTranspose(z, r);
      r -= h;
      CopyEntries(inactive_dofs, h, r);

      MultiplierSchurComplement S(A_inverse, Ct, inactive_dofs);
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
      z.Neg();
      z += g;
      A_inverse.Mult(z, u);
   }
   else
   {
      HybridSaddlePointOperator K(*A_hat, Ct, inactive_dofs);
      BlockDiagonalPreconditioner preconditioner(offsets);
      preconditioner.SetDiagonalBlock(0, element_inverse.get());
      preconditioner.SetDiagonalBlock(1, trace_inverse.get());
      MINRESSolver minres(MPI_COMM_WORLD);
      minres.SetPreconditioner(preconditioner);
      minres.SetOperator(K);
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
