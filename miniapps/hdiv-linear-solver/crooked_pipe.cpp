// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "discrete_divergence.hpp"
#include "hdiv_linear_solver.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref = 0, int par_ref = 0);
void SolveCG(Operator &A, Solver &P, const Vector &B, Vector &X);
void CompareToReference(Vector &x, Vector &x_ref);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "./CrookedPipeMesh_1.mesh";
   const char *device_config = "cpu";
   int order = 3;
   int ser_ref = 0;
   int par_ref = 0;
   bool visualization = false;
   bool use_saddle_point = false;
   bool use_ams = false;
   bool use_lor_ams = false;
   bool use_hybridization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable ParaView.");
   args.AddOption(&use_saddle_point,
                  "-sp", "--saddle-point", "-no-sp", "--no-saddle-point",
                  "Enable or disable saddle-point solver.");
   args.AddOption(&use_ams, "-ams", "--ams", "-no-ams", "--no-ams",
                  "Enable or disable AMS solver.");
   args.AddOption(&use_lor_ams, "-lor", "--lor-ams", "-no-lor", "--no-lor-ams",
                  "Enable or disable LOR-AMS solver.");
   args.AddOption(&use_hybridization,
                  "-hb", "--hybridization", "-no-hb", "--no-hybridization",
                  "Enable or disable hybridization solver.");
   args.ParseCheck();

   if (!use_saddle_point && !use_ams && !use_lor_ams && !use_hybridization)
   {
      if (Mpi::Root()) { cout << "No solver enabled. Exiting.\n"; }
      return 0;
   }

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLegendre;
   RT_FECollection fec_rt(order-1, dim, b1, b2);
   ParFiniteElementSpace fes_rt(&mesh, &fec_rt);
   HYPRE_BigInt ndofs_rt = fes_rt.GlobalTrueVSize();
   if (Mpi::Root()) { cout << "\nRT DOFs: " << ndofs_rt << endl; }

   Array<int> ess_rt_dofs; // empty

   // Crooked pipe coefficients
   const IntegrationRule &ir = IntRules.Get(
      mesh.GetElementGeometry(0),
      2*(order - 1) + mesh.GetElementTransformation(0)->OrderW());
   const IntegrationRule &ir_rt = IntRules.Get(
      mesh.GetElementGeometry(0),
      2*order + mesh.GetElementTransformation(0)->OrderW());

   QuadratureSpace qs_l2(mesh, ir);
   QuadratureSpace qs_rt(mesh, ir_rt);
   QuadratureFunction qf_l2(qs_l2);
   QuadratureFunction qf_rt(qs_rt);

   if (Mpi::Root()) cout << "Coefficients... " << flush;
   tic_toc.Clear();
   tic_toc.Start();
   Vector alpha_vec(mesh.attributes.Max());
   alpha_vec = 1.64098370E+00;
   alpha_vec(0) = 1.88046595E-03;
   PWConstCoefficient alpha_coeff_pw(alpha_vec);
   alpha_coeff_pw.Project(qf_l2);
   QuadratureFunctionCoefficient alpha_coeff(qf_l2);

   Vector beta_vec(mesh.attributes.Max());
   beta_vec = 0.2;
   beta_vec(0) = 2000;
   PWConstCoefficient beta_coeff_pw(beta_vec);
   beta_coeff_pw.Project(qf_rt);
   QuadratureFunctionCoefficient beta_coeff(qf_rt);
   if (Mpi::Root()) cout << "Done. " << tic_toc.RealTime() << endl;

   // Right-hand side
   Vector ones(dim);
   ones = 1.0;
   VectorConstantCoefficient f_coeff(ones);
   if (Mpi::Root()) cout << "Right-hand side... " << flush;
   tic_toc.Clear();
   tic_toc.Start();
   ParLinearForm b(&fes_rt);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_coeff));
   b.UseFastAssembly(true);
   b.Assemble();
   if (Mpi::Root()) cout << "Done. " << tic_toc.RealTime() << endl;

   ParGridFunction x(&fes_rt);
   Vector x_ref;

   if (use_saddle_point)
   {
      if (Mpi::Root()) { cout << "\nSaddle point solver... " << flush; }
      tic_toc.Clear(); tic_toc.Start();

      const int mt = FiniteElement::INTEGRAL;
      L2_FECollection fec_l2(order-1, dim, b2, mt);
      ParFiniteElementSpace fes_l2(&mesh, &fec_l2);

      HdivSaddlePointSolver saddle_point_solver(
         mesh, fes_rt, fes_l2, alpha_coeff, beta_coeff, ess_rt_dofs,
         HdivSaddlePointSolver::Mode::GRAD_DIV);

      const Array<int> &offsets = saddle_point_solver.GetOffsets();

      BlockVector X_block(offsets), B_block(offsets);
      B_block.GetBlock(0) = 0.0;
      b.ParallelAssemble(B_block.GetBlock(1));
      B_block.GetBlock(1) *= -1.0;
      B_block.SyncFromBlocks();

      // No essential BCs, don't need to call SetBC
      // saddle_point_solver.SetBC(X_block.GetBlock(1));

      X_block = 0.0;
      StopWatch sw;
      sw.Start();
      saddle_point_solver.Mult(B_block, X_block);
      if (Mpi::Root()) { cout << "\nSolve: " << sw.RealTime() << endl; }

      if (Mpi::Root())
      {
         cout << "Done.\nIterations: "
              << saddle_point_solver.GetNumIterations()
              << "\nElapsed: " << tic_toc.RealTime() << endl;
      }

      X_block.SyncToBlocks();
      x.SetFromTrueDofs(X_block.GetBlock(1));

      CompareToReference(x, x_ref);
   }

   if (use_ams)
   {
      if (Mpi::Root()) { cout << "\nADS solver... " << flush; }
      tic_toc.Clear(); tic_toc.Start();

      ParBilinearForm a(&fes_rt);
      a.AddDomainIntegrator(new DivDivIntegrator(alpha_coeff, &ir));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(beta_coeff));
      a.Assemble();

      OperatorHandle A;
      Vector B, X;
      b.Assemble();
      x = 0.0;
      a.FormLinearSystem(ess_rt_dofs, x, b, A, X, B);
      HypreParMatrix &Ah = *A.As<HypreParMatrix>();

      std::unique_ptr<Solver> prec;
      if (dim == 2) { prec.reset(new HypreAMS(Ah, &fes_rt)); }
      else  { prec.reset(new HypreADS(Ah, &fes_rt)); }

      SolveCG(Ah, *prec, B, X);
      x.SetFromTrueDofs(X);

      CompareToReference(x, x_ref);
   }

   if (use_lor_ams)
   {
      const int b2_lor = BasisType::IntegratedGLL;
      RT_FECollection fec_rt_lor(order-1, dim, b1, b2_lor);
      ParFiniteElementSpace fes_rt_lor(&mesh, &fec_rt_lor);

      ParLinearForm b_lor(&fes_rt_lor);
      b_lor.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_coeff));
      b_lor.UseFastAssembly(true);
      b_lor.Assemble();

      if (Mpi::Root()) { cout << "\nLOR-ADS solver... " << flush; }
      tic_toc.Clear(); tic_toc.Start();

      ParBilinearForm a(&fes_rt_lor);
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.AddDomainIntegrator(new DivDivIntegrator(alpha_coeff_pw, &ir));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(beta_coeff_pw));
      a.Assemble();

      ParGridFunction x_lor(&fes_rt_lor);
      x_lor = 0.0;

      OperatorHandle A;
      Vector B, X;
      a.FormLinearSystem(ess_rt_dofs, x_lor, b_lor, A, X, B);

      std::unique_ptr<Solver> prec;
      if (dim == 2) { prec.reset(new LORSolver<HypreAMS>(a, ess_rt_dofs)); }
      else { prec.reset(new LORSolver<HypreADS>(a, ess_rt_dofs)); }

      SolveCG(*A, *prec, B, X);
      a.RecoverFEMSolution(X, b_lor, x_lor);
   }

   if (use_hybridization)
   {
      if (Mpi::Root()) { cout << "\nHybridization solver... " << flush; }
      tic_toc.Clear(); tic_toc.Start();

      DG_Interface_FECollection fec_hb(order-1, dim);
      ParFiniteElementSpace fes_hb(&mesh, &fec_hb);

      ParBilinearForm a(&fes_rt);
      a.AddDomainIntegrator(new DivDivIntegrator(alpha_coeff, &ir));
      a.AddDomainIntegrator(new VectorFEMassIntegrator(beta_coeff));
      a.EnableHybridization(&fes_hb, new NormalTraceJumpIntegrator, ess_rt_dofs);
      a.Assemble();

      OperatorHandle A;
      Vector B, X;
      b.Assemble();
      x = 0.0;
      a.FormLinearSystem(ess_rt_dofs, x, b, A, X, B);

      HypreBoomerAMG amg_hb(*A.As<HypreParMatrix>());
      amg_hb.SetPrintLevel(0);

      StopWatch sw;
      sw.Start();
      SolveCG(*A, amg_hb, B, X);
      if (Mpi::Root()) { cout << "   [Solve: " << sw.RealTime() << "]\n"; }
      a.RecoverFEMSolution(X, b, x);

      CompareToReference(x, x_ref);
   }

   if (visualization)
   {
      ParaViewDataCollection pv("CrookedPipe", &mesh);
      pv.SetPrefixPath("ParaView");
      pv.SetHighOrderOutput(true);
      pv.SetLevelsOfDetail(order + 1);
      pv.RegisterField("u", &x);
      pv.SetCycle(0);
      pv.Save();
   }

   return 0;
}

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}

void SolveCG(Operator &A, Solver &P, const Vector &B, Vector &X)
{
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(1000);
   cg.SetPrintLevel(0);
   cg.SetOperator(A);
   cg.SetPreconditioner(P);
   X = 0.0;
   cg.Mult(B, X);
   if (Mpi::Root())
   {
      cout << "Done.\nIterations: " << cg.GetNumIterations()
           << "\nElapsed: " << tic_toc.RealTime() << endl;
   }
};

void CompareToReference(Vector &x, Vector &x_ref)
{
   if (x_ref.Size() > 0)
   {
      Vector y = x_ref;
      y -= x;
      const double nrm2 = sqrt(InnerProduct(MPI_COMM_WORLD, y, y));
      if (Mpi::Root()) { cout << "Norm difference: " << nrm2 << endl; }
   }
   else
   {
      x_ref = x;
   }
}
