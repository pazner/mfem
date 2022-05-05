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
//
//                     ---------------------------------
//                     H(div) saddle-point system solver
//                     ---------------------------------
//

#include "mfem.hpp"
#include "linalg/dtensor.hpp"
#include <fstream>
#include <iostream>
#include <memory>

#include "lor_mms.hpp"

using namespace std;
using namespace mfem;

bool grad_div_problem = true;

void MakeTranspose(SparseMatrix &A, SparseMatrix &B)
{
   SparseMatrix *tmp = Transpose(A);
   B.Swap(*tmp);
   delete tmp;
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../../data/star.mesh";
   int ref_levels = 1;
   int order = 3;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.ParseCheck();

   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

   VectorFunctionCoefficient f_vec_coeff(dim, f_vec), u_vec_coeff(dim, u_vec);

   int b1 = BasisType::GaussLobatto, b2 = BasisType::IntegratedGLL;
   int mt = FiniteElement::INTEGRAL;

   RT_FECollection fec_rt(order-1, dim, b1, b2);
   FiniteElementSpace fes_rt(&mesh, &fec_rt);

   L2_FECollection fec_l2(order-1, dim, b2, mt);
   FiniteElementSpace fes_l2(&mesh, &fec_l2);

   cout << "Number of RT DOFs: " << fes_rt.GetTrueVSize() << endl;
   cout << "Number of L2 DOFs: " << fes_l2.GetTrueVSize() << endl;

   Array<int> ess_dofs;

   DiscreteLinearOperator div(&fes_rt, &fes_l2);
   div.AddDomainInterpolator(new DivergenceInterpolator);
   div.Assemble();
   div.Finalize();
   SparseMatrix &D = div.SpMat();
   SparseMatrix Dt;
   MakeTranspose(D, Dt);
   {
      std::ofstream f("D.txt");
      div.SpMat().PrintMatlab(f);
   }

   BilinearForm mass_rt(&fes_rt);
   mass_rt.AddDomainIntegrator(new VectorFEMassIntegrator);
   mass_rt.Assemble();
   mass_rt.Finalize();
   SparseMatrix &M = mass_rt.SpMat();

   BilinearForm dg_mass(&fes_l2);
   dg_mass.AddDomainIntegrator(new MassIntegrator);
   dg_mass.Assemble();
   dg_mass.Finalize();
   SparseMatrix &W = dg_mass.SpMat();

   BilinearForm dg_mass_inv(&fes_l2);
   dg_mass_inv.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator));
   dg_mass_inv.Assemble();
   dg_mass_inv.Finalize();
   SparseMatrix &W_inv = dg_mass_inv.SpMat();

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = fes_rt.GetTrueVSize();
   offsets[2] = offsets[1] + fes_l2.GetTrueVSize();

   BlockOperator A_block(offsets);
   A_block.SetBlock(0, 0, &M);
   A_block.SetBlock(0, 1, &Dt);
   A_block.SetBlock(1, 0, &D);
   A_block.SetBlock(1, 1, &W_inv, -1.0);

   DSmoother M_jacobi(M);

   Vector M_diag_vec;
   M.GetDiag(M_diag_vec);
   for (int i=0; i<M_diag_vec.Size(); ++i) { M_diag_vec[i] = 1.0/M_diag_vec[i]; }
   SparseMatrix M_diag_inv(M_diag_vec);

   Vector W_diag_vec;
   W.GetDiag(W_diag_vec);
   for (int i=0; i<W_diag_vec.Size(); ++i) { W_diag_vec[i] = 1.0/W_diag_vec[i]; }
   SparseMatrix W_diag_inv(W_diag_vec);

   SparseMatrix S;
   {
      SparseMatrix *tmp = RAP(M_diag_inv, D);
      S.Swap(*tmp);
      delete tmp;
   }
   // S.Add(1.0, W_diag_inv);
   S.Add(1.0, W_inv);

   HYPRE_BigInt row_starts[2];
   row_starts[0] = 0;
   row_starts[1] = S.Height();

   HypreParMatrix S_hypre(MPI_COMM_WORLD, fes_l2.GetTrueVSize(), row_starts, &S);
   HypreBoomerAMG S_inv(S_hypre);
   S_inv.SetPrintLevel(0);
   // UMFPackSolver S_inv(S);

   BlockDiagonalPreconditioner D_prec(offsets);
   D_prec.SetDiagonalBlock(0, &M_diag_inv);
   D_prec.SetDiagonalBlock(1, &S_inv);

   BlockVector X_block(offsets), B_block(offsets);
   X_block = 0.0;
   B_block.GetBlock(0).Randomize(1);
   B_block.GetBlock(1) = 0.0;

   MINRESSolver minres;
   minres.SetAbsTol(0.0);
   minres.SetRelTol(1e-12);
   minres.SetMaxIter(500);
   minres.SetPrintLevel(1);
   minres.SetPrintLevel(IterativeSolver::PrintLevel().Summary().FirstAndLast());
   minres.SetOperator(A_block);
   minres.SetPreconditioner(D_prec);

   tic_toc.Start();
   minres.Mult(B_block, X_block);
   tic_toc.Stop();
   std::cout << "MINRES Elapsed: " << tic_toc.RealTime() << '\n';

   // fes.GetBoundaryTrueDofs(ess_dofs);

   //    BilinearForm a(&fes);
   //    if (H1 || L2)
   //    {
   //       a.AddDomainIntegrator(new MassIntegrator);
   //       a.AddDomainIntegrator(new DiffusionIntegrator);
   //    }
   //    else
   //    {
   //       a.AddDomainIntegrator(new VectorFEMassIntegrator);
   //    }

   //    if (ND) { a.AddDomainIntegrator(new CurlCurlIntegrator); }
   //    else if (RT) { a.AddDomainIntegrator(new DivDivIntegrator); }
   //    else if (L2)
   //    {
   //       a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(-1.0, kappa));
   //       a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(-1.0, kappa));
   //    }
   //    // TODO: L2 diffusion not implemented with partial assembly
   //    if (!L2) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   //    a.Assemble();

   //    LinearForm b(&fes);
   //    if (H1 || L2) { b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff)); }
   //    else { b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_vec_coeff)); }
   //    if (L2)
   //    {
   //       // DG boundary conditions are enforced weakly with this integrator.
   //       b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(u_coeff, -1.0, kappa));
   //    }
   //    b.Assemble();

   //    GridFunction x(&fes);
   //    if (H1 || L2) { x.ProjectCoefficient(u_coeff);}
   //    else { x.ProjectCoefficient(u_vec_coeff); }

   //    Vector X, B;
   //    OperatorHandle A;
   //    a.FormLinearSystem(ess_dofs, x, b, A, X, B);

   // #ifdef MFEM_USE_SUITESPARSE
   //    LORSolver<UMFPackSolver> solv_lor(a, ess_dofs);
   // #else
   //    LORSolver<GSSmoother> solv_lor(a, ess_dofs);
   // #endif

   //    CGSolver cg;
   //    cg.SetAbsTol(0.0);
   //    cg.SetRelTol(1e-12);
   //    cg.SetMaxIter(500);
   //    cg.SetPrintLevel(1);
   //    cg.SetOperator(*A);
   //    cg.SetPreconditioner(solv_lor);
   //    cg.Mult(B, X);

   //    a.RecoverFEMSolution(X, b, x);

   //    double er =
   //       (H1 || L2) ? x.ComputeL2Error(u_coeff) : x.ComputeL2Error(u_vec_coeff);
   //    cout << "L2 error: " << er << endl;

   //    if (visualization)
   //    {
   //       // Save the solution and mesh to disk. The output can be viewed using
   //       // GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
   //       x.Save("sol.gf");
   //       mesh.Save("mesh.mesh");

   //       // Also save the solution for visualization using ParaView
   //       ParaViewDataCollection dc("LOR", &mesh);
   //       dc.SetPrefixPath("ParaView");
   //       dc.SetHighOrderOutput(true);
   //       dc.SetLevelsOfDetail(order);
   //       dc.RegisterField("u", &x);
   //       dc.SetCycle(0);
   //       dc.SetTime(0.0);
   //       dc.Save();
   //    }

   return 0;
}
