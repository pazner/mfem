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
#include "discrete_divergence.hpp"

using namespace std;
using namespace mfem;

bool grad_div_problem = true;

void TestSameMatrices(SparseMatrix &A1, const SparseMatrix &A2,
                      HYPRE_BigInt *cmap1=nullptr,
                      std::unordered_map<HYPRE_BigInt,int> *cmap2inv=nullptr)
{
   MFEM_VERIFY(A1.Height() == A2.Height(), "");
   int n = A1.Height();

   const int *I1 = A1.HostReadI();
   const int *J1 = A1.HostReadJ();
   const double *V1 = A1.HostReadData();

   A2.HostReadI();
   A2.HostReadJ();
   A2.HostReadData();

   double error = 0.0;

   for (int i=0; i<n; ++i)
   {
      for (int jj=I1[i]; jj<I1[i+1]; ++jj)
      {
         int j = J1[jj];
         if (cmap1)
         {
            if (cmap2inv->count(cmap1[j]) > 0)
            {
               j = (*cmap2inv)[cmap1[j]];
            }
            else
            {
               error = std::max(error, std::fabs(V1[jj]));
               continue;
            }
         }
         error = std::max(error, std::fabs(V1[jj] - A2(i,j)));
      }
   }

   MFEM_VERIFY(std::abs(error) <= 1e-10, "");
}

void TestSameMatrices(HypreParMatrix &A1, const HypreParMatrix &A2)
{
   HYPRE_BigInt *cmap1, *cmap2;
   SparseMatrix diag1, offd1, diag2, offd2;

   A1.GetDiag(diag1);
   A2.GetDiag(diag2);
   A1.GetOffd(offd1, cmap1);
   A2.GetOffd(offd2, cmap2);

   TestSameMatrices(diag1, diag2);

   if (cmap1)
   {
      std::unordered_map<HYPRE_BigInt,int> cmap2inv;
      for (int i=0; i<offd2.Width(); ++i) { cmap2inv[cmap2[i]] = i; }
      TestSameMatrices(offd1, offd2, cmap1, &cmap2inv);
   }
   else
   {
      TestSameMatrices(offd1, offd2);
   }
}

void MakeTranspose(SparseMatrix &A, SparseMatrix &B)
{
   SparseMatrix *tmp = Transpose(A);
   B.Swap(*tmp);
   delete tmp;
}

ParMesh LoadParMesh(const char *mesh_file, int ser_ref = 0, int par_ref = 0)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}

HypreParMatrix *DiagonalInverse(Vector &diag_vec, ParFiniteElementSpace &fes)
{
   diag_vec.HostReadWrite();
   for (int i=0; i<diag_vec.Size(); ++i) { diag_vec[i] = 1.0/diag_vec[i]; }
   SparseMatrix diag_inv(diag_vec);

   HYPRE_BigInt global_size = fes.GlobalTrueVSize();
   HYPRE_BigInt *row_starts = fes.GetTrueDofOffsets();
   HypreParMatrix D(MPI_COMM_WORLD, global_size, row_starts, &diag_inv);
   return new HypreParMatrix(D); // make a deep copy
}

HypreParMatrix *DiagonalInverse(HypreParMatrix &A, ParFiniteElementSpace &fes)
{
   Vector diag_vec;
   A.GetDiag(diag_vec);
   return DiagonalInverse(diag_vec, fes);
}

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../../data/star.mesh";
   const char *device_config = "cpu";
   int ser_ref = 1;
   int par_ref = 1;
   int order = 3;
   bool sanity_check = false;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&sanity_check, "-s", "--sanity-check", "-no-s",
                  "--no-sanity-check", "Enable or disable sanity check.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   VectorFunctionCoefficient f_vec_coeff(dim, f_vec), u_vec_coeff(dim, u_vec);

   int b1 = BasisType::GaussLobatto, b2 = BasisType::IntegratedGLL;
   int mt = FiniteElement::INTEGRAL;

   RT_FECollection fec_rt(order-1, dim, b1, b2);
   ParFiniteElementSpace fes_rt(&mesh, &fec_rt);

   L2_FECollection fec_l2(order-1, dim, b2, mt);
   ParFiniteElementSpace fes_l2(&mesh, &fec_l2);

   {
      HYPRE_BigInt ndofs_rt = fes_rt.GlobalTrueVSize();
      HYPRE_BigInt ndofs_l2 = fes_l2.GlobalTrueVSize();
      if (Mpi::Root())
      {
         cout << "Number of RT DOFs: " << ndofs_rt << endl;
         cout << "Number of L2 DOFs: " << ndofs_l2 << endl;
         cout << "Assembling... " << flush;
      }
   }

   Array<int> ess_dofs, empty;
   fes_rt.GetBoundaryTrueDofs(ess_dofs);

   // Form the 2x2 block system
   //
   //     [ M    D^t   ]
   //     [ D  -W^{-1} ]
   //
   // where M is the RT mass matrix, D is the discrete divergence operator,
   // and W is the L2 mass matrix.

   // Form M
   ParBilinearForm mass_rt(&fes_rt);
   mass_rt.AddDomainIntegrator(new VectorFEMassIntegrator);
   mass_rt.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_rt.Assemble();
   OperatorHandle M;
   mass_rt.FormSystemMatrix(ess_dofs, M);

   // Form D
   unique_ptr<HypreParMatrix> D(
      FormDiscreteDivergenceMatrix(fes_rt, fes_l2, ess_dofs));
   unique_ptr<HypreParMatrix> Dt(D->Transpose());

   // Form W^{-1}
   unique_ptr<Solver> W_inv;
   if (order <= 2) { W_inv.reset(new DGMassInverse_Direct(fes_l2)); }
   else { W_inv.reset(new DGMassInverse(fes_l2)); }

   if (Mpi::Root()) { cout << "Done." << endl; }

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = fes_rt.GetTrueVSize();
   offsets[2] = offsets[1] + fes_l2.GetTrueVSize();

   BlockOperator A_block(offsets);
   A_block.SetBlock(0, 0, M.Ptr());
   A_block.SetBlock(0, 1, Dt.get());
   A_block.SetBlock(1, 0, D.get());
   A_block.SetBlock(1, 1, W_inv.get(), -1.0);


   // We precondition the 2x2 block system defined above with the block-diagonal
   // preconditioner
   //
   //     [ M^{-1}     0   ]
   //     [   0     S^{-1} ]
   //
   // where M^{-1} is approximated by the reciprocal of the diagonal of M,
   // and S^{-1} is approximate by one AMG V-cycle applied to the approximate
   // Schur complement S, given by
   //
   //     S = W^{-1} + D M^{-1} D^t
   //
   // where W^{-1} is the recriprocal of the diagonal of W, and M^{-1} is the
   // reciprocal of the diagonal of M.

   // Form the diagonal of M
   Vector M_diag(fes_rt.GetTrueVSize());
   mass_rt.AssembleDiagonal(M_diag);
   OperatorJacobiSmoother M_jacobi(M_diag, ess_dofs);

   // Form the diagonal of W
   Vector W_diag(fes_l2.GetTrueVSize());
   {
      ParBilinearForm mass_l2(&fes_l2);
      mass_l2.AddDomainIntegrator(new MassIntegrator);
      mass_l2.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      mass_l2.Assemble();
      mass_l2.AssembleDiagonal(W_diag);
   }
   unique_ptr<HypreParMatrix> W_diag_inv(DiagonalInverse(W_diag, fes_l2));

   // Form the approximate Schur complement
   unique_ptr<HypreParMatrix> S;
   {
      unique_ptr<HypreParMatrix> M_diag_inv(DiagonalInverse(M_diag, fes_rt));
      unique_ptr<HypreParMatrix> D_Minv_Dt(RAP(M_diag_inv.get(), Dt.get()));
      S.reset(ParAdd(D_Minv_Dt.get(), W_diag_inv.get()));
   }

   // Create the block-diagonal preconditioner
   HypreBoomerAMG S_inv(*S);
   S_inv.SetPrintLevel(0);

   BlockDiagonalPreconditioner D_prec(offsets);
   D_prec.SetDiagonalBlock(0, &M_jacobi);
   D_prec.SetDiagonalBlock(1, &S_inv);

   BlockVector X_block(offsets), B_block(offsets);
   X_block = 0.0;

   B_block.GetBlock(0).Randomize(1);
   B_block.GetBlock(1) = 0.0;

   MINRESSolver minres(MPI_COMM_WORLD);
   minres.SetAbsTol(0.0);
   minres.SetRelTol(1e-12);
   minres.SetMaxIter(500);
   minres.SetPrintLevel(IterativeSolver::PrintLevel().Summary().FirstAndLast());
   minres.SetOperator(A_block);
   minres.SetPreconditioner(D_prec);

   tic_toc.Clear();
   tic_toc.Start();
   minres.Mult(B_block, X_block);
   tic_toc.Stop();
   if (Mpi::Root()) { cout << "MINRES Elapsed: " << tic_toc.RealTime() << '\n'; }

   // If the sanity check is enabled, compare the solution of block 2x2 system
   // to that of the RT div-div system
   if (sanity_check)
   {
      if (Mpi::Root()) { cout << "\nPerforming sanity check...\n" << endl; }

      ParBilinearForm a(&fes_rt);
      a.AddDomainIntegrator(new DivDivIntegrator);
      a.AddDomainIntegrator(new VectorFEMassIntegrator);
      a.Assemble();
      a.Finalize();

      ParGridFunction x(&fes_rt);
      ParLinearForm b(&fes_rt);
      b = 0.0;
      x = 0.0;
      OperatorHandle A;

      a.FormSystemMatrix(ess_dofs, A);

      Vector X(offsets[1]), B(offsets[1]);
      X = 0.0;
      {
         Vector &B0 = B_block.GetBlock(0);
         B0.HostRead();
         B.HostWrite();
         for (int i = 0; i < B0.Size(); ++i)
         {
            B[i] = B0[i];
         }
      }

      unique_ptr<Solver> amg;
      if (dim == 3)
      {
         amg.reset(new HypreADS(*A.As<HypreParMatrix>(), &fes_rt));
      }
      else
      {
         amg.reset(new HypreAMS(*A.As<HypreParMatrix>(), &fes_rt));
      }

      CGSolver cg(MPI_COMM_WORLD);
      cg.SetAbsTol(0.0);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(500);
      cg.SetPrintLevel(1);
      cg.SetPrintLevel(IterativeSolver::PrintLevel().Summary().FirstAndLast());
      cg.SetOperator(*A);
      cg.SetPreconditioner(*amg);

      tic_toc.Clear();
      tic_toc.Start();
      cg.Mult(B, X);
      tic_toc.Stop();
      if (Mpi::Root()) { cout << "CG Elapsed:     " << tic_toc.RealTime() << '\n'; }

      {
         Vector &X0 = X_block;
         X.HostReadWrite();
         X0.HostRead();
         for (int i = 0; i < X.Size(); ++i)
         {
            X[i] -= X0[i];
         }
      }

      double error = X.Normlinf();
      if (Mpi::Root()) { cout << "Error: " << error << '\n'; }
   }

   return 0;
}
