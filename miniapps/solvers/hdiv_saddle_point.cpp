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

ParMesh LoadParMesh(const char *mesh_file)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   return ParMesh(MPI_COMM_WORLD, serial_mesh);
}

HypreParMatrix *DiagonalInverse(HypreParMatrix &A, ParFiniteElementSpace &fes)
{
   Vector diag_vec;
   A.GetDiag(diag_vec);
   for (int i=0; i<diag_vec.Size(); ++i) { diag_vec[i] = 1.0/diag_vec[i]; }
   SparseMatrix diag_inv(diag_vec);

   HYPRE_BigInt global_size = fes.GlobalTrueVSize();
   HYPRE_BigInt *row_starts = fes.GetTrueDofOffsets();
   HypreParMatrix D(MPI_COMM_WORLD, global_size, row_starts, &diag_inv);
   return new HypreParMatrix(D); // make a deep copy
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

   ParMesh mesh = LoadParMesh(mesh_file);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }

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

   Array<int> ess_dofs;

   ParDiscreteLinearOperator div(&fes_rt, &fes_l2);
   div.AddDomainInterpolator(new DivergenceInterpolator);
   div.Assemble();
   div.Finalize();

   unique_ptr<HypreParMatrix> D(div.ParallelAssemble());
   unique_ptr<HypreParMatrix> Dt(D->Transpose());

   ParBilinearForm mass_rt(&fes_rt);
   mass_rt.AddDomainIntegrator(new VectorFEMassIntegrator);
   mass_rt.Assemble();
   mass_rt.Finalize();
   unique_ptr<HypreParMatrix> M(mass_rt.ParallelAssemble());

   ParBilinearForm mass_l2(&fes_l2);
   mass_l2.AddDomainIntegrator(new MassIntegrator);
   mass_l2.Assemble();
   mass_l2.Finalize();
   unique_ptr<HypreParMatrix> W(mass_l2.ParallelAssemble());

   ParBilinearForm mass_l2_inv(&fes_l2);
   mass_l2_inv.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator));
   mass_l2_inv.Assemble();
   mass_l2_inv.Finalize();
   unique_ptr<HypreParMatrix> W_inv(mass_l2_inv.ParallelAssemble());

   if (Mpi::Root()) { cout << "Done." << endl; }

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = fes_rt.GetTrueVSize();
   offsets[2] = offsets[1] + fes_l2.GetTrueVSize();

   BlockOperator A_block(offsets);
   A_block.SetBlock(0, 0, M.get());
   A_block.SetBlock(0, 1, Dt.get());
   A_block.SetBlock(1, 0, D.get());
   A_block.SetBlock(1, 1, W_inv.get(), -1.0);

   HypreSmoother M_jacobi(*M, HypreSmoother::Jacobi);

   unique_ptr<HypreParMatrix> M_diag_inv(DiagonalInverse(*M, fes_rt));
   unique_ptr<HypreParMatrix> W_diag_inv(DiagonalInverse(*W, fes_l2));

   unique_ptr<HypreParMatrix> S;
   {
      unique_ptr<HypreParMatrix> D_Minv_Dt(RAP(M_diag_inv.get(), Dt.get()));
      S.reset(ParAdd(D_Minv_Dt.get(), W_inv.get()));
   }

   HypreBoomerAMG S_inv(*S);
   S_inv.SetPrintLevel(0);

   BlockDiagonalPreconditioner D_prec(offsets);
   D_prec.SetDiagonalBlock(0, M_diag_inv.get());
   D_prec.SetDiagonalBlock(1, &S_inv);

   BlockVector X_block(offsets), B_block(offsets);
   X_block = 0.0;
   B_block.GetBlock(0).Randomize(1);
   B_block.GetBlock(1) = 0.0;

   MINRESSolver minres(MPI_COMM_WORLD);
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
   if (Mpi::Root()) { cout << "MINRES Elapsed: " << tic_toc.RealTime() << '\n'; }

   return 0;
}