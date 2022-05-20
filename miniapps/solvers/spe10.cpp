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

#include "discrete_divergence.hpp"

using namespace std;
using namespace mfem;

static constexpr int nx = 60;
static constexpr int ny = 220;
static constexpr int nz = 850;

static constexpr double hx = 20.0;
static constexpr double hy = 10.0;
static constexpr double hz = 2.0;

ParMesh MakeParMesh()
{
   Mesh mesh = Mesh::MakeCartesian3D(
                  nx, ny, nz, Element::HEXAHEDRON, nx*hx, ny*hy, nz*hz);
   return ParMesh(MPI_COMM_WORLD, mesh);
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

struct SPE10Coefficient : DiagonalMatrixCoefficient
{
   Array<double> coeff_data;
   SPE10Coefficient() : VectorCoefficient(3), coeff_data(3*nx*ny*nz)
   {
      ifstream permfile("spe_perm.dat");
      if (!permfile.good())
      {
         MFEM_ABORT("Cannot open data file spe_perm.dat.")
      }
      int n = 3*nx*ny*nz;
      for (int i = 0; i < n; ++i)
      {
         double val;
         permfile >> val;
         coeff_data[i] = 1.0/val;
      }
   }
   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      double data[3];
      Vector xvec(data, 3);
      T.Transform(ip, xvec);

      const double x = xvec[0], y = xvec[1], z = xvec[2];
      unsigned int i = nx-1-(int)floor(x/hx/(1+3e-16));
      MFEM_ASSERT(i >= 0 && i<nx, "");
      unsigned int j = (int)floor(y/hy/(1+3e-16));
      MFEM_ASSERT(j >= 0 && j < ny, "");
      unsigned int k = nz-1-(int)floor(z/hz/(1+3e-16));
      MFEM_ASSERT(k >= 0 && k < nz, "");

      V[0] = coeff_data[ny*nx*k + nx*j + i];
      V[1] = coeff_data[ny*nx*k + nx*j + i + nx*ny*nz];
      V[2] = coeff_data[ny*nx*k + nx*j + i + 2*nx*ny*nz];
   }
};

int main(int argc, char *argv[])
{
   tic_toc.Start();

   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *device_config = "cpu";
   int order = 3;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = MakeParMesh();
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   int b1 = BasisType::GaussLobatto, b2 = BasisType::IntegratedGLL;
   int mt = FiniteElement::INTEGRAL;

   RT_FECollection fec_rt(order-1, dim, b1, b2);
   ParFiniteElementSpace fes_rt(&mesh, &fec_rt);

   L2_FECollection fec_l2(order-1, dim, b2, mt);
   ParFiniteElementSpace fes_l2(&mesh, &fec_l2);

   tic_toc.Stop();
   if (Mpi::Root()) { cout << "Preamble: " << tic_toc.RealTime() << endl; }

   tic_toc.Clear();
   tic_toc.Start();
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

   SPE10Coefficient beta_coeff;

   // Form the 2x2 block system
   //
   //     [ M    D^t   ]
   //     [ D  -W^{-1} ]
   //
   // where M is the RT mass matrix, D is the discrete divergence operator,
   // and W is the L2 mass matrix.

   // Form M
   ParBilinearForm mass_rt(&fes_rt);
   mass_rt.AddDomainIntegrator(new VectorFEMassIntegrator(&beta_coeff));
   mass_rt.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   mass_rt.Assemble();
   OperatorHandle M;
   mass_rt.FormSystemMatrix(ess_dofs, M);

   // Form D
   unique_ptr<HypreParMatrix> D(
      FormDiscreteDivergenceMatrix(fes_rt, fes_l2, ess_dofs));
   unique_ptr<HypreParMatrix> Dt(D->Transpose());

   // Form W^{-1}
   unique_ptr<Operator> W_inv;
   if (order <= 2) { W_inv.reset(new DGMassInverse_Direct(fes_l2)); }
   else { W_inv.reset(new DGMassInverse(fes_l2)); }

   tic_toc.Stop();
   if (Mpi::Root()) { cout << "Done. Elapsed: " << tic_toc.RealTime() << endl; }
   tic_toc.Clear();
   tic_toc.Start();
   if (Mpi::Root()) { cout << "Preconditioner setup... " << flush; }

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
   // S_inv.SetPrintLevel(0);
   S_inv.SetPrintLevel(1);

   BlockDiagonalPreconditioner D_prec(offsets);
   D_prec.SetDiagonalBlock(0, &M_jacobi);
   D_prec.SetDiagonalBlock(1, &S_inv);

   BlockVector X_block(offsets), B_block(offsets);
   X_block = 0.0;

   B_block.GetBlock(0).Randomize(1);
   B_block.GetBlock(1) = 0.0;

   S_inv.Setup(B_block.GetBlock(1), X_block.GetBlock(1)); // AMG setup

   tic_toc.Stop();
   if (Mpi::Root())
   {
      cout << "Done. Elapsed: " << tic_toc.RealTime() << endl;
   }

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

   return 0;
}
