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

#include "dgmassinv.hpp"
#include "bilinearform.hpp"
#include "dgmassinv_kernels.hpp"
#include "../general/forall.hpp"

#ifdef MFEM_USE_CUDA
#include <cusolverDn.h>
#endif

#define MFEM_NVTX_COLOR Crimson
#include "../general/nvtx.hpp"

namespace mfem
{

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_orig, Coefficient *coeff,
                             const IntegrationRule *ir,
                             int btype)
   : Solver(fes_orig.GetTrueVSize()),
     fec(fes_orig.GetMaxElementOrder(),
         fes_orig.GetMesh()->Dimension(),
         btype,
         fes_orig.GetFE(0)->GetMapType()),
     fes(fes_orig.GetMesh(), &fec)
{
   MFEM_VERIFY(fes.IsDGSpace(), "Space must be DG.");
   MFEM_VERIFY(!fes.IsVariableOrder(), "Variable orders not supported.");

   const int btype_orig =
      static_cast<const L2_FECollection*>(fes_orig.FEColl())->GetBasisType();

   if (btype_orig == btype)
   {
      // No change of basis required
      d2q = nullptr;
   }
   else
   {
      // original basis to solver basis
      const auto mode = DofToQuad::TENSOR;
      d2q = &fes_orig.GetFE(0)->GetDofToQuad(fes.GetFE(0)->GetNodes(), mode);

      int n = d2q->ndof;
      Array<double> B_inv = d2q->B; // deep copy
      Array<int> ipiv(n);
      // solver basis to original
      LUFactors lu(B_inv.HostReadWrite(), ipiv.HostWrite());
      lu.Factor(n);
      B_.SetSize(n*n);
      lu.GetInverseMatrix(n, B_.HostWrite());
      Bt_.SetSize(n*n);
      DenseMatrix B_matrix(B_.HostReadWrite(), n, n);
      DenseMatrix Bt_matrix(Bt_.HostWrite(), n, n);
      Bt_matrix.Transpose(B_matrix);
   }

   if (coeff) { m = new MassIntegrator(*coeff, ir); }
   else { m = new MassIntegrator(ir); }

   diag_inv.SetSize(height);
   // Workspace vectors used for CG
   r_.SetSize(height);
   d_.SetSize(height);
   z_.SetSize(height);
   // Only need transformed RHS if basis is different
   if (btype_orig != btype) { b2_.SetSize(height); }

   M = new BilinearForm(&fes);
   M->AddDomainIntegrator(m); // M assumes ownership of m
   M->SetAssemblyLevel(AssemblyLevel::PARTIAL);

   // Assemble the bilinear form and its diagonal (for preconditioning).
   Update();
}

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_, Coefficient &coeff,
                             int btype)
   : DGMassInverse(fes_, &coeff, nullptr, btype) { }

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_, Coefficient &coeff,
                             const IntegrationRule &ir, int btype)
   : DGMassInverse(fes_, &coeff, &ir, btype) { }

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_,
                             const IntegrationRule &ir, int btype)
   : DGMassInverse(fes_, nullptr, &ir, btype) { }

DGMassInverse::DGMassInverse(FiniteElementSpace &fes_, int btype)
   : DGMassInverse(fes_, nullptr, nullptr, btype) { }

void DGMassInverse::SetOperator(const Operator &op)
{
   MFEM_ABORT("SetOperator not supported with DGMassInverse.")
}

void DGMassInverse::SetRelTol(const double rel_tol_) { rel_tol = rel_tol_; }

void DGMassInverse::SetAbsTol(const double abs_tol_) { abs_tol = abs_tol_; }

void DGMassInverse::SetMaxIter(const double max_iter_) { max_iter = max_iter_; }

void DGMassInverse::Update()
{
   M->Assemble();
   M->AssembleDiagonal(diag_inv);
   internal::MakeReciprocal(diag_inv.Size(), diag_inv.ReadWrite());
}

DGMassInverse::~DGMassInverse()
{
   delete M;
}

template<int DIM, int D1D, int Q1D>
void DGMassInverse::DGMassCGIteration(const Vector &b_, Vector &u_) const
{
   NVTX("DG Mass Inverse");

   if (D1D == 0 || Q1D == 0)
   {
      std::cout << "Fallback!\n"
                << "d1d = " << m->dofs1D << '\n'
                << "q1d = " << m->quad1D << '\n'
                << "dim = " << DIM << '\n';
      MFEM_ABORT("Hit fallback kernel.");
   }

   using namespace internal; // host/device kernel functions

   const int NE = fes.GetNE();
   const int d1d = m->dofs1D;
   const int q1d = m->quad1D;

   const int ND = static_cast<int>(pow(d1d, DIM));

   const auto B = m->maps->B.Read();
   const auto Bt = m->maps->Bt.Read();
   const auto pa_data = m->pa_data.Read();
   const auto dinv = diag_inv.Read();
   auto r = r_.Write();
   auto d = d_.Write();
   auto z = z_.Write();
   auto u = u_.ReadWrite();

   const double RELTOL = rel_tol;
   const double ABSTOL = abs_tol;
   const double MAXIT = max_iter;
   const bool IT_MODE = iterative_mode;
   const bool CHANGE_BASIS = (d2q != nullptr);

   // b is the right-hand side (if no change of basis, this just points to the
   // incoming RHS vector, if we have to change basis, this points to the
   // internal b2 vector where we put the transformed RHS)
   const double *b;
   // the following are non-null if we have to change basis
   double *b2 = nullptr; // non-const access to b2
   const double *b_orig = nullptr; // RHS vector in "original" basis
   const double *d2q_B = nullptr; // matrix to transform initial guess
   const double *q2d_B = nullptr; // matrix to transform solution
   const double *q2d_Bt = nullptr; // matrix to transform RHS
   if (CHANGE_BASIS)
   {
      d2q_B = d2q->B.Read();
      q2d_B = B_.Read();
      q2d_Bt = Bt_.Read();

      b2 = b2_.Write();
      b_orig = b_.Read();
      b = b2;
   }
   else
   {
      b = b_.Read();
   }

   constexpr int NB = Q1D ? Q1D : 1; // block size

   MFEM_FORALL_2D(e, NE, NB, NB, 1,
   {
      constexpr int NB = Q1D ? Q1D : 1; // redefine here for some compilers

      // Perform change of basis if needed
      if (CHANGE_BASIS)
      {
         // Transform RHS
         DGMassBasis<DIM,D1D,MAX_D1D>(e, NE, q2d_Bt, b_orig, b2, d1d);
         if (IT_MODE)
         {
            // Transform initial guess
            DGMassBasis<DIM,D1D,MAX_D1D>(e, NE, d2q_B, u, u, d1d);
         }
      }

      const int tid = MFEM_THREAD_ID(x) + NB*MFEM_THREAD_ID(y);

      // Compute first residual
      if (IT_MODE)
      {
         DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, u, r, d1d, q1d);
         DGMassAxpy(e, NE, ND, 1.0, b, -1.0, r, r); // r = b - r
      }
      else
      {
         // if not in iterative mode, use zero initial guess
         const int BX = MFEM_THREAD_SIZE(x);
         const int BY = MFEM_THREAD_SIZE(y);
         const int bxy = BX*BY;
         const auto B = ConstDeviceMatrix(b, ND, NE);
         auto U = DeviceMatrix(u, ND, NE);
         auto R = DeviceMatrix(r, ND, NE);
         for (int i = tid; i < ND; i += bxy)
         {
            U(i, e) = 0.0;
            R(i, e) = B(i, e);
         }
         MFEM_SYNC_THREAD;
      }

      DGMassPreconditioner(e, NE, ND, dinv, r, z);
      DGMassAxpy(e, NE, ND, 1.0, z, 0.0, z, d); // d = z

      double nom = DGMassDot<NB>(e, NE, ND, d, r);
      if (nom < 0.0) { return; /* Not positive definite */ }
      double r0 = fmax(nom*RELTOL*RELTOL, ABSTOL*ABSTOL);
      if (nom <= r0) { return; /* Converged */ }

      DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, d, z, d1d, q1d);
      double den = DGMassDot<NB>(e, NE, ND, z, d);
      if (den <= 0.0)
      {
         DGMassDot<NB>(e, NE, ND, d, d);
         // d2 > 0 => not positive definite
         if (den == 0.0) { return; }
      }

      // start iteration
      int i = 1;
      while (true)
      {
         const double alpha = nom/den;
         DGMassAxpy(e, NE, ND, 1.0, u, alpha, d, u); // u = u + alpha*d
         DGMassAxpy(e, NE, ND, 1.0, r, -alpha, z, r); // r = r - alpha*A*d

         DGMassPreconditioner(e, NE, ND, dinv, r, z);

         double betanom = DGMassDot<NB>(e, NE, ND, r, z);
         if (betanom < 0.0) { return; /* Not positive definite */ }
         if (betanom <= r0) { break; /* Converged */ }

         if (++i > MAXIT) { break; }

         const double beta = betanom/nom;
         DGMassAxpy(e, NE, ND, 1.0, z, beta, d, d); // d = z + beta*d
         DGMassApply<DIM,D1D,Q1D>(e, NE, B, Bt, pa_data, d, z, d1d, q1d); // z = A d
         den = DGMassDot<NB>(e, NE, ND, d, z);
         if (den <= 0.0)
         {
            DGMassDot<NB>(e, NE, ND, d, d);
            // d2 > 0 => not positive definite
            if (den == 0.0) { break; }
         }
         nom = betanom;
      }

      if (CHANGE_BASIS)
      {
         DGMassBasis<DIM,D1D,MAX_D1D>(e, NE, q2d_B, u, u, d1d);
      }
   });
}

void DGMassInverse::Mult(const Vector &Mu, Vector &u) const
{
   // Dispatch to templated version based on dim, d1d, and q1d.
   const int dim = fes.GetMesh()->Dimension();
   const int d1d = m->dofs1D;
   const int q1d = m->quad1D;

   const int id = (d1d << 4) | q1d;

   if (dim == 2)
   {
      switch (id)
      {
         case 0x11: return DGMassCGIteration<2,1,1>(Mu, u);
         case 0x22: return DGMassCGIteration<2,2,2>(Mu, u);
         case 0x33: return DGMassCGIteration<2,3,3>(Mu, u);
         case 0x35: return DGMassCGIteration<2,3,5>(Mu, u);
         case 0x44: return DGMassCGIteration<2,4,4>(Mu, u);
         case 0x46: return DGMassCGIteration<2,4,6>(Mu, u);
         case 0x55: return DGMassCGIteration<2,5,5>(Mu, u);
         case 0x57: return DGMassCGIteration<2,5,7>(Mu, u);
         case 0x66: return DGMassCGIteration<2,6,6>(Mu, u);
         case 0x68: return DGMassCGIteration<2,6,8>(Mu, u);
         default: return DGMassCGIteration<2>(Mu, u); // Fallback
      }
   }
   else if (dim == 3)
   {
      switch (id)
      {
         case 0x22: return DGMassCGIteration<3,2,2>(Mu, u);
         case 0x23: return DGMassCGIteration<3,2,3>(Mu, u);
         case 0x33: return DGMassCGIteration<3,3,3>(Mu, u);
         case 0x34: return DGMassCGIteration<3,3,4>(Mu, u);
         case 0x35: return DGMassCGIteration<3,3,5>(Mu, u);
         case 0x44: return DGMassCGIteration<3,4,4>(Mu, u);
         case 0x45: return DGMassCGIteration<3,4,5>(Mu, u);
         case 0x46: return DGMassCGIteration<3,4,6>(Mu, u);
         case 0x48: return DGMassCGIteration<3,4,8>(Mu, u);
         case 0x55: return DGMassCGIteration<3,5,5>(Mu, u);
         case 0x56: return DGMassCGIteration<3,5,6>(Mu, u);
         case 0x57: return DGMassCGIteration<3,5,7>(Mu, u);
         case 0x58: return DGMassCGIteration<3,5,8>(Mu, u);
         case 0x66: return DGMassCGIteration<3,6,6>(Mu, u);
         case 0x67: return DGMassCGIteration<3,6,7>(Mu, u);
         case 0x77: return DGMassCGIteration<3,7,7>(Mu, u);
         case 0x78: return DGMassCGIteration<3,7,8>(Mu, u);
         case 0x88: return DGMassCGIteration<3,8,8>(Mu, u);
         case 0x89: return DGMassCGIteration<3,8,9>(Mu, u);
         case 0x99: return DGMassCGIteration<3,9,9>(Mu, u);
         case 0x9A: return DGMassCGIteration<3,9,10>(Mu, u);
         default: return DGMassCGIteration<3>(Mu, u); // Fallback
      }
   }
}

#ifdef MFEM_USE_CUDA
class CuSolver
{
protected:
   cusolverDnHandle_t handle = nullptr;
   CuSolver()
   {
      cusolverStatus_t status = cusolverDnCreate(&handle);
      MFEM_VERIFY(status == CUSOLVER_STATUS_SUCCESS,
                  "Cannot initialize CuSolver.");
   }
   ~CuSolver()
   {
      cusolverDnDestroy(handle);
   }
   static CuSolver &Instance()
   {
      static CuSolver instance;
      return instance;
   }
public:
   static cusolverDnHandle_t Handle()
   {
      return Instance().handle;
   }
};

class CuBLAS
{
protected:
   cublasHandle_t handle = nullptr;
   CuBLAS()
   {
      cublasStatus_t status = cublasCreate(&handle);
      MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "Cannot initialize cuBLAS.");
   }
   ~CuBLAS()
   {
      cublasDestroy(handle);
   }
   static CuBLAS &Instance()
   {
      static CuBLAS instance;
      return instance;
   }
public:
   static cublasHandle_t Handle()
   {
      return Instance().handle;
   }

   static void EnableAtomics()
   {
      cublasStatus_t status = cublasSetAtomicsMode(Handle(), CUBLAS_ATOMICS_ALLOWED);
      MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "cuBLAS error.");
   }

   static void DisableAtomics()
   {
      cublasStatus_t status = cublasSetAtomicsMode(Handle(), CUBLAS_ATOMICS_NOT_ALLOWED);
      MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "cuBLAS error.");
   }
};
#endif

DGMassInverse_Direct::DGMassInverse_Direct(FiniteElementSpace &fes_,
                                           BatchSolverMode mode_)
   : DGMassInverse_Direct(fes_, nullptr, nullptr, mode_) { }

DGMassInverse_Direct::DGMassInverse_Direct(FiniteElementSpace &fes_,
                                           Coefficient &coeff,
                                           const IntegrationRule &ir,
                                           BatchSolverMode mode_)
   : DGMassInverse_Direct(fes_, &coeff, &ir, mode_) { }

DGMassInverse_Direct::DGMassInverse_Direct(FiniteElementSpace &fes_,
                                           const IntegrationRule &ir,
                                           BatchSolverMode mode_)
   : DGMassInverse_Direct(fes_, nullptr, &ir, mode_) { }

DGMassInverse_Direct::DGMassInverse_Direct(FiniteElementSpace &fes_,
                                           Coefficient *coeff,
                                           const IntegrationRule *ir,
                                           BatchSolverMode mode_)
   : Solver(fes_.GetTrueVSize()), fes(fes_), mode(mode_)
{
   const int ne = fes.GetNE();
   const int elem_dofs = fes.GetFE(0)->GetDof();

   blocks.SetSize(ne*elem_dofs*elem_dofs);

   if (coeff) { m = new MassIntegrator(*coeff, ir); }
   else { m = new MassIntegrator(ir); }

   Setup();
}

void DGMassInverse_Direct::Setup()
{
   const int ne = fes.GetNE();
   const int elem_dofs = fes.GetFE(0)->GetDof();
   MFEM_CONTRACT_VAR(ne);
   MFEM_CONTRACT_VAR(elem_dofs);

   if (mode == BatchSolverMode::CUBLAS)
   {
      tmp.SetSize(blocks.Size());
      m->AssembleEA(fes, tmp, false);
   }
   else
   {
      m->AssembleEA(fes, blocks, false);
   }

   tensor.UseExternalData(NULL, elem_dofs, elem_dofs, ne);
   tensor.GetMemory().MakeAlias(blocks.GetMemory(), 0, blocks.Size());

   if ((mode == BatchSolverMode::CUSOLVER || mode == BatchSolverMode::CUBLAS)
       && !Device::Allows(Backend::CUDA))
   {
      MFEM_ABORT("Unsupported mode. Use BatchSolverMode::NATIVE.");
   }

   if (mode == BatchSolverMode::NATIVE || !Device::Allows(Backend::CUDA))
   {
      BatchLUFactor(tensor, ipiv);
   }
   else if (mode == BatchSolverMode::CUSOLVER)
   {
#ifdef MFEM_USE_CUDA
      vector_array.SetSize(ne);
      matrix_array.SetSize(ne);
      double *ptr_base = blocks.ReadWrite();
      double **d_matrix_array = matrix_array.Write();
      MFEM_FORALL(i, ne, d_matrix_array[i] = ptr_base + i*elem_dofs*elem_dofs;);
      info_array.SetSize(ne);

      cusolverStatus_t status = cusolverDnDpotrfBatched(
                                   CuSolver::Handle(),
                                   CUBLAS_FILL_MODE_LOWER,
                                   elem_dofs,
                                   matrix_array.ReadWrite(),
                                   elem_dofs,
                                   info_array.Write(),
                                   ne);
      MFEM_VERIFY(status == CUSOLVER_STATUS_SUCCESS, "");
#else
      MFEM_ABORT("CUDA must be enabled.");
#endif
   }
   else if (mode == BatchSolverMode::CUBLAS)
   {
#ifdef MFEM_USE_CUDA
      ipiv.SetSize(ne*elem_dofs);
      info_array.SetSize(ne);
      tmp_array.SetSize(ne);
      double *ptr_base = tmp.ReadWrite();
      double **d_tmp_array = tmp_array.Write();
      MFEM_FORALL(i, ne, d_tmp_array[i] = ptr_base + i*elem_dofs*elem_dofs;);

      matrix_array.SetSize(ne);
      ptr_base = blocks.ReadWrite();
      double **d_matrix_array = matrix_array.Write();
      MFEM_FORALL(i, ne, d_matrix_array[i] = ptr_base + i*elem_dofs*elem_dofs;);

      cublasStatus_t status = cublasDgetrfBatched(
                                 CuBLAS::Handle(),
                                 elem_dofs,
                                 tmp_array.ReadWrite(),
                                 elem_dofs,
                                 ipiv.Write(),
                                 info_array.Write(),
                                 ne);
      MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "");

      status = cublasDgetriBatched(
                  CuBLAS::Handle(),
                  elem_dofs,
                  tmp_array.ReadWrite(),
                  elem_dofs,
                  ipiv.ReadWrite(),
                  matrix_array.ReadWrite(),
                  elem_dofs,
                  info_array.Write(),
                  ne);
      MFEM_VERIFY(status == CUBLAS_STATUS_SUCCESS, "");
#else
      MFEM_ABORT("CUDA must be enabled.");
#endif
   }
}

void DGMassInverse_Direct::Mult(const Vector &Mu, Vector &u) const
{
   if (mode == BatchSolverMode::CUBLAS)
   {
#ifdef MFEM_USE_CUDA
      const int n = tensor.SizeI();
      const int nblocks = tensor.SizeK();

      const double alpha = 1.0, beta = 0.0;

      cublasStatus_t status = cublasDgemvStridedBatched(
         CuBLAS::Handle(),
         CUBLAS_OP_N,
         n, n,
         &alpha,
         blocks.Read(), n, n*n,
         Mu.Read(), 1, n,
         &beta,
         u.Write(), 1, n,
         nblocks);
      // cublasStatus_t status = cublasDgemmStridedBatched(
      //                            CuBLAS::Handle(),
      //                            CUBLAS_OP_N, CUBLAS_OP_N,
      //                            n, 1, n,
      //                            &alpha,
      //                            blocks.Read(), n, n*n,
      //                            Mu.Read(), n, n,
      //                            &beta,
      //                            u.Write(), n, n,
      //                            nblocks);
#else
      MFEM_ABORT("CUDA must be enabled.");
#endif
   }
   else
   {
      u = Mu;
      Solve(u);
   }
}

void DGMassInverse_Direct::Solve(Vector &u) const
{
   if (mode == BatchSolverMode::NATIVE)
   {
      BatchLUSolve(tensor, ipiv, u);
   }
   else if (mode == BatchSolverMode::CUSOLVER)
   {
#ifdef MFEM_USE_CUDA
      const int n = tensor.SizeI();
      const int nblocks = tensor.SizeK();
      double *ptr_base = u.ReadWrite();
      auto u_ptr = Reshape(vector_array.Write(), nblocks);

      MFEM_FORALL(i, nblocks, u_ptr[i] = ptr_base + i*n; );

      cusolverStatus_t status = cusolverDnDpotrsBatched(
                                   CuSolver::Handle(),
                                   CUBLAS_FILL_MODE_LOWER,
                                   n,
                                   1,
                                   matrix_array.ReadWrite(),
                                   n,
                                   u_ptr,
                                   n,
                                   info_array.Write(),
                                   nblocks);
#else
      MFEM_ABORT("CUDA must be enabled.");
#endif
   }
}

void DGMassInverse_Direct::SetOperator(const Operator &op)
{
   MFEM_ABORT("Not supported.");
}

DGMassInverse_Direct::~DGMassInverse_Direct()
{
   delete m;
}


} // namespace mfem
