// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.
 
#include "mg_agglom.hpp"
#include "partition.hpp"
 
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>
 
using namespace std;
 
namespace mfem
{
 
namespace
{
 
/// True on rank 0 of @a comm.
bool IsRoot(MPI_Comm comm)
{
   int rank;
   MPI_Comm_rank(comm, &rank);
   return rank == 0;
}
 
/// Global (all-rank) l2 norm of a distributed vector.
real_t GlobalNorml2(MPI_Comm comm, const Vector &v)
{
   return std::sqrt(InnerProduct(comm, v, v));
}
 
/// Builds the hypre partitioning of a distributed index space in which this
/// rank owns @a nloc consecutive indices. Returns the global size.
HYPRE_BigInt MakePartitioning(MPI_Comm comm, int nloc,
                              Array<HYPRE_BigInt> &starts)
{
   long long n = nloc, my_end = 0, glob = 0;
   MPI_Scan(&n, &my_end, 1, MPI_LONG_LONG, MPI_SUM, comm);
   MPI_Allreduce(&n, &glob, 1, MPI_LONG_LONG, MPI_SUM, comm);
 
   if (HYPRE_AssumedPartitionCheck())
   {
      starts.SetSize(2);
      starts[0] = static_cast<HYPRE_BigInt>(my_end - n);
      starts[1] = static_cast<HYPRE_BigInt>(my_end);
   }
   else
   {
      int nranks;
      MPI_Comm_size(comm, &nranks);
      std::vector<long long> ends(nranks);
      MPI_Allgather(&my_end, 1, MPI_LONG_LONG, ends.data(), 1, MPI_LONG_LONG,
                    comm);
      starts.SetSize(nranks + 1);
      starts[0] = 0;
      for (int r = 0; r < nranks; r++)
      {
         starts[r + 1] = static_cast<HYPRE_BigInt>(ends[r]);
      }
   }
   return static_cast<HYPRE_BigInt>(glob);
}
 
/** Returns a new matrix equal to @a A with all entries |a_ij| <= tol removed,
    matching SparseMatrix::Threshold(tol, false) in the serial code.
 
    This replaces HypreParMatrix::Threshold(), which segfaults inside its
    internal GenerateDiagAndOffd() helper in some MFEM/hypre builds (that
    helper reads CSR column arrays that recent hypre versions may leave unset,
    and it is host-only code). Here the rank-local rows are rebuilt with global
    column indices and handed to MFEM's general CSR constructor, which goes
    through hypre's IJ interface and copies all input arrays. */
HypreParMatrix *ThresholdCopy(const HypreParMatrix &A, real_t tol)
{
   A.HostRead();
   SparseMatrix diag, offd;
   HYPRE_BigInt *cmap = nullptr;
   A.GetDiag(diag);
   A.GetOffd(offd, cmap);
   hypre_ParCSRMatrix *h = A;
   const HYPRE_BigInt first_col = hypre_ParCSRMatrixFirstColDiag(h);
 
   const int nrows = diag.Height();
   Array<int> I(nrows + 1);
   Array<HYPRE_BigInt> J;
   Array<real_t> V;
   I[0] = 0;
   if (nrows > 0)
   {
      const int *dI = diag.HostReadI();
      const int *dJ = diag.HostReadJ();
      const real_t *dV = diag.HostReadData();
 
      const bool use_offd = (offd.Width() > 0);
      const int *oI = use_offd ? offd.HostReadI() : nullptr;
      const int *oJ = use_offd ? offd.HostReadJ() : nullptr;
      const real_t *oV = use_offd ? offd.HostReadData() : nullptr;
 
      for (int r = 0; r < nrows; r++)
      {
         for (int k = dI[r]; k < dI[r+1]; k++)
         {
            if (std::abs(dV[k]) > tol)
            {
               J.Append(first_col + dJ[k]);
               V.Append(dV[k]);
            }
         }
         if (use_offd)
         {
            for (int k = oI[r]; k < oI[r+1]; k++)
            {
               if (std::abs(oV[k]) > tol)
               {
                  J.Append(cmap[oJ[k]]);
                  V.Append(oV[k]);
               }
            }
         }
         I[r+1] = J.Size();
      }
   }
   A.HypreRead();
 
   return new HypreParMatrix(A.GetComm(), nrows, A.GetGlobalNumRows(),
                             A.GetGlobalNumCols(), I.GetData(), J.GetData(),
                             V.GetData(), A.GetRowStarts(), A.GetColStarts());
}
 
/// Global number of nonzeros of a HypreParMatrix (collective).
long long GlobalNNZ(const HypreParMatrix &A)
{
   hypre_ParCSRMatrix *h = A;
   long long loc =
      static_cast<long long>(hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(h))) +
      static_cast<long long>(hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(h)));
   long long glob = 0;
   MPI_Allreduce(&loc, &glob, 1, MPI_LONG_LONG, MPI_SUM, A.GetComm());
   return glob;
}
 
/** Smoother that applies a (rank-local) block-diagonal approximate inverse,
    stored as a block-diagonal HypreParMatrix. Replaces the serial "dummy"
    BlockJacobi wrapper. Because the action is a HypreParMatrix::Mult, it runs
    on the GPU when hypre is built with CUDA. */
class BlockDiagonalSmoother : public Solver
{
   SparseMatrix local; // owns the data the HypreParMatrix may reference
   std::unique_ptr<HypreParMatrix> S;
 
public:
   /// Takes over the contents of @a local_inv (it is left empty).
   BlockDiagonalSmoother(SparseMatrix &local_inv, MPI_Comm comm,
                         HYPRE_BigInt glob_size, HYPRE_BigInt *row_starts)
      : Solver(local_inv.Height())
   {
      local.Swap(local_inv);
      S.reset(new HypreParMatrix(comm, glob_size, row_starts, &local));
   }
 
   const HypreParMatrix &GetMatrix() const { return *S; }
 
   void SetOperator(const Operator &) override { }
 
   void Mult(const Vector &b, Vector &x) const override { S->Mult(b, x); }
   void MultTranspose(const Vector &b, Vector &x) const override
   {
      S->Mult(b, x);
   }
};
 
/** Host copy of the rank-local diagonal block of a HypreParMatrix in CSR
    form. Gauss-Seidel sweeps and ILU triangular solves are inherently
    sequential, so the two smoothers below work on the host (unlike
    BlockDiagonalSmoother, which can run on the GPU through hypre). */
struct LocalCSR
{
   int n = 0;
   std::vector<int> I, J;
   std::vector<real_t> V;
};

LocalCSR CopyLocalDiag(const HypreParMatrix &A)
{
   LocalCSR csr;
   A.HostRead();
   SparseMatrix diag;
   A.GetDiag(diag);
   csr.n = diag.Height();
   MFEM_VERIFY(diag.Width() == csr.n, "Local diagonal block must be square.");
   csr.I.assign(csr.n + 1, 0);
   if (csr.n > 0)
   {
      const int *dI = diag.HostReadI();
      const int *dJ = diag.HostReadJ();
      const real_t *dV = diag.HostReadData();
      const int nnz = dI[csr.n];
      csr.I.assign(dI, dI + csr.n + 1);
      csr.J.assign(dJ, dJ + nnz);
      csr.V.assign(dV, dV + nnz);
   }
   A.HypreRead();
   return csr;
}

/// C = A B, for small dense blocks.
void BlockMul(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C)
{
   const int m = A.Height(), l = A.Width(), n = B.Width();
   C.SetSize(m, n);
   C = 0.0;
   for (int j = 0; j < n; j++)
   {
      for (int k = 0; k < l; k++)
      {
         const real_t bkj = B(k, j);
         for (int i = 0; i < m; i++) { C(i, j) += A(i, k)*bkj; }
      }
   }
}

/// C -= A B, for small dense blocks.
void BlockSubMul(const DenseMatrix &A, const DenseMatrix &B, DenseMatrix &C)
{
   const int m = A.Height(), l = A.Width(), n = B.Width();
   for (int j = 0; j < n; j++)
   {
      for (int k = 0; k < l; k++)
      {
         const real_t bkj = B(k, j);
         for (int i = 0; i < m; i++) { C(i, j) -= A(i, k)*bkj; }
      }
   }
}

/// t -= M x(cols), where x is a full local vector and cols lists the dofs.
void BlockSubMultVec(const DenseMatrix &M, const real_t *x,
                     const Array<int> &cols, real_t *t)
{
   const int m = M.Height();
   for (int c = 0; c < cols.Size(); c++)
   {
      const real_t xc = x[cols[c]];
      if (xc == 0.0) { continue; }
      for (int r = 0; r < m; r++) { t[r] -= M(r, c)*xc; }
   }
}

/** Block Gauss-Seidel smoother on the rank-local diagonal block of A, with
    the same (possibly non-contiguous) blocks as BlockDiagonalSmoother.

    Mult(b, x) performs one forward block sweep from a zero initial guess,
    i.e. x = w (D + L)^{-1} b, where D is the block diagonal and L the strict
    block lower triangle of the local diagonal block. MultTranspose performs
    the backward sweep x = w (D + U)^{-1} b. With MFEM's multigrid cycle (pre-
    smoothing with Mult, post-smoothing with MultTranspose) this gives a
    symmetric block Gauss-Seidel cycle. For symmetric A the backward sweep is
    exactly the transpose of the forward sweep; for nonsymmetric A it is still
    a smoother for A (not A^T), matching the convention of
    BlockDiagonalSmoother::MultTranspose.

    Across ranks the method is block Jacobi (couplings in the off-diagonal
    part of A are ignored), i.e. the usual "hybrid" Gauss-Seidel. */
class BlockGaussSeidelSmoother : public Solver
{
   LocalCSR A;
   std::vector<Array<int>> blocks;   // non-empty blocks, in sweep order
   std::vector<DenseMatrix> Dinv;    // inverses of the diagonal blocks
   real_t weight;
   int max_bs = 0;

   void Sweep(const real_t *b, real_t *x, bool forward) const
   {
      std::vector<real_t> r(max_bs);
      const int nb = static_cast<int>(blocks.size());
      for (int s = 0; s < nb; s++)
      {
         const int i = forward ? s : nb - 1 - s;
         const Array<int> &blk = blocks[i];
         const int bs = blk.Size();

         // r = (b - A x) restricted to block i.
         for (int c = 0; c < bs; c++)
         {
            const int row = blk[c];
            real_t res = b[row];
            for (int k = A.I[row]; k < A.I[row+1]; k++)
            {
               res -= A.V[k]*x[A.J[k]];
            }
            r[c] = res;
         }

         // x_i += w A_ii^{-1} r
         const DenseMatrix &Di = Dinv[i];
         for (int c = 0; c < bs; c++)
         {
            real_t z = 0.0;
            for (int d = 0; d < bs; d++) { z += Di(c, d)*r[d]; }
            x[blk[c]] += weight*z;
         }
      }
   }

public:
   BlockGaussSeidelSmoother(const HypreParMatrix &Ap,
                            const std::vector<Array<int>> &blks,
                            real_t w = 1.0)
      : Solver(Ap.Height()), A(CopyLocalDiag(Ap)), weight(w)
   {
      Array<int> loc(A.n); // dof -> position within the current block
      loc = -1;
      for (const Array<int> &b : blks)
      {
         const int bs = b.Size();
         if (bs == 0) { continue; }
         for (int c = 0; c < bs; c++)
         {
            MFEM_ASSERT(b[c] >= 0 && b[c] < A.n, "Block index out of local range.");
            loc[b[c]] = c;
         }

         DenseMatrix Ab(bs);
         Ab = 0.0;
         for (int r = 0; r < bs; r++)
         {
            for (int k = A.I[b[r]]; k < A.I[b[r]+1]; k++)
            {
               if (loc[A.J[k]] >= 0) { Ab(r, loc[A.J[k]]) = A.V[k]; }
            }
         }
         Ab.Invert();

         for (int c = 0; c < bs; c++) { loc[b[c]] = -1; }
         blocks.push_back(b);
         Dinv.push_back(Ab);
         max_bs = std::max(max_bs, bs);
      }
   }

   void SetOperator(const Operator &) override { }

   void Mult(const Vector &b, Vector &x) const override
   {
      MFEM_ASSERT(b.Size() == height && x.Size() == height, "Size mismatch.");
      const real_t *bh = b.HostRead();
      real_t *xh = x.HostWrite();
      std::fill(xh, xh + height, real_t(0.0));
      Sweep(bh, xh, true);
   }

   void MultTranspose(const Vector &b, Vector &x) const override
   {
      MFEM_ASSERT(b.Size() == height && x.Size() == height, "Size mismatch.");
      const real_t *bh = b.HostRead();
      real_t *xh = x.HostWrite();
      std::fill(xh, xh + height, real_t(0.0));
      Sweep(bh, xh, false);
   }
};

/** Block ILU(0) smoother on the rank-local diagonal block of A, with the same
    (possibly non-contiguous, variable-size) blocks as BlockDiagonalSmoother.

    The local diagonal block is viewed as a block matrix whose (I,J) block is
    A restricted to (dofs of block I) x (dofs of block J). It is factored as
    A ~= L U, with L block unit lower triangular and U block upper triangular,
    keeping only the block sparsity pattern of A (no fill-in). Blocks are
    ordered as given. Mult(b, x) computes x = U^{-1} L^{-1} b.

    MultTranspose applies the same operator: for symmetric A, ILU(0) gives
    L U = L D L^T, whose inverse is symmetric, so this is the transpose; for
    nonsymmetric A it keeps the post-smoother an approximate inverse of A,
    matching BlockDiagonalSmoother::MultTranspose.

    Across ranks the method is block Jacobi (off-diagonal couplings ignored).
    Blocks must be disjoint; dofs not in any block are left at zero, as in
    BlockDiagonalSmoother. */
class BlockILUSmoother : public Solver
{
   std::vector<Array<int>> blocks;    // non-empty blocks (dof lists)
   std::vector<int> row_ptr, col;     // block CSR pattern, sorted columns
   std::vector<int> diag_pos;         // position of block (i,i) in col/val
   std::vector<DenseMatrix> val;      // L_ij (j<i), U_ii^{-1}, U_ij (j>i)
   int max_bs = 0;

   void Factor()
   {
      const int nb = static_cast<int>(blocks.size());
      std::vector<int> marker(nb, -1); // block column -> position in row i
      DenseMatrix Lik;
      for (int i = 0; i < nb; i++)
      {
         for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) { marker[col[p]] = p; }

         // IKJ variant: eliminate with the previous block rows k < i, in
         // increasing order of k.
         for (int p = row_ptr[i]; p < diag_pos[i]; p++)
         {
            const int k = col[p];
            // L_ik = A_ik U_kk^{-1} (block k is already factored).
            BlockMul(val[p], val[diag_pos[k]], Lik);
            val[p] = Lik;
            // A_ij -= L_ik U_kj for j > k in the pattern of row i.
            for (int q = diag_pos[k] + 1; q < row_ptr[k+1]; q++)
            {
               const int m = marker[col[q]];
               if (m >= 0) { BlockSubMul(val[p], val[q], val[m]); }
            }
         }

         // Store U_ii^{-1}.
         val[diag_pos[i]].Invert();

         for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) { marker[col[p]] = -1; }
      }
   }

public:
   BlockILUSmoother(const HypreParMatrix &Ap,
                    const std::vector<Array<int>> &blks)
      : Solver(Ap.Height())
   {
      const LocalCSR A = CopyLocalDiag(Ap);
      const int n = A.n;

      // dof -> (block, position within block)
      std::vector<int> dof2blk(n, -1), dof2loc(n, -1);
      for (const Array<int> &b : blks)
      {
         if (b.Size() == 0) { continue; }
         const int bi = static_cast<int>(blocks.size());
         for (int c = 0; c < b.Size(); c++)
         {
            MFEM_ASSERT(b[c] >= 0 && b[c] < n, "Block index out of local range.");
            MFEM_VERIFY(dof2blk[b[c]] == -1,
                        "BlockILUSmoother requires disjoint blocks.");
            dof2blk[b[c]] = bi;
            dof2loc[b[c]] = c;
         }
         blocks.push_back(b);
         max_bs = std::max(max_bs, b.Size());
      }
      const int nb = static_cast<int>(blocks.size());

      // Block sparsity pattern (diagonal block always included).
      row_ptr.assign(nb + 1, 0);
      diag_pos.assign(nb, -1);
      std::vector<int> seen(nb, -1), row_cols;
      for (int i = 0; i < nb; i++)
      {
         row_cols.clear();
         row_cols.push_back(i);
         seen[i] = i;
         const Array<int> &blk = blocks[i];
         for (int c = 0; c < blk.Size(); c++)
         {
            const int row = blk[c];
            for (int k = A.I[row]; k < A.I[row+1]; k++)
            {
               const int bj = dof2blk[A.J[k]];
               if (bj >= 0 && seen[bj] != i)
               {
                  seen[bj] = i;
                  row_cols.push_back(bj);
               }
            }
         }
         std::sort(row_cols.begin(), row_cols.end());
         const int start = static_cast<int>(col.size());
         col.insert(col.end(), row_cols.begin(), row_cols.end());
         row_ptr[i+1] = static_cast<int>(col.size());
         diag_pos[i] = start + static_cast<int>(
                          std::lower_bound(row_cols.begin(), row_cols.end(), i) -
                          row_cols.begin());
      }

      // Fill the dense blocks with the entries of A.
      val.resize(col.size());
      std::vector<int> marker(nb, -1);
      for (int i = 0; i < nb; i++)
      {
         const Array<int> &blk = blocks[i];
         for (int p = row_ptr[i]; p < row_ptr[i+1]; p++)
         {
            val[p].SetSize(blk.Size(), blocks[col[p]].Size());
            val[p] = 0.0;
            marker[col[p]] = p;
         }
         for (int c = 0; c < blk.Size(); c++)
         {
            const int row = blk[c];
            for (int k = A.I[row]; k < A.I[row+1]; k++)
            {
               const int j = A.J[k];
               const int bj = dof2blk[j];
               if (bj >= 0) { val[marker[bj]](c, dof2loc[j]) += A.V[k]; }
            }
         }
         for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) { marker[col[p]] = -1; }
      }

      Factor();
   }

   void SetOperator(const Operator &) override { }

   void Mult(const Vector &b, Vector &x) const override
   {
      MFEM_ASSERT(b.Size() == height && x.Size() == height, "Size mismatch.");
      const real_t *bh = b.HostRead();
      real_t *xh = x.HostWrite();
      std::fill(xh, xh + height, real_t(0.0));

      const int nb = static_cast<int>(blocks.size());
      std::vector<real_t> t(max_bs);

      // Forward solve L y = b (L block unit lower); y is stored in x.
      for (int i = 0; i < nb; i++)
      {
         const Array<int> &blk = blocks[i];
         for (int c = 0; c < blk.Size(); c++) { t[c] = bh[blk[c]]; }
         for (int p = row_ptr[i]; p < diag_pos[i]; p++)
         {
            BlockSubMultVec(val[p], xh, blocks[col[p]], t.data());
         }
         for (int c = 0; c < blk.Size(); c++) { xh[blk[c]] = t[c]; }
      }

      // Backward solve U x = y.
      for (int i = nb - 1; i >= 0; i--)
      {
         const Array<int> &blk = blocks[i];
         const int bs = blk.Size();
         for (int c = 0; c < bs; c++) { t[c] = xh[blk[c]]; }
         for (int p = diag_pos[i] + 1; p < row_ptr[i+1]; p++)
         {
            BlockSubMultVec(val[p], xh, blocks[col[p]], t.data());
         }
         const DenseMatrix &Uinv = val[diag_pos[i]];
         for (int r = 0; r < bs; r++)
         {
            real_t z = 0.0;
            for (int c = 0; c < bs; c++) { z += Uinv(r, c)*t[c]; }
            xh[blk[r]] = z;
         }
      }
   }

   void MultTranspose(const Vector &b, Vector &x) const override
   {
      Mult(b, x);
   }
};

/** Coarse-grid solver.
    - On a single MPI rank (including a plain serial launch, which MPI treats
      as one rank), and if MFEM was built with SuiteSparse: UMFPACK, as in the
      serial code. With one rank, the local diagonal block of the
      HypreParMatrix is the entire matrix, so it is copied into a SparseMatrix
      and factored directly.
    - Otherwise (more than one rank, or no SuiteSparse). The acoustics operator
      is not assumed symmetric, so this must handle general matrices:
      - with MUMPS: a parallel direct LU solve (closest to UMFPACK);
      - otherwise: GMRES preconditioned with BoomerAMG to a tight tolerance. */
class CoarseSolver : public Solver
{
#ifdef MFEM_USE_SUITESPARSE
   SparseMatrix A_serial; // owned copy of the matrix factored by UMFPACK
   std::unique_ptr<UMFPackSolver> umf;
#endif
#ifdef MFEM_USE_MUMPS
   std::unique_ptr<MUMPSSolver> mumps;
#else
   std::unique_ptr<HypreBoomerAMG> amg;
   std::unique_ptr<GMRESSolver> gmres;
#endif
   const Solver *solver = nullptr; // the solver actually in use
 
public:
   CoarseSolver(HypreParMatrix &A) : Solver(A.Height())
   {
      MPI_Comm comm = A.GetComm();
      int nranks, rank;
      MPI_Comm_size(comm, &nranks);
      MPI_Comm_rank(comm, &rank);
 
#ifdef MFEM_USE_SUITESPARSE
      if (nranks == 1)
      {
         hypre_ParCSRMatrix *h = A;
         MFEM_VERIFY(hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(h)) == 0,
                     "Expected no off-diagonal block on a single rank.");
 
         // Copy the local block to host memory so the matrix UMFPACK uses is
         // independent of the HypreParMatrix.
         A.HostRead();
         SparseMatrix diag;
         A.GetDiag(diag);
         SparseMatrix copy(diag, true, MemoryType::HOST);
         A.HypreRead();
         A_serial.Swap(copy);
 
         // hypre stores the diagonal entry first in each row; UMFPACK needs
         // sorted column indices.
         A_serial.SortColumnIndices();
 
         umf.reset(new UMFPackSolver(A_serial));
         solver = umf.get();
         mfem::out << "Coarse solver: UMFPACK (single rank)\n";
         return;
      }
#else
      if (nranks == 1 && rank == 0)
      {
         mfem::out << "Note: MFEM was built without SuiteSparse, so UMFPACK "
                   << "is unavailable; using the parallel coarse solver.\n";
      }
#endif
 
#ifdef MFEM_USE_MUMPS
      mumps.reset(new MUMPSSolver(comm));
      mumps->SetPrintLevel(0);
      mumps->SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
      mumps->SetOperator(A);
      solver = mumps.get();
      if (rank == 0) { mfem::out << "Coarse solver: MUMPS\n"; }
#else
      amg.reset(new HypreBoomerAMG(A));
      amg->SetPrintLevel(0);
      gmres.reset(new GMRESSolver(comm));
      gmres->SetKDim(100);
      gmres->SetRelTol(1e-10);
      gmres->SetAbsTol(0.0);
      gmres->SetMaxIter(500);
      gmres->SetPrintLevel(0);
      gmres->iterative_mode = false;
      gmres->SetOperator(A);
      gmres->SetPreconditioner(*amg);
      solver = gmres.get();
      if (rank == 0) { mfem::out << "Coarse solver: GMRES + BoomerAMG\n"; }
#endif
   }
 
   void SetOperator(const Operator &) override { }
 
   void Mult(const Vector &b, Vector &x) const override { solver->Mult(b, x); }
};

 
} // anonymous namespace
 
TruncatedMultigrid::TruncatedMultigrid(const AgglomerationMultigrid &other)
{
   MFEM_VERIFY(other.NumLevels() >= 2, "");
   const int nlevels = other.NumLevels() - 1;
 
   operators = other.operators;
   operators.DeleteLast();
   ownedOperators.SetSize(nlevels, false);
 
   smoothers = other.smoothers;
   smoothers.DeleteLast();
   ownedSmoothers.SetSize(nlevels, false);
 
   prolongations = other.prolongations;
   prolongations.DeleteLast();
   ownedProlongations.SetSize(nlevels - 1, false);
 
   height = width = operators.Last()->Height();
}
 
const HypreParMatrix &AgglomerationMultigrid::GetFinestProlongation() const
{
   return static_cast<const HypreParMatrix&>(*prolongations.Last());
}
 
std::vector<std::vector<std::vector<int>>> Agglomerate(ParMesh &mesh,
                                                       int ncoarse,
                                                       int num_levels)
{
   MFEM_VERIFY(num_levels >= 1, "num_levels must be at least 1.");
   MFEM_VERIFY(ncoarse >= 1, "ncoarse must be at least 1.");
 
   MPI_Comm comm = mesh.GetComm();
   const int ne = mesh.GetNE(); // rank-local number of elements
 
   // Number of coarsest-level macro elements on this rank. Clamped to at least
   // 1 so ranks with few elements still produce a valid hierarchy.
   int ne_coarsest = 0;
   if (ne > 0)
   {
      const double denom = std::pow(static_cast<double>(ncoarse), num_levels - 1);
      ne_coarsest = std::max(1, static_cast<int>(ne / denom));
   }
 
   {
      long long loc[2] = {ne, ne_coarsest};
      long long glob[2] = {0, 0};
      MPI_Allreduce(loc, glob, 2, MPI_LONG_LONG, MPI_SUM, comm);
      if (IsRoot(comm))
      {
         mfem::out << "number of fine elements: " << glob[0] << '\n'
                   << "number of elements per macro elements: " << ncoarse << '\n'
                   << "number of coarsest level elements: " << glob[1] << '\n';
      }
   }
 
   // See the description in mg_agglom.hpp. E[0] is a single "root" entry, as
   // in the serial version.
   std::vector<std::vector<std::vector<int>>> E(num_levels);
   E[0].push_back(std::vector<int>(ne_coarsest, 0));
 
   // p[i] = index of the current-level macro element containing fine element i.
   Array<int> p(ne);
   if (ne > 0) { p = PartitionMesh(mesh, ne_coarsest, 4); }
 
   int num_macros = ne_coarsest; // number of macro elements on current level
   for (int j = 1; j < num_levels; ++j)
   {
      // macro_elements[m] = fine elements belonging to macro element m.
      std::vector<std::vector<int>> macro_elements(num_macros);
      for (int i = 0; i < ne; ++i) { macro_elements[p[i]].push_back(i); }
 
      // On the finest level, the entries are the fine element indices.
      if (j + 1 == num_levels)
      {
         E[j] = std::move(macro_elements);
         break;
      }
 
      // Otherwise partition each macro element into (up to) ncoarse pieces.
      int total_new_macros = 0;
      for (int e = 0; e < num_macros; ++e)
      {
         const std::vector<int> &fine = macro_elements[e];
         const int nfine = static_cast<int>(fine.size());
 
         int new_macros = 0;
         if (nfine > 0)
         {
            Array<int> subset(nfine);
            for (int i = 0; i < nfine; i++) { subset[i] = fine[i]; }
 
            Array<int> part = PartitionMesh(mesh, ncoarse, 3, subset);
            new_macros = part.Max() + 1;
            for (int ip = 0; ip < part.Size(); ++ip)
            {
               p[subset[ip]] = part[ip] + total_new_macros;
            }
         }
 
         std::vector<int> macro_vec(new_macros);
         for (int kk = 0; kk < new_macros; ++kk)
         {
            macro_vec[kk] = total_new_macros + kk;
         }
         E[j].push_back(std::move(macro_vec));
 
         total_new_macros += new_macros;
      }
      num_macros = total_new_macros;
   }
 
   return E;
}
 
/** Construct Block Jacobi Smoother
 *  Extracts the block diagonal of A (the rank-local diagonal block of the
 *  parallel operator) and inverts each block. Each block is given as a list of
 *  (local) dof indices, which need not be contiguous; this supports
 *  vector-valued spaces (e.g. acoustics, vdim > 1) where an element's vdofs
 *  are strided. Since aggregates never cross rank boundaries, every block lies
 *  entirely inside the local diagonal block. Empty blocks are skipped.
 */
SparseMatrix ExtractBlockDiagonalInverse(const SparseMatrix &A,
                                         const std::vector<Array<int>> &blocks)
{
   const int n = A.Height();
   SparseMatrix D(n, A.Width());
   const int *I = A.HostReadI();
   const int *J = A.HostReadJ();
   const real_t *V = A.HostReadData();
 
   Array<int> loc(A.Width()); // column -> position within the current block
   loc = -1;
   DenseMatrix Ab;
   for (const Array<int> &b : blocks)
   {
      const int bs = b.Size();
      if (bs == 0) { continue; }
      for (int c = 0; c < bs; c++)
      {
         MFEM_ASSERT(b[c] >= 0 && b[c] < n, "Block index out of local range.");
         loc[b[c]] = c;
      }
 
      // Extract the block from the matrix A
      Ab.SetSize(bs);
      Ab = 0.0;
      for (int r = 0; r < bs; r++)
      {
         for (int k = I[b[r]]; k < I[b[r]+1]; k++)
         {
            if (loc[J[k]] >= 0) { Ab(r, loc[J[k]]) = V[k]; }
         }
      }
 
      // Invert the block and insert it into the block diagonal matrix
      Ab.Invert();
      for (int r = 0; r < bs; r++)
      {
         for (int c = 0; c < bs; c++)
         {
            D.Set(b[r], b[c], Ab(r, c));
         }
      }
      for (int c = 0; c < bs; c++) { loc[b[c]] = -1; }
   }
 
   D.Finalize();
   return D;
}
 
/** Adaptive Smoother
 *  Returns the (rank-local) block Jacobi smoother with appropriate damping
 *  coefficient. In addition, also smooths the columns of the matrix B (whose
 *  rows are this rank's rows of the operator).
 *
 *  All norms are global, so every rank computes the same damping w.
 */
SparseMatrix AdaptiveSmoother(const HypreParMatrix &A,
                              const std::vector<Array<int>> &blocks,
                              const Vector &x_random, DenseMatrix &B, int level)
{
   MPI_Comm comm = A.GetComm();
 
   // Block-diagonal inverse from the rank-local diagonal block of A.
   A.HostRead();
   SparseMatrix A_diag;
   A.GetDiag(A_diag);
   SparseMatrix smoother = ExtractBlockDiagonalInverse(A_diag, blocks);
   A.HypreRead();
 
   // Find optimal damping coefficient through power iteration. The smoother is
   // block diagonal and rank-local, so it can be applied to local vectors
   // directly; only A needs the parallel matvec.
   Vector PinvAx(x_random);
   Vector Ax(PinvAx.Size());
   for (int i = 0; i < 4; i++)
   {
      A.Mult(PinvAx, Ax);
      smoother.Mult(Ax, PinvAx);
      if (i != 3) { PinvAx /= GlobalNorml2(comm, PinvAx); }
   }
   const real_t rho = GlobalNorml2(comm, PinvAx);
   const real_t w = 4.0/3.0/rho;
   if (IsRoot(comm))
   {
      mfem::out << "level: " << level << " rho: " << rho << " w: " << w << '\n';
   }
   smoother *= w;
 
   // Smooth the columns of B: b <- (I - S A) b, 20 times.
   const int num_B_cols = B.Width();
   const int num_B_rows = B.Height();
   Vector b(num_B_rows), b_prev(num_B_rows), Ab(num_B_rows);
   for (int j = 0; j < num_B_cols; j++)
   {
      {
         // Write through HostWrite() so the memory flags stay correct on GPU.
         real_t *bh = b.HostWrite();
         for (int i = 0; i < num_B_rows; i++) { bh[i] = B(i, j); }
      }
 
      for (int it = 0; it < 20; it++)
      {
         b_prev = b;
         A.Mult(b, Ab);
         smoother.Mult(Ab, b);
         b *= -1.0;
         b += b_prev;
      }
 
      const real_t *bh = b.HostRead();
      for (int i = 0; i < num_B_rows; i++) { B(i, j) = bh[i]; }
   }
   return smoother;
}
 
SmoothedAggregationGMG::SmoothedAggregationGMG(ParFiniteElementSpace &fes,
                                               HypreParMatrix &Af,
                                               int ncoarse, int num_levels,
                                               bool paraview_vis,
                                               SmootherType smoother_type)
{
   (void) paraview_vis; // not used by this method (as in the serial version)
 
   MPI_Comm comm = fes.GetComm();
   int rank;
   MPI_Comm_rank(comm, &rank);
   const bool root = (rank == 0);
 
   ParMesh &mesh = *fes.GetParMesh();
   const int vdim = fes.GetVDim();
   const int ne = mesh.GetNE(); // rank-local
   const int dim = mesh.Dimension();
   const int n_cut = (dim == 2) ? 2*2 - 2 + 1 : 2*2*2 - 3 + 1;
 
   // For DG (L2) spaces there are no shared dofs, so the local vdofs are the
   // true dofs, in the same order, and they are the local rows of Af.
   MFEM_VERIFY(fes.GetTrueVSize() == fes.GetVSize(),
               "SmoothedAggregationGMG requires a DG (L2) space with no shared "
               "dofs.");
   MFEM_VERIFY(Af.Height() == fes.GetVSize(),
               "The local rows of Af must be the local vdofs of fes.");
 
   // Create the (rank-local) mesh hierarchy.
   auto E = Agglomerate(mesh, ncoarse, num_levels);
   if (root)
   {
      mfem::out << "num agglomerated levels " << E.size() << '\n';
      const char *name = "block Jacobi";
      switch (smoother_type)
      {
         case SmootherType::BlockJacobi: name = "block Jacobi"; break;
         case SmootherType::BlockGaussSeidel: name = "block Gauss-Seidel"; break;
         case SmootherType::BlockILU: name = "block ILU(0)"; break;
      }
      mfem::out << "smoother: " << name << '\n';
   }
 
   height = width = Af.Height();
 
   // Populate the arrays: operators, smoothers, ownedOperators, ownedSmoothers
   // from the MultigridBase class. (All smoothers are owned, all operators
   // except the finest are owned).
   operators.SetSize(num_levels);
   smoothers.SetSize(num_levels);
   ownedOperators.SetSize(num_levels);
   ownedSmoothers.SetSize(num_levels);
   prolongations.SetSize(num_levels-1);
   ownedProlongations.SetSize(num_levels-1);
   for (int l = 0; l < num_levels-1; ++l)
   {
      ownedOperators[l] = true;
      ownedSmoothers[l] = true;
      ownedProlongations[l] = true;
   }
   ownedOperators[num_levels-1] = false;
   ownedSmoothers[num_levels-1] = true;
   operators[num_levels - 1] = &Af;
 
   coarse_starts.resize(num_levels);
 
   // prev_blocks[i] lists the (local) dof indices of block i on the "previous"
   // (finer) level. On the finest level, block i holds the vdofs of element i
   // (all vdim components, which are strided for byNODES ordering). On coarser
   // levels, block j holds the coarse dofs kept by aggregate j.
   std::vector<Array<int>> prev_blocks(ne);
   for (int i = 0; i < ne; i++)
   {
      fes.GetElementVDofs(i, prev_blocks[i]);
   }
 
   for (int k = num_levels - 2; k >= 0; --k)
   {
      HypreParMatrix &A_prev = static_cast<HypreParMatrix&>(*operators[k + 1]);
      const int nloc = A_prev.Height();
 
      // Initialize B and x_random. The seed differs per rank so ranks do not
      // sample identical patterns; on rank 0 it matches the serial seed.
      const int num_samp = vdim*ncoarse*ncoarse;
      std::mt19937 gen(12345u + 7919u*static_cast<unsigned>(rank));
      std::normal_distribution<double> dist(0.0, 1.0);
      DenseMatrix B(nloc, num_samp);
      for (int i = 0; i < B.Height(); i++)
      {
         for (int j = 0; j < B.Width(); j++)
         {
            B(i,j) = dist(gen);
         }
      }
      Vector x_random(nloc);
      {
         real_t *xr = x_random.HostWrite();
         for (int i = 0; i < nloc; i++) { xr[i] = dist(gen); }
      }
 
      // Find the damping coefficient. Smooth columns of B. The damped block
      // Jacobi operator is always built, because it is also used below to
      // smooth the tentative prolongation (an explicit sparse matrix is
      // needed there; Gauss-Seidel/ILU would produce dense prolongations).
      SparseMatrix S_local = AdaptiveSmoother(A_prev, prev_blocks, x_random, B, k);
      std::unique_ptr<BlockDiagonalSmoother> jacobi(
         new BlockDiagonalSmoother(S_local, comm, A_prev.GetGlobalNumRows(),
                                   A_prev.GetRowStarts()));
      const HypreParMatrix &S_prol = jacobi->GetMatrix();

      // Smoother used in the multigrid cycle on level k+1.
      switch (smoother_type)
      {
         case SmootherType::BlockJacobi:
            smoothers[k + 1] = jacobi.release();
            break;
         case SmootherType::BlockGaussSeidel:
            smoothers[k + 1] = new BlockGaussSeidelSmoother(A_prev, prev_blocks);
            break;
         case SmootherType::BlockILU:
            smoothers[k + 1] = new BlockILUSmoother(A_prev, prev_blocks);
            break;
         default:
            MFEM_ABORT("Unknown smoother type.");
      }
 
      // Collect the (local) dofs of each aggregate on this level and decide
      // how many coarse dofs each aggregate keeps. Coarse dofs are numbered
      // consecutively, aggregate by aggregate.
      const int num_agg = static_cast<int>(E[k+1].size());
      std::vector<std::vector<int>> agg_dofs(num_agg);
      std::vector<Array<int>> curr_blocks(num_agg);
      Array<int> agg_offset(num_agg + 1);
      agg_offset = 0;
      long long num_clamped = 0;
      for (int j = 0; j < num_agg; j++)
      {
         for (int kk : E[k+1][j])
         {
            MFEM_ASSERT(kk >= 0 && kk < static_cast<int>(prev_blocks.size()), "");
            const Array<int> &blk = prev_blocks[kk];
            for (int m = 0; m < blk.Size(); m++) { agg_dofs[j].push_back(blk[m]); }
         }
         const int nrows = static_cast<int>(agg_dofs[j].size());
         int num_keep = nrows / n_cut + (nrows % n_cut != 0);
         // The thin SVD has only min(nrows, num_samp) left singular vectors.
         // The serial version printed a warning here and then read past the
         // end of U; instead, clamp and report.
         const int max_keep = std::min(nrows, num_samp);
         if (num_keep > max_keep)
         {
            num_keep = max_keep;
            num_clamped++;
         }
         curr_blocks[j].SetSize(num_keep);
         for (int c = 0; c < num_keep; c++) { curr_blocks[j][c] = agg_offset[j] + c; }
         agg_offset[j+1] = agg_offset[j] + num_keep;
      }
      const int nloc_coarse = agg_offset.Last();
 
      {
         long long glob_clamped = 0;
         MPI_Allreduce(&num_clamped, &glob_clamped, 1, MPI_LONG_LONG, MPI_SUM,
                       comm);
         if (root && glob_clamped > 0)
         {
            mfem::out << "WARNING: level " << k << ": " << glob_clamped
                      << " aggregate(s) requested more coarse dofs than "
                      << "available singular vectors; num_keep was clamped.\n";
         }
      }
 
      // Build the rank-local tentative prolongation from the left singular
      // vectors of each aggregate's block of B.
      SparseMatrix P_local(nloc, nloc_coarse);
      for (int j = 0; j < num_agg; j++)
      {
         const int num_keep = curr_blocks[j].Size();
         if (num_keep == 0) { continue; }
         const std::vector<int> &indices = agg_dofs[j];
         const int nrows = static_cast<int>(indices.size());
 
         DenseMatrix B_sub(nrows, num_samp);
         for (int i = 0; i < nrows; i++)
         {
            for (int c = 0; c < num_samp; c++)
            {
               B_sub(i, c) = B(indices[i], c);
            }
         }
 
         DenseMatrixSVD Bsvd(B_sub, 'S', 'N');
         Bsvd.Eval(B_sub);
         const DenseMatrix &U = Bsvd.LeftSingularvectors();
 
         for (int r = 0; r < nrows; r++)
         {
            for (int c = 0; c < num_keep; c++)
            {
               P_local.Set(indices[r], agg_offset[j] + c, U(r, c));
            }
         }
      }
      P_local.Finalize();
 
      // Wrap the tentative prolongation as a parallel matrix. It has no
      // off-processor entries because aggregates are rank-local.
      const HYPRE_BigInt glob_coarse =
         MakePartitioning(comm, nloc_coarse, coarse_starts[k]);
      HypreParMatrix P_tent(comm, A_prev.GetGlobalNumRows(), glob_coarse,
                            A_prev.GetRowStarts(), coarse_starts[k].GetData(),
                            &P_local);
 
      // Construct the prolongation T = (I - \tilde{A}^{-1} A) P, computed as
      // P - \tilde{A}^{-1} (A P) to avoid forming the identity.
      std::unique_ptr<HypreParMatrix> AP(ParMult(&A_prev, &P_tent, true));
      std::unique_ptr<HypreParMatrix> SAP(ParMult(&S_prol, AP.get(), true));
      std::unique_ptr<HypreParMatrix> T_full(Add(1.0, P_tent, -1.0, *SAP));
      SAP.reset();
      AP.reset();
      HypreParMatrix *T = ThresholdCopy(*T_full, 1e-10);
      T_full.reset();
      prolongations[k] = T;
 
      // Coarse operator: T^T A T.
      std::unique_ptr<HypreParMatrix> Anew_full(RAP(&A_prev, T));
      HypreParMatrix *Anew = ThresholdCopy(*Anew_full, 1e-10);
      Anew_full.reset();
      operators[k] = Anew;
 
      prev_blocks = std::move(curr_blocks);
   }
 
   smoothers[0] = new CoarseSolver(static_cast<HypreParMatrix&>(*operators[0]));
 
   for (int i = num_levels - 1; i >= 0; --i)
   {
      const HypreParMatrix &Ai = static_cast<const HypreParMatrix&>(*operators[i]);
      const long long nnz = GlobalNNZ(Ai); // collective: call on all ranks
      if (root)
      {
         mfem::out << "Level " << i << ": " << Ai.GetGlobalNumRows()
                   << " DOFs. nnz = " << nnz << ".\n";
      }
   }
}
 
} // namespace mfem
