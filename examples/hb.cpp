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

// Compile with: make hb (requires MFEM_USE_SUITESPARSE=YES)
//
// Sample runs: hb -m ../data/inline-quad.mesh -o 3 -verify
//              hb -m ../data/beam-hex.mesh -o 2 -verify
//              hb -m ../data/star.mesh -o 2 -r 1
//
// Solve -grad(div u) + u = f on a conforming quadrilateral/hexahedral mesh,
// prescribing the normal component of a manufactured solution on the boundary.
// The order p >= 1 denotes RT_{p-1}. Integrated GLL flux coordinates identify
// the high-order and low-order refined (LOR) degrees of freedom.
//
// MINRES solves the uncondensed hybridized saddle-point system
//
//       [ A_hat  C^T ] [ u_hat  ] = [ f_hat ]
//       [ C       0  ] [ lambda ]   [   0   ].
//
// Its SPD block preconditioner consists of dense Cholesky solves with the
// high-order element matrices and J^T H_lor^{-1} J on the multipliers. Here
// H_lor hybridizes EVERY refined subcell, and J injects only the macro-interface
// residuals. UMFPACK solves the full H_lor system, including internal subcell
// multipliers. No high-order multiplier Schur complement is assembled.
// This initial example stores dense high-order blocks; it is not matrix-free.

#include "mfem.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace mfem;
using namespace std;

#ifdef MFEM_USE_SUITESPARSE

namespace
{

int Index(int d) { return d < 0 ? -1-d : d; }
real_t Sign(int d) { return d < 0 ? -1.0 : 1.0; }

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

// Local free DOFs retain the finite element's orientation. Each constraint is
// the difference of two copies expressed in the global flux orientation, so
// there is exactly one independent row per shared integrated flux DOF. Physical
// boundary faces have no multipliers, and essential DOFs are omitted entirely.
struct BrokenSpace
{
   Array<int> offsets, dofs, owners, multiplicity, first, row;
   vector<Array<int>> local;
   unique_ptr<SparseMatrix> C;

   BrokenSpace(FiniteElementSpace &fes, const Array<int> &essential)
      : offsets(fes.GetNE()+1), multiplicity(fes.GetVSize()),
        first(fes.GetVSize()), row(fes.GetVSize()), local(fes.GetNE())
   {
      multiplicity = 0;
      first = -1;
      row = -1;
      offsets[0] = 0;
      Array<int> edofs;
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
            local[e].Append(j);
            dofs.Append(edofs[j]);
            owners.Append(e);
         }
         offsets[e+1] = dofs.Size();
      }
      int nr = 0;
      for (int d = 0; d < fes.GetVSize(); ++d)
      {
         MFEM_VERIFY(multiplicity[d] <= 2, "Expected a conforming RT space.");
         if (multiplicity[d] == 2) { row[d] = nr++; }
      }
      C.reset(new SparseMatrix(nr, dofs.Size()));
      for (int i = 0; i < dofs.Size(); ++i)
      {
         const int d = Index(dofs[i]);
         if (row[d] >= 0) { C->Add(row[d], i, ConstraintSign(i)); }
      }
      C->Finalize();
   }

   real_t ConstraintSign(int i) const
   {
      return (first[Index(dofs[i])] == i ? 1.0 : -1.0)*Sign(dofs[i]);
   }

   void Prolongate(const Vector &x, Vector &broken) const
   {
      broken.SetSize(dofs.Size());
      for (int i = 0; i < dofs.Size(); ++i)
      {
         broken(i) = Sign(dofs[i])*x(Index(dofs[i]));
      }
   }

   void Assemble(const Vector &broken, Vector &x) const
   {
      x.SetSize(multiplicity.Size());
      x = 0.0;
      for (int i = 0; i < dofs.Size(); ++i)
      {
         x(Index(dofs[i])) += Sign(dofs[i])*broken(i);
      }
   }

   void Average(const Vector &broken, Vector &x) const
   {
      Assemble(broken, x);
      for (int d = 0; d < x.Size(); ++d)
      {
         if (multiplicity[d]) { x(d) /= multiplicity[d]; }
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
   ElementOperator(FiniteElementSpace &fes, const BrokenSpace &space_,
                   const Vector *lift = nullptr, VectorCoefficient *f = nullptr,
                   Vector *rhs = nullptr)
      : Operator(space_.dofs.Size()), space(space_), blocks(fes.GetNE())
   {
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
            // Assemble the broken load by local quadrature, then subtract the
            // same element operator applied to the prescribed normal-flux lift.
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

   void Mult(const Vector &x, Vector &y) const override
   {
      Apply(x, y, false);
   }

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

   // Assemble H_lor = C_lor A_hat_lor^{-1} C_lor^T from constant-size RT0
   // subcell blocks. All internal subcell constraints remain in this matrix.
   unique_ptr<SparseMatrix> HybridizedMatrix() const
   {
      const int nr = space.C->Height();
      unique_ptr<SparseMatrix> H(new SparseMatrix(nr));
      DenseMatrix inv;
      for (int e = 0; e < static_cast<int>(blocks.size()); ++e)
      {
         if (!blocks[e]) { continue; }
         blocks[e]->inverse->GetInverseMatrix(inv);
         for (int j = 0; j < inv.Width(); ++j)
         {
            const int jj = space.offsets[e]+j;
            const int rj = space.row[Index(space.dofs[jj])];
            if (rj < 0) { continue; }
            for (int i = 0; i < inv.Height(); ++i)
            {
               const int ii = space.offsets[e]+i;
               const int ri = space.row[Index(space.dofs[ii])];
               if (ri < 0) { continue; }
               H->Add(ri, rj, space.ConstraintSign(ii)*inv(i,j)*
                      space.ConstraintSign(jj));
            }
         }
      }
      H->Finalize();
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

class MultiplierInverse : public Solver
{
   const SparseMatrix &J;
   UMFPackSolver inverse;
   mutable Vector rhs, solution;
public:
   MultiplierInverse(SparseMatrix &H, const SparseMatrix &J_)
      : Solver(J_.Width()), J(J_), rhs(H.Height()), solution(H.Height())
   {
      if (H.Height()) { inverse.SetOperator(H); }
   }
   void SetOperator(const Operator &) override { MFEM_ABORT("Fixed operator."); }
   void Mult(const Vector &x, Vector &y) const override
   {
      J.Mult(x, rhs);
      if (rhs.Size()) { inverse.Mult(rhs, solution); }
      J.MultTranspose(solution, y);
   }
};

// The LOR permutation is SIGNED. In addition, the first refined copy of a
// face need not belong to the first macroelement copy used to orient C. Both
// signs must be included in J to use the same dual multiplier coordinates.
unique_ptr<SparseMatrix> MultiplierInjection(const BrokenSpace &ho,
                                             const BrokenSpace &lo,
                                             const Array<int> &perm,
                                             const CoarseFineTransformations &cf)
{
   unique_ptr<SparseMatrix> J(new SparseMatrix(lo.C->Height(), ho.C->Height()));
   Array<int> found(ho.C->Height());
   found = 0;
   for (int d = 0; d < lo.row.Size(); ++d)
   {
      const int hd = Index(perm[d]);
      const int hr = ho.row[hd];
      if (hr < 0) { continue; }
      MFEM_VERIFY(lo.row[d] >= 0, "Missing LOR macro-interface constraint.");
      const int first_parent = cf.embeddings[lo.owners[lo.first[d]]].parent;
      const int first_macro = ho.owners[ho.first[hd]];
      J->Add(lo.row[d], hr, Sign(perm[d])*
             (first_parent == first_macro ? 1.0 : -1.0));
      ++found[hr];
   }
   for (int r = 0; r < found.Size(); ++r)
   {
      MFEM_VERIFY(found[r] == 1, "Expected one LOR face per IH macro-face DOF.");
   }
   J->Finalize();
   return J;
}

// Check C_lor P_mac = J C on arbitrary broken high-order coordinates, including
// zero jumps on every internal subcell face. This detects transfer sign errors
// that a test on an already conforming vector would miss.
real_t CheckTransfer(const BrokenSpace &ho, const BrokenSpace &lo,
                     const Array<int> &perm,
                     const CoarseFineTransformations &cf, const SparseMatrix &J)
{
   Vector u(ho.dofs.Size()), q(lo.dofs.Size());
   for (int i = 0; i < u.Size(); ++i) { u(i) = sin(0.7*(i+1)); }
   for (int i = 0; i < q.Size(); ++i)
   {
      const int ld = lo.dofs[i], hd = perm[Index(ld)];
      const int parent = cf.embeddings[lo.owners[i]].parent;
      int hi = -1;
      for (int j = ho.offsets[parent]; j < ho.offsets[parent+1]; ++j)
      {
         if (Index(ho.dofs[j]) == Index(hd)) { hi = j; break; }
      }
      MFEM_VERIFY(hi >= 0, "Missing macro-broken LOR flux coordinate.");
      q(i) = Sign(ld)*Sign(hd)*Sign(ho.dofs[hi])*u(hi);
   }
   Vector jump(ho.C->Height()), expected(lo.C->Height()), actual(lo.C->Height());
   ho.C->Mult(u, jump);
   J.Mult(jump, expected);
   lo.C->Mult(q, actual);
   actual -= expected;
   return actual.Norml2()/max(real_t(1.0), expected.Norml2());
}

} // namespace

#endif // MFEM_USE_SUITESPARSE

int main(int argc, char *argv[])
{
#ifndef MFEM_USE_SUITESPARSE
   cerr << "This example requires MFEM_USE_SUITESPARSE=YES.\n";
   return 1;
#else
   const char *mesh_file = "../data/inline-quad.mesh";
   int order = 2, refinements = 0, max_iter = 1000;
   real_t tolerance = 1e-10;
   bool verify = false, visualization = false;
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Conforming quad/hex mesh file.");
   args.AddOption(&order, "-o", "--order", "Order p >= 1 (RT_{p-1}).");
   args.AddOption(&refinements, "-r", "--refine", "Uniform mesh refinements.");
   args.AddOption(&max_iter, "-mi", "--max-iterations",
                  "Maximum MINRES iterations.");
   args.AddOption(&tolerance, "-tol", "--tolerance",
                  "Relative residual tolerance.");
   args.AddOption(&verify, "-verify", "--verify", "-no-verify", "--no-verify",
                  "Check the transfer and compare with a conforming direct solve.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable GLVis visualization.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);
   if (order < 1 || refinements < 0 || max_iter < 1 ||
       !isfinite(tolerance) || tolerance <= 0.0 || tolerance >= 1.0)
   {
      cerr << "Require order >= 1, refine >= 0, max-iterations >= 1, and 0 < tol < 1.\n";
      return 1;
   }

   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();
   if ((dim != 2 && dim != 3) || mesh.SpaceDimension() != dim ||
       mesh.Nonconforming() || mesh.GetNE() == 0)
   {
      cerr << "A conforming volume mesh in dimension 2 or 3 is required.\n";
      return 1;
   }
   const Geometry::Type geometry = dim == 2 ? Geometry::SQUARE : Geometry::CUBE;
   for (int e = 0; e < mesh.GetNE(); ++e)
   {
      if (mesh.GetElementBaseGeometry(e) != geometry)
      {
         cerr << "Only quadrilateral or hexahedral elements are supported.\n";
         return 1;
      }
   }
   for (int r = 0; r < refinements; ++r) { mesh.UniformRefinement(); }

   RT_FECollection fec(order-1, dim, BasisType::GaussLobatto,
                       BasisType::IntegratedGLL);
   FiniteElementSpace fes(&mesh, &fec);
   Array<int> ess_dofs, essential(fes.GetVSize());
   essential = 0;
   if (mesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_dofs);
      for (int d : ess_dofs) { essential[d] = 1; }
   }
   VectorFunctionCoefficient exact(dim, ExactSolution), force(dim, RightHandSide);
   GridFunction lift(&fes);
   lift.ProjectCoefficient(exact);
   lift.SetSubVectorComplement(ess_dofs, 0.0);

   StopWatch timer;
   timer.Start();
   BrokenSpace ho(fes, essential);
   Vector load;
   ElementOperator A(fes, ho, &lift, &force, &load);
   ElementInverse local_inverse(A);

   LORDiscretization lor(fes);
   FiniteElementSpace &lor_fes = lor.GetFESpace();
   const Array<int> &perm = lor.GetDofPermutation();
   Array<int> lor_essential(lor_fes.GetVSize());
   for (int d = 0; d < perm.Size(); ++d)
   {
      lor_essential[d] = essential[Index(perm[d])];
   }
   BrokenSpace lo(lor_fes, lor_essential);
   ElementOperator A_lor(lor_fes, lo);
   unique_ptr<SparseMatrix> H_lor = A_lor.HybridizedMatrix();
   const CoarseFineTransformations &cf =
      lor_fes.GetMesh()->GetRefinementTransforms();
   unique_ptr<SparseMatrix> J = MultiplierInjection(ho, lo, perm, cf);
   MultiplierInverse multiplier_inverse(*H_lor, *J);
   timer.Stop();
   cout << "Macro elements: " << mesh.GetNE()
        << "\nConforming flux DOFs: " << fes.GetVSize()
        << "\nFree broken flux DOFs: " << A.Height()
        << "\nMacro-interface multipliers: " << ho.C->Height()
        << "\nLOR elements: " << lor_fes.GetNE()
        << "\nFull LOR multipliers: " << H_lor->Height()
        << "\nSetup time: " << timer.RealTime() << " seconds\n";

   const int nblocks = ho.C->Height() ? 2 : 1;
   Array<int> offsets(nblocks+1);
   offsets[0] = 0;
   offsets[1] = A.Height();
   if (nblocks == 2) { offsets[2] = A.Height()+ho.C->Height(); }
   BlockOperator K(offsets);
   BlockDiagonalPreconditioner B(offsets);
   TransposeOperator Ct(ho.C.get());
   K.SetDiagonalBlock(0, &A);
   B.SetDiagonalBlock(0, &local_inverse);
   if (nblocks == 2)
   {
      K.SetBlock(1, 0, ho.C.get());
      K.SetBlock(0, 1, &Ct);
      B.SetDiagonalBlock(1, &multiplier_inverse);
   }
   BlockVector rhs(offsets), solution(offsets), residual(offsets);
   rhs = 0.0;
   rhs.GetBlock(0) = load;
   solution = 0.0;
   MINRESSolver minres;
   minres.SetOperator(K);
   minres.SetPreconditioner(B);
   minres.SetRelTol(tolerance);
   minres.SetAbsTol(0.0);
   minres.SetMaxIter(max_iter);
   minres.SetPrintLevel(1);
   const real_t scale = max(real_t(1.0), load.Norml2());
   Vector correction, assembled_load, broken, product, conforming_residual;
   ho.Assemble(load, assembled_load);
   real_t flux_residual = 0.0, jump_residual = 0.0, conf_residual = 0.0;
   bool success = false;
   int iterations = 0;
   timer.Clear();
   timer.Start();
   while (true)
   {
      if (K.Height())
      {
         minres.SetMaxIter(max_iter-iterations);
         minres.Mult(rhs, solution);
         iterations += minres.GetNumIterations();
      }
      K.Mult(solution, residual);
      residual -= rhs;
      flux_residual = residual.GetBlock(0).Norml2()/scale;
      jump_residual = nblocks == 2 ? residual.GetBlock(1).Norml2()/scale : 0.0;
      ho.Average(solution.GetBlock(0), correction);
      ho.Prolongate(correction, broken);
      A.Mult(broken, product);
      ho.Assemble(product, conforming_residual);
      conforming_residual -= assembled_load;
      conf_residual = conforming_residual.Norml2()/
                      max(real_t(1.0), assembled_load.Norml2());
      success = isfinite(flux_residual) && flux_residual <= tolerance &&
                isfinite(jump_residual) && jump_residual <= tolerance &&
                isfinite(conf_residual) && conf_residual <= tolerance;
      if (success || iterations >= max_iter || !minres.GetConverged() ||
          minres.GetNumIterations() == 0 ||
          !isfinite(flux_residual+jump_residual+conf_residual)) { break; }
      break;
      // Averaging can amplify a small jump in the conforming energy. Restart
      // from the current iterate until the actual equations meet the requested
      // tolerance. Each restart still uses the same fixed SPD preconditioner.
      cout << "Refining the iterate to meet the true residual tolerance.\n";
      minres.iterative_mode = true;
      minres.SetRelTol(min(real_t(0.1), real_t(0.1)*tolerance/
                           max(flux_residual, max(jump_residual, conf_residual))));
   }
   timer.Stop();
   cout << "Total MINRES iterations: " << iterations
        << "\nSolve time: " << timer.RealTime() << " seconds\n";
   cout << "True flux residual (scaled): " << flux_residual
        << "\nContinuity residual (scaled): " << jump_residual
        << "\nConforming residual (scaled): " << conf_residual << '\n';
   // The solution comparison is a forward-error check, with a small allowance
   // for conditioning; the actual residuals above must meet --tolerance.
   const real_t check_tol = max(real_t(100.0)*tolerance, real_t(1e-12));

   if (verify)
   {
      const real_t transfer_error = CheckTransfer(ho, lo, perm, cf, *J);
      cout << "Two-level constraint transfer error: " << transfer_error << '\n';
      success = success && isfinite(transfer_error) && transfer_error <= 1e-12;
      // Independently assemble the original conforming problem through the
      // standard forms, including its load and essential boundary lifting.
      // High-order global assembly is confined to this verification path.
      ConstantCoefficient one(1.0);
      BilinearForm reference(&fes);
      reference.AddDomainIntegrator(new DivDivIntegrator(one));
      reference.AddDomainIntegrator(new VectorFEMassIntegrator(one));
      reference.Assemble();
      LinearForm reference_load(&fes);
      reference_load.AddDomainIntegrator(new VectorFEDomainLFIntegrator(force));
      reference_load.Assemble();
      GridFunction reference_solution(&fes);
      reference_solution = lift;
      OperatorPtr reference_matrix;
      Vector b, x;
      reference.FormLinearSystem(ess_dofs, reference_solution, reference_load,
                                 reference_matrix, x, b);
      UMFPackSolver direct;
      direct.SetOperator(*reference_matrix);
      direct.Mult(b, x);
      reference.RecoverFEMSolution(x, reference_load, reference_solution);
      reference_solution -= lift;
      const real_t norm = max(real_t(1.0), reference_solution.Norml2());
      reference_solution -= correction;
      const real_t error = reference_solution.Norml2()/norm;
      cout << "Conforming direct-solve difference (scaled): " << error << '\n';
      success = success && isfinite(error) && error <= check_tol;
   }

   GridFunction u(&fes);
   u = lift;
   u += correction;
   cout << "L2 error: " << u.ComputeL2Error(exact) << '\n';
   if (!success)
   {
      cerr << "Solve failed the convergence or verification checks.\n";
      return 2;
   }
   ofstream mesh_out("hb.mesh"), solution_out("hb.gf");
   mesh_out.precision(16);
   solution_out.precision(16);
   mesh.Print(mesh_out);
   u.Save(solution_out);
   if (visualization)
   {
      socketstream socket("localhost", 19916);
      socket.precision(8);
      socket << "solution\n" << mesh << u << flush;
   }
   return 0;
#endif
}
