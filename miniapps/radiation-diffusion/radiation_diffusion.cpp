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

#include "radiation_diffusion.hpp"
#include "energy_integrator.hpp"

namespace mfem
{

RadiationDiffusionOperator::RadiationDiffusionOperator(ParMesh &mesh, int order)
   : dim(mesh.Dimension()),
     fec_l2(order-1, dim, b2, FiniteElement::INTEGRAL),
     fes_l2(&mesh, &fec_l2),
     fec_rt(order-1, dim, b1, b2),
     fes_rt(&mesh, &fec_rt),
     e_gf(&fes_l2),
     H_form(&fes_l2),
     L_form(&fes_l2),
     R_form(&fes_rt),
     D_form(&fes_rt, &fes_l2),
     Q_e_coeff(MMS::MaterialEnergySource),
     S_E_coeff(MMS::RadiationEnergySource),
     E_bdr_coeff(MMS::ExactRadiationEnergy),
     Q_e_form(&fes_l2),
     S_E_form(&fes_l2),
     b_n_form(&fes_rt)
{
   const int n_l2 = fes_l2.GetTrueVSize();
   const int n_rt = fes_rt.GetTrueVSize();

   // Unknowns: material energy, radiation energy (L2), radiation flux (RT)
   width = height = 2*n_l2 + n_rt;

   offsets = Array<int>({0, n_l2, 2*n_l2, 2*n_l2 + n_rt});

   dt = 0.0;

   H_form.AddDomainIntegrator(new NonlinearEnergyIntegrator(*this));

   L_form.AddDomainIntegrator(new MassIntegrator);
   L_form.Assemble();
   L_form.Finalize();
   L.reset(L_form.ParallelAssemble());

   R_form.AddDomainIntegrator(new VectorFEMassIntegrator);
   R_form.Assemble();
   R_form.Finalize();
   R.reset(R_form.ParallelAssemble());

   D_form.AddDomainIntegrator(new MixedScalarDivergenceIntegrator);
   D_form.Assemble();
   D_form.Finalize();
   D.reset(D_form.ParallelAssemble());
   Dt.reset(D->Transpose());

   Q_e_form.AddDomainIntegrator(new DomainLFIntegrator(Q_e_coeff));
   S_E_form.AddDomainIntegrator(new DomainLFIntegrator(S_E_coeff));
   b_n_form.AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(
                                     E_bdr_coeff));

   nonlinear_solver.reset(new BrunnerNowackIteration(*this));

   SetTime(0);
}

void RadiationDiffusionOperator::ImplicitSolve(
   const double dt_, const Vector &x, Vector &k)
{
   using namespace MMS;

   // Solve the system k = f(x + dt*k)

   const int n_l2 = fes_l2.GetTrueVSize();
   const int n_rt = fes_rt.GetTrueVSize();

   b.SetSize(x.Size());
   Vector b_e(b, offsets[0], n_l2);
   Vector b_E(b, offsets[1], n_l2);
   Vector b_F(b, offsets[2], n_rt);

   const Vector x_e(const_cast<Vector&>(x), offsets[0], n_l2);
   const Vector x_E(const_cast<Vector&>(x), offsets[1], n_l2);

   dt = dt_;
   e_gf.SetFromTrueDofs(x_e);  // Set state needed by nonlinear operator H

   // Form the right-hand side by moving all terms that do not depend on k
   // into the vector b.
   z.SetSize(n_l2);
   L->Mult(x_E, z);

   b_e.Set(c*eta*sigma, z);
   b_e += Q_e;

   b_E.Set(-c*eta*sigma, z);
   b_E += S_E;

   z.SetSize(n_rt);
   Dt->Mult(x_E, z);
   add(1.0/dt, b_n, -1.0/dt, z, b_F); // Include the boundary flux term.

   k = 0.0; // zero initial guess
   nonlinear_solver->Setup();
   nonlinear_solver->Mult(b, k);
}

void RadiationDiffusionOperator::SetTime(const double t_)
{
   t = t_;

   // Set the time for the time-dependent coefficients
   Q_e_coeff.SetTime(t);
   S_E_coeff.SetTime(t);
   E_bdr_coeff.SetTime(t);

   Q_e.SetSize(fes_l2.GetTrueVSize());
   S_E.SetSize(fes_l2.GetTrueVSize());
   b_n.SetSize(fes_rt.GetTrueVSize());

   // Reassemble the source terms
   Q_e_form.Assemble();
   Q_e_form.ParallelAssemble(Q_e);

   S_E_form.Assemble();
   S_E_form.ParallelAssemble(S_E);

   b_n_form.Assemble();
   b_n_form.ParallelAssemble(b_n);

}

void RadiationDiffusionOperator::ComputeFlux(Vector &x) const
{
   const int n_l2 = fes_l2.GetTrueVSize();
   const int n_rt = fes_rt.GetTrueVSize();

   const Vector x_E(x, offsets[1], n_l2);
   Vector x_F(x, offsets[2], n_rt);

   b.SetSize(n_rt);
   Dt->Mult(x_E, b);
   double coeff = MMS::c/MMS::sigma/3.0;
   add(-coeff, b_n, coeff, b, b);

   Array<int> empty;
   OperatorHandle R_op;
   const_cast<ParBilinearForm&>(R_form).FormSystemMatrix(empty, R_op);
   OperatorJacobiSmoother jacobi(R_form, empty);

   CGSolver cg(GetComm());
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(200);
   cg.SetOperator(*R);
   cg.SetPreconditioner(jacobi);
   cg.SetPrintLevel(IterativeSolver::PrintLevel().None());

   x_F = 0.0;
   cg.Mult(b, x_F);
}

} // namespace mfem
