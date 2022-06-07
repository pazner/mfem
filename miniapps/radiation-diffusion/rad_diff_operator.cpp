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

#include "rad_diff_operator.hpp"

namespace mfem
{

NonlinearEnergyIntegrator::NonlinearEnergyIntegrator(
   RadiationDiffusionOperator &rad_diff)
   : fes(rad_diff.fes_l2), coeff(rad_diff), integ(coeff),
     deriv_coeff(rad_diff), mass(deriv_coeff) { }

void NonlinearEnergyIntegrator::AssembleElementVector(
   const FiniteElement &el, ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   fes.GetElementDofs(Tr.ElementNo, dofs);
   coeff.b_gf.SetSubVector(dofs, elfun);
   integ.AssembleRHSElementVect(el, Tr, elvect);
}

void NonlinearEnergyIntegrator::AssembleElementGrad(
   const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
   DenseMatrix &elmat)
{
   fes.GetElementDofs(Tr.ElementNo, dofs);
   deriv_coeff.b_gf.SetSubVector(dofs, elfun);
   mass.AssembleElementMatrix(el, Tr, elmat);
}

RadiationDiffusionOperator::RadiationDiffusionOperator(ParMesh &mesh, int order)
   : dim(mesh.Dimension()),
     fec_l2(order-1, dim, b2, mt),
     fes_l2(&mesh, &fec_l2),
     fec_rt(order-1, dim, b1, b2),
     fes_rt(&mesh, &fec_rt),
     e_gf(&fes_l2),
     E_gf(&fes_l2),
     F_gf(&fes_rt),
     S_e_coeff(MMS::MaterialEnergySource),
     S_E_coeff(MMS::RadiationEnergySource),
     H_form(&fes_l2),
     L_form(&fes_l2),
     R_form(&fes_rt),
     D_form(&fes_rt, &fes_l2)
{
   const int n_l2 = fes_l2.GetTrueVSize();
   const int n_rt = fes_rt.GetTrueVSize();

   // Unknowns: material energy, radiation energy (L2), flux (RT)
   width = height = 2*n_l2 + n_rt;

   offsets = Array<int>({0, n_l2, 2*n_l2, 2*n_l2 + n_rt});

   double h_min, h_max, kappa_min, kappa_max;
   mesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
   dt = h_min*0.05/MMS::tau;

   FunctionCoefficient e_coeff(MMS::InitialMaterialEnergy);
   e_gf.ProjectCoefficient(e_coeff);

   FunctionCoefficient E_coeff(MMS::InitialRadiationEnergy);
   E_gf.ProjectCoefficient(E_coeff);

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

   // J.SetBlock(0, 0, J00.get());
   // J.SetBlock(0, 1, L.get(), -c*dt*sigma);
   // J.SetBlock(1, 0, &H_form, -1.0);
   // J.SetBlock(1, 1, &L, 1.0 + c*dt*sigma);
   // J.SetBlock(1, 2, &D);
   // J.SetBlock(2, 1, &Dt);
   // J.SetBlock(2, 2, &R, -3*sigma/c/dt);

   linear_solver.reset(new RadiationDiffusionLinearSolver(*this));

   newton.SetAbsTol(1e-12);
   newton.SetRelTol(1e-12);
   newton.SetMaxIter(20);
   newton.SetPrintLevel(IterativeSolver::PrintLevel().Iterations());
   newton.SetSolver(*linear_solver);
   newton.SetOperator(*this);
}

void RadiationDiffusionOperator::Mult(const Vector &x, Vector &y) const
{
   using namespace MMS;

   const int n_l2 = fes_l2.GetTrueVSize();
   const int n_rt = fes_rt.GetTrueVSize();

   const Vector x_e(const_cast<Vector&>(x), offsets[0], n_l2);
   const Vector x_E(const_cast<Vector&>(x), offsets[1], n_l2);
   const Vector x_F(const_cast<Vector&>(x), offsets[2], n_rt);

   Vector y_e(y, offsets[0], n_l2);
   Vector y_E(y, offsets[1], n_l2);
   Vector y_F(y, offsets[2], n_rt);

   // Material energy
   z.SetSize(n_l2);
   L->Mult(x_e, y_e);
   y_e *= rho;

   H_form.Mult(x_e, z);
   y_e += z;
   y_E.Set(-1, z); // Contribution to radiation energy

   L->Mult(x_E, z);
   y_E.Add(1 + c*dt*sigma, z); // Contribution to radiation energy
   y_e.Add(-c*dt*sigma, z);

   // Radiation energy
   D->Mult(x_F, z);
   y_E += z;

   // Radiation flux
   z.SetSize(n_rt);
   R->Mult(x_F, y_F);
   y_F *= -3*sigma/c/dt;

   Dt->Mult(x_E, z);
   y_F += z;

   // y_F = 0.0;
}

Operator &RadiationDiffusionOperator::GetGradient(const Vector &x) const
{
   linear_solver->Update(x);
   const Operator &op = *this;
   return const_cast<Operator&>(op);
}

void RadiationDiffusionOperator::ImplicitSolve(
   const double dt_, const Vector &x, Vector &k)
{
   using namespace MMS;

   dt = dt_;

   // Solve the system k = f(x + dt*k)

   const int n_l2 = fes_l2.GetTrueVSize();
   const int n_rt = fes_rt.GetTrueVSize();

   b.SetSize(x.Size());
   Vector b_e(b, offsets[0], n_l2);
   Vector b_E(b, offsets[1], n_l2);
   Vector b_F(b, offsets[2], n_rt);

   const Vector x_e(const_cast<Vector&>(x), offsets[0], n_l2);
   const Vector x_E(const_cast<Vector&>(x), offsets[1], n_l2);
   const Vector x_F(const_cast<Vector&>(x), offsets[2], n_rt);

   e_gf = x_e; // Set state needed by nonlinear operator H

   z.SetSize(n_l2);
   L->Mult(x_E, z);

   z.Randomize(2);

   b_e.Set(c*eta*sigma, z);
   // TODO: add source to b_e

   b_E.Set(-c*eta*sigma, z);
   // TODO: add source to b_E

   z.SetSize(n_rt);
   Dt->Mult(x_E, z);
   b_F.Set(-1.0/dt, z);
   // TODO: add boundary flux term to b_F

   // TEMPORARY
   // b_F = 0.0;

   k = 0.0; // zero initial guess

   newton.Mult(b, k);
}

} // namespace mfem
