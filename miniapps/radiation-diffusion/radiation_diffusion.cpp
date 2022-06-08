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
     S_e_coeff(MMS::MaterialEnergySource),
     S_E_coeff(MMS::RadiationEnergySource),
     H_form(&fes_l2),
     L_form(&fes_l2),
     R_form(&fes_rt),
     D_form(&fes_rt, &fes_l2)
{
   const int n_l2 = fes_l2.GetTrueVSize();
   const int n_rt = fes_rt.GetTrueVSize();

   // Unknowns: material energy, radiation energy (L2), radiation flux (RT)
   width = height = 2*n_l2 + n_rt;

   offsets = Array<int>({0, n_l2, 2*n_l2, 2*n_l2 + n_rt});

   double h_min, h_max, kappa_min, kappa_max;
   mesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
   dt = h_min*0.05/MMS::tau;

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

   nonlinear_solver.reset(new BrunnerNowackIteration(*this));
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

   k = 0.0; // zero initial guess

   nonlinear_solver->Mult(b, k);
}

} // namespace mfem
