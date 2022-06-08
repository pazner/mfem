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
#include "linear_solver.hpp"

namespace mfem
{

RadiationDiffusionLinearSolver::RadiationDiffusionLinearSolver(
   RadiationDiffusionOperator &rad_diff_)
   : Solver(rad_diff_.Height()),
     rad_diff(rad_diff_)
{
   using namespace MMS;

   const double dt = rad_diff.dt;

   Array2D<HypreParMatrix*> EF_blocks(2,2);
   Array2D<double> EF_coeff(2,2);

   EF_blocks(0,0) = rad_diff.L.get();
   EF_blocks(0,1) = rad_diff.D.get();
   EF_blocks(1,0) = rad_diff.Dt.get();
   EF_blocks(1,1) = rad_diff.R.get();

   EF_coeff(0,0) = 1.0 + c*dt*sigma;
   EF_coeff(0,1) = 1.0;
   EF_coeff(1,0) = 1.0;
   EF_coeff(1,1) = -3*sigma/c/dt;

   JEF.reset(HypreParMatrixFromBlocks(EF_blocks, &EF_coeff));
   EF_solver.reset(new SerialDirectSolver(*JEF));
}

void RadiationDiffusionLinearSolver::Mult(const Vector &b, Vector &x) const
{
   EF_solver->Mult(b, x);
   // J_solver->Mult(b, x);

   // const int n_eE = rad_diff.offsets[2];
   // Vector x_eE(x, 0, n_eE);
   // const Vector b_eE(const_cast<Vector&>(b), 0, n_eE);
   // r.SetSize(x.Size());
   // eE_solver->Mult(b_eE, x_eE);

   // r.SetSize(x.Size());
   // J.Mult(x, r);
   // subtract(b, r, r); // Set r = b - J*x
   // const double r_norm = r.Norml2();
   // std::cout << "Right-hand side norm:   " << std::setw(10) << std::scientific <<
   //           b.Norml2() << std::endl;
   // std::cout << "Linear solver residual: " << std::setw(10) << std::scientific <<
   //           r_norm << std::endl;

   // Brunner-Nowak iteration

   // const int maxit = 100;
   // const double tol = 1e-12;

   // std::cout << " Brunner-Nowack iteration\n"
   //           << " It.    Resnorm\n"
   //           << "=====================\n";

   // const int n_eE = rad_diff.offsets[2];
   // const int n_EF = rad_diff.offsets[3] - rad_diff.offsets[1];

   // Vector x_eE(x, 0, n_eE);
   // Vector x_EF(x, rad_diff.offsets[1], n_EF);

   // r.SetSize(x.Size());
   // Vector r_eE(r, 0, n_eE);
   // Vector r_EF(r, rad_diff.offsets[1], n_EF);

   // c_eE.SetSize(n_eE);
   // c_EF.SetSize(n_EF);

   // for (int it = 0; it < maxit; ++it)
   // {
   //    std::cout << " " << std::setw(3) << it << "    " << std::flush;
   //    // Compute full residual
   //    J.Mult(x, r);
   //    subtract(b, r, r); // Set r = b - J*x
   //    const double r_norm = r.Norml2();
   //    std::cout << std::setw(10) << std::scientific << r_norm << std::endl;
   //    if (r.Norml2() < tol) { break; }

   //    // Solve for correction to x_e, x_E
   //    eE_solver->Mult(r_eE, c_eE);

   //    // Update x given the correction c_EE
   //    x_eE += c_eE;

   //    // Compute residual again
   //    J.Mult(x, r);
   //    subtract(b, r, r); // Set r = b - J*x
   //    if (r.Norml2() < tol) { break; }

   //    // Solve for correction to x_E, x_F
   //    EF_solver->Mult(r_EF, c_EF);

   //    // Update x given the correction c_EF
   //    x_EF += c_EF;
   // }
}

// void RadiationDiffusionLinearSolver::Update(const Vector &x)
// {
//    using namespace MMS;

//    const double dt = rad_diff.dt;

//    HypreParMatrix &L = *rad_diff.L;
//    HypreParMatrix &D = *rad_diff.D;
//    HypreParMatrix &Dt = *rad_diff.Dt;
//    HypreParMatrix &R = *rad_diff.R;

//    const Vector x_e(const_cast<Vector&>(x), 0, rad_diff.offsets[1]);
//    lin_coeff.b_gf = x_e;

//    dH_form.Update();
//    dH_form.Assemble();
//    dH_form.Finalize();
//    dH.reset(dH_form.ParallelAssemble());
//    J00.reset(Add(rho, *rad_diff.L, 1.0, *dH));

//    J.SetBlock(0, 0, J00.get());
//    J.SetBlock(0, 1, &L, -c*dt*sigma);
//    J.SetBlock(1, 0, dH.get(), -1.0);
//    J.SetBlock(1, 1, &L, 1.0 + c*dt*sigma);
//    J.SetBlock(1, 2, &D);
//    J.SetBlock(2, 1, &Dt);
//    J.SetBlock(2, 2, &R, -3*sigma/c/dt);

//    Array2D<HypreParMatrix*> eE_blocks(2,2), EF_blocks(2,2);
//    Array2D<double> eE_coeff(2,2), EF_coeff(2,2);

//    eE_blocks(0,0) = J00.get();
//    eE_blocks(0,1) = &L;
//    eE_blocks(1,0) = dH.get();
//    eE_blocks(1,1) = &L;

//    eE_coeff(0,0) = 1.0;
//    eE_coeff(0,1) = -c*dt*sigma;
//    eE_coeff(1,0) = -1.0;
//    eE_coeff(1,1) = 1.0 + c*dt*sigma;

//    EF_blocks(0,0) = &L;
//    EF_blocks(0,1) = &D;
//    EF_blocks(1,0) = &Dt;
//    EF_blocks(1,1) = &R;

//    EF_coeff(0,0) = 1.0 + c*dt*sigma;
//    EF_coeff(0,1) = 1.0;
//    EF_coeff(1,0) = 1.0;
//    EF_coeff(1,1) = -3*sigma/c/dt;

//    JeE.reset(HypreParMatrixFromBlocks(eE_blocks, &eE_coeff));
//    JEF.reset(HypreParMatrixFromBlocks(EF_blocks, &EF_coeff));

//    eE_solver.reset(new SerialDirectSolver(*JeE));
//    EF_solver.reset(new SerialDirectSolver(*JEF));

//    Array2D<HypreParMatrix*> J_blocks(3,3);
//    Array2D<double> J_coeff(3,3);
//    J_coeff(0, 0) = 1.0;
//    J_coeff(0, 1) = -c*dt*sigma;
//    J_coeff(1, 0) = -1.0;
//    J_coeff(1, 1) = 1.0 + c*dt*sigma;
//    J_coeff(1, 2) = 1.0;
//    J_coeff(2, 1) = 1.0;
//    J_coeff(2, 2) = -3*sigma/c/dt;

//    J_blocks(0, 0) = J00.get();
//    J_blocks(0, 1) = &L;
//    J_blocks(0, 2) = nullptr;
//    J_blocks(1, 0) = dH.get();
//    J_blocks(1, 1) = &L;
//    J_blocks(1, 2) = &D;
//    J_blocks(2, 0) = nullptr;
//    J_blocks(2, 1) = &Dt;
//    J_blocks(2, 2) = &R;

//    JJ.reset(HypreParMatrixFromBlocks(J_blocks, &J_coeff));
//    J_solver.reset(new SerialDirectSolver(*JJ));
// }

void RadiationDiffusionLinearSolver::SetOperator(const Operator &op) { }

} // namespace mfem
