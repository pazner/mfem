// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

namespace
{
#ifdef MFEM_USE_SINGLE
constexpr real_t fd_step = 1e-3, fd_tol = 2e-3;
#else
constexpr real_t fd_step = 1e-6, fd_tol = 1e-8;
#endif
}

TEST_CASE("Hyperbolic flux and face derivatives", "[Hyperbolic]")
{
   Mesh mesh = Mesh::MakeCartesian2D(2,1,Element::QUADRILATERAL,true);
   FaceElementTransformations *face = nullptr;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
      auto *candidate = mesh.GetFaceElementTransformations(i);
      if (candidate->Elem2No >= 0) { face = candidate; break; }
   }
   REQUIRE(face != nullptr);
   const auto &ip = Geometries.GetCenter(Geometry::SEGMENT);
   face->SetAllIntPoints(&ip);
   Vector velocity(2); velocity(0) = 1.3; velocity(1) = -.4;
   VectorConstantCoefficient coefficient(velocity);
   AdvectionFlux flux(coefficient);
   RusanovFlux numerical_flux(flux);
   Vector normal(2), a(1), b(1), plus(1), minus(1);
   a(0) = .3; b(0) = 1.4;
   DenseMatrix derivative(1);
   for (real_t orientation : {-1.0,1.0})
   {
      normal(0) = 2.1*orientation; normal(1) = .7*orientation;
      for (int side : {1,2})
      {
         numerical_flux.Grad(side,a,b,normal,*face,derivative);
         Vector &state = side == 1 ? a : b;
         const real_t eps = fd_step;
         state(0) += eps; numerical_flux.Eval(a,b,normal,*face,plus);
         state(0) -= 2*eps; numerical_flux.Eval(a,b,normal,*face,minus);
         state(0) += eps;
         REQUIRE(derivative(0,0) == MFEM_Approx((plus(0)-minus(0))/(2*eps),fd_tol));
      }
   }
   DG_FECollection fec(2,2);
   const FiniteElement &fe = *fec.FiniteElementForGeometry(Geometry::SQUARE);
   Vector state(2*fe.GetDof()), rp, rm;
   for (int i = 0; i < state.Size(); i++) { state(i) = .7+.03*i; }
   HyperbolicFormIntegrator integrator(numerical_flux,3);
   integrator.AssembleFaceGrad(fe,fe,*face,state,derivative);
   for (int j = 0; j < state.Size(); j++)
   {
      state(j) += fd_step; integrator.AssembleFaceVector(fe,fe,*face,state,rp);
      state(j) -= 2*fd_step; integrator.AssembleFaceVector(fe,fe,*face,state,rm);
      state(j) += fd_step;
      for (int i = 0; i < state.Size(); i++)
      {
         REQUIRE(derivative(i,j) == MFEM_Approx((rp(i)-rm(i))/(2*fd_step),fd_tol));
      }
   }
}

TEST_CASE("Euler conservative flux Jacobian", "[Hyperbolic]")
{
   Mesh mesh = Mesh::MakeCartesian2D(1,1,Element::QUADRILATERAL,true);
   auto &tr = *mesh.GetElementTransformation(0);
   tr.SetIntPoint(&Geometries.GetCenter(Geometry::SQUARE));
   for (int dim : {1,2,3})
   {
      EulerFlux flux(dim,1.4);
      Vector u(dim+2);
      u(0) = 1.2; u(dim+1) = 4.0;
      for (int d = 0; d < dim; d++) { u(d+1) = .3*(d+1); }
      DenseTensor jac;
      flux.ComputeFluxJacobian(u,tr,jac);
      DenseMatrix fp(dim+2,dim), fm(dim+2,dim);
      for (int j = 0; j < dim+2; j++)
      {
         u(j) += fd_step; flux.ComputeFlux(u,tr,fp);
         u(j) -= 2*fd_step; flux.ComputeFlux(u,tr,fm);
         u(j) += fd_step;
         for (int i = 0; i < dim+2; i++)
            for (int d = 0; d < dim; d++)
            {
               REQUIRE(jac(i,j,d) == MFEM_Approx((fp(i,d)-fm(i,d))/(2*fd_step),fd_tol));
            }
      }
   }
}
