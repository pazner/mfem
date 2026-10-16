// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "acoustics.hpp"
#include <cmath>
#include <random>

namespace acoustics
{
real_t AcousticFlux::ComputeFlux(const Vector &u, ElementTransformation &,
                                 DenseMatrix &f) const
{
   f.SetSize(3,2);
   f = 0.0;
   for (int d = 0; d < 2; d++)
   {
      f(0,d) = medium.density*medium.speed*medium.speed*u(d+1);
      f(d+1,d) = u(0)/medium.density;
   }
   return medium.speed;
}
real_t AcousticFlux::ComputeFluxDotN(const Vector &u, const Vector &n,
                                     FaceElementTransformations &, Vector &f) const
{
   f.SetSize(3);
   f(0) = medium.density*medium.speed*medium.speed*(u(1)*n(0)+u(2)*n(1));
   f(1) = u(0)*n(0)/medium.density;
   f(2) = u(0)*n(1)/medium.density;
   // RusanovFlux applies the normal magnitude to the dissipation itself.
   return medium.speed;
}
void AcousticFlux::ComputeFluxJacobian(const Vector &, ElementTransformation &,
                                       DenseTensor &j) const
{
   j.SetSize(3,3,2);
   j = 0.0;
   for (int d = 0; d < 2; d++)
   {
      j(0,d+1,d) = medium.density*medium.speed*medium.speed;
      j(d+1,0,d) = 1/medium.density;
   }
}
void AcousticFlux::ComputeFluxJacobianDotN(const Vector &, const Vector &n,
                                           ElementTransformation &,
                                           DenseMatrix &j) const
{
   j.SetSize(3);
   j = 0.0;
   for (int d = 0; d < 2; d++)
   {
      j(0,d+1) = medium.density*medium.speed*medium.speed*n(d);
      j(d+1,0) = n(d)/medium.density;
   }
}
Problem::Problem(const Medium &m, const std::string &n) : medium(m), name(n)
{
   MFEM_VERIFY(name == "wave" || name == "multimode" || name == "constant",
               "Unknown acoustic problem.");
   if (name != "multimode")
   {
      return;
   }

   // One representative of each pair k and -k; sample without replacement.
   for (int mx = 0; mx <= 8; mx++)
   {
      for (int my = -8; my <= 8; my++)
      {
         if ((mx == 0 && my <= 0) || mx*mx+my*my > 64)
         {
            continue;
         }
         const real_t kx = (M_PI/2)*mx;
         const real_t ky = (M_PI/2)*my;
         modes.push_back({kx,ky,std::hypot(kx,ky),0});
      }
   }
   std::mt19937 generator(1742);
   for (size_t i = modes.size()-1; i > 0; i--)
   {
      std::swap(modes[i],modes[generator()%(i+1)]);
   }
   modes.resize(48);
   for (auto &mode : modes)
   {
      mode.phase = 2*M_PI*(real_t(generator())/4294967296.0);
   }
}

void Problem::State(const Vector &x, real_t time, Vector &u) const
{
   u.SetSize(3);
   if (name == "constant")
   {
      u(0) = -.7;
      u(1) = .3;
      u(2) = -.2;
      return;
   }
   if (name == "multimode")
   {
      u = 0.0;
      const real_t amplitude = 1/std::sqrt(real_t(modes.size()));
      for (const auto &mode : modes)
      {
         const real_t p = amplitude*std::cos(mode.kx*x(0)+mode.ky*x(1)
                                             -medium.speed*mode.magnitude*time+mode.phase);
         const real_t velocity = p/(medium.density*medium.speed*mode.magnitude);
         u(0) += p;
         u(1) += mode.kx*velocity;
         u(2) += mode.ky*velocity;
      }
      return;
   }
   u(0) = std::sin((M_PI/2)*(x(0)+x(1)-std::sqrt(2.0)*medium.speed*time));
   u(1) = u(2) = u(0)/(std::sqrt(2.0)*medium.density*medium.speed);
}
void RandomRightHandSide(ParFiniteElementSpace &fes, Vector &rhs)
{
   std::mt19937 generator(1742);
   generator.discard(fes.GetMyTDofOffset());
   rhs.SetSize(fes.GetTrueVSize());
   for (int i = 0; i < rhs.Size(); i++)
   {
      rhs(i) = 2*(real_t(generator())/4294967296.0)-1;
   }
   rhs /= GlobalNorm(rhs,fes.GetComm());
}
Mesh PeriodicMesh(int n, bool triangles)
{
   Mesh mesh = Mesh::MakeCartesian2D(n,n,
                                     triangles ? Element::TRIANGLE : Element::QUADRILATERAL,
                                     true,4.0,4.0);
   for (int i = 0; i < mesh.GetNV(); i++)
   {
      mesh.GetVertex(i)[0] -= 2;
      mesh.GetVertex(i)[1] -= 2;
   }
   Vector tx(2), ty(2);
   tx = 0.0;
   ty = 0.0;
   tx(0) = 4;
   ty(1) = 4;
   return Mesh::MakePeriodic(mesh,mesh.CreatePeriodicVertexMapping({tx,ty}));
}
void VerifyCollectively(bool ok, MPI_Comm comm, const char *message)
{
   int local = ok, global;
   MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MIN,comm);
   MFEM_VERIFY(global, message);
}
real_t GlobalNorm(const Vector &x, MPI_Comm comm)
{
   real_t local = x*x, global;
   MPI_Allreduce(&local,&global,1,MPITypeMap<real_t>::mpi_type,MPI_SUM,comm);
   return std::sqrt(global);
}
void CheckFinite(const Vector &x, MPI_Comm comm)
{
   bool ok = true;
   for (int i = 0; i < x.Size(); i++)
   {
      ok = ok && std::isfinite(x(i));
   }
   VerifyCollectively(ok,comm,"Nonfinite acoustic state.");
}
}
