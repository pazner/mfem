// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Internal header, included only by .cpp files.
// Template function implementations.

#ifndef MFEM_QINTERP_HDIV_HPP
#define MFEM_QINTERP_HDIV_HPP

#include "../quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include "../kernels.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

// Template compute kernel for H(div) Values in 2D: tensor product version.
template<QVectorLayout Q_LAYOUT, int MAX_D1D = 0, int MAX_Q1D = 0>
static void HdivValues2D(const int NE,
                         const double *J_,
                         const double *detJ_,
                         const double *bc_,
                         const double *bo_,
                         const double *x_,
                         double *y_,
                         const int VDIM,
                         const int D1D,
                         const int Q1D)
{
   auto J = Reshape(J_, Q1D, Q1D, 2, 2, NE);
   const auto detJ = Reshape(detJ_, Q1D, Q1D, NE);
   const auto bc = Reshape(bc_, Q1D, D1D);
   const auto bo = Reshape(bo_, Q1D, D1D - 1);
   const auto x = Reshape(x_, 2*(D1D-1)*D1D, VDIM, NE);

   auto y = Q_LAYOUT == QVectorLayout::byNODES ?
            Reshape(y_, Q1D, Q1D, 2*VDIM, NE):
            Reshape(y_, 2*VDIM, Q1D, Q1D, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ = MAX_Q1D ? MAX_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD = MAX_D1D ? MAX_D1D : DofQuadLimits::MAX_D1D;
      constexpr int MDQ = (MQ > MD) ? MQ : MD;

      MFEM_SHARED double sBc[MQ*MD];
      MFEM_SHARED double sBo[MQ*MD];
      MFEM_SHARED double sm0[MDQ*MDQ];
      MFEM_SHARED double sm1[MDQ*MDQ];
      MFEM_SHARED double sm2[2*MDQ*MDQ];

      kernels::internal::LoadB<MD,MQ>(D1D,Q1D,bc,sBc);
      kernels::internal::LoadB<MD,MQ>(D1D-1,Q1D,bo,sBo);

      ConstDeviceMatrix Bc(sBc, D1D,Q1D);
      ConstDeviceMatrix Bo(sBo, D1D-1,Q1D);

      DeviceMatrix DD(sm0, MD, MD);
      DeviceMatrix DQ(sm1, MD, MQ);
      DeviceCube QQ(sm2, MQ, MQ, 2);

      for (int c = 0; c < VDIM; c++)
      {
         for (int d = 0; d < 2; ++d)
         {
            const int offset = d*D1D*(D1D - 1);
            const int d1d_x = (d == 0) ? D1D : D1D - 1;
            const int d1d_y = (d == 1) ? D1D : D1D - 1;
            const auto &Bx = (d == 0) ? Bc : Bo;
            const auto &By = (d == 1) ? Bc : Bo;

            // Load x into shared memory
            MFEM_FOREACH_THREAD(dy,y,d1d_y)
            {
               MFEM_FOREACH_THREAD(dx,x,d1d_x)
               {
                  DD(dx,dy) = x(dx + (dy * d1d_x) + offset, c, e);
               }
            }
            MFEM_SYNC_THREAD;

            // Eval x
            MFEM_FOREACH_THREAD(dy,y,d1d_y)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  for (int dx = 0; dx < d1d_x; ++dx)
                  {
                     u += Bx(dx,qx) * DD(dx,dy);
                  }
                  DQ(dy,qx) = u;
               }
            }
            MFEM_SYNC_THREAD;

            // Eval y
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  double u = 0.0;
                  for (int dy = 0; dy < d1d_y; ++dy)
                  {
                     u += DQ(dy,qx) * By(dy,qy);
                  }
                  QQ(qx,qy,d) = u;
               }
            }
            MFEM_SYNC_THREAD;
         }
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const double u0 = QQ(qx,qy,0);
               const double u1 = QQ(qx,qy,1);

               const double J00 = J(qx,qy,0,0,e);
               const double J10 = J(qx,qy,1,0,e);
               const double J01 = J(qx,qy,0,1,e);
               const double J11 = J(qx,qy,1,1,e);
               const double inv_det = 1.0/detJ(qx,qy,e);

               const double u0_phys = inv_det*(J00*u0 + J01*u1);
               const double u1_phys = inv_det*(J10*u0 + J11*u1);

               const int c0 = 0 + c*2;
               const int c1 = 1 + c*2;

               if (Q_LAYOUT == QVectorLayout::byVDIM)
               {
                  y(c0,qx,qy,e) = u0_phys;
                  y(c1,qx,qy,e) = u1_phys;
               }
               if (Q_LAYOUT == QVectorLayout::byNODES)
               {
                  y(qx,qy,c0,e) = u0_phys;
                  y(qx,qy,c1,e) = u1_phys;
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Template compute kernel for H(div) Values in 3D: tensor product version.
template<QVectorLayout Q_LAYOUT, int MAX_D1D = 0, int MAX_Q1D = 0>
static void HdivValues3D(const int NE,
                         const double *J_,
                         const double *detJ_,
                         const double *bc_,
                         const double *bo_,
                         const double *x_,
                         double *y_,
                         const int VDIM,
                         const int D1D,
                         const int Q1D)
{
   auto J = Reshape(J_, Q1D, Q1D, Q1D, 3, 3, NE);
   const auto detJ = Reshape(detJ_, Q1D, Q1D, Q1D, NE);
   const auto Bc = Reshape(bc_, Q1D, D1D);
   const auto Bo = Reshape(bo_, Q1D, D1D - 1);
   const auto x = Reshape(x_, 3*(D1D-1)*(D1D-1)*D1D, VDIM, NE);

   auto y = Q_LAYOUT == QVectorLayout::byNODES ?
            Reshape(y_, Q1D, Q1D, Q1D, 3*VDIM, NE):
            Reshape(y_, 3*VDIM, Q1D, Q1D, Q1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ = MAX_Q1D ? MAX_Q1D : 8;
      constexpr int MD = MAX_D1D ? MAX_D1D : 8;

      for (int c = 0; c < VDIM; c++)
      {
         double QQQ_[MQ*MQ*MQ*3];
         DeviceTensor<4> QQQ(QQQ_, Q1D, Q1D, Q1D, 3);
         for (int d = 0; d < 3; ++d)
         {
            const int offset = d*(D1D-1)*(D1D-1)*D1D;
            const int d1d_x = (d == 0) ? D1D : D1D - 1;
            const int d1d_y = (d == 1) ? D1D : D1D - 1;
            const int d1d_z = (d == 2) ? D1D : D1D - 1;
            const auto &Bx = (d == 0) ? Bc : Bo;
            const auto &By = (d == 1) ? Bc : Bo;
            const auto &Bz = (d == 2) ? Bc : Bo;

            double DQQ[MD*MQ*MQ];

            for (int dz = 0; dz < d1d_z; ++dz)
            {
               double DDQ[MD*MD*MQ];

               // Eval x
               for (int dy = 0; dy < d1d_y; ++dy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     double u = 0.0;
                     for (int dx = 0; dx < d1d_x; ++dx)
                     {
                        const int i = offset + dx + dy*d1d_x + dz*d1d_x*d1d_y;
                        u += Bx(qx,dx) * x(i, c, e);
                     }
                     DDQ[qx + dy*Q1D + dz*Q1D*d1d_y] = u;
                  }
               }

               // Eval y
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     double u = 0.0;
                     for (int dy = 0; dy < d1d_y; ++dy)
                     {
                        u += By(qy,dy) * DDQ[qx + dy*Q1D + dz*Q1D*d1d_y];
                     }
                     DQQ[qx + qy*Q1D + dz*Q1D*Q1D] = u;
                  }
               }
            }
            // Eval z
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     double u = 0.0;
                     for (int dz = 0; dz < d1d_z; ++dz)
                     {
                        u += Bz(qz,dz) * DQQ[qx + qy*Q1D + dz*Q1D*Q1D];
                     }
                     QQQ(qx,qy,qz,d) = u;
                  }
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double u0 = QQQ(qx,qy,qz,0);
                  const double u1 = QQQ(qx,qy,qz,1);
                  const double u2 = QQQ(qx,qy,qz,2);

                  const double inv_det = 1.0/detJ(qx,qy,qz,e);
                  for (int d = 0; d < 3; ++ d)
                  {
                     const double J0 = J(qx,qy,qz,d,0,e);
                     const double J1 = J(qx,qy,qz,d,1,e);
                     const double J2 = J(qx,qy,qz,d,2,e);

                     const double u_phys = inv_det*(u0*J0 + u1*J1 + u2*J2);

                     const int ci = d + c*3;

                     if (Q_LAYOUT == QVectorLayout::byVDIM) { y(ci,qx,qy,qz,e) = u_phys; }
                     if (Q_LAYOUT == QVectorLayout::byNODES) { y(qx,qy,qz,ci,e) = u_phys; }
                  }
               }
            }
         }
      }
   });
}

template<QVectorLayout VL>
void HdivTensorValues(const int NE,
                      const int vdim,
                      const GeometricFactors *geom,
                      const DofToQuad &maps,
                      const DofToQuad &maps_o,
                      const Vector &e_vec,
                      Vector &q_val)
{
   if (NE == 0) { return; }
   const int dim = maps.FE->GetDim();
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const double *Bc = maps.B.Read();
   const double *Bo = maps_o.B.Read();
   const double *X = e_vec.Read();
   const double *J = geom->J.Read();
   const double *detJ = geom->detJ.Read();
   double *Y = q_val.Write();

   constexpr QVectorLayout L = VL;

   if (dim == 2)
   {
      const int MD = DeviceDofQuadLimits::Get().MAX_D1D;
      const int MQ = DeviceDofQuadLimits::Get().MAX_Q1D;
      MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                  << " are not supported!");
      MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                  << MQ << " 1D points are not supported!");
      HdivValues2D<L>(NE,J,detJ,Bc,Bo,X,Y,vdim,D1D,Q1D);
   }
   else if (dim == 3)
   {
      const int MD = 8;
      const int MQ = 8;
      MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                  << " are not supported!");
      MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                  << MQ << " 1D points are not supported!");
      HdivValues3D<L>(NE,J,detJ,Bc,Bo,X,Y,vdim,D1D,Q1D);
   }
   else
   {
      MFEM_ABORT("Dimension not supported");
   }
}

// Template compute kernel for H(div) divergence in 2D: tensor product version.
template<QVectorLayout Q_LAYOUT, int MAX_D1D = 0, int MAX_Q1D = 0>
static void HdivDivergence2D(const int NE,
                             const double *detJ_,
                             const double *bo_,
                             const double *gc_,
                             const double *x_,
                             double *y_,
                             const int VDIM,
                             const int D1D,
                             const int Q1D)
{
   const auto detJ = Reshape(detJ_, Q1D, Q1D, NE);
   const auto Bo = Reshape(bo_, Q1D, D1D - 1);
   const auto Gc = Reshape(gc_, Q1D, D1D);
   const auto x = Reshape(x_, 2*(D1D-1)*D1D, VDIM, NE);

   auto y = Q_LAYOUT == QVectorLayout::byNODES ?
            Reshape(y_, Q1D, Q1D, VDIM, NE):
            Reshape(y_, VDIM, Q1D, Q1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ = MAX_Q1D ? MAX_Q1D : DofQuadLimits::MAX_Q1D;
      constexpr int MD = MAX_D1D ? MAX_D1D : DofQuadLimits::MAX_D1D;

      double DQ_[MD*MQ];
      double QQ_[MD*MQ];

      DeviceMatrix DQ(DQ_, MQ, MD);
      DeviceMatrix QQ(QQ_, MQ, MQ);

      for (int c = 0; c < VDIM; c++)
      {
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               QQ(qx,qy) = 0.0;
            }
         }

         for (int d = 0; d < 2; ++d)
         {
            const int offset = d*D1D*(D1D - 1);
            const int d1d_x = (d == 0) ? D1D : D1D - 1;
            const int d1d_y = (d == 1) ? D1D : D1D - 1;
            const auto &Bx = (d == 0) ? Gc : Bo;
            const auto &By = (d == 1) ? Gc : Bo;

            // Eval x
            for (int dy = 0; dy < d1d_y; ++dy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  double u = 0.0;
                  for (int dx = 0; dx < d1d_x; ++dx)
                  {
                     u += Bx(qx,dx) * x(offset + dx + dy*d1d_x, c, e);
                  }
                  DQ(qx, dy) = u;
               }
            }
            // Eval y
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  double u = 0.0;
                  for (int dy = 0; dy < d1d_y; ++dy)
                  {
                     u += By(qy,dy) * DQ(qx, dy);
                  }
                  QQ(qx, qy) += u;
               }
            }
         }
         for (int qy = 0; qy < Q1D; ++qy)
         {
            for (int qx = 0; qx < Q1D; ++qx)
            {
               const double u = QQ(qx, qy);
               const double inv_det = 1.0/detJ(qx,qy,e);
               const double phys_u = inv_det * u;
               if (Q_LAYOUT == QVectorLayout::byVDIM) { y(c,qx,qy,e) = phys_u; }
               if (Q_LAYOUT == QVectorLayout::byNODES) { y(qx,qy,c,e) = phys_u; }
            }
         }
      }
   });
}

// Template compute kernel for H(div) divergence in 3D: tensor product version.
template<QVectorLayout Q_LAYOUT, int MAX_D1D = 0, int MAX_Q1D = 0>
static void HdivDivergence3D(const int NE,
                             const double *detJ_,
                             const double *bo_,
                             const double *gc_,
                             const double *x_,
                             double *y_,
                             const int VDIM,
                             const int D1D,
                             const int Q1D)
{
   const auto detJ = Reshape(detJ_, Q1D, Q1D, Q1D, NE);
   const auto Bo = Reshape(bo_, Q1D, D1D - 1);
   const auto Gc = Reshape(gc_, Q1D, D1D);
   const auto x = Reshape(x_, 3*(D1D-1)*(D1D-1)*D1D, VDIM, NE);

   auto y = Q_LAYOUT == QVectorLayout::byNODES ?
            Reshape(y_, Q1D, Q1D, Q1D, VDIM, NE):
            Reshape(y_, VDIM, Q1D, Q1D, Q1D, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      constexpr int MQ = MAX_Q1D ? MAX_Q1D : 8;
      constexpr int MD = MAX_D1D ? MAX_D1D : 8;

      for (int c = 0; c < VDIM; c++)
      {
         double QQQ_[MQ*MQ*MQ*3];
         DeviceCube QQQ(QQQ_, Q1D, Q1D, Q1D);

         for (int qz = 0; qz < Q1D; ++qz)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  QQQ(qx,qy,qz) = 0.0;
               }
            }
         }
         for (int d = 0; d < 3; ++d)
         {
            const int offset = d*(D1D-1)*(D1D-1)*D1D;
            const int d1d_x = (d == 0) ? D1D : D1D - 1;
            const int d1d_y = (d == 1) ? D1D : D1D - 1;
            const int d1d_z = (d == 2) ? D1D : D1D - 1;
            const auto &Bx = (d == 0) ? Gc : Bo;
            const auto &By = (d == 1) ? Gc : Bo;
            const auto &Bz = (d == 2) ? Gc : Bo;

            double DQQ[MD*MQ*MQ];

            for (int dz = 0; dz < d1d_z; ++dz)
            {
               double DDQ[MD*MD*MQ];

               // Eval x
               for (int dy = 0; dy < d1d_y; ++dy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     double u = 0.0;
                     for (int dx = 0; dx < d1d_x; ++dx)
                     {
                        const int i = offset + dx + dy*d1d_x + dz*d1d_x*d1d_y;
                        u += Bx(qx,dx) * x(i, c, e);
                     }
                     DDQ[qx + dy*Q1D + dz*Q1D*d1d_y] = u;
                  }
               }

               // Eval y
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     double u = 0.0;
                     for (int dy = 0; dy < d1d_y; ++dy)
                     {
                        u += By(qy,dy) * DDQ[qx + dy*Q1D + dz*Q1D*d1d_y];
                     }
                     DQQ[qx + qy*Q1D + dz*Q1D*Q1D] = u;
                  }
               }
            }
            // Eval z
            for (int qz = 0; qz < Q1D; ++qz)
            {
               for (int qy = 0; qy < Q1D; ++qy)
               {
                  for (int qx = 0; qx < Q1D; ++qx)
                  {
                     double u = 0.0;
                     for (int dz = 0; dz < d1d_z; ++dz)
                     {
                        u += Bz(qz,dz) * DQQ[qx + qy*Q1D + dz*Q1D*Q1D];
                     }
                     QQQ(qx,qy,qz) += u;
                  }
               }
            }
         }
         for (int qz = 0; qz < Q1D; ++qz)
         {
            for (int qy = 0; qy < Q1D; ++qy)
            {
               for (int qx = 0; qx < Q1D; ++qx)
               {
                  const double u = QQQ(qx,qy,qz);
                  const double inv_det = 1.0/detJ(qx,qy,qz,e);
                  const double u_phys = inv_det*u;
                  if (Q_LAYOUT == QVectorLayout::byVDIM) { y(c,qx,qy,qz,e) = u_phys; }
                  if (Q_LAYOUT == QVectorLayout::byNODES) { y(qx,qy,qz,c,e) = u_phys; }
               }
            }
         }
      }
   });
}

template<QVectorLayout VL>
void HdivTensorDivergence(const int NE,
                          const int vdim,
                          const GeometricFactors *geom,
                          const DofToQuad &maps,
                          const DofToQuad &maps_o,
                          const Vector &e_vec,
                          Vector &q_val)
{
   if (NE == 0) { return; }
   const int dim = maps.FE->GetDim();
   const int D1D = maps.ndof;
   const int Q1D = maps.nqpt;
   const double *Bo = maps_o.B.Read();
   const double *Gc = maps.G.Read();
   const double *X = e_vec.Read();
   const double *detJ = geom->detJ.Read();
   double *Y = q_val.Write();

   constexpr QVectorLayout L = VL;

   if (dim == 2)
   {
      const int MD = DeviceDofQuadLimits::Get().MAX_D1D;
      const int MQ = DeviceDofQuadLimits::Get().MAX_Q1D;
      MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                  << " are not supported!");
      MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                  << MQ << " 1D points are not supported!");
      HdivDivergence2D<L>(NE,detJ,Bo,Gc,X,Y,vdim,D1D,Q1D);
   }
   else if (dim == 3)
   {
      const int MD = 8;
      const int MQ = 8;
      MFEM_VERIFY(D1D <= MD, "Orders higher than " << MD-1
                  << " are not supported!");
      MFEM_VERIFY(Q1D <= MQ, "Quadrature rules with more than "
                  << MQ << " 1D points are not supported!");
      HdivDivergence3D<L>(NE,detJ,Bo,Gc,X,Y,vdim,D1D,Q1D);
   }
   else
   {
      MFEM_ABORT("Dimension not supported");
   }
}

template void HdivTensorValues<QVectorLayout::byNODES>(
   const int NE,
   const int vdim,
   const GeometricFactors *geom,
   const DofToQuad &maps,
   const DofToQuad &maps_o,
   const Vector &e_vec,
   Vector &q_val);

template void HdivTensorValues<QVectorLayout::byVDIM>(
   const int NE,
   const int vdim,
   const GeometricFactors *geom,
   const DofToQuad &maps,
   const DofToQuad &maps_o,
   const Vector &e_vec,
   Vector &q_val);

template void HdivTensorDivergence<QVectorLayout::byNODES>(
   const int NE,
   const int vdim,
   const GeometricFactors *geom,
   const DofToQuad &maps,
   const DofToQuad &maps_o,
   const Vector &e_vec,
   Vector &q_val);

template void HdivTensorDivergence<QVectorLayout::byVDIM>(
   const int NE,
   const int vdim,
   const GeometricFactors *geom,
   const DofToQuad &maps,
   const DofToQuad &maps_o,
   const Vector &e_vec,
   Vector &q_val);

} // namespace quadrature_interpolator

} // namespace internal

} // namespace mfem

#endif
