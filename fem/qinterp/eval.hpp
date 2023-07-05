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

#ifndef MFEM_EVAL_HPP
#define MFEM_EVAL_HPP

#include "../quadinterpolator.hpp"
#include "../../general/forall.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../linalg/kernels.hpp"
#include "../kernels.hpp"

namespace mfem
{

namespace internal
{

namespace quadrature_interpolator
{

template<QVectorLayout Q_LAYOUT>
static void Values1D(const int NE,
                     const double *b_,
                     const double *x_,
                     double *y_,
                     const int vdim,
                     const int d1d,
                     const int q1d)
{
   const auto b = Reshape(b_, q1d, d1d);
   const auto x = Reshape(x_, d1d, vdim, NE);
   auto y = Q_LAYOUT == QVectorLayout::byNODES ?
            Reshape(y_, q1d, vdim, NE):
            Reshape(y_, vdim, q1d, NE);

   mfem::forall(NE, [=] MFEM_HOST_DEVICE (int e)
   {
      for (int c = 0; c < vdim; c++)
      {
         for (int q = 0; q < q1d; q++)
         {
            double u = 0.0;
            for (int d = 0; d < d1d; d++)
            {
               u += b(q, d) * x(d, c, e);
            }
            if (Q_LAYOUT == QVectorLayout::byVDIM)  { y(c, q, e) = u; }
            if (Q_LAYOUT == QVectorLayout::byNODES) { y(q, c, e) = u; }
         }
      }
   });
}

// Template compute kernel for Values in 2D: tensor product version.
template<QVectorLayout Q_LAYOUT,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int T_NBZ = 1, int MAX_D1D = 0, int MAX_Q1D = 0>
static void Values2D(const int NE,
                     const double *b_,
                     const double *x_,
                     double *y_,
                     const int vdim = 0,
                     const int d1d = 0,
                     const int q1d = 0)
{
   static constexpr int NBZ = T_NBZ ? T_NBZ : 1;

   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, VDIM, NE);
   auto y = Q_LAYOUT == QVectorLayout::byNODES ?
            Reshape(y_, Q1D, Q1D, VDIM, NE):
            Reshape(y_, VDIM, Q1D, Q1D, NE);

   mfem::forall_2D_batch(NE, Q1D, Q1D, NBZ, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      const int tidz = MFEM_THREAD_ID(z);

      MFEM_SHARED double sB[MQ1*MD1];
      MFEM_SHARED double sm0[NBZ][MDQ*MDQ];
      MFEM_SHARED double sm1[NBZ][MDQ*MDQ];

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,sB);

      ConstDeviceMatrix B(sB, D1D,Q1D);
      DeviceMatrix DD(sm0[tidz], MD1, MD1);
      DeviceMatrix DQ(sm1[tidz], MD1, MQ1);
      DeviceMatrix QQ(sm0[tidz], MQ1, MQ1);

      for (int c = 0; c < VDIM; c++)
      {
         kernels::internal::LoadX(e,D1D,c,x,DD);
         kernels::internal::EvalX(D1D,Q1D,B,DD,DQ);
         kernels::internal::EvalY(D1D,Q1D,B,DQ,QQ);
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               double u = QQ(qx,qy);
               if (Q_LAYOUT == QVectorLayout::byVDIM) { y(c,qx,qy,e) = u; }
               if (Q_LAYOUT == QVectorLayout::byNODES) { y(qx,qy,c,e) = u; }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Template compute kernel for Values in 3D: tensor product version.
template<QVectorLayout Q_LAYOUT,
         int T_VDIM = 0, int T_D1D = 0, int T_Q1D = 0,
         int MAX_D1D = 0, int MAX_Q1D = 0>
static void Values3D(const int NE,
                     const double *b_,
                     const double *x_,
                     double *y_,
                     const int vdim = 0,
                     const int d1d = 0,
                     const int q1d = 0)
{
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int VDIM = T_VDIM ? T_VDIM : vdim;

   const auto b = Reshape(b_, Q1D, D1D);
   const auto x = Reshape(x_, D1D, D1D, D1D, VDIM, NE);
   auto y = Q_LAYOUT == QVectorLayout:: byNODES ?
            Reshape(y_, Q1D, Q1D, Q1D, VDIM, NE):
            Reshape(y_, VDIM, Q1D, Q1D, Q1D, NE);

   mfem::forall_3D(NE, Q1D, Q1D, Q1D, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int VDIM = T_VDIM ? T_VDIM : vdim;
      constexpr int MQ1 = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED double sB[MQ1*MD1];
      MFEM_SHARED double sm0[MDQ*MDQ*MDQ];
      MFEM_SHARED double sm1[MDQ*MDQ*MDQ];

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,b,sB);

      ConstDeviceMatrix B(sB, D1D,Q1D);
      DeviceCube DDD(sm0, MD1,MD1,MD1);
      DeviceCube DDQ(sm1, MD1,MD1,MQ1);
      DeviceCube DQQ(sm0, MD1,MQ1,MQ1);
      DeviceCube QQQ(sm1, MQ1,MQ1,MQ1);

      for (int c = 0; c < VDIM; c++)
      {
         kernels::internal::LoadX(e,D1D,c,x,DDD);
         kernels::internal::EvalX(D1D,Q1D,B,DDD,DDQ);
         kernels::internal::EvalY(D1D,Q1D,B,DDQ,DQQ);
         kernels::internal::EvalZ(D1D,Q1D,B,DQQ,QQQ);
         MFEM_FOREACH_THREAD(qz,z,Q1D)
         {
            MFEM_FOREACH_THREAD(qy,y,Q1D)
            {
               MFEM_FOREACH_THREAD(qx,x,Q1D)
               {
                  const double u = QQQ(qz,qy,qx);
                  if (Q_LAYOUT == QVectorLayout::byVDIM) { y(c,qx,qy,qz,e) = u; }
                  if (Q_LAYOUT == QVectorLayout::byNODES) { y(qx,qy,qz,c,e) = u; }
               }
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

// Template compute kernel for Values in 2D: tensor product version.
template<QVectorLayout Q_LAYOUT, int MAX_D1D, int MAX_Q1D>
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
      constexpr int MQ1 = MAX_Q1D;
      constexpr int MD1 = MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED double sBc[MQ1*MD1];
      MFEM_SHARED double sBo[MQ1*MD1];
      MFEM_SHARED double sm0[MDQ*MDQ];
      MFEM_SHARED double sm1[MDQ*MDQ];
      MFEM_SHARED double sm2[2*MDQ*MDQ];

      kernels::internal::LoadB<MD1,MQ1>(D1D,Q1D,bc,sBc);
      kernels::internal::LoadB<MD1,MQ1>(D1D-1,Q1D,bo,sBo);

      ConstDeviceMatrix Bc(sBc, D1D,Q1D);
      ConstDeviceMatrix Bo(sBo, D1D-1,Q1D);

      DeviceMatrix DD(sm0, MD1, MD1);
      DeviceMatrix DQ(sm1, MD1, MQ1);
      DeviceCube QQ(sm2, MQ1, MQ1, 2);

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

// Template compute kernel for Values in 3D: tensor product version.
template<QVectorLayout Q_LAYOUT, int MAX_D1D, int MAX_Q1D>
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
      for (int c = 0; c < VDIM; c++)
      {
         double QQQ_[MAX_Q1D*MAX_Q1D*MAX_Q1D*3];
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

            double DQQ[MAX_D1D*MAX_Q1D*MAX_Q1D];

            for (int dz = 0; dz < d1d_z; ++dz)
            {
               double DDQ[MAX_D1D*MAX_D1D*MAX_Q1D];

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

} // namespace quadrature_interpolator

} // namespace internal

} // namespace mfem

#endif
