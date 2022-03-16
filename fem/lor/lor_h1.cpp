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

#include "lor_h1.hpp"
#include "../../linalg/dtensor.hpp"
#include "../../general/forall.hpp"

namespace mfem
{

template <int ORDER>
void BatchedLOR_H1::Assemble2D()
{
   const int nel_ho = fes_ho.GetNE();

   static constexpr int nv = 4;
   static constexpr int dim = 2;
   static constexpr int ddm2 = (dim*(dim+1))/2;
   static constexpr int nd1d = ORDER + 1;
   static constexpr int ndof_per_el = nd1d*nd1d;
   static constexpr int nnz_per_row = 9;
   static constexpr int sz_local_mat = nv*nv;

   const double DQ = diffusion_coeff;
   const double MQ = mass_coeff;

   sparse_ij.SetSize(nnz_per_row*ndof_per_el*nel_ho);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, nd1d, nd1d, nel_ho);

   auto X = X_vert.Read();

   MFEM_FORALL_2D(iel_ho, nel_ho, ORDER, ORDER, 1,
   {
      // Assemble a sparse matrix over the macro-element by looping over each
      // subelement.
      // V(j,ix,iy) stores the jth nonzero in the row of the sparse matrix
      // corresponding to local DOF (ix, iy).
      MFEM_FOREACH_THREAD(iy,y,nd1d)
      {
         MFEM_FOREACH_THREAD(ix,x,nd1d)
         {
            for (int j=0; j<nnz_per_row; ++j)
            {
               V(j,ix,iy,iel_ho) = 0.0;
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Compute geometric factors at quadrature points
      MFEM_FOREACH_THREAD(ky,y,ORDER)
      {
         MFEM_FOREACH_THREAD(kx,x,ORDER)
         {
            double Q_[(ddm2 + 1)*nv];
            double local_mat_[sz_local_mat];

            DeviceTensor<3> Q(Q_, ddm2 + 1, 2, 2);
            DeviceTensor<2> local_mat(local_mat_, nv, nv);

            // local_mat is the local (dense) stiffness matrix
            for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

            const int v0 = kx + nd1d*ky;
            const int v1 = kx + 1 + nd1d*ky;
            const int v2 = kx + 1 + nd1d*(ky + 1);
            const int v3 = kx + nd1d*(ky + 1);

            const int e0 = dim*(v0 + ndof_per_el*iel_ho);
            const int e1 = dim*(v1 + ndof_per_el*iel_ho);
            const int e2 = dim*(v2 + ndof_per_el*iel_ho);
            const int e3 = dim*(v3 + ndof_per_el*iel_ho);

            // Vertex coordinates
            const double v0x = X[e0 + 0];
            const double v0y = X[e0 + 1];

            const double v1x = X[e1 + 0];
            const double v1y = X[e1 + 1];

            const double v2x = X[e2 + 0];
            const double v2y = X[e2 + 1];

            const double v3x = X[e3 + 0];
            const double v3y = X[e3 + 1];

            for (int iqy=0; iqy<2; ++iqy)
            {
               for (int iqx=0; iqx<2; ++iqx)
               {
                  const double x = iqx;
                  const double y = iqy;
                  const double w = 1.0/4.0;

                  const double J11 = -(1-y)*v0x + (1-y)*v1x + y*v2x - y*v3x;
                  const double J12 = -(1-x)*v0x - x*v1x + x*v2x + (1-x)*v3x;

                  const double J21 = -(1-y)*v0y + (1-y)*v1y + y*v2y - y*v3y;
                  const double J22 = -(1-x)*v0y - x*v1y + x*v2y + (1-x)*v3y;

                  const double detJ = J11*J22 - J21*J12;
                  const double w_detJ = w/detJ;

                  Q(0,iqy,iqx) = w_detJ * (J12*J12 + J22*J22); // 1,1
                  Q(1,iqy,iqx) = -w_detJ * (J12*J11 + J22*J21); // 1,2
                  Q(2,iqy,iqx) = w_detJ * (J11*J11 + J21*J21); // 2,2
                  Q(3,iqy,iqx) = w*detJ;
               }
            }
            for (int iqx=0; iqx<2; ++iqx)
            {
               for (int iqy=0; iqy<2; ++iqy)
               {
                  for (int jy=0; jy<2; ++jy)
                  {
                     const double bjy = (jy == iqy) ? 1.0 : 0.0;
                     const double gjy = (jy == 0) ? -1.0 : 1.0;
                     for (int jx=0; jx<2; ++jx)
                     {
                        const double bjx = (jx == iqx) ? 1.0 : 0.0;
                        const double gjx = (jx == 0) ? -1.0 : 1.0;

                        const double djx = gjx*bjy;
                        const double djy = bjx*gjy;

                        int jj_loc = jx + 2*jy;

                        for (int iy=0; iy<2; ++iy)
                        {
                           const double biy = (iy == iqy) ? 1.0 : 0.0;
                           const double giy = (iy == 0) ? -1.0 : 1.0;
                           for (int ix=0; ix<2; ++ix)
                           {
                              const double bix = (ix == iqx) ? 1.0 : 0.0;
                              const double gix = (ix == 0) ? -1.0 : 1.0;

                              const double dix = gix*biy;
                              const double diy = bix*giy;

                              int ii_loc = ix + 2*iy;

                              // Only store the lower-triangular part of
                              // the matrix (by symmetry).
                              if (jj_loc > ii_loc) { continue; }

                              double val = 0.0;
                              val += dix*djx*Q(0,iqy,iqx);
                              val += (dix*djy + diy*djx)*Q(1,iqy,iqx);
                              val += diy*djy*Q(2,iqy,iqx);
                              val *= DQ;

                              val += MQ*bix*biy*bjx*bjy*Q(3,iqy,iqx);

                              local_mat(ii_loc, jj_loc) += val;
                           }
                        }
                     }
                  }
               }
            }
            // Assemble the local matrix into the macro-element sparse matrix
            // in a format similar to coordinate format. The (I,J) arrays
            // are implicit (not stored explicitly).
            for (int ii_loc=0; ii_loc<nv; ++ii_loc)
            {
               const int ix = ii_loc%2;
               const int iy = ii_loc/2;
               for (int jj_loc=0; jj_loc<nv; ++jj_loc)
               {
                  const int jx = jj_loc%2;
                  const int jy = jj_loc/2;
                  const int jj_off = (jx-ix+1) + 3*(jy-iy+1);

                  // Symmetry
                  if (jj_loc <= ii_loc)
                  {
                     AtomicAdd(V(jj_off, ix+kx, iy+ky, iel_ho), local_mat(ii_loc, jj_loc));
                  }
                  else
                  {
                     AtomicAdd(V(jj_off, ix+kx, iy+ky, iel_ho), local_mat(jj_loc, ii_loc));
                  }
               }
            }
         }
      }
   });

   sparse_mapping.SetSize(nnz_per_row, ndof_per_el);
   sparse_mapping = -1;
   for (int iy=0; iy<nd1d; ++iy)
   {
      const int jy_begin = (iy > 0) ? iy - 1 : 0;
      const int jy_end = (iy < ORDER) ? iy + 1 : ORDER;
      for (int ix=0; ix<nd1d; ++ix)
      {
         const int jx_begin = (ix > 0) ? ix - 1 : 0;
         const int jx_end = (ix < ORDER) ? ix + 1 : ORDER;
         const int ii_el = ix + nd1d*iy;
         for (int jy=jy_begin; jy<=jy_end; ++jy)
         {
            for (int jx=jx_begin; jx<=jx_end; ++jx)
            {
               const int jj_off = (jx-ix+1) + 3*(jy-iy+1);
               const int jj_el = jx + nd1d*jy;
               sparse_mapping(jj_off, ii_el) = jj_el;
            }
         }
      }
   }
}

template <int ORDER>
void BatchedLOR_H1::Assemble3D()
{
   const int nel_ho = fes_ho.GetNE();

   const double DQ = diffusion_coeff;
   const double MQ = mass_coeff;

   static constexpr int nv = 8;
   static constexpr int dim = 3;
   static constexpr int ddm2 = (dim*(dim+1))/2;
   static constexpr int nd1d = ORDER + 1;
   static constexpr int ndof_per_el = nd1d*nd1d*nd1d;
   static constexpr int nnz_per_row = 27;
   static constexpr int sz_grad_A = 3*3*2*2*2*2;
   static constexpr int sz_grad_B = sz_grad_A*2;
   static constexpr int sz_mass_A = 2*2*2*2;
   static constexpr int sz_mass_B = sz_mass_A*2;
   static constexpr int sz_local_mat = nv*nv;

   sparse_ij.SetSize(nel_ho*ndof_per_el*nnz_per_row);
   auto V = Reshape(sparse_ij.Write(), nnz_per_row, nd1d, nd1d, nd1d, nel_ho);

   auto X = X_vert.Read();

   // Last thread dimension is lowered to avoid "too many resources" error
   MFEM_FORALL_3D(iel_ho, nel_ho, ORDER, ORDER, (ORDER>6)?4:ORDER,
   {
      // Assemble a sparse matrix over the macro-element by looping over each
      // subelement.
      // V(j,i) stores the jth nonzero in the ith row of the sparse matrix.
      MFEM_FOREACH_THREAD(iz,z,nd1d)
      {
         MFEM_FOREACH_THREAD(iy,y,nd1d)
         {
            MFEM_FOREACH_THREAD(ix,x,nd1d)
            {
               MFEM_UNROLL(nnz_per_row)
               for (int j=0; j<nnz_per_row; ++j)
               {
                  V(j,ix,iy,iz,iel_ho) = 0.0;
               }
            }
         }
      }
      MFEM_SYNC_THREAD;

      // Compute geometric factors at quadrature points
      MFEM_FOREACH_THREAD(kz,z,ORDER)
      {
         MFEM_FOREACH_THREAD(ky,y,ORDER)
         {
            MFEM_FOREACH_THREAD(kx,x,ORDER)
            {
               double Q_[(ddm2 + 1)*nv];
               double grad_A_[sz_grad_A];
               double grad_B_[sz_grad_B];
               double mass_A_[sz_mass_A];
               double mass_B_[sz_mass_B];
               double local_mat_[sz_local_mat];

               DeviceTensor<4> Q(Q_, ddm2 + 1, 2, 2, 2);
               DeviceTensor<2> local_mat(local_mat_, nv, nv);
               DeviceTensor<6> grad_A(grad_A_, 3, 3, 2, 2, 2, 2);
               DeviceTensor<7> grad_B(grad_B_, 3, 3, 2, 2, 2, 2, 2);
               DeviceTensor<4> mass_A(mass_A_, 2, 2, 2, 2);
               DeviceTensor<5> mass_B(mass_B_, 2, 2, 2, 2, 2);

               // local_mat is the local (dense) stiffness matrix
               for (int i=0; i<sz_local_mat; ++i) { local_mat[i] = 0.0; }

               // Intermediate quantities
               // (see e.g. Mora and Demkowicz for notation).
               for (int i=0; i<sz_grad_A; ++i) { grad_A[i] = 0.0; }
               for (int i=0; i<sz_grad_B; ++i) { grad_B[i] = 0.0; }

               for (int i=0; i<sz_mass_A; ++i) { mass_A[i] = 0.0; }
               for (int i=0; i<sz_mass_B; ++i) { mass_B[i] = 0.0; }

               const int v0 = kx + nd1d*(ky + nd1d*kz);
               const int v1 = kx + 1 + nd1d*(ky + nd1d*kz);
               const int v2 = kx + 1 + nd1d*(ky + 1 + nd1d*kz);
               const int v3 = kx + nd1d*(ky + 1 + nd1d*kz);
               const int v4 = kx + nd1d*(ky + nd1d*(kz + 1));
               const int v5 = kx + 1 + nd1d*(ky + nd1d*(kz + 1));
               const int v6 = kx + 1 + nd1d*(ky + 1 + nd1d*(kz + 1));
               const int v7 = kx + nd1d*(ky + 1 + nd1d*(kz + 1));

               const int e0 = dim*(v0 + ndof_per_el*iel_ho);
               const int e1 = dim*(v1 + ndof_per_el*iel_ho);
               const int e2 = dim*(v2 + ndof_per_el*iel_ho);
               const int e3 = dim*(v3 + ndof_per_el*iel_ho);
               const int e4 = dim*(v4 + ndof_per_el*iel_ho);
               const int e5 = dim*(v5 + ndof_per_el*iel_ho);
               const int e6 = dim*(v6 + ndof_per_el*iel_ho);
               const int e7 = dim*(v7 + ndof_per_el*iel_ho);

               const double v0x = X[e0 + 0];
               const double v0y = X[e0 + 1];
               const double v0z = X[e0 + 2];

               const double v1x = X[e1 + 0];
               const double v1y = X[e1 + 1];
               const double v1z = X[e1 + 2];

               const double v2x = X[e2 + 0];
               const double v2y = X[e2 + 1];
               const double v2z = X[e2 + 2];

               const double v3x = X[e3 + 0];
               const double v3y = X[e3 + 1];
               const double v3z = X[e3 + 2];

               const double v4x = X[e4 + 0];
               const double v4y = X[e4 + 1];
               const double v4z = X[e4 + 2];

               const double v5x = X[e5 + 0];
               const double v5y = X[e5 + 1];
               const double v5z = X[e5 + 2];

               const double v6x = X[e6 + 0];
               const double v6y = X[e6 + 1];
               const double v6z = X[e6 + 2];

               const double v7x = X[e7 + 0];
               const double v7y = X[e7 + 1];
               const double v7z = X[e7 + 2];

               //MFEM_UNROLL(2)
               for (int iqz=0; iqz<2; ++iqz)
               {
                  //MFEM_UNROLL(2)
                  for (int iqy=0; iqy<2; ++iqy)
                  {
                     //MFEM_UNROLL(2)
                     for (int iqx=0; iqx<2; ++iqx)
                     {

                        const double x = iqx;
                        const double y = iqy;
                        const double z = iqz;
                        const double w = 1.0/8.0;

                        // c: (1-x)(1-y)(1-z)v0[c] + x (1-y)(1-z)v1[c] + x y (1-z)v2[c] + (1-x) y (1-z)v3[c]
                        //  + (1-x)(1-y) z   v4[c] + x (1-y) z   v5[c] + x y z    v6[c] + (1-x) y z    v7[c]
                        const double J11 = -(1-y)*(1-z)*v0x
                                           + (1-y)*(1-z)*v1x + y*(1-z)*v2x - y*(1-z)*v3x
                                           - (1-y)*z*v4x + (1-y)*z*v5x + y*z*v6x - y*z*v7x;

                        const double J12 = -(1-x)*(1-z)*v0x
                                           - x*(1-z)*v1x + x*(1-z)*v2x + (1-x)*(1-z)*v3x
                                           - (1-x)*z*v4x - x*z*v5x + x*z*v6x + (1-x)*z*v7x;

                        const double J13 = -(1-x)*(1-y)*v0x - x*(1-y)*v1x
                                           - x*y*v2x - (1-x)*y*v3x + (1-x)*(1-y)*v4x
                                           + x*(1-y)*v5x + x*y*v6x + (1-x)*y*v7x;

                        const double J21 = -(1-y)*(1-z)*v0y + (1-y)*(1-z)*v1y
                                           + y*(1-z)*v2y - y*(1-z)*v3y - (1-y)*z*v4y
                                           + (1-y)*z*v5y + y*z*v6y - y*z*v7y;

                        const double J22 = -(1-x)*(1-z)*v0y - x*(1-z)*v1y
                                           + x*(1-z)*v2y + (1-x)*(1-z)*v3y- (1-x)*z*v4y -
                                           x*z*v5y + x*z*v6y + (1-x)*z*v7y;

                        const double J23 = -(1-x)*(1-y)*v0y - x*(1-y)*v1y
                                           - x*y*v2y - (1-x)*y*v3y + (1-x)*(1-y)*v4y
                                           + x*(1-y)*v5y + x*y*v6y + (1-x)*y*v7y;

                        const double J31 = -(1-y)*(1-z)*v0z + (1-y)*(1-z)*v1z
                                           + y*(1-z)*v2z - y*(1-z)*v3z- (1-y)*z*v4z +
                                           (1-y)*z*v5z + y*z*v6z - y*z*v7z;

                        const double J32 = -(1-x)*(1-z)*v0z - x*(1-z)*v1z
                                           + x*(1-z)*v2z + (1-x)*(1-z)*v3z - (1-x)*z*v4z
                                           - x*z*v5z + x*z*v6z + (1-x)*z*v7z;

                        const double J33 = -(1-x)*(1-y)*v0z - x*(1-y)*v1z
                                           - x*y*v2z - (1-x)*y*v3z + (1-x)*(1-y)*v4z
                                           + x*(1-y)*v5z + x*y*v6z + (1-x)*y*v7z;

                        const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                            J21 * (J12 * J33 - J32 * J13) +
                                            J31 * (J12 * J23 - J22 * J13);
                        const double w_detJ = w/detJ;

                        // adj(J)
                        const double A11 = (J22 * J33) - (J23 * J32);
                        const double A12 = (J32 * J13) - (J12 * J33);
                        const double A13 = (J12 * J23) - (J22 * J13);
                        const double A21 = (J31 * J23) - (J21 * J33);
                        const double A22 = (J11 * J33) - (J13 * J31);
                        const double A23 = (J21 * J13) - (J11 * J23);
                        const double A31 = (J21 * J32) - (J31 * J22);
                        const double A32 = (J31 * J12) - (J11 * J32);
                        const double A33 = (J11 * J22) - (J12 * J21);

                        Q(0,iqz,iqy,iqx) = w_detJ * (A11*A11 + A12*A12 + A13*A13); // 1,1
                        Q(1,iqz,iqy,iqx) = w_detJ * (A11*A21 + A12*A22 + A13*A23); // 2,1
                        Q(2,iqz,iqy,iqx) = w_detJ * (A11*A31 + A12*A32 + A13*A33); // 3,1
                        Q(3,iqz,iqy,iqx) = w_detJ * (A21*A21 + A22*A22 + A23*A23); // 2,2
                        Q(4,iqz,iqy,iqx) = w_detJ * (A21*A31 + A22*A32 + A23*A33); // 3,2
                        Q(5,iqz,iqy,iqx) = w_detJ * (A31*A31 + A32*A32 + A33*A33); // 3,3
                        Q(6,iqz,iqy,iqx) = w*detJ;
                     }
                  }
               }

               //MFEM_UNROLL(2)
               for (int iqx=0; iqx<2; ++iqx)
               {
                  //MFEM_UNROLL(2)
                  for (int jz=0; jz<2; ++jz)
                  {
                     // Note loop starts at iz=jz here, taking advantage of
                     // symmetries.
                     //MFEM_UNROLL(2)
                     for (int iz=jz; iz<2; ++iz)
                     {
                        //MFEM_UNROLL(2)
                        for (int iqy=0; iqy<2; ++iqy)
                        {
                           //MFEM_UNROLL(2)
                           for (int iqz=0; iqz<2; ++iqz)
                           {
                              const double biz = (iz == iqz) ? 1.0 : 0.0;
                              const double giz = (iz == 0) ? -1.0 : 1.0;

                              const double bjz = (jz == iqz) ? 1.0 : 0.0;
                              const double gjz = (jz == 0) ? -1.0 : 1.0;

                              const double J11 = Q(0,iqz,iqy,iqx);
                              const double J21 = Q(1,iqz,iqy,iqx);
                              const double J31 = Q(2,iqz,iqy,iqx);
                              const double J12 = J21;
                              const double J22 = Q(3,iqz,iqy,iqx);
                              const double J32 = Q(4,iqz,iqy,iqx);
                              const double J13 = J31;
                              const double J23 = J32;
                              const double J33 = Q(5,iqz,iqy,iqx);

                              grad_A(0,0,iqy,iz,jz,iqx) += J11*biz*bjz;
                              grad_A(1,0,iqy,iz,jz,iqx) += J21*biz*bjz;
                              grad_A(2,0,iqy,iz,jz,iqx) += J31*giz*bjz;
                              grad_A(0,1,iqy,iz,jz,iqx) += J12*biz*bjz;
                              grad_A(1,1,iqy,iz,jz,iqx) += J22*biz*bjz;
                              grad_A(2,1,iqy,iz,jz,iqx) += J32*giz*bjz;
                              grad_A(0,2,iqy,iz,jz,iqx) += J13*biz*gjz;
                              grad_A(1,2,iqy,iz,jz,iqx) += J23*biz*gjz;
                              grad_A(2,2,iqy,iz,jz,iqx) += J33*giz*gjz;

                              double wdetJ = Q(6,iqz,iqy,iqx);
                              mass_A(iqy,iz,jz,iqx) += wdetJ*biz*bjz;
                           }
                           //MFEM_UNROLL(2)
                           for (int jy=0; jy<2; ++jy)
                           {
                              //MFEM_UNROLL(2)
                              for (int iy=0; iy<2; ++iy)
                              {
                                 const double biy = (iy == iqy) ? 1.0 : 0.0;
                                 const double giy = (iy == 0) ? -1.0 : 1.0;

                                 const double bjy = (jy == iqy) ? 1.0 : 0.0;
                                 const double gjy = (jy == 0) ? -1.0 : 1.0;

                                 grad_B(0,0,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(0,0,iqy,iz,jz,iqx);
                                 grad_B(1,0,iy,jy,iz,jz,iqx) += giy*bjy*grad_A(1,0,iqy,iz,jz,iqx);
                                 grad_B(2,0,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(2,0,iqy,iz,jz,iqx);
                                 grad_B(0,1,iy,jy,iz,jz,iqx) += biy*gjy*grad_A(0,1,iqy,iz,jz,iqx);
                                 grad_B(1,1,iy,jy,iz,jz,iqx) += giy*gjy*grad_A(1,1,iqy,iz,jz,iqx);
                                 grad_B(2,1,iy,jy,iz,jz,iqx) += biy*gjy*grad_A(2,1,iqy,iz,jz,iqx);
                                 grad_B(0,2,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(0,2,iqy,iz,jz,iqx);
                                 grad_B(1,2,iy,jy,iz,jz,iqx) += giy*bjy*grad_A(1,2,iqy,iz,jz,iqx);
                                 grad_B(2,2,iy,jy,iz,jz,iqx) += biy*bjy*grad_A(2,2,iqy,iz,jz,iqx);

                                 mass_B(iy,jy,iz,jz,iqx) += biy*bjy*mass_A(iqy,iz,jz,iqx);
                              }
                           }
                        }
                        //MFEM_UNROLL(2)
                        for (int jy=0; jy<2; ++jy)
                        {
                           //MFEM_UNROLL(2)
                           for (int jx=0; jx<2; ++jx)
                           {
                              //MFEM_UNROLL(2)
                              for (int iy=0; iy<2; ++iy)
                              {
                                 //MFEM_UNROLL(2)
                                 for (int ix=0; ix<2; ++ix)
                                 {
                                    const double bix = (ix == iqx) ? 1.0 : 0.0;
                                    const double gix = (ix == 0) ? -1.0 : 1.0;

                                    const double bjx = (jx == iqx) ? 1.0 : 0.0;
                                    const double gjx = (jx == 0) ? -1.0 : 1.0;

                                    int ii_loc = ix + 2*iy + 4*iz;
                                    int jj_loc = jx + 2*jy + 4*jz;

                                    // Only store the lower-triangular part of
                                    // the matrix (by symmetry).
                                    if (jj_loc > ii_loc) { continue; }

                                    double val = 0.0;
                                    val += gix*gjx*grad_B(0,0,iy,jy,iz,jz,iqx);
                                    val += bix*gjx*grad_B(1,0,iy,jy,iz,jz,iqx);
                                    val += bix*gjx*grad_B(2,0,iy,jy,iz,jz,iqx);
                                    val += gix*bjx*grad_B(0,1,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(1,1,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(2,1,iy,jy,iz,jz,iqx);
                                    val += gix*bjx*grad_B(0,2,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(2,2,iy,jy,iz,jz,iqx);
                                    val += bix*bjx*grad_B(1,2,iy,jy,iz,jz,iqx);

                                    val *= DQ;

                                    val += MQ*bix*bjx*mass_B(iy,jy,iz,jz,iqx);

                                    local_mat(ii_loc, jj_loc) += val;
                                 }
                              }
                           }
                        }
                     }
                  }
               }
               // Assemble the local matrix into the macro-element sparse matrix
               // in a format similar to coordinate format. The (I,J) arrays
               // are implicit (not stored explicitly).
               //MFEM_UNROLL(8)
               for (int ii_loc=0; ii_loc<nv; ++ii_loc)
               {
                  const int ix = ii_loc%2;
                  const int iy = (ii_loc/2)%2;
                  const int iz = ii_loc/2/2;

                  for (int jj_loc=0; jj_loc<nv; ++jj_loc)
                  {
                     const int jx = jj_loc%2;
                     const int jy = (jj_loc/2)%2;
                     const int jz = jj_loc/2/2;
                     const int jj_off = (jx-ix+1) + 3*(jy-iy+1) + 9*(jz-iz+1);

                     if (jj_loc <= ii_loc)
                     {
                        AtomicAdd(V(jj_off, ix+kx, iy+ky, iz+kz, iel_ho), local_mat(ii_loc, jj_loc));
                     }
                     else
                     {
                        AtomicAdd(V(jj_off, ix+kx, iy+ky, iz+kz, iel_ho), local_mat(jj_loc, ii_loc));
                     }
                  }
               }
            }
         }
      }
   });

   sparse_mapping.SetSize(nnz_per_row, ndof_per_el);
   sparse_mapping = -1;
   for (int iz=0; iz<nd1d; ++iz)
   {
      const int jz_begin = (iz > 0) ? iz - 1 : 0;
      const int jz_end = (iz < ORDER) ? iz + 1 : ORDER;
      for (int iy=0; iy<nd1d; ++iy)
      {
         const int jy_begin = (iy > 0) ? iy - 1 : 0;
         const int jy_end = (iy < ORDER) ? iy + 1 : ORDER;
         for (int ix=0; ix<nd1d; ++ix)
         {
            const int jx_begin = (ix > 0) ? ix - 1 : 0;
            const int jx_end = (ix < ORDER) ? ix + 1 : ORDER;

            const int ii_el = ix + nd1d*(iy + nd1d*iz);

            for (int jz=jz_begin; jz<=jz_end; ++jz)
            {
               for (int jy=jy_begin; jy<=jy_end; ++jy)
               {
                  for (int jx=jx_begin; jx<=jx_end; ++jx)
                  {
                     const int jj_off = (jx-ix+1) + 3*(jy-iy+1) + 9*(jz-iz+1);
                     const int jj_el = jx + nd1d*(jy + nd1d*jz);
                     sparse_mapping(jj_off, ii_el) = jj_el;
                  }
               }
            }
         }
      }
   }
}

void BatchedLOR_H1::AssemblyKernel()
{
   const int dim = fes_ho.GetMesh()->Dimension();
   const int order = fes_ho.GetMaxElementOrder();

   if (dim == 2)
   {
      switch (order)
      {
         case 1: Assemble2D<1>(); break;
         case 2: Assemble2D<2>(); break;
         case 3: Assemble2D<3>(); break;
         case 4: Assemble2D<4>(); break;
         case 5: Assemble2D<5>(); break;
         case 6: Assemble2D<6>(); break;
         case 7: Assemble2D<7>(); break;
         case 8: Assemble2D<8>(); break;
         default: MFEM_ABORT("No kernel order " << order << "!");
      }
   }
   else if (dim == 3)
   {
      switch (order)
      {
         case 1: Assemble3D<1>(); break;
         case 2: Assemble3D<2>(); break;
         case 3: Assemble3D<3>(); break;
         case 4: Assemble3D<4>(); break;
         case 5: Assemble3D<5>(); break;
         case 6: Assemble3D<6>(); break;
         case 7: Assemble3D<7>(); break;
         case 8: Assemble3D<8>(); break;
         default: MFEM_ABORT("No kernel order " << order << "!");
      }
   }
}

// This is a static member function, called by the virtual member
// FormIsSupported.
bool BatchedLOR_H1::FormIsSupported_(BilinearForm &a)
{
   const FiniteElementCollection *fec = a.FESpace()->FEColl();
   if (dynamic_cast<const H1_FECollection*>(fec))
   {
      if (HasIntegrators<DiffusionIntegrator, MassIntegrator>(a)) { return true; }
   }
   return false;
}

bool BatchedLOR_H1::FormIsSupported(BilinearForm &a)
{
   return FormIsSupported_(a);
}

void BatchedLOR_H1::SetForm(BilinearForm &a)
{
   MassIntegrator *mass = GetIntegrator<MassIntegrator>(a);
   DiffusionIntegrator *diffusion = GetIntegrator<DiffusionIntegrator>(a);

   if (mass != nullptr)
   {
      auto *coeff = dynamic_cast<const ConstantCoefficient*>(mass->GetCoefficient());
      mass_coeff = coeff ? coeff->constant : 1.0;
   }
   else
   {
      mass_coeff = 0.0;
   }

   if (diffusion != nullptr)
   {
      auto *coeff = dynamic_cast<const ConstantCoefficient*>
                    (diffusion->GetCoefficient());
      diffusion_coeff = coeff ? coeff->constant : 1.0;
   }
   else
   {
      diffusion_coeff = 0.0;
   }
}

BatchedLOR_H1::BatchedLOR_H1(FiniteElementSpace &fes_ho_)
   : BatchedLORAssembly(fes_ho_)
{ }

} // namespace mfem
