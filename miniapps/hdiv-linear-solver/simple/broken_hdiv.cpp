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

#include "broken_hdiv.hpp"
#include "fast_diag.hpp"
#include <cmath>

namespace mfem
{

// The element vector of the lexicographic broken space stores component c
// (contiguously, at offset c*NC) as a tensor that is closed (D1D = p+1 points)
// in direction c and open (P = p points) in the other directions. The
// divergence d = D_hat x lives on the open tensor basis (P^dim coefficients):
// in direction c, it is the difference of the neighboring closed coefficients
// of component c. The kernels treat d as one more field, interpolated with the
// open basis in every direction, and add D_hat^T of its transposed
// interpolation to the output.

/// Is the DOF with closed index k (of P+1) along its component's direction
/// constrained, given the element's face masks for that direction?
MFEM_HOST_DEVICE inline bool IsConstrained(const int *mask, int f, int k,
                                           int P)
{
   if (!mask) { return false; }
   const int m = mask[f];
   return (k == 0 && (m & 1)) || (k == P && (m & 2));
}

/// Quadrature data in 2D: w/det J times (J^T alpha J)_{00}, (J^T alpha J)_{01},
/// (J^T alpha J)_{11}, and beta. alpha is scalar.
static void SetupQuadratureData2D(const int NE, const int NQ,
                                  const Array<real_t> &w, const Vector &j,
                                  const Vector &alpha,
                                  const Vector &beta, Vector &qd)
{
   const auto W = w.Read();
   const auto J = Reshape(j.Read(), NQ, 2, 2, NE);
   const auto A = Reshape(alpha.Read(), NQ, NE);
   const auto B = Reshape(beta.Read(), NQ, NE);
   auto D = Reshape(qd.Write(), NQ, 4, NE);
   mfem::forall(NE*NQ, [=] MFEM_HOST_DEVICE (int i)
   {
      const int q = i%NQ, e = i/NQ;
      const real_t J11 = J(q,0,0,e), J21 = J(q,1,0,e);
      const real_t J12 = J(q,0,1,e), J22 = J(q,1,1,e);
      const real_t c = W[q]/(J11*J22 - J21*J12);
      const real_t a = A(q,e);
      D(q,0,e) = c*(a*J11*J11 + a*J21*J21);
      D(q,1,e) = c*(a*J11*J12 + a*J21*J22);
      D(q,2,e) = c*(a*J12*J12 + a*J22*J22);
      D(q,3,e) = c*B(q,e);
   });
}

/// Quadrature data in 3D: w/det J times the upper triangle of J^T alpha J
/// (row-major), and beta. alpha is scalar.
static void SetupQuadratureData3D(const int NE, const int NQ,
                                  const Array<real_t> &w, const Vector &j,
                                  const Vector &alpha,
                                  const Vector &beta, Vector &qd)
{
   const auto W = w.Read();
   const auto J = Reshape(j.Read(), NQ, 3, 3, NE);
   const auto A = Reshape(alpha.Read(), NQ, NE);
   const auto B = Reshape(beta.Read(), NQ, NE);
   auto D = Reshape(qd.Write(), NQ, 7, NE);
   mfem::forall(NE*NQ, [=] MFEM_HOST_DEVICE (int i)
   {
      const int q = i%NQ, e = i/NQ;
      const real_t J11 = J(q,0,0,e), J21 = J(q,1,0,e), J31 = J(q,2,0,e);
      const real_t J12 = J(q,0,1,e), J22 = J(q,1,1,e), J32 = J(q,2,1,e);
      const real_t J13 = J(q,0,2,e), J23 = J(q,1,2,e), J33 = J(q,2,2,e);
      const real_t detJ = J11*(J22*J33 - J32*J23) - J21*(J12*J33 - J32*J13) +
                          J31*(J12*J23 - J22*J13);
      const real_t c = W[q]/detJ;
      int idx = 0;
      for (int a = 0; a < 3; ++a)
      {
         for (int b = a; b < 3; ++b)
         {
            real_t s = 0.0;
            for (int k = 0; k < 3; ++k)
            {
               s += A(q,e)*J(q,k,a,e)*J(q,k,b,e);
            }
            D(q,idx++,e) = c*s;
         }
      }
      D(q,6,e) = c*B(q,e);
   });
}

template <int T_D1D = 0, int T_Q1D = 0>
static void BrokenHdivApply2D(const int NE, const Array<real_t> &Bo_,
                              const Array<real_t> &Bc_, const Vector &qd_,
                              const Vector &x_, Vector &y_,
                              const int *masks,
                              const int d1d = 0, const int q1d = 0)
{
   static constexpr int NF = 3; // the two components and the divergence
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int ND = 2*D1D*(D1D-1);
   const auto bo = Reshape(Bo_.Read(), Q1D, D1D-1);
   const auto bc = Reshape(Bc_.Read(), Q1D, D1D);
   const auto QD = Reshape(qd_.Read(), Q1D, Q1D, 4, NE);
   const auto x = Reshape(x_.Read(), ND, NE);
   auto y = Reshape(y_.Write(), ND, NE);

   mfem::forall_3D(NE, Q1D, Q1D, NF, [=] MFEM_HOST_DEVICE (int e)
   {
      const int tidz = MFEM_THREAD_ID(z);
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int P = D1D - 1;
      const int NC = D1D*P; // DOFs per component
      const int *mask = masks ? masks + 2*e : nullptr;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      MFEM_SHARED real_t smo[MQ1*(MD1-1)];
      MFEM_SHARED real_t smc[MQ1*MD1];
      DeviceMatrix Bo(smo, D1D-1, Q1D);
      DeviceMatrix Bc(smc, D1D, Q1D);

      MFEM_SHARED real_t sm0[NF*MDQ*MDQ];
      MFEM_SHARED real_t sm1[NF*MDQ*MDQ];
      // Fields f at offsets f*NC, each lexicographic with its own sizes.
      real_t *X = sm0;
      DeviceTensor<3> QD_(sm1, Q1D, D1D, NF);
      DeviceTensor<3> QQ(sm0, Q1D, Q1D, NF);
      DeviceTensor<3> DQ(sm1, D1D, Q1D, NF);
      real_t *Y = sm0;

      MFEM_FOREACH_THREAD(f,z,2)
      {
         const int nx = f == 0 ? D1D : P, ny = f == 1 ? D1D : P;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               const int i = f*NC + dx + nx*dy;
               const int k = f == 0 ? dx : dy; // closed index
               X[i] = IsConstrained(mask, f, k, P) ? 0.0 : x(i,e);
            }
         }
      }
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(d,y,D1D-1)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D) { Bo(d,q) = bo(q,d); }
         }
         MFEM_FOREACH_THREAD(d,y,D1D)
         {
            MFEM_FOREACH_THREAD(q,x,Q1D) { Bc(d,q) = bc(q,d); }
         }
      }
      MFEM_SYNC_THREAD;
      // Divergence coefficients d = D_hat x.
      if (tidz == 0)
      {
         const real_t *X0 = X, *X1 = X + NC;
         MFEM_FOREACH_THREAD(dy,y,P)
         {
            MFEM_FOREACH_THREAD(dx,x,P)
            {
               X[2*NC + dx + P*dy] = X0[dx+1 + D1D*dy] - X0[dx + D1D*dy] +
                                     X1[dx + P*(dy+1)] - X1[dx + P*dy];
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Interpolate to quadrature points.
      MFEM_FOREACH_THREAD(f,z,NF)
      {
         const int nx = f == 0 ? D1D : P, ny = f == 1 ? D1D : P;
         const real_t *Xf = X + f*NC;
         const DeviceMatrix B = f == 0 ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u = 0.0;
               MFEM_UNROLL(MD1)
               for (int dx = 0; dx < nx; ++dx) { u += Xf[dx + nx*dy]*B(dx,qx); }
               QD_(qx,dy,f) = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(f,z,NF)
      {
         const int ny = f == 1 ? D1D : P;
         const DeviceMatrix B = f == 1 ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               real_t u = 0.0;
               MFEM_UNROLL(MD1)
               for (int dy = 0; dy < ny; ++dy) { u += QD_(qx,dy,f)*B(dy,qy); }
               QQ(qx,qy,f) = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Apply the quadrature data.
      if (tidz == 0)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(qx,x,Q1D)
            {
               const real_t u0 = QQ(qx,qy,0), u1 = QQ(qx,qy,1);
               const real_t D00 = QD(qx,qy,0,e), D01 = QD(qx,qy,1,e);
               const real_t D11 = QD(qx,qy,2,e);
               QQ(qx,qy,0) = D00*u0 + D01*u1;
               QQ(qx,qy,1) = D01*u0 + D11*u1;
               QQ(qx,qy,2) *= QD(qx,qy,3,e);
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Integrate against the test functions.
      MFEM_FOREACH_THREAD(f,z,NF)
      {
         const int nx = f == 0 ? D1D : P;
         const DeviceMatrix B = f == 0 ? Bc : Bo;
         MFEM_FOREACH_THREAD(qy,y,Q1D)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u = 0.0;
               MFEM_UNROLL(MQ1)
               for (int qx = 0; qx < Q1D; ++qx) { u += QQ(qx,qy,f)*B(dx,qx); }
               DQ(dx,qy,f) = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      MFEM_FOREACH_THREAD(f,z,NF)
      {
         const int nx = f == 0 ? D1D : P, ny = f == 1 ? D1D : P;
         const DeviceMatrix B = f == 1 ? Bc : Bo;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               real_t u = 0.0;
               MFEM_UNROLL(MQ1)
               for (int qy = 0; qy < Q1D; ++qy) { u += DQ(dx,qy,f)*B(dy,qy); }
               Y[f*NC + dx + nx*dy] = u;
            }
         }
      }
      MFEM_SYNC_THREAD;
      // Add D_hat^T of the divergence field and write the output.
      const real_t *R = Y + 2*NC;
      MFEM_FOREACH_THREAD(f,z,2)
      {
         const int nx = f == 0 ? D1D : P, ny = f == 1 ? D1D : P;
         MFEM_FOREACH_THREAD(dy,y,ny)
         {
            MFEM_FOREACH_THREAD(dx,x,nx)
            {
               const int k = f == 0 ? dx : dy; // closed index
               const int s = f == 0 ? 1 : P;   // stride in R along f
               const int r = dx + P*dy;        // R index of subcell k
               real_t div = 0.0;
               if (k > 0) { div += R[r - s]; }
               if (k < P) { div -= R[r]; }
               const int i = f*NC + dx + nx*dy;
               y(i,e) = IsConstrained(mask, f, k, P) ? x(i,e) : Y[i] + div;
            }
         }
      }
   });
}

/// Size in direction d of field f of the 3D kernel (component f closed in
/// direction f, divergence field f = 3 open in all directions).
MFEM_HOST_DEVICE inline int FieldSize(int f, int d, int D1D)
{
   return d == f ? D1D : D1D - 1;
}

/// Contract direction a of all NF fields of the 3D kernel, from in to out,
/// using flat thread indexing over "pencils" along a. The forward
/// (interpolation) passes map DOF sizes to Q1D in the order a = 0, 1, 2; the
/// backward (integration) passes map Q1D back to DOF sizes in the same order.
/// Field f is stored at offset f*S, lexicographically with its current sizes.
/// The bases are stored as B(d,q) at d + n*q, with n = D1D for Bc and D1D-1
/// for Bo.
template <int MQD>
MFEM_HOST_DEVICE inline void ContractPencils3D(
   const int NF, const int D1D, const int Q1D, const int S, const int a,
   const bool forward, const real_t *Bc, const real_t *Bo, const real_t *in,
   real_t *out)
{
   // Sizes of field f in direction d before (in) and after (out) this pass.
   auto size_in = [=](int f, int d)
   {
      const bool quad = forward ? d < a : d >= a;
      return quad ? Q1D : FieldSize(f, d, D1D);
   };
   auto size_out = [=](int f, int d)
   {
      const bool quad = forward ? d <= a : d > a;
      return quad ? Q1D : FieldSize(f, d, D1D);
   };
   // The other two directions; their sizes do not change in this pass.
   const int b = a == 0 ? 1 : 0, c = a == 2 ? 1 : 2;
   int total = 0;
   for (int f = 0; f < NF; ++f) { total += size_in(f,b)*size_in(f,c); }

   MFEM_FOREACH_THREAD(t,x,total)
   {
      int f = 0, r = t;
      for (; f < NF-1; ++f)
      {
         const int count = size_in(f,b)*size_in(f,c);
         if (r < count) { break; }
         r -= count;
      }
      const int si0 = size_in(f,0), si1 = size_in(f,1);
      const int so0 = size_out(f,0), so1 = size_out(f,1);
      const int stin[3] = {1, si0, si0*si1};
      const int stout[3] = {1, so0, so0*so1};
      const int sb = size_in(f,b);
      const int ib = r%sb, ic = r/sb;
      const real_t *pin = in + f*S + ib*stin[b] + ic*stin[c];
      real_t *pout = out + f*S + ib*stout[b] + ic*stout[c];
      const int n = FieldSize(f, a, D1D);
      const real_t *B = a == f ? Bc : Bo;
      real_t v[MQD];
      if (forward)
      {
         for (int d = 0; d < n; ++d) { v[d] = pin[d*stin[a]]; }
         for (int q = 0; q < Q1D; ++q)
         {
            real_t s = 0.0;
            for (int d = 0; d < n; ++d) { s += B[d + n*q]*v[d]; }
            pout[q*stout[a]] = s;
         }
      }
      else
      {
         for (int q = 0; q < Q1D; ++q) { v[q] = pin[q*stin[a]]; }
         for (int d = 0; d < n; ++d)
         {
            real_t s = 0.0;
            for (int q = 0; q < Q1D; ++q) { s += B[d + n*q]*v[q]; }
            pout[d*stout[a]] = s;
         }
      }
   }
}

template <int T_D1D = 0, int T_Q1D = 0>
static void BrokenHdivApply3D(const int NE, const Array<real_t> &Bo_,
                              const Array<real_t> &Bc_, const Vector &qd_,
                              const Vector &x_, Vector &y_,
                              const int *masks,
                              const int d1d = 0, const int q1d = 0)
{
   static constexpr int NF = 4; // the three components and the divergence
   static constexpr int NT = 128; // threads per element
   const int D1D = T_D1D ? T_D1D : d1d;
   const int Q1D = T_Q1D ? T_Q1D : q1d;
   const int ND = 3*D1D*(D1D-1)*(D1D-1);
   const auto bo = Bo_.Read();
   const auto bc = Bc_.Read();
   const auto QD = Reshape(qd_.Read(), Q1D*Q1D*Q1D, 7, NE);
   const auto x = Reshape(x_.Read(), ND, NE);
   auto y = Reshape(y_.Write(), ND, NE);

   mfem::forall_2D(NE, NT, 1, [=] MFEM_HOST_DEVICE (int e)
   {
      const int D1D = T_D1D ? T_D1D : d1d;
      const int Q1D = T_Q1D ? T_Q1D : q1d;
      const int P = D1D - 1;
      const int NC = D1D*P*P; // DOFs per component
      const int NQ = Q1D*Q1D*Q1D;
      constexpr int MQ1 = T_Q1D ? T_Q1D : DofQuadLimits::HDIV_MAX_Q1D;
      constexpr int MD1 = T_D1D ? T_D1D : DofQuadLimits::HDIV_MAX_D1D;
      constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
      // Field stride: every stage of a field fits in max(D1D,Q1D)^3.
      const int M = (Q1D > D1D) ? Q1D : D1D;
      const int S = M*M*M;
      const int *mask = masks ? masks + 3*e : nullptr;

      MFEM_SHARED real_t sBc[MD1*MQ1];
      MFEM_SHARED real_t sBo[(MD1-1)*MQ1];
      MFEM_SHARED real_t s0[NF*MDQ*MDQ*MDQ];
      MFEM_SHARED real_t s1[NF*MDQ*MDQ*MDQ];

      // Load the element vector (components at offsets f*S) and the bases.
      MFEM_FOREACH_THREAD(i,x,ND)
      {
         const int f = i/NC, j = i - f*NC;
         const int nx = FieldSize(f,0,D1D), ny = FieldSize(f,1,D1D);
         const int k = f == 0 ? j%nx : (f == 1 ? (j/nx)%ny : j/(nx*ny));
         s0[f*S + j] = IsConstrained(mask, f, k, P) ? 0.0 : x(i,e);
      }
      MFEM_FOREACH_THREAD(i,x,D1D*Q1D)
      {
         const int d = i%D1D, q = i/D1D;
         sBc[i] = bc[q + Q1D*d];
      }
      MFEM_FOREACH_THREAD(i,x,P*Q1D)
      {
         const int d = i%P, q = i/P;
         sBo[i] = bo[q + Q1D*d];
      }
      MFEM_SYNC_THREAD;
      // Divergence coefficients d = D_hat x.
      MFEM_FOREACH_THREAD(i,x,P*P*P)
      {
         const int dx = i%P, dy = (i/P)%P, dz = i/(P*P);
         const real_t *X0 = s0, *X1 = s0 + S, *X2 = s0 + 2*S;
         s0[3*S + i] =
            X0[dx+1 + D1D*(dy + P*dz)] - X0[dx + D1D*(dy + P*dz)] +
            X1[dx + P*(dy+1 + D1D*dz)] - X1[dx + P*(dy + D1D*dz)] +
            X2[dx + P*(dy + P*(dz+1))] - X2[dx + P*(dy + P*dz)];
      }
      MFEM_SYNC_THREAD;
      // Interpolate to quadrature points, one direction at a time.
      ContractPencils3D<MDQ>(NF, D1D, Q1D, S, 0, true, sBc, sBo, s0, s1);
      MFEM_SYNC_THREAD;
      ContractPencils3D<MDQ>(NF, D1D, Q1D, S, 1, true, sBc, sBo, s1, s0);
      MFEM_SYNC_THREAD;
      ContractPencils3D<MDQ>(NF, D1D, Q1D, S, 2, true, sBc, sBo, s0, s1);
      MFEM_SYNC_THREAD;
      // Apply the quadrature data.
      MFEM_FOREACH_THREAD(q,x,NQ)
      {
         const real_t u0 = s1[q], u1 = s1[S + q], u2 = s1[2*S + q];
         const real_t D00 = QD(q,0,e), D01 = QD(q,1,e), D02 = QD(q,2,e);
         const real_t D11 = QD(q,3,e), D12 = QD(q,4,e), D22 = QD(q,5,e);
         s1[q] = D00*u0 + D01*u1 + D02*u2;
         s1[S + q] = D01*u0 + D11*u1 + D12*u2;
         s1[2*S + q] = D02*u0 + D12*u1 + D22*u2;
         s1[3*S + q] *= QD(q,6,e);
      }
      MFEM_SYNC_THREAD;
      // Integrate against the test functions, one direction at a time.
      ContractPencils3D<MDQ>(NF, D1D, Q1D, S, 0, false, sBc, sBo, s1, s0);
      MFEM_SYNC_THREAD;
      ContractPencils3D<MDQ>(NF, D1D, Q1D, S, 1, false, sBc, sBo, s0, s1);
      MFEM_SYNC_THREAD;
      ContractPencils3D<MDQ>(NF, D1D, Q1D, S, 2, false, sBc, sBo, s1, s0);
      MFEM_SYNC_THREAD;
      // Add D_hat^T of the divergence field and write the output.
      const real_t *R = s0 + 3*S;
      MFEM_FOREACH_THREAD(i,x,ND)
      {
         const int f = i/NC, j = i - f*NC;
         const int nx = FieldSize(f,0,D1D), ny = FieldSize(f,1,D1D);
         const int dx = j%nx, dy = (j/nx)%ny, dz = j/(nx*ny);
         const int k = f == 0 ? dx : (f == 1 ? dy : dz); // closed index
         const int s = f == 0 ? 1 : (f == 1 ? P : P*P); // stride in R along f
         const int r = dx + P*(dy + P*dz); // R index of subcell k
         real_t div = 0.0;
         if (k > 0) { div += R[r - s]; }
         if (k < P) { div -= R[r]; }
         y(i,e) = IsConstrained(mask, f, k, P) ? x(i,e) : s0[f*S + j] + div;
      }
   });
}

BrokenHdivOperator::BrokenHdivOperator(const FiniteElementSpace &fes,
                                       Coefficient &alpha, Coefficient &beta)
   : Operator(fes.GetNE()*fes.GetTypicalFE()->GetDof()),
     dim(fes.GetMesh()->Dimension()), ne(fes.GetNE())
{
   const auto *el =
      dynamic_cast<const VectorTensorFiniteElement*>(fes.GetTypicalFE());
   const auto *fec = dynamic_cast<const RT_FECollection*>(fes.FEColl());
   MFEM_VERIFY(el && fec && fec->GetClosedBasisType() == BasisType::GaussLobatto
               && fec->GetOpenBasisType() == BasisType::IntegratedGLL,
               "Expected GaussLobatto/IntegratedGLL RT elements.");
   MFEM_VERIFY((dim == 2 || dim == 3) && fes.GetVDim() == 1 &&
               fes.GetMesh()->SpaceDimension() == dim && !fes.IsVariableOrder(),
               "Expected a uniform-order 2D or 3D RT space.");

   Mesh &mesh = *fes.GetMesh();
   const IntegrationRule &ir = MassIntegrator::GetRule(
                                  *el, *el, *mesh.GetTypicalElementTransformation());

   const DofToQuad &maps_c = el->GetDofToQuad(ir, DofToQuad::TENSOR);
   const DofToQuad &maps_o = el->GetDofToQuadOpen(ir, DofToQuad::TENSOR);
   d1d = maps_c.ndof;
   q1d = maps_c.nqpt;
   MFEM_VERIFY(maps_o.ndof == d1d-1 && maps_o.nqpt == q1d,
               "Inconsistent open and closed bases.");
   MFEM_VERIFY(d1d <= DeviceDofQuadLimits::Get().HDIV_MAX_D1D &&
               q1d <= DeviceDofQuadLimits::Get().HDIV_MAX_Q1D,
               "Order exceeds the H(div) limits HDIV_MAX_D1D/HDIV_MAX_Q1D.");
   Bc = &maps_c.B;
   Bo = &maps_o.B;

   const int nq = ir.GetNPoints();
   MFEM_VERIFY(nq == static_cast<int>(std::pow(q1d, dim)),
               "Expected a tensor-product quadrature rule.");
   const auto *geom = mesh.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);
   QuadratureSpace qs(mesh, ir);
   CoefficientVector alpha_values(alpha, qs, CoefficientStorage::FULL);
   CoefficientVector beta_values(beta, qs, CoefficientStorage::FULL);
   qdata.SetSize((dim == 3 ? 7 : 4)*nq*ne);
   qdata.UseDevice(true);
   if (dim == 3)
   {
      SetupQuadratureData3D(ne, nq, ir.GetWeights(), geom->J, alpha_values,
                            beta_values, qdata);
   }
   else
   {
      SetupQuadratureData2D(ne, nq, ir.GetWeights(), geom->J, alpha_values,
                            beta_values, qdata);
   }
}

void BrokenHdivOperator::SetEssentialDofs(const FiniteElementSpace &fes,
                                          const Array<int> &ess_dofs)
{
   GetConstrainedFaceMasks(fes, ess_dofs, masks);
}

void BrokenHdivOperator::Mult(const Vector &x, Vector &y) const
{
   MFEM_ASSERT(x.Size() == width && y.Size() == height, "Invalid vector size.");
   const int *d_masks = masks.Size() ? masks.Read() : nullptr;
   const int id = (d1d << 4) | q1d;
   if (dim == 3)
   {
      switch (id)
      {
         case 0x23: return BrokenHdivApply3D<2,3>(ne, *Bo, *Bc, qdata, x, y, d_masks);
         case 0x34: return BrokenHdivApply3D<3,4>(ne, *Bo, *Bc, qdata, x, y, d_masks);
         case 0x45: return BrokenHdivApply3D<4,5>(ne, *Bo, *Bc, qdata, x, y, d_masks);
         case 0x56: return BrokenHdivApply3D<5,6>(ne, *Bo, *Bc, qdata, x, y, d_masks);
         default:
            return BrokenHdivApply3D(ne, *Bo, *Bc, qdata, x, y, d_masks, d1d, q1d);
      }
   }
   switch (id)
   {
      case 0x22: return BrokenHdivApply2D<2,2>(ne, *Bo, *Bc, qdata, x, y, d_masks);
      case 0x33: return BrokenHdivApply2D<3,3>(ne, *Bo, *Bc, qdata, x, y, d_masks);
      case 0x44: return BrokenHdivApply2D<4,4>(ne, *Bo, *Bc, qdata, x, y, d_masks);
      case 0x55: return BrokenHdivApply2D<5,5>(ne, *Bo, *Bc, qdata, x, y, d_masks);
      default:
         return BrokenHdivApply2D(ne, *Bo, *Bc, qdata, x, y, d_masks, d1d, q1d);
   }
}

} // namespace mfem
