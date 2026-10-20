// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "flow.hpp"
#include <cmath>
#include <limits>

namespace compressible
{
real_t Gas::Pressure(const Vector &u) const
{
   return (gamma - 1.0) * (u(3) - 0.5*(u(1)*u(1)+u(2)*u(2))/u(0));
}

real_t EulerRusanovFlux::Speed(const Vector &u, const Vector &normal,
                               Vector &ds) const
{
   ds.SetSize(4);
   const real_t rho = u(0), vx = u(1)/rho, vy = u(2)/rho;
   const real_t v2 = vx*vx + vy*vy;
   const real_t norm = normal.Norml2();
   const real_t vn = (vx*normal(0)+vy*normal(1))/norm;
   const real_t sign = vn > 0 ? 1.0 : (vn < 0 ? -1.0 : 0.0);
   const real_t p = gas.Pressure(u), c = std::sqrt(gas.gamma*p/rho);
   real_t dp[4] = {0.5*(gas.gamma-1)*v2, -(gas.gamma-1)*vx,
                   -(gas.gamma-1)*vy, gas.gamma-1
                  };
   for (int j = 0; j < 4; j++)
   {
      const real_t dv = sign*((j == 1 ? normal(0)/norm : 0.0) +
                              (j == 2 ? normal(1)/norm : 0.0) - vn*(j == 0))/rho;
      ds(j) = dv + gas.gamma/(2*c*rho)*(dp[j] - p/rho*(j == 0));
   }
   return std::abs(vn)+c;
}

void EulerRusanovFlux::Grad(int side, const Vector &u1, const Vector &u2,
                            const Vector &normal, FaceElementTransformations &tr,
                            DenseMatrix &grad) const
{
   RusanovFlux::Grad(side, u1, u2, normal, tr, grad);
   Vector ds1, ds2;
   const real_t s1 = Speed(u1, normal, ds1), s2 = Speed(u2, normal, ds2);
   const bool tied = std::abs(s1-s2) <= 32*std::numeric_limits<real_t>::epsilon()
                     *std::max(s1,s2);
   const real_t weight = tied ? 0.5 :
                         ((side == 1 ? s1 > s2 : s2 > s1) ? 1.0 : 0.0);
   const Vector &ds = side == 1 ? ds1 : ds2;
   for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
      {
         grad(i,j) += 0.5*normal.Norml2()*weight*(u1(i)-u2(i))*ds(j);
      }
}

// Differentiate primitive gradients explicitly. This is a closed-form
// directional differential, not finite differences or automatic differentiation.
static void Viscous(const Gas &g, const Vector &u, const DenseMatrix &q,
                    const Vector *du, const DenseMatrix *dq, DenseMatrix &f)
{
   const bool deriv = du != nullptr;
   const real_t rho = u(0), dr = deriv ? (*du)(0) : 0.0;
   real_t v[2], dv[2], a[2][2], da[2][2], tau[2][2], dtau[2][2];
   for (int i = 0; i < 2; i++)
   {
      v[i] = u(i+1)/rho;
      dv[i] = deriv ? ((*du)(i+1)-v[i]*dr)/rho : 0.0;
      for (int d = 0; d < 2; d++)
      {
         a[i][d] = (q(i+1,d)-v[i]*q(0,d))/rho;
         da[i][d] = deriv ? ((*dq)(i+1,d)-dv[i]*q(0,d)-
                             v[i]*(*dq)(0,d)-a[i][d]*dr)/rho : 0.0;
      }
   }
   const real_t div = a[0][0]+a[1][1], ddiv = da[0][0]+da[1][1];
   for (int i = 0; i < 2; i++)
      for (int d = 0; d < 2; d++)
      {
         tau[i][d] = g.viscosity*(a[i][d]+a[d][i]-(i == d)*2.0/3.0*div);
         dtau[i][d] = g.viscosity*(da[i][d]+da[d][i]-(i == d)*2.0/3.0*ddiv);
      }
   const real_t specific_energy = u(3)/rho;
   const real_t de = deriv ? ((*du)(3)-specific_energy*dr)/rho : 0.0;
   // T = (E/rho - |v|^2/2)/cv.
   f.SetSize(4,2);
   f = 0.0;
   for (int d = 0; d < 2; d++)
   {
      const real_t eg = (q(3,d)-specific_energy*q(0,d))/rho;
      real_t tg = eg, dtg = deriv ? ((*dq)(3,d)-de*q(0,d)-
                                     specific_energy*(*dq)(0,d)-eg*dr)/rho : 0.0;
      for (int i = 0; i < 2; i++)
      {
         tg -= v[i]*a[i][d];
         dtg -= dv[i]*a[i][d]+v[i]*da[i][d];
         f(i+1,d) = deriv ? dtau[i][d] : tau[i][d];
         f(3,d) += deriv ? dv[i]*tau[i][d]+v[i]*dtau[i][d] : v[i]*tau[i][d];
      }
      f(3,d) += g.Conductivity()/g.Cv()*(deriv ? dtg : tg);
   }
}
void ViscousFlux(const Gas &g, const Vector &u, const DenseMatrix &q,
                 DenseMatrix &f)
{ Viscous(g,u,q,nullptr,nullptr,f); }
void ViscousDerivative(const Gas &g, const Vector &u, const DenseMatrix &q,
                       const Vector &du, const DenseMatrix &dq, DenseMatrix &df)
{ Viscous(g,u,q,&du,&dq,df); }

void ViscousIntegrator::Element(const FiniteElement &el,
                                ElementTransformation &tr,
                                const Vector &x, Vector *r, DenseMatrix *jac)
{
   const int n = el.GetDof();
   if (r) { r->SetSize(4*n); *r = 0.0; }
   if (jac) { jac->SetSize(4*n); *jac = 0.0; }
   const IntegrationRule &ir = IntRule ? *IntRule :
                               IntRules.Get(el.GetGeomType(), 2*el.GetOrder()+offset);
   DenseMatrix values(x.GetData(), n, 4), ds(n,2), q(4,2), f, dq(4,2);
   Vector shape(n), u(4), du(4);
   for (int ip = 0; ip < ir.GetNPoints(); ip++)
   {
      tr.SetIntPoint(&ir.IntPoint(ip));
      el.CalcShape(ir.IntPoint(ip),shape);
      el.CalcPhysDShape(tr,ds);
      values.MultTranspose(shape,u);
      MultAtB(values,ds,q);
      const real_t w = ir.IntPoint(ip).weight*tr.Weight();
      if (r)
      {
         ViscousFlux(gas,u,q,f);
         for (int a = 0; a < 4; a++)
            for (int i = 0; i < n; i++)
               for (int d = 0; d < 2; d++) { (*r)(a*n+i) += w*ds(i,d)*f(a,d); }
      }
      if (jac)
      {
         // Physics derivatives are evaluated in the 4 state and 8 gradient
         // directions, independently of the number of element DOFs.
         DenseMatrix fu[4], fq[4][2];
         dq = 0.0;
         for (int b = 0; b < 4; b++)
         {
            du = 0.0; du(b) = 1.0;
            ViscousDerivative(gas,u,q,du,dq,fu[b]);
            for (int e = 0; e < 2; e++)
            {
               dq = 0.0; dq(b,e) = 1.0;
               ViscousFlux(gas,u,dq,fq[b][e]);
            }
            dq = 0.0;
         }
         for (int a = 0; a < 4; a++)
            for (int b = 0; b < 4; b++)
               for (int i = 0; i < n; i++)
                  for (int j = 0; j < n; j++)
                     for (int d = 0; d < 2; d++)
                     {
                        real_t value = fu[b](a,d)*shape(j);
                        for (int e = 0; e < 2; e++) { value += fq[b][e](a,d)*ds(j,e); }
                        (*jac)(a*n+i,b*n+j) += w*ds(i,d)*value;
                     }
      }
   }
}

// K maps conservative gradients to viscous flux; DK[b] is its derivative in
// state direction b. Indices: output component/direction, input component/direction.
struct DiffusionTensor
{
   real_t k[4][2][4][2], dk[4][4][2][4][2];
   DiffusionTensor(const Gas &g, const Vector &u, bool derivative)
   {
      DenseMatrix q(4,2), zero(4,2), f;
      Vector du(4); zero = 0.0;
      for (int b = 0; b < 4; b++)
         for (int e = 0; e < 2; e++)
         {
            q = 0.0; q(b,e) = 1.0;
            ViscousFlux(g,u,q,f);
            for (int a = 0; a < 4; a++)
               for (int d = 0; d < 2; d++) { k[a][d][b][e] = f(a,d); }
            if (derivative)
               for (int c = 0; c < 4; c++)
               {
                  du = 0.0; du(c) = 1.0;
                  ViscousDerivative(g,u,q,du,zero,f);
                  for (int a = 0; a < 4; a++)
                     for (int d = 0; d < 2; d++) { dk[c][a][d][b][e] = f(a,d); }
               }
         }
   }
};

void ViscousIntegrator::Face(const FiniteElement &el1, const FiniteElement &el2,
                             FaceElementTransformations &tr, const Vector &x,
                             Vector *r, DenseMatrix *jac)
{
   MFEM_VERIFY(tr.Elem2No >= 0, "Viscous faces must be interior or periodic.");
   const int n[2] = {el1.GetDof(),el2.GetDof()}, off[2] = {0,4*n[0]};
   const int size = 4*(n[0]+n[1]);
   if (r) { r->SetSize(size); *r = 0.0; }
   if (jac) { jac->SetSize(size); *jac = 0.0; }
   const FiniteElement *el[2] = {&el1,&el2};
   const int p = std::max(el1.GetOrder(),el2.GetOrder());
   const IntegrationRule &ir = IntRule ? *IntRule : IntRules.Get(
                                  tr.GetGeometryType(),2*p+offset);
   // Integrate physical volumes/face length: also valid on curved conforming meshes.
   auto measure = [](ElementTransformation &t)
   {
      real_t value = 0;
      const IntegrationRule &rule = IntRules.Get(t.GetGeometryType(),2*t.OrderJ()+4);
      for (int i = 0; i < rule.GetNPoints(); i++)
      { t.SetIntPoint(&rule.IntPoint(i)); value += rule.IntPoint(i).weight*t.Weight(); }
      return value;
   };
   const real_t length = measure(tr);
   const real_t h = std::min(measure(*tr.Elem1),measure(*tr.Elem2))/length;
   const real_t eta = penalty*(p+1)*(p+1)/h;
   Vector shape[2] = {Vector(n[0]),Vector(n[1])}, u[2] = {Vector(4),Vector(4)},
                                                         normal(2);
   DenseMatrix ds[2] = {DenseMatrix(n[0],2),DenseMatrix(n[1],2)};
   DenseMatrix q[2] = {DenseMatrix(4,2),DenseMatrix(4,2)}, f[2];
   DenseMatrix values[2] = {DenseMatrix(x.GetData(),n[0],4),DenseMatrix(x.GetData()+off[1],n[1],4)};
   for (int ip = 0; ip < ir.GetNPoints(); ip++)
   {
      tr.SetAllIntPoints(&ir.IntPoint(ip));
      CalcOrtho(tr.Jacobian(),normal);
      const real_t norm = normal.Norml2(), w = ir.IntPoint(ip).weight*norm;
      normal /= norm;
      for (int s = 0; s < 2; s++)
      {
         ElementTransformation &t = s == 0 ? *tr.Elem1 : *tr.Elem2;
         el[s]->CalcShape(t.GetIntPoint(),shape[s]);
         el[s]->CalcPhysDShape(t,ds[s]);
         values[s].MultTranspose(shape[s],u[s]);
         MultAtB(values[s],ds[s],q[s]);
         ViscousFlux(gas,u[s],q[s],f[s]);
      }
      DiffusionTensor k[2] = {DiffusionTensor(gas,u[0],jac),DiffusionTensor(gas,u[1],jac)};
      real_t jump[4], avg[4] = {}, pen[4] = {}, adj[2][4][2] = {};
      for (int a = 0; a < 4; a++) { jump[a] = u[0](a)-u[1](a); }
      for (int a = 0; a < 4; a++)
         for (int d = 0; d < 2; d++)
         {
            avg[a] += 0.5*(f[0](a,d)+f[1](a,d))*normal(d);
            for (int b = 0; b < 4; b++)
               for (int e = 0; e < 2; e++)
                  for (int s = 0; s < 2; s++)
                  {
                     pen[a] += 0.5*k[s].k[a][d][b][e]*jump[b]*normal(e)*normal(d);
                     adj[s][b][e] += k[s].k[a][d][b][e]*jump[a]*normal(d);
                  }
         }
      if (r)
         for (int s = 0; s < 2; s++)
            for (int a = 0; a < 4; a++)
               for (int i = 0; i < n[s]; i++)
               {
                  real_t value = (s == 0 ? 1 : -1)*shape[s](i)*(-avg[a]+eta*pen[a]);
                  for (int d = 0; d < 2; d++) { value -= 0.5*ds[s](i,d)*adj[s][a][d]; }
                  (*r)(off[s]+a*n[s]+i) += w*value;
               }
      if (jac)
      {
         // Differentiate face quantities in each trace state/gradient direction,
         // then contract with trial shape functions. 12 directions per side.
         for (int t = 0; t < 2; t++)
            for (int c = 0; c < 4; c++)
               for (int direction = -1; direction < 2; direction++)
               {
                  const bool state = direction == -1;
                  real_t davg[4] = {}, dpen[4] = {}, dadj[2][4][2] = {};
                  const real_t jump_sign = t == 0 ? 1.0 : -1.0;
                  for (int a = 0; a < 4; a++)
                     for (int d = 0; d < 2; d++)
                     {
                        if (!state) { davg[a] += 0.5*k[t].k[a][d][c][direction]*normal(d); }
                        else
                           for (int b = 0; b < 4; b++)
                              for (int e = 0; e < 2; e++)
                              {
                                 davg[a] += 0.5*k[t].dk[c][a][d][b][e]*q[t](b,e)*normal(d);
                                 dpen[a] += 0.5*k[t].dk[c][a][d][b][e]*jump[b]*normal(e)*normal(d);
                                 for (int s = 0; s < 2; s++)
                                 {
                                    dpen[a] += 0.5*k[s].k[a][d][b][e]*(b == c)*jump_sign*normal(e)*normal(d);
                                    dadj[s][b][e] += (s == t ? k[s].dk[c][a][d][b][e]*jump[a] : 0.0)*normal(d)
                                                     + k[s].k[a][d][b][e]*(a == c)*jump_sign*normal(d);
                                 }
                              }
                     }
                  for (int s = 0; s < 2; s++)
                     for (int a = 0; a < 4; a++)
                        for (int i = 0; i < n[s]; i++)
                        {
                           real_t value = (s == 0 ? 1 : -1)*shape[s](i)*(-davg[a]+eta*dpen[a]);
                           for (int d = 0; d < 2; d++) { value -= 0.5*ds[s](i,d)*dadj[s][a][d]; }
                           for (int j = 0; j < n[t]; j++)
                           {
                              (*jac)(off[s]+a*n[s]+i,off[t]+c*n[t]+j) += w*value*
                                                                         (state ? shape[t](j) : ds[t](j,direction));
                           }
                        }
               }
      }
   }
}
}
