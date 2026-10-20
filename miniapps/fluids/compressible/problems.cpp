// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#include "flow.hpp"
#include <cmath>

namespace compressible
{
namespace
{
// Analytic primitive value, first derivatives (x,y,t), and spatial Hessian.
struct Field
{
   real_t value, d[3], h[2][2];
   Field(real_t mean, real_t amplitude, real_t kx, real_t ky, real_t kt,
         real_t phase, const Vector &x, real_t t)
   {
      const real_t k[3] = {kx,ky,kt};
      const real_t arg = kx*x(0)+ky*x(1)+kt*t+phase;
      value = mean+amplitude*std::sin(arg);
      for (int i = 0; i < 3; i++) { d[i] = amplitude*k[i]*std::cos(arg); }
      for (int i = 0; i < 2; i++)
         for (int j = 0; j < 2; j++) { h[i][j] = -amplitude*k[i]*k[j]*std::sin(arg); }
   }
};
void Manufactured(const Gas &g, const Vector &x, real_t t, Vector *u,
                  Vector *source)
{
   const real_t k = M_PI/2;
   Field rho(1,.1,k,k,-1,0,x,t), temp(1,.1,k,-k,.5,0,x,t);
   Field v[2] = {Field(.7,.1,k,0,.3,0,x,t),Field(.3,.1,0,k,-.2,M_PI/2,x,t)};
   const real_t e = g.Cv()*temp.value+0.5*(v[0].value*v[0].value
                                           +v[1].value*v[1].value);
   const real_t energy = rho.value*e,
                pressure = rho.value*g.gas_constant*temp.value;
   if (u)
   {
      u->SetSize(4); (*u)(0) = rho.value;
      (*u)(1) = rho.value*v[0].value; (*u)(2) = rho.value*v[1].value;
      (*u)(3) = energy;
   }
   if (!source) { return; }
   // Source is derived independently in primitive variables; no DG kernels or
   // numerical derivatives enter this evaluation.
   source->SetSize(4);
   const real_t div = v[0].d[0]+v[1].d[1];
   real_t mass = rho.d[2]+rho.value*div;
   for (int d = 0; d < 2; d++) { mass += v[d].value*rho.d[d]; }
   (*source)(0) = mass;
   real_t dtau[2], tau[2][2], dp[2], de[3];
   for (int d = 0; d < 3; d++)
   {
      de[d] = rho.d[d]*e+rho.value*(g.Cv()*temp.d[d]+
                                    v[0].value*v[0].d[d]+v[1].value*v[1].d[d]);
   }
   for (int i = 0; i < 2; i++)
   {
      dp[i] = g.gas_constant*(rho.d[i]*temp.value+rho.value*temp.d[i]);
      dtau[i] = g.viscosity*(v[i].h[0][0]+v[i].h[1][1]+
                             (v[0].h[i][0]+v[1].h[i][1])/3.0);
      real_t accel = v[i].d[2];
      for (int d = 0; d < 2; d++)
      {
         accel += v[d].value*v[i].d[d];
         tau[i][d] = g.viscosity*(v[i].d[d]+v[d].d[i]-(i == d)*2.0/3.0*div);
      }
      (*source)(i+1) = rho.value*accel+v[i].value*mass+dp[i]-dtau[i];
   }
   (*source)(3) = de[2]+(energy+pressure)*div-
                  g.Conductivity()*(temp.h[0][0]+temp.h[1][1]);
   for (int d = 0; d < 2; d++)
   {
      (*source)(3) += v[d].value*(de[d]+dp[d])-v[d].value*dtau[d];
      for (int i = 0; i < 2; i++) { (*source)(3) -= v[i].d[d]*tau[i][d]; }
   }
}
}

void Problem::State(const Vector &x, real_t t, Vector &u) const
{
   if (name == "manufactured") { Manufactured(gas,x,t,&u,nullptr); return; }
   u.SetSize(4);
   if (name == "constant")
   {
      u(0) = 1; u(1) = .7; u(2) = .3;
      u(3) = 1/(gas.gamma-1)+.5*(.7*.7+.3*.3);
      return;
   }
   MFEM_VERIFY(name == "vortex", "Unknown problem: " << name);
   // Example 18 parameters, in a larger periodic box: the boundary is ten
   // vortex radii from the center, so the Gaussian seam is below roundoff.
   const real_t radius = .2, beta = .2, mach = .5;
   const real_t xx = std::remainder(x(0)-t,4.0), yy = std::remainder(x(1),4.0);
   const real_t r2 = (xx*xx+yy*yy)/(radius*radius);
   const real_t temp_inf = 1/(gas.gamma*mach*mach*gas.gas_constant);
   const real_t temp = temp_inf-.5*beta*beta/(gas.gamma*gas.Cv())*std::exp(-r2);
   const real_t rho = std::pow(temp/temp_inf,1/(gas.gamma-1));
   const real_t vx = 1-beta*yy/radius*std::exp(-.5*r2);
   const real_t vy = beta*xx/radius*std::exp(-.5*r2);
   u(0) = rho; u(1) = rho*vx; u(2) = rho*vy;
   u(3) = rho*(gas.Cv()*temp+.5*(vx*vx+vy*vy));
}
void Problem::Source(const Vector &x, real_t t, Vector &s) const
{
   if (HasSource()) { Manufactured(gas,x,t,nullptr,&s); }
   else { s.SetSize(4); s = 0.0; }
}

Mesh PeriodicMesh(int n, bool triangles)
{
   Mesh mesh = Mesh::MakeCartesian2D(n,n,
                                     triangles ? Element::TRIANGLE : Element::QUADRILATERAL,
                                     true,4.0,4.0);
   for (int i = 0; i < mesh.GetNV(); i++)
   { mesh.GetVertex(i)[0] -= 2; mesh.GetVertex(i)[1] -= 2; }
   Vector tx(2), ty(2); tx = 0.0; ty = 0.0; tx(0) = 4; ty(1) = 4;
   return Mesh::MakePeriodic(mesh,mesh.CreatePeriodicVertexMapping({tx,ty}));
}
}
