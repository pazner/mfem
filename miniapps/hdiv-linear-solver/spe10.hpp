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

#ifndef MFEM_SPE10_HPP
#define MFEM_SPE10_HPP

#include "mfem.hpp"

namespace mfem
{

struct SPE10Data
{
   static constexpr int nx = 60;
   static constexpr int ny = 220;
   static constexpr int nz = 85;

   Array<double> coeff_data;
   SPE10Data() : coeff_data(3*nx*ny*nz)
   {
      std::ifstream permfile("data/spe_perm.dat");
      if (!permfile.good())
      {
         MFEM_ABORT("Cannot open data file spe_perm.dat.")
      }
      double tmp;
      double *ptr = coeff_data.begin();
      for (int l=0; l<3; l++)
      {
         for (int k=0; k<nz; k++)
         {
            for (int j=0; j<ny; j++)
            {
               for (int i=0; i<nx; i++)
               {
                  permfile >> *ptr;
                  *ptr = 1.0/(*ptr);
                  ptr++;
               }
               for (int i=0; i<60-nx; i++)
               {
                  permfile >> tmp;   // skip unneeded part
               }
            }
            for (int j=0; j<220-ny; j++)
               for (int i=0; i<60; i++)
               {
                  permfile >> tmp;   // skip unneeded part
               }
         }
         if (l<2) // if not processing Kz, we must skip unneeded part
         {
            for (int k=0; k<85-nz; k++)
               for (int j=0; j<220; j++)
                  for (int i=0; i<60; i++)
                  {
                     permfile >> tmp;   // skip unneeded part
                  }
         }
      }
   }
   void Invert()
   {
      for (int i = 0; i < coeff_data.Size(); ++i)
      {
         coeff_data[i] = 1.0/coeff_data[i];
      }
   }
};

struct SPE10Coefficient : DiagonalMatrixCoefficient
{
   SPE10Data spe10;
   SPE10Coefficient() : VectorCoefficient(3) { }
   using VectorCoefficient::Eval;
   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      constexpr int nx = SPE10Data::nx;
      constexpr int ny = SPE10Data::ny;
      constexpr int nz = SPE10Data::nz;
      double data[3];
      Vector xvec(data, 3);
      T.Transform(ip, xvec);

      const double x = xvec[0], y = xvec[1], z = xvec[2];
      const int i = (nx*x)/1200;
      MFEM_VERIFY(i >= 0 && i < nx, "");
      const int j = (ny*y)/2200;
      MFEM_VERIFY(j >= 0 && j < ny, "");
      const int k = (nz*z)/170;
      MFEM_VERIFY(k >= 0 && k < nz, "");

      V[0] = spe10.coeff_data[ny*nx*k + nx*j + i];
      V[1] = spe10.coeff_data[ny*nx*k + nx*j + i + nx*ny*nz];
      V[2] = spe10.coeff_data[ny*nx*k + nx*j + i + 2*nx*ny*nz];

      // std::cout << x << '\t' << y << '\t' << z << '\n';
      // std::cout << i << '\t' << j << '\t' << k << '\t' << V[0] << '\t' << V[1] << '\t' << V[2] << '\n';
   }
};

/// The SPE10 domain [0,1200] x [0,2200] x [0,170] with one hexahedron per
/// permeability cell (60 x 220 x 85), refined ref_par times in parallel. Sets
/// negative ny, nz from nx (the reported grid size).
inline ParMesh MakeSPE10Mesh(int &nx, int &ny, int &nz, int ref_par)
{
   if (ny < 0) { ny = (nx*220)/60; }
   if (nz < 0) { nz = (nx*85)/60; }
   if (Mpi::Root())
   {
      std::cout << "Grid size: " << nx << " x " << ny << " x " << nz
                << std::endl;
   }
   Mesh serial_mesh = Mesh::MakeCartesian3D(60, 220, 85, Element::HEXAHEDRON,
                                            1200, 2200, 170);
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < ref_par; ++i) { mesh.UniformRefinement(); }
   return mesh;
}

/// Set u to the interpolant of the constant vector field g in its RT space, on
/// the device. Requires an affine hexahedral mesh and tensor-product RT
/// elements of uniform order (as for SPE10): the Piola pullback adj(J) g of g
/// is then constant on each element, so the element DOFs are combinations of
/// the reference interpolants of the unit vectors, which are computed once on
/// the host. Agrees with ProjectCoefficient (which runs on the host, element
/// by element) up to roundoff.
inline void ProjectConstantFlux(const Vector &g, ParGridFunction &u)
{
   ParFiniteElementSpace &fes = *u.ParFESpace();
   Mesh &mesh = *fes.GetMesh();
   const FiniteElement &fe = *fes.GetTypicalFE();
   const auto *tensor_fe = dynamic_cast<const TensorBasisElement*>(&fe);
   MFEM_VERIFY(mesh.Dimension() == 3 && g.Size() == 3 &&
               mesh.GetNumGeometries(3) <= 1 &&
               fe.GetGeomType() == Geometry::CUBE && tensor_fe,
               "Expected a 3D mesh with tensor-product hexahedral elements.");
   const int nd = fe.GetDof();
   const int ne = mesh.GetNE();

   // Reference interpolants of the unit vectors, in the lexicographic element
   // ordering, with the DOF map signs (as applied by the element restriction).
   Vector ref(3*nd);
   ref.UseDevice(true);
   {
      const Array<int> &dof_map = tensor_fe->GetDofMap();
      const int *h_dof_map = dof_map.HostRead();
      IsoparametricTransformation T;
      T.SetIdentityTransformation(Geometry::CUBE);
      Vector unit(3), native(nd);
      real_t *h_ref = ref.HostWrite();
      for (int c = 0; c < 3; ++c)
      {
         unit = 0.0;
         unit(c) = 1.0;
         VectorConstantCoefficient unit_coeff(unit);
         fe.Project(unit_coeff, T, native);
         for (int i = 0; i < nd; ++i)
         {
            const int j = dof_map.Size() ? h_dof_map[i] : i;
            h_ref[c*nd + i] = (j >= 0) ? native(j) : -native(-1-j);
         }
      }
   }

   // The (constant) Jacobian of each element, at a single point.
   const IntegrationRule &ir = IntRules.Get(Geometry::CUBE, 0);
   const int nq = ir.GetNPoints();
   const GeometricFactors *geom =
      mesh.GetGeometricFactors(ir, GeometricFactors::JACOBIANS);

   const real_t *h_g = g.HostRead();
   const real_t g0 = h_g[0], g1 = h_g[1], g2 = h_g[2];
   const auto J = Reshape(geom->J.Read(), nq, 3, 3, ne);
   const auto R = Reshape(ref.Read(), nd, 3);
   Vector u_elem(nd*ne);
   u_elem.UseDevice(true);
   auto U = Reshape(u_elem.Write(), nd, ne);
   mfem::forall(nd*ne, [=] MFEM_HOST_DEVICE (int i)
   {
      const int d = i % nd;
      const int e = i / nd;
      // Row a of adj(J) is the cross product of columns a+1 and a+2 of J.
      real_t value = 0.0;
      for (int a = 0; a < 3; ++a)
      {
         const int b = (a+1)%3, c = (a+2)%3;
         const real_t n0 = J(0,1,b,e)*J(0,2,c,e) - J(0,2,b,e)*J(0,1,c,e);
         const real_t n1 = J(0,2,b,e)*J(0,0,c,e) - J(0,0,b,e)*J(0,2,c,e);
         const real_t n2 = J(0,0,b,e)*J(0,1,c,e) - J(0,1,b,e)*J(0,0,c,e);
         value += (n0*g0 + n1*g1 + n2*g2)*R(d,a);
      }
      U(d,e) = value;
   });

   // The interpolant is continuous, so the value from any element will do.
   const auto *restriction = dynamic_cast<const ElementRestriction*>(
                                fes.GetElementRestriction(
                                   ElementDofOrdering::LEXICOGRAPHIC));
   MFEM_VERIFY(restriction, "Expected an element restriction.");
   u.UseDevice(true);
   restriction->MultLeftInverse(u_elem, u);
}

struct ScalarSPE10Coefficient : Coefficient
{
   SPE10Coefficient vector_coeff;
   Vector val = Vector(3);
   using Coefficient::Eval;
   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      vector_coeff.Eval(val, T, ip);
      // std::cout << val.Norml2() << '\n';
      const double nrm = val.Norml2();
      MFEM_VERIFY(nrm >= 0, "");
      return nrm;
   }
};

}

#endif
