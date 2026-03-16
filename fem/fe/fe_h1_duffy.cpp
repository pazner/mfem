// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fe_h1_duffy.hpp"

namespace mfem
{

H1Duffy_TriangleElement::H1Duffy_TriangleElement(int p, int btype)
   : NodalFiniteElement(2, Geometry::TRIANGLE, p*p + p + 1, p,
                        FunctionSpace::Pk),
     quad_fe(p, btype)
{
   auto set_duffy = [](IntegrationPoint &ip, const IntegrationPoint &ip_quad)
   {
      ip.x = ip_quad.x * (1.0 - ip_quad.y);
      ip.y = ip_quad.y;
   };
   auto idx = [p](int i, int j) { return i + (p+1)*j; };

   lex_ordering.SetSize(dof);

   // bottom left
   set_duffy(Nodes[0], quad_fe.GetNodes()[0]);
   lex_ordering[idx(0,0)] = 0;
   // bottom right
   set_duffy(Nodes[1], quad_fe.GetNodes()[1]);
   lex_ordering[idx(p,0)] = 1;
   // top
   set_duffy(Nodes[2], quad_fe.GetNodes()[3]);
   lex_ordering[idx(0,p)] = 2;

   int o = 3;
   int oq = 4;
   // bottom edge
   for (int i = 0; i < p-1; ++i)
   {
      lex_ordering[idx(i+1,0)] = o;
      set_duffy(Nodes[o++], quad_fe.GetNodes()[oq++]);
   }
   // diagonal edge
   for (int i = 0; i < p-1; ++i)
   {
      lex_ordering[idx(p,i+1)] = o;
      set_duffy(Nodes[o++], quad_fe.GetNodes()[oq++]);
   }
   // skip top edge
   oq += p-1;
   // left edge
   for (int i = 0; i < p-1; ++i)
   {
      lex_ordering[idx(0,p-i-1)] = o;
      set_duffy(Nodes[o++], quad_fe.GetNodes()[oq++]);
   }

   // interior
   for (int i = 0; i < (p-1)*(p-1); ++i)
   {
      const int ix = 1 + (i % (p-1));
      const int iy = 1 + (i / (p-1));
      lex_ordering[idx(ix,iy)] = o;
      set_duffy(Nodes[o++], quad_fe.GetNodes()[oq++]);
   }
}

void H1Duffy_TriangleElement::CalcShape(const IntegrationPoint &ip,
                                        Vector &shape) const
{
   const int p = order;

   IntegrationPoint ip_quad = ip;
   if (ip_quad.x > 0)
   {
      ip_quad.x /= 1.0 - ip_quad.y;
   }

   Vector shape_quad((p+1)*(p+1));
   quad_fe.CalcShape(ip_quad, shape_quad);

   // bottom left
   shape[0] = shape_quad[0];
   // bottom right
   shape[1] = shape_quad[1];
   // top - contribution from top right
   shape[2] = shape_quad[2];
   // top - contribution from top left
   shape[2] += shape_quad[3];

   int o = 3;
   int oq = 4;
   // bottom edge
   for (int i = 0; i < p-1; ++i) { shape[o++] = shape_quad[oq++]; }
   // diagonal edge
   for (int i = 0; i < p-1; ++i) { shape[o++] = shape_quad[oq++]; }
   // top edge contributions to top vertex
   for (int i = 0; i < p-1; ++i) { shape[2] += shape_quad[oq++]; }
   // left edge
   for (int i = 0; i < p-1; ++i) { shape[o++] = shape_quad[oq++]; }

   // interior
   for (int i = 0; i < (p-1)*(p-1); ++i) { shape[o++] = shape_quad[oq++]; }
}

void H1Duffy_TriangleElement::CalcDShape(const IntegrationPoint &ip,
                                         DenseMatrix &dshape) const
{
   const int p = order;

   IntegrationPoint ip_quad = ip;
   if (ip_quad.x > 0)
   {
      ip_quad.x /= 1.0 - ip_quad.y;
   }

   const real_t a1 = 1.0 / (1.0 - ip_quad.y);
   const real_t a2 = ip_quad.x / (1.0 - ip_quad.y);

   DenseMatrix dshape_quad((p+1)*(p+1), 2);
   quad_fe.CalcDShape(ip_quad, dshape_quad);

   // bottom left
   dshape(0,0) = a1 * dshape_quad(0,0);
   dshape(0,1) = a2 * dshape_quad(0,0) + dshape_quad(0,1);
   // bottom right
   dshape(1,0) = a1 * dshape_quad(1,0);
   dshape(1,1) = a2 * dshape_quad(1,0) + dshape_quad(1,1);
   // top - contribution from top right
   dshape(2,0) = a1 * dshape_quad(2,0);
   // dshape(2,1) = a2 * dshape_quad(2,0) + dshape_quad(2,1);
   dshape(2,1) = dshape_quad(2,1);
   // top - contribution from top left
   dshape(2,0) += a1 * dshape_quad(3,0);
   // dshape(2,1) += 21 * dshape_quad(3,0) + dshape_quad(3,1);
   dshape(2,1) += dshape_quad(3,1);

   int o = 3;
   int oq = 4;
   // bottom edge
   for (int i = 0; i < p-1; ++i)
   {
      dshape(o,0) = a1 * dshape_quad(oq,0);
      dshape(o,1) = a2 * dshape_quad(oq,0) + dshape_quad(oq,1);
      ++o;
      ++oq;
   }
   // diagonal edge
   for (int i = 0; i < p-1; ++i)
   {
      dshape(o,0) = a1 * dshape_quad(oq,0);
      dshape(o,1) = a2 * dshape_quad(oq,0) + dshape_quad(oq,1);
      ++o;
      ++oq;
   }
   // top edge contributions to top vertex
   for (int i = 0; i < p-1; ++i)
   {
      dshape(2,0) += a1 * dshape_quad(oq,0);
      // dshape(2,1) += a2 * dshape_quad(oq,0) + dshape_quad(oq,1);
      // dshape(2,1) += a2 * dshape_quad(oq,0) + dshape_quad(oq,1);
      dshape(2,1) += dshape_quad(oq,1);
      ++oq;
   }
   // left edge
   for (int i = 0; i < p-1; ++i)
   {
      dshape(o,0) = a1 * dshape_quad(oq,0);
      dshape(o,1) = a2 * dshape_quad(oq,0) + dshape_quad(oq,1);
      ++o;
      ++oq;
   }

   // interior
   for (int i = 0; i < (p-1)*(p-1); ++i)
   {
      dshape(o,0) = a1 * dshape_quad(oq,0);
      dshape(o,1) = a2 * dshape_quad(oq,0) + dshape_quad(oq,1);
      ++o;
      ++oq;
   }
}

}
