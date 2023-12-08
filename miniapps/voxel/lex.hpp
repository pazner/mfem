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

#ifndef LEX_HPP
#define LEX_HPP

#include <array>

struct LexIndex
{
   int ndim = 0;
   std::array<int, 3> coords;

   LexIndex() = default;
   LexIndex(int x) : ndim(1), coords({x}) { }
   LexIndex(int x, int y) : ndim(2), coords({x, y}) { }
   LexIndex(int x, int y, int z) : ndim(3), coords({x, y, z}) { }
   LexIndex(const int *xx, int ndim_) : ndim(ndim_)
   {
      std::copy(xx, xx + ndim, coords.begin());
   }

   int operator[](int i) const { return coords[i]; }

   int LinearIndex(const std::vector<int> &n) const
   {
      int shift = 1;
      int idx = 0;
      for (int i = 0; i < ndim; ++i)
      {
         idx += coords[i]*shift;
         shift *= n[i];
      }
      return idx;
   }
};

#endif
