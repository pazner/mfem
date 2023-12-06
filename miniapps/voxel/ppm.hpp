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

#ifndef PPM_HPP
#define PPM_HPP

#include "mfem.hpp"

namespace mfem
{

class PixelImage
{
public:
   PixelImage(const std::string &filename);
   PixelImage(const int width_, const int height_, const std::vector<int> &data_);

   PixelImage Coarsen() const;

   int Height() const { return height; }
   int Width() const { return width; }

   int operator()(int i, int j) const;
   int operator[](int i) const;

   bool Empty() const;

private:
   int width = -1;
   int height = -1;

   std::vector<int> data;

   void ReadMagicNumber(std::istream &in);
   void ReadComments(std::istream &in);
   void ReadDimensions(std::istream &in);
   int ReadDepth(std::istream &in);
   void ReadPGM(std::istream &in, const int depth);
};

}

#endif
