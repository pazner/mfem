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

#include "ppm.hpp"
#include <fstream>

namespace mfem
{

using namespace std;

PixelImage::PixelImage(const std::string &filename)
{
   ifstream in(filename);
   if (!in.good())
   {
      MFEM_ABORT("Image file not found: " << filename);
   }

   ReadMagicNumber(in);
   ReadDimensions(in);
   const int depth = ReadDepth(in);
   ReadPGM(in, depth);
}

PixelImage::PixelImage(const int width_, const int height_,
                       const std::vector<int> &data_)
   : width(width_), height(height_), data(data_)
{ }

PixelImage PixelImage::Coarsen() const
{
   const int new_width = static_cast<int>(std::ceil(0.5*width));
   const int new_height = static_cast<int>(std::ceil(0.5*height));
   std::cout << "Old: " << width << "x" << height
             << "\tNew: " << new_width << "x" << new_height << '\n';

   std::vector<int> new_data(new_width * new_height);

   for (int j = 0; j < new_height; ++j)
   {
      for (int i = 0; i < new_width; ++i)
      {
         int value = 0;
         // If we want contained rather than containing coarse meshes, then
         // switch the above line with:
         //    int value = 1;
         // and below, take the min:
         //    value = std::min(value, fine_value);
         // instead of taking the max.
         for (int ii = 0; ii < 2; ++ii)
         {
            const int ix = 2*i + ii;
            for (int jj = 0; jj < 2; ++jj)
            {
               const int iy = 2*j + jj;
               int fine_value = 0;
               if (ix < width && iy < height)
               {
                  fine_value = (*this)(ix, iy);
               }
               value = std::max(value, fine_value);
            }
         }
         new_data[i + new_width*j] = value;
      }
   }
   return PixelImage(new_width, new_height, new_data);
}

bool PixelImage::Empty() const
{
   for (int i = 0; i < width; ++i)
   {
      for (int j = 0; j < height; ++j)
      {
         if ((*this)(i,j) != 0) { return false; }
      }
   }
   return true;
}

int PixelImage::operator()(int i, int j) const { return data[i + width*j]; }

int PixelImage::operator[](int i) const { return data[i]; }

void PixelImage::ReadMagicNumber(istream &in)
{
   char c;
   int p;
   in >> c >> p; // Read magic number which should be P2 or P5
   MFEM_VERIFY(c == 'P' && (p == 2 || p == 5),
               "Invalid PGM file! Unrecognized magic number\""
               << c << p << "\".");
   ReadComments(in);
}

void PixelImage::ReadComments(istream &in)
{
   string buf;
   in >> std::ws; // absorb any white space
   while (in.peek() == '#')
   {
      std::getline(in,buf);
   }
   in >> std::ws; // absorb any white space
}

void PixelImage::ReadDimensions(istream &in)
{
   in >> width;
   ReadComments(in);
   in >> height;
   ReadComments(in);
}

int PixelImage::ReadDepth(istream &in)
{
   int depth;
   in >> depth;
   ReadComments(in);
   return depth;
}

void PixelImage::ReadPGM(istream &in, const int depth)
{
   const int N = width*height;
   data.resize(N);

   int value;
   for (int i = 0; i < N; ++i)
   {
      in >> value;
      data[i] = value;
   }
}

}
