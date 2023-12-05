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

#include "image_mesh.hpp"
#include "ppm.hpp"

namespace mfem
{

ImageMesh::ImageMesh(const std::string &filename)
   : ImageMesh(PixelImage(filename)) { }

ImageMesh::ImageMesh(const PixelImage &image_) : image(image_)
{
   const int width = image.Width();
   const int height = image.Height();

   const int n_vertices = (width + 1)*(height + 1);

   int n_elements = 0;
   // Count how many elements there are. Each non-zero pixel corresponds to an
   // element.
   for (int i = 0; i < width*height; ++i)
   {
      if (image[i] != 0) { ++n_elements; }
   }

   const int dim = 2;
   InitMesh(dim, dim, n_vertices, n_elements, 0);

   // Add all verties in the grid, even if they are unused. We will remove
   // unused vertices later.
   double h = 1.0 / width;
   for (int j = 0; j < height + 1; ++j)
   {
      for (int i = 0; i < width + 1; ++i)
      {
         AddVertex(i*h, j*h);
      }
   }

   // Add the elements.
   for (int j = 0; j < height; ++j)
   {
      for (int i = 0; i < width; ++i)
      {
         if (image(i, j) != 0)
         {
            const int lex_idx = i + j*width;
            lex2idx[lex_idx] = NumOfElements;
            idx2lex.emplace_back(i, j);

            const int v1 = j*(width + 1) + i;
            const int v2 = v1 + 1;
            const int v3 = v1 + width + 1 + 1;
            const int v4 = v3 - 1;
            AddQuad(v1, v2, v3, v4);
         }
      }
   }

   RemoveUnusedVertices();
   FinalizeMesh(); // The boundary elements will be generated here.
}

ImageMesh ImageMesh::Coarsen() const
{
   return ImageMesh(image.Coarsen());
}

int ImageMesh::GetElementIndex(int i, int j) const
{
   const int lex = i + j*image.Width();
   const auto result = lex2idx.find(lex);

   if (result != lex2idx.end()) { return result->second; }

   return -1; // The requested index is not occupied.
}

LexIndex ImageMesh::GetLexicographicIndex(int idx) const
{
   return idx2lex[idx];
}

}
