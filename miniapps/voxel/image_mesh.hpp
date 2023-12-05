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

#ifndef IMAGE_MESH_HPP
#define IMAGE_MESH_HPP

#include "mfem.hpp"
#include "ppm.hpp"

#include <unordered_map>

namespace mfem
{

struct LexIndex
{
   int x;
   int y;
   LexIndex(int x_, int y_) : x(x_), y(y_) { }
};

class ImageMesh : public Mesh
{
private:
   PixelImage image;
   std::unordered_map<int,int> lex2idx;
   std::vector<LexIndex> idx2lex;
public:
   ImageMesh(const std::string &filename);
   ImageMesh(const PixelImage &image_);
   ImageMesh Coarsen() const;
   int GetElementIndex(int i, int j) const;
   LexIndex GetLexicographicIndex(int idx) const;
   PixelImage &GetImage() { return image; }
   const PixelImage &GetImage() const { return image; }
};

}

#endif
