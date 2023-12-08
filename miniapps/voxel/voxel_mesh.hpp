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

#ifndef VOXEL_MESH_HPP
#define VOXEL_MESH_HPP

#include "mfem.hpp"
#include "lex.hpp"

namespace mfem
{

class VoxelMesh : public Mesh
{
protected:
   double h;
   std::vector<int> n;
   std::unordered_map<int,int> lex2idx;
   std::vector<LexIndex> idx2lex;

   VoxelMesh(double h_, const std::vector<int> &n_);

public:
   VoxelMesh(const std::string &filename, double h);

   VoxelMesh Coarsen() const;
};

}

#endif
