// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_MINIAPPS_LOR_MESH_HPP
#define MFEM_MINIAPPS_LOR_MESH_HPP

#include "mfem.hpp"

namespace tetlor
{

/// Coarse vertices in the order used to map the reference LOR patch.
void GetPatchVertices(const mfem::Mesh &mesh, int e, bool duffy,
                      mfem::Array<int> &vertices);

mfem::Mesh MakeReferenceLORMesh(int q, const std::string &grid_type);

/// Each coarse tetrahedron has q^3 consecutive children in the returned mesh.
mfem::Mesh MakeLORMesh(const mfem::Mesh &mesh, int q,
                       const std::string &grid_type);

} // namespace tetlor

#endif
