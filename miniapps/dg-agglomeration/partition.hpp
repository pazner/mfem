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

#ifndef MFEM_PARTITION_HPP
#define MFEM_PARTITION_HPP

#include "mfem.hpp"
#include "optref.hpp"

namespace mfem
{

/** Partition the elements of @a mesh (or only the elements listed in
    @a subset) into @a npart parts using METIS.

    Works on a ParMesh as well: only the rank-local elements are partitioned,
    and adjacency to elements on other ranks (face neighbors) is ignored.

    part_method: 0/3 = METIS_PartGraphRecursive, 1/4 = METIS_PartGraphKway;
    0-2 sort the neighbor lists first, 3-4 do not.

    Returns an array with one part index per partitioned element, in the order
    of @a subset (or of the mesh elements if no subset is given). */
Array<int> PartitionMesh(Mesh &mesh,
                         const int npart, const int part_method,
                         OptRef<Array<int>> subset = NullOpt);

} // namespace mfem

#endif
