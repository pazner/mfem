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

// Electrostatic nodes from Nektar++ LibUtilities/Foundations/{NodalTriElecData,
// NodalTetElecData}.h (5.10). The coordinate tables carry the following license.
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
#include "lor_mesh.hpp"
#include <algorithm>
#include <numeric>

namespace tetlor
{
namespace
{

// One representative per permutation orbit of open face and volume nodes.
// Integer barycentric labels identify topology independently of coordinates.
struct Orbit
{
   BarycentricLabel label;
   std::array<double,4> bary;
};

constexpr Orbit orbits[] =
{
   // q = 3
   {{0,1,1,1}, {0.0, 0.3333333333, 0.3333333333, 0.3333333333}},
   // q = 4
   {{0,1,1,2}, {0.0, 0.2371200168, 0.2371200168, 0.5257599664}},
   {{1,1,1,1}, {0.2500000000, 0.2500000000, 0.2500000000, 0.2500000000}},
   // q = 5
   {{0,1,1,3}, {0.0, 0.1575181512, 0.1575181512, 0.6849636976}},
   {{0,1,2,2}, {0.0, 0.1789696980, 0.4105151510, 0.4105151510}},
   {{1,1,1,2}, {0.1834903473, 0.1834903473, 0.1834903473, 0.4495289581}},
   // q = 6
   {{0,1,1,4}, {0.0, 0.1061169285, 0.1061169285, 0.7877661430}},
   {{0,1,2,3}, {0.0, 0.1332918645, 0.3097982151, 0.5569099204}},
   {{0,2,2,2}, {0.0, 0.3333333333, 0.3333333333, 0.3333333333}},
   {{1,1,1,3}, {0.1402705801, 0.1402705801, 0.1402705801, 0.5791882597}},
   {{1,1,2,2}, {0.1457947417, 0.1457947417, 0.3542052583, 0.3542052583}},
   // q = 7
   {{0,1,1,5}, {0.0, 0.0660520784, 0.0660520784, 0.8678958432}},
   {{0,1,2,4}, {0.0, 0.0914849272, 0.2325524777, 0.6759625951}},
   {{0,1,3,3}, {0.0, 0.1044549894, 0.4477725053, 0.4477725053}},
   {{0,2,2,3}, {0.0, 0.2604038024, 0.2604038024, 0.4791923952}},
   {{1,1,1,4}, {0.1144606542, 0.1144606542, 0.1144606542, 0.6566180374}},
   {{1,1,2,3}, {0.1208429970, 0.1208429970, 0.2812936703, 0.4770203357}},
   {{1,2,2,2}, {0.1248991534, 0.2917002822, 0.2917002822, 0.2917002822}},
   // q = 8
   {{0,1,1,6}, {0.0, 0.0469685351, 0.0469685351, 0.9060629298}},
   {{0,1,2,5}, {0.0, 0.0679538347, 0.1801396087, 0.7519065566}},
   {{0,1,3,4}, {0.0, 0.0840385433, 0.3617970895, 0.5541643672}},
   {{0,2,2,4}, {0.0, 0.2033467796, 0.2033467796, 0.5933064408}},
   {{0,2,3,3}, {0.0, 0.2189007568, 0.3905496216, 0.3905496216}},
   {{1,1,1,5}, {0.0991203900, 0.0991203900, 0.0991203900, 0.7026388300}},
   {{1,1,2,4}, {0.1047451941, 0.1047451941, 0.2323149656, 0.5581946462}},
   {{1,1,3,3}, {0.1079468963, 0.1079468963, 0.3920531037, 0.3920531037}},
   {{1,2,2,3}, {0.1099065340, 0.2419418605, 0.2419418605, 0.4062097450}},
   {{2,2,2,2}, {0.2500000000, 0.2500000000, 0.2500000000, 0.2500000000}},
   // q = 9
   {{0,1,1,7}, {0.0, 0.0355775717, 0.0355775717, 0.9288448566}},
   {{0,1,2,6}, {0.0, 0.0529419344, 0.1439089974, 0.8031490682}},
   {{0,1,3,5}, {0.0, 0.0684032457, 0.2966333890, 0.6349633653}},
   {{0,1,4,4}, {0.0, 0.0719393950, 0.4640303025, 0.4640303025}},
   {{0,2,2,5}, {0.0, 0.1633923069, 0.1633923069, 0.6732153862}},
   {{0,2,3,4}, {0.0, 0.1806100558, 0.3225890045, 0.4968009397}},
   {{0,3,3,3}, {0.0, 0.3333333333, 0.3333333333, 0.3333333333}},
   {{1,1,1,6}, {0.0823287303, 0.0823287303, 0.0823287303, 0.7530138091}},
   {{1,1,2,5}, {0.0873980781, 0.0873980781, 0.1954980564, 0.6297057875}},
   {{1,1,3,4}, {0.0916714679, 0.0916714679, 0.3347047619, 0.4819523024}},
   {{1,2,2,4}, {0.0923029247, 0.2040338880, 0.2040338880, 0.4996292993}},
   {{1,2,3,3}, {0.0956734931, 0.2075502723, 0.3483881173, 0.3483881173}},
   {{2,2,2,3}, {0.2123055477, 0.2123055477, 0.2123055477, 0.3630833569}},
   // q = 10
   {{0,1,1,8}, {0.0, 0.0265250690, 0.0265250690, 0.9469498620}},
   {{0,1,2,7}, {0.0, 0.0409265838, 0.1163195333, 0.8427538829}},
   {{0,1,3,6}, {0.0, 0.0545659823, 0.2462883939, 0.6991456238}},
   {{0,1,4,5}, {0.0, 0.0592706549, 0.3934913008, 0.5472380443}},
   {{0,2,2,6}, {0.0, 0.1330857076, 0.1330857076, 0.7338285848}},
   {{0,2,3,5}, {0.0, 0.1481684519, 0.2707097521, 0.5811217960}},
   {{0,2,4,4}, {0.0, 0.1535875376, 0.4232062312, 0.4232062312}},
   {{0,3,3,4}, {0.0, 0.2833924371, 0.2833924371, 0.4332151258}},
   {{1,1,1,7}, {0.0678316144, 0.0678316144, 0.0678316144, 0.7965051568}},
   {{1,1,2,6}, {0.0724033935, 0.0724033935, 0.1658367169, 0.6893564961}},
   {{1,1,3,5}, {0.0768451848, 0.0768451848, 0.2889363346, 0.5573732958}},
   {{1,1,4,4}, {0.0780456199, 0.0780456199, 0.4219543801, 0.4219543801}},
   {{1,2,2,5}, {0.0768158196, 0.1734555313, 0.1734555313, 0.5762731177}},
   {{1,2,3,4}, {0.0809350066, 0.1776946096, 0.3019928872, 0.4393774966}},
   {{1,3,3,3}, {0.0845418628, 0.3051527124, 0.3051527124, 0.3051527124}},
   {{2,2,2,4}, {0.1805746957, 0.1805746957, 0.1805746957, 0.4582759129}},
   {{2,2,3,3}, {0.1835663764, 0.1835663764, 0.3164336236, 0.3164336236}},
};

} // namespace

std::array<mfem::real_t,4> ElectrostaticNode(const BarycentricLabel &label,
                                             const mfem::real_t *gll)
{
   std::array<mfem::real_t,4> bary;
   if (std::count(label.begin(),label.end(),0) >= 2)
   {
      for (int i = 0; i < 4; ++i) { bary[i] = gll[label[i]]; }
      return bary;
   }

   std::array<int,4> perm = {0,1,2,3};
   std::sort(perm.begin(),perm.end(),
   [&](int i, int j) { return label[i] < label[j]; });
   BarycentricLabel key;
   for (int i = 0; i < 4; ++i) { key[i] = label[perm[i]]; }
   for (const auto &orbit : orbits)
   {
      if (orbit.label != key) { continue; }
      // Restore partition of unity after decimal rounding of the tables.
      const auto sum = std::accumulate(orbit.bary.begin(),orbit.bary.end(),0.0);
      for (int i = 0; i < 4; ++i) { bary[perm[i]] = orbit.bary[i]/sum; }
      return bary;
   }
   MFEM_ABORT("Missing electrostatic node for the requested lattice label.");
}

} // namespace tetlor
