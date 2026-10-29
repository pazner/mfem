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

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

TEST_CASE("Prescribed-pattern incomplete Cholesky",
          "[IncompleteCholesky][Parallel]")
{
   const int n = 6;
   SparseMatrix a(n), pattern(n);
   DenseMatrix dense(n); dense = 0.;
   for (int i = 0; i < n; ++i)
   {
      dense(i, i) = 4.; a.Add(i, i, 4.);
      for (int j = 0; j < n; ++j) { pattern.Add(i, j, 0.); }
      if (i + 1 < n) { dense(i, i + 1) = dense(i + 1, i) = -1.; a.Add(i, i + 1, -1.); a.Add(i + 1, i, -1.); }
   }
   dense(0, n - 1) = dense(n - 1, 0) = -1.; a.Add(0, n - 1, -1.);
   a.Add(n - 1, 0, -1.);
   a.Finalize(); pattern.Finalize(0);
   Array<int> permutation({2, 4, 0, 5, 1, 3});
   IncompleteCholeskySolver solver;
   REQUIRE(solver.Factor(a, pattern, permutation) == -1);
   REQUIRE(solver.NumNonZeroElems() == n * (n + 1) / 2);
   Vector b(n), x(n), reference(n);
   for (int i = 0; i < n; ++i) { b[i] = i + 1.; }
   DenseMatrixInverse direct(dense, true); direct.Mult(b, reference);
   solver.Mult(b, x);
   x -= reference; REQUIRE(x.Norml2() < 1.e-12);
   // A second setup must not retain a stale permutation or factor graph.
   solver.SetOperator(a); solver.Mult(b, x);
   REQUIRE(x * b > 0.);
   REQUIRE(solver.NumNonZeroElems() == 2 * n);
   SparseMatrix bad(2); bad.Add(0, 0, 1.); bad.Add(0, 1, 2.);
   bad.Add(1, 0, 2.); bad.Add(1, 1, 1.); bad.Finalize();
   Array<int> identity({0, 1});
   REQUIRE(solver.Factor(bad, bad, identity) == 1);
   REQUIRE(solver.Factor(bad, bad, identity, 2.) == -1);
}
