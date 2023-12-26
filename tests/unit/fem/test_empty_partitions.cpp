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

#include "unit_tests.hpp"
#include "mfem.hpp"

using namespace mfem;

#ifdef MFEM_USE_MPI

TEST_CASE("Empty Partitions", "[PartialAssembly], [Parallel]")
{
   const int order = 1;
   ParMesh mesh = []()
   {
      Mesh serial_mesh = Mesh::MakeCartesian2D(10, 10, Element::QUADRILATERAL);
      // The mesh partition for MPI rank 0 is actually the whole mesh, every
      // other MPI rank has an empty partition.
      Array<int> partitioning(serial_mesh.GetNE());
      partitioning = 0;
      return ParMesh(MPI_COMM_WORLD, serial_mesh, partitioning.GetData());
   }();
   const int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&mesh, &fec);

   ParGridFunction x(&fes), y_pa(&fes), y_fa(&fes);
   Array<int> empty;
   OperatorHandle A_pa, A_fa;

   ParBilinearForm a_pa(&fes);
   a_pa.AddBoundaryIntegrator(new MassIntegrator);
   a_pa.AddDomainIntegrator(new DiffusionIntegrator);
   a_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a_pa.Assemble();
   a_pa.FormSystemMatrix(empty, A_pa);

   ParBilinearForm a_fa(&fes);
   a_fa.AddBoundaryIntegrator(new MassIntegrator);
   a_fa.AddDomainIntegrator(new DiffusionIntegrator);
   a_fa.Assemble();
   a_fa.Finalize();
   a_fa.FormSystemMatrix(empty, A_fa);

   x.Randomize(1);
   A_pa->Mult(x, y_pa);
   A_fa->Mult(x, y_fa);

   y_pa -= y_fa;
   REQUIRE(y_pa.Normlinf() == MFEM_Approx(0.0));

   Vector diag_pa(fes.GetTrueVSize()), diag_fa(fes.GetTrueVSize());
   a_pa.AssembleDiagonal(diag_pa);
   a_fa.AssembleDiagonal(diag_fa);

   diag_pa -= diag_fa;
   REQUIRE(diag_pa.Normlinf() == MFEM_Approx(0.0));
}

#endif
