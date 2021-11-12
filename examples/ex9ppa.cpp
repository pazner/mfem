#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "../linalg/dtensor.hpp"

using namespace std;
using namespace mfem;

void AddDGIntegrators(BilinearForm &k, VectorCoefficient &velocity)
{
   double alpha = 1.0;
   double beta = -0.5;
   k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -alpha));
   k.AddDomainIntegrator(new MassIntegrator);
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, alpha, beta)));
   k.AddBdrFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, alpha, beta)));
}

static double u0_f(const Vector&)
{
   static int n = 0;
   return n++/(M_PI*M_PI);
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
   const int num_procs = mpi.WorldSize();
   int myid = mpi.WorldRank();

   // 1. Parse command-line options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int ref_levels = 2;
   int order = 2;
   const char *device_config = "cpu";
   bool visualization = false;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }

   if (mpi.Root()) { args.PrintOptions(cout); }

   Device device(device_config);
   device.Print();

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   mesh.EnsureNodes();

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.RandomRefinement(0.6, false, 1, 4);
   }

   int* partitioning = mesh.GeneratePartitioning(mpi.WorldSize());
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      partitioning[i] = i*mpi.WorldSize()/mesh.GetNE();
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning);

   pmesh.ExchangeFaceNbrData();


   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace serial_fes(&mesh, &fec);
   ParFiniteElementSpace fes(&pmesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;

   Vector velocity_vector(dim);
   for (int i = 0; i < dim; ++i) { velocity_vector[i] = -M_PI; }
   VectorConstantCoefficient velocity(velocity_vector);
   ParBilinearForm k_test(&fes), k_ref(&fes);
   k_test.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   AddDGIntegrators(k_test, velocity);
   AddDGIntegrators(k_ref, velocity);

   k_test.Assemble();
   k_ref.Assemble();
   k_ref.Finalize();

   GridFunction serial_u(&serial_fes);
   FunctionCoefficient u0_fct_coeff(u0_f);
   GridFunction l2_u(&serial_fes);
   l2_u.ProjectCoefficient(u0_fct_coeff);
   serial_u.ProjectGridFunction(l2_u);
   ParGridFunction u(&pmesh, &serial_u, partitioning);

   ParGridFunction r_test(&fes), r_ref(&fes), diff(&fes);

   Array<int> bdofs;
   OperatorHandle A_ref;
   k_ref.FormSystemMatrix(bdofs,A_ref);

   A_ref->Mult(u, r_ref);
   k_test.Mult(u, r_test);

   const double rr = InnerProduct(MPI_COMM_WORLD,r_ref,r_ref);
   const double tt = InnerProduct(MPI_COMM_WORLD,r_test,r_test);
   const double eps = fabs(rr-tt)/fabs(rr+tt);

   MFEM_VERIFY(eps < 1e-14,"Test error");

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      {
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << pmesh << r_ref << flush;
      }
      {
         socketstream sol_sock(vishost, visport);
         sol_sock << "parallel " << num_procs << " " << myid << "\n";
         sol_sock.precision(8);
         sol_sock << "solution\n" << pmesh << r_test << flush;
      }
   }

   return 0;
}
