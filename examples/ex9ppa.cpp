#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "../linalg/dtensor.hpp"

#define MFEM_DEBUG_COLOR 226
#include "../general/debug.hpp"

using namespace std;
using namespace mfem;

int problem;

void AddDGIntegrators(BilinearForm &k, VectorCoefficient &velocity)
{
   double alpha = 1.0;
   double beta = -0.5;
   ///k.AddDomainIntegrator(new ConvectionIntegrator(velocity, -alpha));
   ///k.AddDomainIntegrator(new MassIntegrator);
   k.AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(velocity, alpha, beta)));
   ///k.AddBdrFaceIntegrator(
   ///   new TransposeIntegrator(new DGTraceIntegrator(velocity, alpha, beta)));
   // k.AddInteriorFaceIntegrator(new DGTraceIntegrator(velocity, alpha, beta));
   // k.AddBdrFaceIntegrator(new DGTraceIntegrator(velocity, alpha, beta));
}

////////////////////////////////////////////////////////////////////////////////
#undef USE_SHIFT

static double u0_f(const Vector &x)
{
   static int n = 0;
   int shifted_n = -1;
#ifdef USE_SHIFT
   int mpi_size, mpi_rank, shift;
   MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
   if (mpi_size == 1) { shift = 0; }
   // 0:32, 1:32
   if (mpi_size == 2) { shift = mpi_rank>0 ? 32 : 0; }
   // 0:24, 1:20, 2:20
   if (mpi_size == 3) { shift = mpi_rank>1 ? 20+24 : mpi_rank>0?24: 0; }
   assert(mpi_size==1 || mpi_size==2 || mpi_size==3);
   shifted_n = shift + n;
   dbg("shifted_n: %d",shifted_n);
#else
   shifted_n = n;
#endif
   n++;
   assert(shifted_n >= 0);
   return shifted_n;///(32.*M_PI);
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
   dbg();

   // 1. Parse command-line options.
   problem = 0;
   const char *mesh_file = "../data/inline-quad.mesh";
   int ref_levels = 0;
   int order = 3;
   const char *device_config = "cpu";
   bool visualization = true;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   int pi(0), pj(0);
   args.AddOption(&pi, "-pi", "--permi", "Permutation i.");
   args.AddOption(&pj, "-pj", "--permj", "Permutation j.");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root())
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (mpi.Root())
   {
      args.PrintOptions(cout);
   }

   Device device(device_config);
   device.Print();


   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   mesh.EnsureNodes();

   for (int lev = 0; lev < ref_levels; lev++)
   {
      //mesh.UniformRefinement();
      mesh.RandomRefinement(0.6, false, 1, 4);
   }

#ifdef USE_SHIFT
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
#else
   int* partitioning = mesh.GeneratePartitioning(mpi.WorldSize());
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      partitioning[i] = i*mpi.WorldSize()/mesh.GetNE();
   }
   ParMesh pmesh(MPI_COMM_WORLD, mesh, partitioning);
#endif

   pmesh.ExchangeFaceNbrData();

   dbg("pmesh:%d",pmesh.GetNE());

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   // H1_FECollection fec(order, dim);
   FiniteElementSpace serial_fes(&mesh, &fec);
   ParFiniteElementSpace fes(&pmesh, &fec);

   cout << "Number of unknowns: " << fes.GetVSize() << endl;
   dbg("Number of unknowns: %d",fes.GetVSize());

   Vector velocity_vector(dim);
   for (int i = 0; i < dim; ++i)
   {
      velocity_vector[i] = -M_PI;
   }
   VectorConstantCoefficient velocity(velocity_vector);
   ParBilinearForm k_test(&fes), k_ref(&fes);
   k_test.SetAssemblyLevel(AssemblyLevel::PARTIAL);

   AddDGIntegrators(k_test, velocity);
   AddDGIntegrators(k_ref, velocity);

   tic_toc.Clear();
   tic_toc.Start();
   k_test.Assemble();
   k_test.Finalize();
   tic_toc.Stop();
   cout << "test assembly time: " << tic_toc.RealTime() << " sec." << endl;

   tic_toc.Clear();
   tic_toc.Start();
   k_ref.Assemble();
   k_ref.Finalize();
   tic_toc.Stop();
   cout << "ref assembly time: " << tic_toc.RealTime() << " sec." << endl;


   MPI_Barrier(MPI_COMM_WORLD);

#ifdef USE_SHIFT
   ParGridFunction u(&fes);
   FunctionCoefficient u0_fct_coeff(u0_f);
   ParGridFunction l2_u(&fes);
   l2_u.ProjectCoefficient(u0_fct_coeff);
   u.ProjectGridFunction(l2_u);
#else
   GridFunction serial_u(&serial_fes);
   FunctionCoefficient u0_fct_coeff(u0_f);
   GridFunction l2_u(&serial_fes);
   l2_u.ProjectCoefficient(u0_fct_coeff);
   serial_u.ProjectGridFunction(l2_u);
   ParGridFunction u(&pmesh, &serial_u, partitioning);
#endif

   ParGridFunction r_test(&fes), r_ref(&fes), diff(&fes);

   Array<int> bdofs;
   OperatorHandle A_ref;
   k_ref.FormSystemMatrix(bdofs,A_ref);

   const double EPS = 1e-14;

   //dbg("u(%d):",u.Size());
   //for (int i=0; i<u.Size(); ++i) { dbg("u[%d] %f", i, u[i]);}

   // all: 1578.18566715073
   dbg("u: %.15e",sqrt(InnerProduct(MPI_COMM_WORLD,u,u)));
   r_ref = 0.0;
   r_test = 0.0;
   A_ref->Mult(u, r_ref);
   k_test.Mult(u, r_test);

   //             2216.34303831963006814

   // 1, shifted: 2216.34303831963006814
   // 1, parting: 2216.34303831963006814

   // 2, shifted: 1994.27693929511883653
   // 2, parting: 2216.34303831962915865

   // 3, shifted: 2296.5335740784794325
   // 3, parting: 2216.3430383196287039

   const double rr = InnerProduct(MPI_COMM_WORLD,r_ref,r_ref);
   dbg("rr: %.21e",rr);
   const double tt = InnerProduct(MPI_COMM_WORLD,r_test,r_test);
   dbg("tt: %.21e",tt);

   const double eps = fabs(rr-tt)/fabs(rr+tt);
   dbg("eps: %.21e",eps);
   assert(eps < EPS);

   diff = r_test;
   diff -= r_ref;

   dbg("Save");
   pmesh.Save("ex9ppa");
   u.Save("u");
   diff.Save("diff");
   r_ref.Save("ref");
   r_test.Save("test");

   return 0;
}
