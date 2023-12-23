//                       MFEM Example 2 - Parallel Version
//
// Compile with: make ex2p
//
// Sample runs:  mpirun -np 4 ex2p -m ../data/beam-tri.mesh
//               mpirun -np 4 ex2p -m ../data/beam-quad.mesh
//               mpirun -np 4 ex2p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex2p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex2p -m ../data/beam-wedge.mesh
//               mpirun -np 4 ex2p -m ../data/beam-tri.mesh -o 2 -sys
//               mpirun -np 4 ex2p -m ../data/beam-quad.mesh -o 3 -elast
//               mpirun -np 4 ex2p -m ../data/beam-quad.mesh -o 3 -sc
//               mpirun -np 4 ex2p -m ../data/beam-quad-nurbs.mesh
//               mpirun -np 4 ex2p -m ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise. The geometry of the domain is assumed to be as
//               follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (pull down)
//
//               The example demonstrates the use of high-order and NURBS vector
//               finite element spaces with the linear elasticity bilinear form,
//               meshes with curved elements, and the definition of piece-wise
//               constant and vector coefficient objects. Static condensation is
//               also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-tri.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   bool amg_elast = 0;
   bool reorder_space = false;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
                  "--amg-for-systems",
                  "Use the special AMG elasticity solver (GM/LN approaches), "
                  "or standard AMG for systems (unknown approach).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&reorder_space, "-nodes", "--by-nodes", "-vdim", "--by-vdim",
                  "Use byNODES ordering of vector space instead of byVDIM");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.ParseCheck();

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh mesh(mesh_file);
   const int dim = mesh.Dimension();
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   H1_FECollection fec(order, dim);
   auto ordering = reorder_space ? Ordering::byNODES : Ordering::byVDIM;
   ParFiniteElementSpace fespace(&pmesh, &fec, dim, ordering);

   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of finite element unknowns: " << size << endl
           << "Assembling: " << flush;
   }

   // 9. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined by
   //    marking only boundary attribute 1 from the mesh as essential and
   //    converting it to a list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;

   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 10. Set up the parallel linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system. In this case, b_i equals the
   //     boundary integral of f*phi_i where f represents a "pull down" force on
   //     the Neumann part of the boundary and phi_i are the basis functions in
   //     the finite element fespace. The force is defined by the object f, which
   //     is a vector of Coefficient objects. The fact that f is non-zero on
   //     boundary attribute 2 is indicated by the use of piece-wise constants
   //     coefficient for its last component.
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(pmesh.bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   ParLinearForm b(&fespace);
   b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   if (Mpi::Root())
   {
      cout << "r.h.s. ... " << flush;
   }
   b.Assemble();

   b.Randomize(1);

   // 11. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 12. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu.
   ConstantCoefficient lambda_func(100.0);
   ConstantCoefficient mu_func(50.0);

   ParBilinearForm a(&fespace);
   auto integ = new ElasticityIntegrator(lambda_func, mu_func);
   a.AddDomainIntegrator(integ);

   // IntegrationRules irs(0, Quadrature1D::GaussLobatto);
   // integ->SetIntRule(&irs.Get(pmesh.GetElementGeometry(0), 2*order - 1));

   // 13. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (Mpi::Root()) { cout << "matrix ... " << flush; }
   if (static_cond) { a.EnableStaticCondensation(); }
   a.Assemble();

   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   if (Mpi::Root())
   {
      cout << "done." << endl;
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   // 14. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
   //     preconditioner from hypre.
   HypreBoomerAMG amg(A);
   if (amg_elast && !a.StaticCondensationIsEnabled())
   {
      amg.SetElasticityOptions(&fespace);
   }
   else
   {
      amg.SetSystemsOptions(dim, reorder_space);
   }
   HyprePCG pcg(A);
   pcg.SetTol(1e-8);
   pcg.SetMaxIter(500);
   pcg.SetPrintLevel(2);
   pcg.SetPreconditioner(amg);
   pcg.Mult(B, X);

   // 15. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   // 16. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element.  This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   pmesh.SetNodalFESpace(&fespace);

   // 17. Save in parallel the displaced mesh and the inverted solution (which
   //     gives the backward displacements to the original grid). This output
   //     can be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      GridFunction *nodes = pmesh.GetNodes();
      *nodes += x;
      x *= -1;

      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << Mpi::WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << Mpi::WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 18. Send the above data by socket to a GLVis server.  Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   return 0;
}
