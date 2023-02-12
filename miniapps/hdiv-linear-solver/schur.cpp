#include "mfem.hpp"
#include <iostream>
#include <memory>

#include "discrete_divergence.hpp"
#include "hdiv_linear_solver.hpp"

#include "../solvers/lor_mms.hpp"

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref = 0, int par_ref = 0);

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../../data/star.mesh";
   const char *device_config = "cpu";
   int ser_ref = 1;
   int par_ref = 1;
   int order = 3;
   double alpha = 0.0;

   OptionsParser args(argc, argv);
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&alpha, "-a", "--alpha", "Value of alpha coefficient.");
   args.ParseCheck();

   Device device(device_config);
   if (Mpi::Root()) { device.Print(); }

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);
   const int dim = mesh.Dimension();
   MFEM_VERIFY(dim == 2 || dim == 3, "Spatial dimension must be 2 or 3.");

   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLegendre;
   const int mt = FiniteElement::INTEGRAL;
   RT_FECollection fec_rt(order-1, dim, b1, b2);
   L2_FECollection fec_l2(order-1, dim, b2, mt);
   ParFiniteElementSpace fes_rt(&mesh, &fec_rt);
   ParFiniteElementSpace fes_l2(&mesh, &fec_l2);

   HYPRE_BigInt ndofs_rt = fes_rt.GlobalTrueVSize();
   HYPRE_BigInt ndofs_l2 = fes_l2.GlobalTrueVSize();

   if (Mpi::Root())
   {
      cout << "\nRT DOFs: " << ndofs_rt << "\nL2 DOFs: " << ndofs_l2 << endl;
   }

   Array<int> ess_rt_dofs; // empty

   // f is the RHS, u is the exact solution
   FunctionCoefficient f_coeff(f(alpha)), u_coeff(u);

   // Assemble the right-hand side for the scalar (L2) unknown.
   ParLinearForm b_l2(&fes_l2);
   b_l2.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
   b_l2.UseFastAssembly(true);
   b_l2.Assemble();

   ConstantCoefficient one(1.0);
   ConstantCoefficient alpha_coeff(alpha);
   const auto solver_mode = HdivSaddlePointSolver::Mode::DARCY;
   HdivSaddlePointSolver saddle_point_solver(
      mesh, fes_rt, fes_l2, alpha_coeff, one, ess_rt_dofs, solver_mode);

   HypreParMatrix &S = saddle_point_solver.GetApproxSchurComplement();
   HypreBoomerAMG &S_inv = saddle_point_solver.GetSchurComplementAMG();

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(500);
   cg.SetPrintLevel(1);
   cg.SetOperator(S);
   cg.SetPreconditioner(S_inv);
   Vector X(fes_l2.GetTrueVSize());
   X = 0.0;
   tic_toc.Clear();
   tic_toc.Start();
   cg.Mult(b_l2, X);

   if (Mpi::Root())
   {
      cout << "Done.\nIterations: "
            << cg.GetNumIterations()
            << "\nElapsed: " << tic_toc.RealTime() << endl;
   }

   return 0;
}

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref)
{
   Mesh serial_mesh = Mesh::LoadFromFile(mesh_file);
   for (int i = 0; i < ser_ref; ++i) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int i = 0; i < par_ref; ++i) { mesh.UniformRefinement(); }
   return mesh;
}
