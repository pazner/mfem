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
#include <eigen3/Eigen/Dense>
#include <vector>


using namespace std;
using namespace mfem;


class StrainVectorCoefficient: public VectorCoefficient
{
private:
   GridFunction &u;
   DenseMatrix eps;
   int vd;
   Vector eigenvalues;
   DenseMatrix eigenvectors;

public:
   StrainVectorCoefficient(int vd, GridFunction &_u)
      : VectorCoefficient(vd), u(_u), eigenvalues(3), eigenvectors(3,
                                                                   3) {}  // we define a StrainVectorCoefficient that needs an input of int vd (components number of the output vector ) and GridFunction &_u (the calculated displacement grid function x)

   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint
                     &ip); //the Eval function outputs the strain vector V at the integration points (IP)
   virtual ~StrainVectorCoefficient() { }
};




int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   const char *mesh_file = "../data/beam-tri.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 1;
   bool amg_elast = 0;
   bool reorder_space = false;
   const char *device_config = "cpu";
   bool par_format = false;
   //int nev = 3;
   //int seed = 66;

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
   args.AddOption(&par_format, "-pf", "--parallel-format", "-sf",
                  "--serial-format",
                  "Format to use when saving the results for VisIt.");
   //args.AddOption(&nev, "-n", "--num-eigs",
   // "Number of desired eigenmodes.");
   //args.AddOption(&seed, "-s", "--seed",
   // "Random seed used to initialize LOBPCG.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
   {
      if (myid == 0)
         cerr << "\nInput mesh should have at least two materials and "
              << "two boundary attributes! (See schematic in ex2.cpp)\n"
              << endl;
      return 3;
   }

   // 5. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext)
   {
      mesh->DegreeElevate(order, order);
   }

   // 6. PREVENT REFINMENT OF THE MESH! Refine the serial mesh on all processors to increase the resolution. In
   //    this example we do 'ref_levels' of uniform refinement. We choose
   //    'ref_levels' to be the largest number that gives a final mesh with no
   //    more than 1,000 elements.
   {
      int ref_levels = ;
      (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         //mesh->UniformRefinement();
      }
   }

   // 7. PREVENT REFINMENT OF THE MESH! Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   {
      int par_ref_levels = 2.5;
      for (int l = 0; l < par_ref_levels; l++)
      {
         //pmesh->UniformRefinement();
      }
   }

   // 8. Define a parallel finite element space on the parallel mesh. Here we
   //    use vector finite elements, i.e. dim copies of a scalar finite element
   //    space. We use the ordering by vector dimension (the last argument of
   //    the FiniteElementSpace constructor) which is expected in the systems
   //    version of BoomerAMG preconditioner. For NURBS meshes, we use the
   //    (degree elevated) NURBS space associated with the mesh nodes.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;
   const bool use_nodal_fespace = pmesh->NURBSext && !amg_elast;
   if (use_nodal_fespace)
   {
      fec = NULL;
      fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      if (reorder_space)
      {
         fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byNODES);
      }
      else
      {
         fespace = new ParFiniteElementSpace(pmesh, fec, dim, Ordering::byVDIM);
      }
   }
   HYPRE_BigInt size = fespace->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl
           << "Assembling: " << flush;
   }

   // 9. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined by
   //    marking only boundary attribute 1 from the mesh as essential and
   //    converting it to a list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   std::cout << "ess_tdof_list size: " << ess_tdof_list.Size() << std::endl;

   // 10. Set up the parallel linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system. In this case, b_i equals the
   //     boundary integral of FORCE*phi_i where FORCE represents the force components on
   //     the Neumann part of the boundary and phi_i are the basis functions in
   //     the finite element fespace. The FORCE is defined by the function F_val, which
   //     is a vector of Coefficient objects. The FORCE is applied only on
   //     boundary attribute 2 by the use of 'AddBoundaryIntegrator' which applied the FORCE
   //     only to bc_bdr that are flfed as non-zero
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(
               0.0)); //f is the Surface Traction vector which defines the direction of the force. here we set fx=0 & fy=0.
   }
   {
      Vector pull_force(
         pmesh->bdr_attributes.Max());//pull_force is a vector with size of the number of boundary attributes (number of outer surfaces in the domain)
      pull_force = 0.0;
      //pull_force(1) = -235.0; //defines the value/magnitude asigned to attribut 2 faces
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(
               pull_force)); // here fz (dim-1)=value asigned in pull force
   }

   ParLinearForm *b = new ParLinearForm(fespace);
   b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   if (myid == 0)
   {
      cout << "r.h.s. ... " << flush;
   }
   b->Assemble();
   std::cout << "rhs norm: " << b->Norml2() << std::endl;//RHS (right hand side).

   // 11. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   x = 0.0;
   std::cout << "x grid function size: " << x.Size() <<
             std::endl;//number of dof with overlapping decomposition across different processors.

   // 12. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu.
   Vector lambda(pmesh->attributes.Max());
   //lambda = 3846.0;
   lambda = 100.0;
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh->attributes.Max());
   //mu = 5769.0;
   mu = 50.0;
   PWConstCoefficient mu_func(mu);
   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

   // 13. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (myid == 0) { cout << "matrix ... " << flush; }
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   HypreParMatrix A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
   std::cout << "linear system formed" << std::endl;
   std::cout << "Before solve B size: " << B.Size() << "   Before solve X size: "
             << X.Size() << std::endl;
   std::cout << "Before solve x size: " << x.Size() <<
             "   Before solve ess_tdof_list size: " << ess_tdof_list.Size() << std::endl;
   std::cout << "b size: " << b->Size() << std::endl;
   std::cout << "A Height: " << A.Height() << std::endl;
   std::cout << "a Height: " << a->Height() << std::endl;

   if (myid == 0)
   {
      cout << "done." << endl;
      cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
   }

   // 14. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
   //     preconditioner from hypre.
   HypreBoomerAMG *amg = new HypreBoomerAMG(A);
   if (amg_elast && !a->StaticCondensationIsEnabled())
   {
      amg->SetElasticityOptions(fespace);
   }
   else
   {
      amg->SetSystemsOptions(dim, reorder_space);
   }
   HyprePCG *pcg = new HyprePCG(A);
   pcg->SetTol(1e-8);
   pcg->SetMaxIter(500);
   pcg->SetPrintLevel(2);
   pcg->SetPreconditioner(*amg);
   pcg->Mult(B, X);

   std::cout << "After solve X size: " << X.Size() << std::endl;

   // 15. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a->RecoverFEMSolution(X, *b, x);
   std::cout << "After recovery X size: " << X.Size() <<
             "   After recovery x size: " << x.Size() << std::endl;

   // 16. For non-NURBS meshes, make the mesh curved based on the finite element
   //     space. This means that we define the mesh elements through a fespace
   //     based transformation of the reference element.  This allows us to save
   //     the displaced mesh as a curved mesh when using high-order finite
   //     element displacement field. We assume that the initial mesh (read from
   //     the file) is not higher order curved mesh compared to the chosen FE
   //     space.
   if (!use_nodal_fespace)
   {
      pmesh->SetNodalFESpace(fespace);
   }

   GridFunction *nodes = pmesh->GetNodes();
   *nodes += x;
   //x *= -1; no need to inverse the displacments value

   // 17. Calculating stress tensor H1_FECollection L2_FECollection
   L2_FECollection eps_fec(1,
                           dim); // eps_fec is non continues FE collection (L2_FECollection) with linear shape functions (1) for 3D domain (dim=3).
   ParFiniteElementSpace eps_fespace(pmesh, &eps_fec,
                                     2); // eps_fespace is a FE space build by the pmash (mesh divided by MPI), eps_fec (linear shape functions, 3D) for 9 components (dim*dim; Max principal, Mid principal, Min principal, E11, E22, E33, E12, E13, E23)  
   ParGridFunction eps_gf(
      &eps_fespace); // eps_gf is a grid function (represents the nodes) for the strain tensor
   StrainVectorCoefficient strain_coef(2,
                                       x); //strain_coef is the strain vector coefficient  with 9 components and input of the displacement x.
   eps_gf.ProjectCoefficient(
      strain_coef); //Projects the calculations of the Eval at IP to the nodes of the grid function.


   // 18. Save the displaced mesh and the inverted solution (which gives the
   //     backward displacements to the original grid). This output can be
   //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
   {

      ostringstream mesh_name, sol_name, strain_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;
      strain_name << "strain." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);

      ofstream strain_ofs(strain_name.str().c_str());
      strain_ofs.precision(8);
      eps_gf.Save(strain_ofs);
   }

   // ParaView save Displacements and Strain
   {
      ParaViewDataCollection *pd = NULL;
      pd = new ParaViewDataCollection("Example02", pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("Displacements", &x);
      pd->RegisterField("Strain", &eps_gf);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }


   // 19. Free the used memory.
   //delete lobpcg;

   delete pcg;
   delete amg;
   delete a;
   delete b;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete pmesh;

   return 0;
}

void StrainVectorCoefficient::Eval(Vector &V, ElementTransformation &T,
                                   const IntegrationPoint &ip)
{
   u.GetVectorGradient(T, eps);  // eps = grad(u)
   eps.Symmetrize();// eps = (1/2)*(grad(u) + grad(u)^t)
   eps.CalcEigenvalues(eigenvalues.GetData(),
                       eigenvectors.Data()); //calculating the principal strains of eps dense matrix and storing in eigenvalues array in increasing order
   V(0)=eigenvalues(2);//max principal strain
   // V(1)=eigenvalues(1);//mid principal strain
   V(1)=eigenvalues(0);//min principal strain
   //V(3)=eps(0,0);//strain11
   //V(4)=eps(1,1);//strain22
   //V(5)=eps(2,2);//strain33
   //V(6)=eps(0,1);//strain12
   //V(7)=eps(0,2);//strain13
   //V(8)=eps(1,2);//strain23
}



