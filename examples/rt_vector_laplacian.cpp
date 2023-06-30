#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class RepeatedCoefficient : public VectorCoefficient
{
   Coefficient &coeff;
public:
   RepeatedCoefficient(int dim, Coefficient &coeff_)
      : VectorCoefficient(dim), coeff(coeff_)
   { }
   void Eval(Vector &V, ElementTransformation &T, const IntegrationPoint &ip)
   {
      V.SetSize(vdim);
      V = coeff.Eval(T, ip);
   }
};

double u(const Vector &xvec);
double f(const Vector &xvec);

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

struct GeomDof
{
   const int dof_idx;
   const int local_id;

};

void FillFaceMap(const std::vector<int> &n_face_dofs_per_component,
                 const std::vector<int> offsets,
                 const std::vector<int> &strides,
                 const std::vector<int> &n_dofs_per_dim,
                 Array<int> &face_map)
{
   int total_n_face_dofs = 0;
   for (int n : n_face_dofs_per_component) { total_n_face_dofs += n; }
   face_map.SetSize(total_n_face_dofs);

   const int n_components = offsets.size();
   const int face_dim = strides.size() / n_components;
   int comp_offset = 0;
   for (int comp = 0; comp < n_components; ++comp)
   {
      const int offset = offsets[comp];
      const int n_face_dofs = n_face_dofs_per_component[comp];
      for (int i = 0; i < n_face_dofs; ++i)
      {
         int idx = offset;
         int j = i;
         for (int d = 0; d < face_dim; ++d)
         {
            const int dof1d = n_dofs_per_dim[comp*(face_dim) + d];
            idx += strides[comp*(face_dim) + d]*(j % dof1d);
            j /= dof1d;
         }
         face_map[comp_offset + i] = idx;
      }
      comp_offset += n_face_dofs;
   }
}

void GetRTQuadEdgeMap(const int p, const int face_id, Array<int> &face_map)
{
   const int pp1 = p + 1;

   std::vector<int> n_face_dofs =
   {
      (face_id == 0 || face_id == 2) ? pp1 : p,
      (face_id == 0 || face_id == 2) ? p : pp1
   };
   std::vector<int> offsets;
   std::vector<int> strides;
   switch (face_id)
   {
      case 0: // y = 0
         offsets = {0, p*pp1};
         strides = {1, 1};
         break;
      case 1: // x = 1
         offsets = {pp1 - 1, p*pp1 + p - 1};
         strides = {p+1, p};
         break;
      case 2: // y = 1
         offsets = {pp1*(p-1), p*pp1 + p*(pp1 - 1)};
         strides = {1, 1};
         break;
      case 3: // x = 0
         offsets = {0, p*pp1};
         strides = {p+1, p};
         break;
   }

   std::vector<int> n_dofs = n_face_dofs;
   FillFaceMap(n_face_dofs, offsets, strides, n_dofs, face_map);
}

void GetRTQuadVertexMap(const int p, const int vert_id, Array<int> &vert_map)
{
   const int pp1 = p + 1;
   vert_map.SetSize(2);
   switch (vert_id)
   {
      case 0: // (0,0)
         vert_map[0] = 0;
         vert_map[1] = p*pp1;
         break;
      case 1: // (1,0)
         vert_map[0] = pp1 - 1;
         vert_map[1] = p*pp1 + p - 1;
         break;
      case 2: // (1,1)
         vert_map[0] = p*pp1 - 1;
         vert_map[1] = 2*p*pp1 - 1;
         break;
      case 3: // (0,1)
         vert_map[0] = pp1*(p-1);
         vert_map[1] = p*pp1 + p*(pp1 - 1);
         break;
   }
}

struct GeometricEntity
{
   std::unordered_map<int,int> dofs;
public:
   void AddDOFs(const Array<int> &el_vdofs, const Array<int> &dof_map,
                const Array<int> &local_lex_dofs)
   {
      for (int i = 0; i < local_lex_dofs.Size(); ++i)
      {
         const int s_local_dof = dof_map[local_lex_dofs[i]];
         const int local_dof = (s_local_dof >= 0) ? s_local_dof : -1 - s_local_dof;
         const int s1 = (s_local_dof >= 0) ? 1 : 0;
         const int s_global_dof = el_vdofs[local_dof];
         const int global_dof = (s_global_dof >= 0) ? s_global_dof : -1 - s_global_dof;
         const int s2 = (s_global_dof >= 0) ? 1 : 0;

         const auto dof = dofs.find(global_dof);
         if (dof != dofs.end())
         {

         }
         else
         {
         }
      }
   }
};

struct Entities
{
   std::vector<GeometricEntity> verts;
   std::vector<GeometricEntity> edges;
   std::vector<GeometricEntity> elems;

   Entities(FiniteElementSpace &fes)
      : verts(fes.GetMesh()->GetNV()),
        edges(fes.GetMesh()->GetNEdges()),
        elems(fes.GetMesh()->GetNE())
   {
      Mesh &mesh = *fes.GetMesh();
      const int dim = mesh.Dimension();
      MFEM_VERIFY(dim == 2, "");

      const int nv_per_el = 4;
      const int ne_per_el = 4;

      for (int iel = 0; iel < mesh.GetNE(); ++iel)
      {
         const FiniteElement *fe = fes.GetFE(iel);
         const auto *tbe = dynamic_cast<const TensorBasisElement*>(fe);
         MFEM_VERIFY(tbe, "");
         const Array<int> &dof_map = tbe->GetDofMap();
         const int p = fe->GetOrder();

         Array<int> v, e, o;
         mesh.GetElementVertices(iel, v);
         mesh.GetElementEdges(iel, e, o);

         Array<int> vdofs;
         fes.GetElementVDofs(iel, vdofs);

         for (int d = 0; d < dim; ++d)
         {
            const int nx = (d == 0) ? p + 1 : p;
            const int ny = (d == 1) ? p + 1 : p;

            // vertices
            for (int iv = 0; iv < nv_per_el; ++iv)
            {
               Array<int> vert_map;
               GetRTQuadVertexMap(p, iv, vert_map);
            }

            // edges
         }
      }
   }
};

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command line options.
   const char *mesh_file = "../data/star.mesh";
   const char *vis_vector = "";
   int order = 1;
   int ser_ref = 1;
   int par_ref = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.AddOption(&par_ref, "-vis", "--visualize", "Vector to visualize.");
   args.ParseCheck();

   // std::cout << '\n';

   // for (int iv = 0; iv < 4; ++iv)
   // {
   //    Array<int> vert_map;
   //    GetRTQuadVertexMap(2, iv, vert_map);
   //    std::cout << "Vertex ID " << iv << ":   ";
   //    for (int i = 0; i < vert_map.Size(); ++i)
   //    {
   //       std::cout << vert_map[i];
   //       if (i < vert_map.Size() - 1) { std::cout << ", "; }
   //       else { std::cout << std::endl; }
   //    }
   // }

   // std::cout << '\n';

   // for (int ie = 0; ie < 4; ++ie)
   // {
   //    Array<int> edge_map;
   //    GetRTQuadEdgeMap(2, ie, edge_map);
   //    std::cout << "Edge ID " << ie << ":   ";
   //    for (int i = 0; i < edge_map.Size(); ++i)
   //    {
   //       std::cout << edge_map[i];
   //       if (i < edge_map.Size() - 1) { std::cout << ", "; }
   //       else { std::cout << std::endl; }
   //    }
   // }
   // return 0;

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);

   // 5. Define a finite element space on the mesh. Here we use H1 continuous
   //    high-order Lagrange finite elements of the given order.
   const int dim = mesh.Dimension();
   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLobatto;
   RT_FECollection fec(order-1, dim, b1, b2);
   ParFiniteElementSpace fespace(&mesh, &fec);
   HYPRE_BigInt total_num_dofs = fespace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << total_num_dofs << endl;
   }

   // 6. Extract the list of all the boundary DOFs. These will be marked as
   //    Dirichlet in order to enforce zero boundary conditions.
   Array<int> boundary_dofs;
   fespace.GetBoundaryTrueDofs(boundary_dofs);

   FunctionCoefficient scalar_f_coeff(f), scalar_u_coeff(u);
   RepeatedCoefficient f_coeff(dim, scalar_f_coeff);
   RepeatedCoefficient u_coeff(dim, scalar_u_coeff);

   // 7. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   ParGridFunction x(&fespace);
   x.ProjectCoefficient(u_coeff);

   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_coeff));
   b.Assemble();

   // 9. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
   ParBilinearForm a(&fespace);
   a.AddDomainIntegrator(new VectorFEDiffusionIntegrator);
   a.AddInteriorFaceIntegrator(new VectorFE_DGDiffusionIntegrator(20.0));
   a.AddBdrFaceIntegrator(new VectorFE_DGDiffusionIntegrator(20.0));
   a.Assemble();

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.
   HypreParMatrix A;
   Vector B, X;
   a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

   // SparseMatrix A_diag;
   // A.GetDiag(A_diag);
   // {
   //    std::ofstream f("A.txt");
   //    A_diag.PrintMatlab(f);
   // }

   // 11. Solve the system using PCG with hypre's BoomerAMG preconditioner.
   HypreBoomerAMG M(A);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(M);
   cg.SetOperator(A);
   cg.Mult(B, X);

   a.RecoverFEMSolution(X, b, x);

   ParaViewDataCollection pv("RTLaplacian", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   // pv.SetLevelsOfDetail(order + 1);
   pv.SetLevelsOfDetail(11);
   pv.RegisterField("u", &x);
   pv.SetCycle(0);
   pv.SetTime(0.0);
   pv.Save();

   double er = x.ComputeL2Error(u_coeff);
   if (Mpi::Root()) { cout << "L2 error: " << er << endl; }

   Vector one_vec(dim);
   one_vec = 1.0;
   VectorConstantCoefficient one(one_vec);
   x.ProjectCoefficient(one);
   for (double &xi : x) { xi = 1.0 / xi; }

   SparseMatrix x_diag(x);
   HypreParMatrix D(MPI_COMM_WORLD, fespace.GlobalTrueVSize(),
                    fespace.GetTrueDofOffsets(), &x_diag);
   HypreParMatrix *DtAD = RAP(&A, &D);

   {
      HypreBoomerAMG M(*DtAD);
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(2000);
      cg.SetPrintLevel(1);
      cg.SetPreconditioner(M);
      cg.SetOperator(*DtAD);
      B.Randomize(1);
      X = 0.0;
      cg.Mult(B, X);
   }

   // x = 0.0;
   // for (int i = 0; i < x.Size(); ++ i)
   // {
   //    x[i] = 1.0;
   //    pv.SetCycle(pv.GetCycle() + 1);
   //    pv.SetTime(pv.GetTime() + 1);
   //    pv.Save();
   //    x[i] = 0.0;
   // }

   return 0;
}

constexpr double pi = M_PI, pi2 = pi*pi;

double u(const Vector &xvec)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];
   if (dim == 2) { return sin(x)*sin(y); }
   else { double z = pi*xvec[2]; return sin(x)*sin(y)*sin(z); }
}

double f(const Vector &xvec)
{
   int dim = xvec.Size();
   double x = pi*xvec[0], y = pi*xvec[1];

   if (dim == 2)
   {
      return 2*pi2*sin(x)*sin(y);
   }
   else // dim == 3
   {
      double z = pi*xvec[2];
      return 3*pi2*sin(x)*sin(y)*sin(z);
   }
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
