#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

ParMesh LoadParMesh(const char *mesh_file, int ser_ref, int par_ref);

struct LocalIndexOrientation
{
   const int index, orientation;
   LocalIndexOrientation(const int i, const int o) : index(i), orientation(o) { }
};

struct Entity
{
   const Mesh &mesh;
   const int dim;
   const int entity_index;

   Entity(const Mesh &mesh_, const int dim_, const int entity_index_)
      : mesh(mesh_), dim(dim_), entity_index(entity_index_) { }

   Geometry::Type GetGeometry() const
   {
      switch (dim)
      {
         case 0: return Geometry::POINT;
         case 1: return Geometry::SEGMENT;
         case 2: return Geometry::SQUARE;
         case 3: return Geometry::CUBE;
         default: MFEM_ABORT("Invalid entity dimension.")
      }
      return Geometry::INVALID;
   }

   const IntegrationRule &GetIntegrationRule(const int npts) const
   {
      const int order = (dim == 0) ? 1 : 2*npts - 1;
      return IntRules.Get(GetGeometry(), order);
   }

   LocalIndexOrientation GetLocalIndexOrientation(const int element_index) const
   {
      const int mesh_dim = mesh.Dimension();
      Array<int> indices, orientations;

      if (dim == 0)
      {
         mesh.GetElementVertices(element_index, indices);
      }
      else if (dim == mesh_dim)
      {
         return LocalIndexOrientation(0, 0);
      }
      else if (dim == 1)
      {
         mesh.GetElementEdges(element_index, indices, orientations);
      }
      else if (dim == 2)
      {
         mesh.GetElementFaces(element_index, indices, orientations);
      }

      auto found = std::find(indices.begin(), indices.end(), entity_index);
      MFEM_VERIFY(found != indices.end(), "Did not find index.");
      const int i = std::distance(indices.begin(), found);
      const int o = (orientations.Size() > 0) ? orientations[i] : 0;
      return LocalIndexOrientation(i, o);
   }

   IntegrationPoint Transform(const IntegrationPoint &ip,
                              const LocalIndexOrientation &io) const
   {
      MFEM_VERIFY(mesh.Dimension() == 2, "Unsupported");

      if (dim == 0)
      {
         const Geometry::Type geom = mesh.GetElementGeometry(0);
         return (*Geometries.GetVertices(geom))[io.index];
      }
      else if (dim == 1)
      {
         IntegrationPoint ipt;
         ipt.Init(0);
         const double x1 = (io.orientation > 0) ? ip.x : 1.0 - ip.x;
         const double x2 = (io.index == 0 || io.index == 3) ? 0.0 : 1.0;
         if (io.index == 0 || io.index == 2)
         {
            ipt.x = x1;
            ipt.y = x2;
         }
         else
         {
            ipt.x = x2;
            ipt.y = x1;
         }
         return ipt;
      }
      else // dim == 2
      {
         // No need to transform element interior DOFs
         return ip;
      }
   }
};

struct ElementEntityDofs
{
   const Entity &entity;
   const int element_index;
   LocalIndexOrientation local_index_orientation;
   Array<int> local_dofs;
   Array<int> global_dofs;
   IntegrationRule ir_el;

   ElementEntityDofs(const FiniteElementSpace &fes, const Entity &entity_,
                     const int element_index_)
      : entity(entity_),
        element_index(element_index_),
        local_index_orientation(entity.GetLocalIndexOrientation(element_index))
   {
      Array<int> element_dofs;
      fes.GetElementDofs(element_index, element_dofs);

      const Mesh &mesh = *fes.GetMesh();
      const int dim = mesh.Dimension();

      const FiniteElement *fe = fes.GetFE(element_index);
      auto tfe = dynamic_cast<const TensorBasisElement*>(fe);
      MFEM_VERIFY(tfe != NULL, "Only tensor elements supported.");
      const auto dof_map = tfe->GetDofMap();

      const int p = fes.GetElementOrder(element_index);
      const int loc = local_index_orientation.index;

      // Set the local DOFs
      MFEM_VERIFY(dim == 2, "Not yet implemented.");
      if (entity.dim == 0)
      {
         // 3 -- 2
         // |    |
         // 0 -- 1
         local_dofs.SetSize(2);
         const int x_offset = (loc == 0 || loc == 3) ? 0 : 1;
         const int y_offset = (loc == 0 || loc == 1) ? 0 : 1;

         const int ndof_per_dim = p*(p+1);

         local_dofs[0] = x_offset*p + y_offset*(p+1)*(p-1);
         local_dofs[1] = ndof_per_dim + x_offset*(p-1) + y_offset*p*p;
      }
      else if (entity.dim == 1)
      {
         // * - 2 - *
         // |       |
         // 3       1
         // |       |
         // * - 0 - *
         local_dofs.SetSize(2*p + 1);

         // direction of edge (parallel to x or y axis?)
         const int dir = (loc == 0 || loc == 2) ? 0 : 1;

         const int stride_x = (dir == 0) ? 1 : p + 1;
         const int stride_y = (dir == 0) ? 1 : p;

         // Subtract two since excluding the vertices
         const int nx = ((dir == 0) ? p + 1 : p) - 2;
         const int ny = ((dir == 0) ? p : p + 1) - 2;

         local_dofs.SetSize(nx + ny);

         const int offset_x = (loc == 1) ? p : (loc == 2) ? (p+1)*(p-1) : 0;

         for (int ix = 0; ix < nx; ++ ix)
         {
            // (ix + 1) below since skipping the first DOF (vertex)
            local_dofs[ix] = offset_x + (ix+1)*stride_x;
         }

         const int ndof_per_dim = p*(p+1);

         const int offset_y = (loc == 1) ? p - 1 : (loc == 2) ? p*p : 0;
         for (int iy = 0; iy < ny; ++iy)
         {
            // (iy + 1) below since skipping the first DOF (vertex)
            local_dofs[nx + iy] = ndof_per_dim + offset_y + (iy+1)*stride_y;
         }
      }
      else if (entity.dim == 2)
      {
         // Exclude edges --- only include interior element DOFs
         // local_dofs.SetSize(2*(p-1)*(p-2));
         if (p > 2) { MFEM_ABORT("Not implemented."); }
      }

      auto set_global_dofs = [&](Array<int> &local, Array<int> &global)
      {
         global.SetSize(local.Size());
         for (int i = 0; i < local.Size(); ++i)
         {
            const int sd = dof_map[local[i]];
            const int d = (sd >= 0) ? sd : -1 - sd;
            // Remap from lexicographic to MFEM-native ordering
            local[i] = d;
            // global will incorporate the sign (orientation) of the DOF
            global[i] = element_dofs[d];
         }
      };

      set_global_dofs(local_dofs, global_dofs);

      // Set up the integration rules
      const int ndof = local_dofs.Size();
      const int npts = std::ceil(0.5*ndof);
      const IntegrationRule &ir = entity.GetIntegrationRule(npts);
      ir_el.SetSize(ir.Size());

      for (int i = 0; i < ir.Size(); ++i)
      {
         ir_el[i] = entity.Transform(ir[i], local_index_orientation);
      }

      /// Debug, delete me:
      cout << "Entity " << entity.entity_index << ":\n";
      cout << "    " << "Dimension: " << entity.dim << '\n';
      cout << "    " << "DOFs: ";
      for (int i = 0; i < global_dofs.Size(); ++i)
      {
         const int s_d = global_dofs[i];
         const int d = (s_d >= 0) ? s_d : -1 - s_d;
         std::cout << d;
         if (s_d < 0) { std::cout << "*"; }
         if (i < global_dofs.Size() - 1) { std::cout << ", "; }
         else { std::cout << std::endl; }
      }
   }

   DenseMatrix FormDofMatrix(const FiniteElementSpace &fes,
                             const IntegrationRule &ir) const
   {
      const int ndof = local_dofs.Size();
      const int npts = ir.Size();

      const FiniteElement &fe = *fes.GetFE(element_index);
      const int sdim = fe.GetDim();
      ElementTransformation &T = *fes.GetElementTransformation(element_index);
      DenseMatrix shape(fe.GetDof(), sdim);
      DenseMatrix V(npts*sdim, ndof);
      for (int i = 0; i < npts; ++i)
      {
         const IntegrationPoint &ip = ir[i];
         T.SetIntPoint(&ip);
         fe.CalcPhysVShape(T, shape);
         for (int j = 0; j < ndof; ++j)
         {
            for (int d = 0; d < sdim; ++d)
            {
               V(d + i*sdim, j) = shape(local_dofs[j], d);
            }
         }
      }
      return V;
   }

   DenseMatrix FormDofMatrix(const FiniteElementSpace &fes) const
   {
      return FormDofMatrix(fes, ir_el);
   }
};

struct PrimaryElementEntity
{
   const Entity &entity;
   const ElementEntityDofs dofs;
   const int dof_offset;

   std::vector<std::reference_wrapper<const PrimaryElementEntity>>
                                                                primary_children;
   std::vector<std::reference_wrapper<const struct ConstrainedElementEntity>>
                                                                              constrained_children;

   // For elements, no entity list is required
   PrimaryElementEntity(const FiniteElementSpace &fes, Entity &entity_,
                        int element_index, int dof_offset_)
      : entity(entity_),
        dofs(fes, entity, element_index),
        dof_offset(dof_offset_)
   { }

   PrimaryElementEntity(const FiniteElementSpace &fes, Entity &entity_,
                        int element_index, int dof_offset_,
                        const std::vector<struct EntityDofs> &entities)
      : PrimaryElementEntity(fes, entity_, element_index, dof_offset_)
   {
      // This is an edge: add the vertices as children
      if (entity.dim == 1)
      {
         Array<int> v;
         fes.GetMesh()->GetEdgeVertices(entity.entity_index, v);
         for (const int iv : v)
         {
            AddChild(entities[iv]);
         }
      }
   }

   void AddChild(const struct EntityDofs &child_entity);

   int GetNumChildDofs() const;
};

struct ConstrainedElementEntity
{
   const Entity &entity;
   const PrimaryElementEntity &primary;
   ElementEntityDofs dofs;
   DenseMatrix Pi;
   std::vector<DenseMatrix> Pi_sub_1, Pi_sub_2;

   ConstrainedElementEntity(const FiniteElementSpace &fes,
                            const PrimaryElementEntity &primary_, int element_index)
      : entity(primary_.entity),
        primary(primary_),
        dofs(fes, entity, element_index)
   {
      DenseMatrix Phi = dofs.FormDofMatrix(fes);
      const int ndof1 = Phi.Width();
      Phi.Transpose();
      DenseMatrix PhiTPhi(ndof1, ndof1);
      MultAAt(Phi, PhiTPhi);
      DenseMatrixInverse PhiTPhi_inv(PhiTPhi);

      auto compute_constraint_matrix = [&](const DenseMatrix &Psi)
      {
         const int ndof2 = Psi.Width();
         DenseMatrix PhiTPsi(ndof1, ndof2);
         Mult(Phi, Psi, PhiTPsi);
         DenseMatrix C(ndof1, ndof2);
         PhiTPhi_inv.Mult(PhiTPsi, C);
         return C;
      };

      const IntegrationRule &ir_el = primary.dofs.ir_el;
      Pi = compute_constraint_matrix(primary.dofs.FormDofMatrix(fes, ir_el));

      for (const PrimaryElementEntity &primary_child : primary.primary_children)
      {
         Pi_sub_1.push_back(compute_constraint_matrix(
                               primary_child.dofs.FormDofMatrix(fes, ir_el)));
      }

      for (const ConstrainedElementEntity &constrained_child :
           primary.constrained_children)
      {
         const DenseMatrix &C1 = constrained_child.Pi;
         DenseMatrix C2 = compute_constraint_matrix(
                             constrained_child.dofs.FormDofMatrix(fes, ir_el));
         const int ndof2 = C1.Width();
         Pi_sub_2.emplace_back(ndof1, ndof2);
         Mult(C2, C1, Pi_sub_2.back());
      }

      Pi.Print(std::cout, 20);
   }
};

struct EntityDofs
{
   Entity entity;
   PrimaryElementEntity primary;
   std::vector<ConstrainedElementEntity> constrained;

   EntityDofs(const FiniteElementSpace &fes, int entity_index, int entity_dim,
              const Table &el_table, int dof_offset, const std::vector<EntityDofs> &entities)
      : entity(*fes.GetMesh(), entity_dim, entity_index),
        primary(fes, entity, GetPrimaryIndex(el_table, entity_index), dof_offset,
                entities)
   {
      Array<int> row;
      el_table.GetRow(entity_index, row);
      for (int i = 1; i < row.Size(); ++i)
      {
         constrained.emplace_back(fes, primary, row[i]);
      }
   }

   // Version of the constructor for elements
   EntityDofs(const FiniteElementSpace &fes, int element_index, int dof_offset)
      : entity(*fes.GetMesh(), fes.GetMesh()->Dimension(), element_index),
        primary(fes, entity, element_index, dof_offset)
   { }

   int GetNPrimaryDofs() const
   {
      return primary.dofs.local_dofs.Size();
   }

private:
   static int GetPrimaryIndex(const Table &table, int index)
   {
      Array<int> row;
      table.GetRow(index, row);
      MFEM_VERIFY(row.Size() > 0, "");
      return row[0];
   }
};

class RT_ContinuityConstraints
{
public: // temporary
   std::vector<EntityDofs> entities;
   int n_primary_dofs;
   FiniteElementSpace &fes;
   SparseMatrix P;
public:
   RT_ContinuityConstraints(FiniteElementSpace &fes_) : fes(fes_)
   {
      MFEM_VERIFY(fes.GetMaxElementOrder() >= 2, "Order must be at least 2.");
      Mesh &mesh = *fes.GetMesh();
      const int dim = mesh.Dimension();
      int nentities = mesh.GetNV() + mesh.GetNE();
      if (dim >= 2) { nentities += mesh.GetNEdges(); }
      if (dim == 3) { nentities += mesh.GetNFaces(); }
      entities.reserve(nentities);

      MFEM_VERIFY(dim == 2, "3D not yet implemented.");

      n_primary_dofs = 0;

      std::unique_ptr<Table> el_table(mesh.GetVertexToElementTable());
      for (int iv = 0; iv < mesh.GetNV(); ++iv)
      {
         entities.emplace_back(fes, iv, 0, *el_table, n_primary_dofs, entities);
         n_primary_dofs += entities.back().GetNPrimaryDofs();
      }

      el_table.reset(Transpose(mesh.ElementToEdgeTable()));
      for (int ie = 0; ie < mesh.GetNEdges(); ++ie)
      {
         entities.emplace_back(fes, ie, 1, *el_table, n_primary_dofs, entities);
         n_primary_dofs += entities.back().GetNPrimaryDofs();
      }

      for (int iel = 0; iel < mesh.GetNE(); ++iel)
      {
         entities.emplace_back(fes, iel, n_primary_dofs);
         n_primary_dofs += entities.back().GetNPrimaryDofs();
      }
   }

   void FormProlongationMatrix()
   {
      SparseMatrix P_(fes.GetTrueVSize(), n_primary_dofs);

      for (const EntityDofs &entity : entities)
      {
         const PrimaryElementEntity &primary = entity.primary;
         // Set identity for primary DOFs
         for (int k = 0; k < primary.dofs.global_dofs.Size(); ++k)
         {
            const int j = primary.dof_offset + k;
            const int s_i = primary.dofs.global_dofs[k];
            // const int s = (s_i >= 0) ? 1 : -1;
            const int i = (s_i >= 0) ? s_i : -1 - s_i;

            // Debug check...
            // const double val = P_.SearchRow(i, j);
            // MFEM_ASSERT(val == 0 || val == s, "");

            // P_.Set(i, j, s);

            const double val = P_.SearchRow(i, j);
            MFEM_ASSERT(val == 0.0 || val == 1.0, "");

            std::cout << "Setting (" << i << "," << j << ") = " << 1 << '\n';
            P_.Set(i, j, 1.0);
         }

         // Place the interpolation matrices for constrained DOFs
         for (const ConstrainedElementEntity &constrained : entity.constrained)
         {
            auto set_constraint_submatrix = [&](const DenseMatrix &Pi,
                                                const PrimaryElementEntity &p)
            {
               const int ndof1 = Pi.Height();
               const int ndof2 = Pi.Width();

               MFEM_VERIFY(ndof1 == constrained.dofs.local_dofs.Size(), "");
               MFEM_VERIFY(ndof2 == p.dofs.local_dofs.Size(), "");

               for (int kj = 0; kj < ndof2; ++kj)
               {
                  const int j_s = p.dofs.global_dofs[kj];
                  const int s1 = (j_s >= 0) ? 1 : -1;
                  const int j = p.dof_offset + kj;
                  for (int ki = 0; ki < ndof1; ++ki)
                  {
                     const int i_s = constrained.dofs.global_dofs[ki];
                     const int s2 = (i_s >= 0) ? 1 : -1;
                     const int i = (i_s >= 0) ? i_s : -1 - i_s;

                     const double val = Pi(ki, kj)*s1*s2;

                     const double val_prev = P_.SearchRow(i, j);
                     MFEM_ASSERT(val_prev == 0.0 || std::abs(val - val_prev) < 1e-10, "");

                     std::cout << "Setting (" << i << "," << j << ") = " << val << '\n';
                     P_.Set(i, j, val);
                  }
               }
            };

            set_constraint_submatrix(constrained.Pi, primary);
            for (int i = 0; i < primary.primary_children.size(); ++i)
            {
               set_constraint_submatrix(constrained.Pi_sub_1[i], primary.primary_children[i]);
            }
            for (int i = 0; i < primary.constrained_children.size(); ++i)
            {
               const ConstrainedElementEntity &c = primary.constrained_children[i];
               set_constraint_submatrix(constrained.Pi_sub_2[i], c.primary);
            }
         }
      }

      P_.Finalize();
      P.Swap(P_);
   }

   SparseMatrix &GetProlongationMatrix()
   {
      if (P.Empty()) { FormProlongationMatrix(); }
      return P;
   }
};

void PrimaryElementEntity::AddChild(const struct EntityDofs &child_entity)
{
   const int iel = dofs.element_index;
   if (child_entity.primary.dofs.element_index == iel)
   {
      primary_children.push_back(child_entity.primary);
   }
   else
   {
      for (const ConstrainedElementEntity &constrained : child_entity.constrained)
      {
         if (constrained.dofs.element_index == iel)
         {
            constrained_children.push_back(constrained);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();

   const char *mesh_file = "../data/star.mesh";
   int order = 2;
   int ser_ref = 0;
   int par_ref = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ser_ref, "-rs", "--serial-refine",
                  "Number of times to refine the mesh in serial.");
   args.AddOption(&par_ref, "-rp", "--parallel-refine",
                  "Number of times to refine the mesh in parallel.");
   args.ParseCheck();

   ParMesh mesh = LoadParMesh(mesh_file, ser_ref, par_ref);

   const int dim = mesh.Dimension();
   const int b1 = BasisType::GaussLobatto, b2 = BasisType::GaussLobatto;
   RT_FECollection fec(order-1, dim, b1, b2);
   ParFiniteElementSpace fes(&mesh, &fec);

   HYPRE_BigInt total_num_dofs = fes.GlobalTrueVSize();

   RT_ContinuityConstraints constraints(fes);

   cout << "Number of total DOFs:  " << total_num_dofs << '\n';
   cout << "Number of primary DOFs:" << constraints.n_primary_dofs << '\n';
   cout << "Number of entities:    " << constraints.entities.size() << '\n';

   for (int i = 0; i < constraints.entities.size(); ++i)
   {
      cout << "Entity " << i << ":\n";

      cout << "    " << "Dimension: " << constraints.entities[i].entity.dim << '\n';
      cout << "    " << "Number of DOFs: " <<
           constraints.entities[i].GetNPrimaryDofs() << '\n';
   }

   SparseMatrix &P = constraints.GetProlongationMatrix();
   {
      std::ofstream f("P.txt");
      P.PrintMatlab(f);
   }

   HYPRE_BigInt row_starts[2] = {0, P.Height()};
   HYPRE_BigInt col_starts[2] = {0, P.Width()};
   HypreParMatrix P_par(MPI_COMM_WORLD, P.Height(), P.Width(), row_starts,
                        col_starts, &P);
   P_par.Print("P.txt");

   Array<int> boundary_dofs;
   fes.GetBoundaryTrueDofs(boundary_dofs);

   FunctionCoefficient scalar_f_coeff(f), scalar_u_coeff(u);
   RepeatedCoefficient f_coeff(dim, scalar_f_coeff);
   RepeatedCoefficient u_coeff(dim, scalar_u_coeff);

   // 7. Define the solution x as a finite element grid function in fespace. Set
   //    the initial guess to zero, which also sets the boundary conditions.
   ParGridFunction x(&fes);
   x.ProjectCoefficient(u_coeff);

   // 8. Set up the linear form b(.) corresponding to the right-hand side.
   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(f_coeff));
   b.Assemble();

   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new VectorFEDiffusionIntegrator);
   a.AddInteriorFaceIntegrator(new VectorFE_DGDiffusionIntegrator(20.0));
   a.AddBdrFaceIntegrator(new VectorFE_DGDiffusionIntegrator(20.0));
   a.Assemble();

   // 10. Form the linear system A X = B. This includes eliminating boundary
   //     conditions, applying AMR constraints, parallel assembly, etc.
   HypreParMatrix A;
   a.FormSystemMatrix(boundary_dofs, A);
   A.Print("A.txt");

   std::unique_ptr<HypreParMatrix> A0(RAP(&A, &P_par));
   A0->Print("A0.txt");

   HypreBoomerAMG amg(*A0);
   CGSolver cg(MPI_COMM_WORLD);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(2000);
   cg.SetPrintLevel(1);
   cg.SetPreconditioner(amg);
   cg.SetOperator(*A0);

   Vector B(P.Width());
   P.MultTranspose(b, B);

   Vector X(P.Width());
   X = 0.0;
   cg.Mult(B, X);

   P.Mult(X, x);

   ParaViewDataCollection pv("RTProlongation", &mesh);
   pv.SetPrefixPath("ParaView");
   pv.SetHighOrderOutput(true);
   pv.SetLevelsOfDetail(order + 1);
   pv.RegisterField("u", &x);
   pv.SetCycle(0);
   pv.SetTime(0.0);
   pv.Save();

   X = 0.0;
   for (int i = 0; i < X.Size(); ++ i)
   {
      X[i] = 1.0;
      P.Mult(X, x);
      pv.SetCycle(pv.GetCycle() + 1);
      pv.SetTime(pv.GetTime() + 1);
      pv.Save();
      X[i] = 0.0;
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
