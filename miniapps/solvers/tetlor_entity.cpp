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

#include "tetlor_entity.hpp"
#include <algorithm>
#include <unordered_map>

namespace tetlor
{
namespace
{

// Free DOFs on the closed entity, interior first. Each of its two lists is
// ordered by true DOF number so incident cells and ranks assemble the same form.
struct EntityDofs
{
   Array<int> all;
   int interior = 0;

   void Init(FiniteElementSpace &fes, int dim, int id, const Array<int> &essential)
   {
      Array<int> dofs;
      switch (dim)
      {
         case 0: fes.GetVertexDofs(id,dofs); break;
         case 1: fes.GetEdgeDofs(id,dofs); break;
         case 2: fes.GetFaceDofs(id,dofs); break;
         case 3: fes.GetElementDofs(id,dofs); break;
      }
      const Geometry::Type geom[] = {Geometry::POINT, Geometry::SEGMENT,
                                     Geometry::TRIANGLE, Geometry::TETRAHEDRON
                                    };
      const int boundary = dofs.Size()-fes.FEColl()->DofForGeometry(geom[dim]);
      for (int j = boundary; j < dofs.Size(); ++j)
      {
         if (!essential[dofs[j]]) { all.Append(dofs[j]); }
      }
      interior = all.Size();
      for (int j = 0; j < boundary; ++j)
      {
         if (!essential[dofs[j]]) { all.Append(dofs[j]); }
      }
      if (all.Size() == 0) { return; }
      auto number = [&](int d) -> long long
      {
#ifdef MFEM_USE_MPI
         if (auto *pf = dynamic_cast<ParFiniteElementSpace*>(&fes))
         { return pf->GetGlobalTDofNumber(d); }
#endif
         return d;
      };
      auto less = [&](int a, int b) { return number(a) < number(b); };
      std::sort(all.begin(), all.begin()+interior, less);
      std::sort(all.begin()+interior, all.end(), less);
   }
};

struct Entity
{
   EntityDofs low, high;
   DenseMatrix low_energy, high_energy, sampling;
   DenseMatrix from_low, from_high;
   bool owner = true;

   void BuildMaps()
   {
      if (owner)
      {
         entity::PolarBlocks(low_energy,high_energy,sampling,from_low,from_high);
      }
      else
      {
         from_low.SetSize(high.interior,low.all.Size());
         from_high.SetSize(high.interior,high.all.Size()-high.interior);
      }
   }
};

Array<int> BoundaryMarker(const FiniteElementSpace &fes)
{
   const auto &attr = fes.GetMesh()->bdr_attributes;
   Array<int> boundary(attr.Size() ? attr.Max() : 0), marker;
   boundary = 1;
   fes.GetEssentialVDofs(boundary,marker);
   marker.HostRead();
   return marker;
}

// One contribution, indexed by rank-local DOF. Eliminate the cell interior
// once before condensing to individual edges and faces.
class CellEnergy
{
   DenseMatrix K;
   std::unordered_map<int,int> index;

public:
   CellEnergy(DenseMatrix matrix, const Array<int> &dofs) : K(std::move(matrix))
   {
      for (int i = 0; i < dofs.Size(); ++i) { index.emplace(dofs[i],i); }
   }

   Array<int> Indices(const Array<int> &dofs) const
   {
      Array<int> ids(dofs.Size());
      for (int i = 0; i < ids.Size(); ++i) { ids[i] = index.at(dofs[i]); }
      return ids;
   }

   DenseMatrix FreeBlock(const EntityDofs &cell) const
   {
      DenseMatrix block;
      K.GetSubMatrix(Indices(cell.all),block);
      return block;
   }

   CellEnergy Skeleton(const EntityDofs &cell) const
   {
      Array<int> eliminate, keep;
      for (int i = 0; i < cell.all.Size(); ++i)
      {
         (i < cell.interior ? eliminate : keep).Append(cell.all[i]);
      }
      if (!keep.Size()) { return CellEnergy(DenseMatrix(),keep); }
      return CellEnergy(entity::Condense(K,Indices(keep),Indices(eliminate)),keep);
   }

   void AddTrace(const EntityDofs &entity_dofs, DenseMatrix &sum) const
   {
      Array<int> keep = Indices(entity_dofs.all), mask(K.Height()), eliminate;
      mask = 0;
      for (int i : keep) { mask[i] = 1; }
      for (int i = 0; i < mask.Size(); ++i) { if (!mask[i]) { eliminate.Append(i); } }
      DenseMatrix S = entity::Condense(K,keep,eliminate);
      if (sum.Height()) { sum += S; }
      else { sum = std::move(S); }
   }
};

// The nonlinear setup (Schur complement and polar factor) produces linear
// stages. Each stage writes distinct interior DOFs and reads completed traces.
class Stage
{
   Array<int> output;
   SparseMatrix low, lift;
   mutable Vector values;

public:
   Stage(int n, int nh, int nl) : low(n,nl), lift(n,nh), values(n)
   { values.UseDevice(true); }

   void Add(const Entity &e, const DenseMatrix &from_low,
            const DenseMatrix &from_high)
   {
      for (int i = 0; i < e.high.interior; ++i)
      {
         const int row = output.Size();
         output.Append(e.high.all[i]);
         for (int j = 0; j < e.low.all.Size(); ++j)
         { low.Set(row,e.low.all[j],from_low(i,j)); }
         for (int j = 0; j < from_high.Width(); ++j)
         { lift.Set(row,e.high.all[e.high.interior+j],from_high(i,j)); }
      }
   }

   void Finalize()
   {
      MFEM_VERIFY(output.Size() == low.Height(), "Missing entity transfer rows.");
      low.Finalize();
      lift.Finalize();
      if (low.NumNonZeroElems()) { low.BuildTranspose(); }
      if (lift.NumNonZeroElems()) { lift.BuildTranspose(); }
   }

   void Forward(const Vector &x, Vector &y) const
   {
      if (!output.Size()) { return; }
      low.Mult(x,values);
      if (lift.NumNonZeroElems()) { lift.AddMult(y,values); }
      y.SetSubVector(output,values);
   }

   void Reverse(Vector &adjoint, Vector &y) const
   {
      if (!output.Size()) { return; }
      adjoint.GetSubVector(output,values);
      low.AddMultTranspose(values,y);
      if (lift.NumNonZeroElems()) { lift.AddMultTranspose(values,adjoint); }
   }
};

// Sample native Bernstein basis functions at canonical LOR vertices.
DenseMatrix CellSampling(const FiniteElement &fe, BatchedLOR &lor, int e)
{
   const Mesh &patch = lor.ReferencePatch();
   const int *perm = lor.VertexPermutation(e);
   DenseMatrix sample(patch.GetNV(),fe.GetDof());
   Vector shape(fe.GetDof());
   for (int j = 0; j < patch.GetNV(); ++j)
   {
      const auto x = patch.GetVertex(j);
      const real_t b[] = {1-x[0]-x[1]-x[2], x[0], x[1], x[2]};
      IntegrationPoint ip;
      ip.Set3(b[perm[1]],b[perm[2]],b[perm[3]]);
      fe.CalcShape(ip,shape);
      sample.SetRow(j,shape);
   }
   return sample;
}

void SetSampling(Entity &e, const CellEnergy &lo, const CellEnergy &hi,
                 const DenseMatrix &sample)
{
   if (e.sampling.Height()) { return; }
   auto rows = lo.Indices(e.low.all), cols = hi.Indices(e.high.all);
   rows.SetSize(e.low.interior);
   cols.SetSize(e.high.interior);
   sample.GetSubMatrix(rows,cols,e.sampling);
}

#ifdef MFEM_USE_MPI
// ParMesh supplies a common order of entities in each sharing group; the
// matrices within an entity are already in a common true-DOF order.
class SharedEntities
{
   ParMesh &mesh;
   std::vector<std::pair<int,Entity*>> shared;

public:
   SharedEntities(ParMesh &mesh_, int dim, std::vector<Entity> &entities)
      : mesh(mesh_)
   {
      for (int g = 1; g < mesh.GetNGroups(); ++g)
      {
         const int count = dim == 1 ? mesh.GroupNEdges(g) : mesh.GroupNTriangles(g);
         for (int j = 0; j < count; ++j)
         {
            const int id = dim == 1 ? mesh.GroupEdge(g,j) : mesh.GroupTriangle(g,j);
            Entity &e = entities[id];
            if (!e.high.interior) { continue; }
            e.owner = mesh.gtopo.IAmMaster(g);
            shared.emplace_back(g,&e);
         }
      }
   }

   void Exchange(bool reduce_forms) const
   {
      Array<int> groups;
      Array<real_t> entries;
      std::vector<DenseMatrix*> blocks;
      for (auto [group, e] : shared)
      {
         DenseMatrix *low = reduce_forms ? &e->low_energy : &e->from_low;
         DenseMatrix *high = reduce_forms ? &e->high_energy : &e->from_high;
         for (DenseMatrix *S : {low,high})
         {
            const int n = S->Height()*S->Width();
            if (!n) { continue; }
            blocks.push_back(S);
            const int offset = entries.Size();
            entries.SetSize(offset+n);
            if (reduce_forms || e->owner)
            { std::copy_n(S->Data(),n,entries.GetData()+offset); }
            for (int k = 0; k < n; ++k) { groups.Append(group); }
         }
      }
      GroupCommunicator comm(mesh.gtopo);
      comm.Create(groups);
      if (reduce_forms) { comm.Reduce(entries,GroupCommunicator::Sum); }
      else { comm.Bcast(entries); }
      int offset = 0;
      for (auto *S : blocks)
      {
         const int n = S->Height()*S->Width();
         std::copy_n(entries.GetData()+offset,n,S->Data());
         offset += n;
      }
   }
};
#endif

void BuildTraceStage(Mesh &mesh, int dim, std::vector<Entity> &entities,
                     Stage &stage)
{
#ifdef MFEM_USE_MPI
   std::unique_ptr<SharedEntities> shared;
   if (auto *pm = dynamic_cast<ParMesh*>(&mesh))
   {
      shared = std::make_unique<SharedEntities>(*pm,dim,entities);
      shared->Exchange(true);
   }
#endif
   for (auto &e : entities) { if (e.high.interior) { e.BuildMaps(); } }
#ifdef MFEM_USE_MPI
   if (shared) { shared->Exchange(false); }
#endif
   for (const auto &e : entities)
   {
      if (e.high.interior) { stage.Add(e,e.from_low,e.from_high); }
   }
}

void AddCell(int k, FiniteElementSpace &high, BatchedLOR &lor,
             const Array<int> &eh, const Array<int> &el,
             std::array<std::vector<Entity>,3> &entities, Stage &volume)
{
   Mesh &mesh = *high.GetMesh();
   const int nl = lor.ReferencePatch().GetNV();
   DiffusionIntegrator diffusion;
   DenseMatrix A, H;
   lor.GetPatchStiffness(k,A);
   diffusion.AssembleElementMatrix(*high.GetFE(k),
                                   *mesh.GetElementTransformation(k),H);
   Array<int> hd, ld(nl);
   high.GetElementDofs(k,hd);
   std::copy_n(lor.PatchDofs().HostRead()+k*nl,nl,ld.begin());
   CellEnergy hi(std::move(H),hd), lo(std::move(A),ld);
   DenseMatrix sample = CellSampling(*high.GetFE(k),lor,k);

   Entity cell;
   cell.high.Init(high,3,k,eh);
   cell.low.Init(lor.Space(),3,k,el);
   if (cell.high.interior)
   {
      cell.high_energy = hi.FreeBlock(cell.high);
      cell.low_energy = lo.FreeBlock(cell.low);
      SetSampling(cell,lo,hi,sample);
      cell.BuildMaps();
      volume.Add(cell,cell.from_low,cell.from_high);
   }

   // Condense onto the free skeleton before forming incident trace forms.
   Array<int> edges, faces, orientation;
   mesh.GetElementEdges(k,edges,orientation);
   mesh.GetElementFaces(k,faces,orientation);
   CellEnergy hi_skeleton = hi.Skeleton(cell.high);
   CellEnergy lo_skeleton = lo.Skeleton(cell.low);
   for (int dim : {1,2})
   {
      for (int id : (dim == 1 ? edges : faces))
      {
         auto &e = entities[dim][id];
         if (!e.high.interior) { continue; }
         hi_skeleton.AddTrace(e.high,e.high_energy);
         lo_skeleton.AddTrace(e.low,e.low_energy);
         SetSampling(e,lo,hi,sample);
      }
   }
}

class EntityTransfer : public Operator
{
   std::array<std::unique_ptr<Stage>,4> stages;
   mutable Vector adjoint;

public:
   EntityTransfer(FiniteElementSpace &high, BatchedLOR &lor);
   MemoryClass GetMemoryClass() const override { return Device::GetMemoryClass(); }
   void Mult(const Vector &x, Vector &y) const override
   {
      y.UseDevice(true);
      y = 0.0;
      for (const auto &stage : stages) { stage->Forward(x,y); }
   }
   void MultTranspose(const Vector &x, Vector &y) const override
   {
      y.UseDevice(true);
      y = 0.0;
      adjoint = x;
      for (int d = 3; d >= 0; --d) { stages[d]->Reverse(adjoint,y); }
   }
};

EntityTransfer::EntityTransfer(FiniteElementSpace &high, BatchedLOR &lor)
   : Operator(high.GetVSize(),lor.Space().GetVSize()), adjoint(height)
{
   adjoint.UseDevice(true);
   Mesh &mesh = *high.GetMesh();
   const int counts[] = {mesh.GetNV(),mesh.GetNEdges(),mesh.GetNFaces()};
   const auto eh = BoundaryMarker(high), el = BoundaryMarker(lor.Space());
   std::array<std::vector<Entity>,3> entities;
   for (int d = 0; d < 3; ++d)
   {
      entities[d].resize(counts[d]);
      int rows = 0;
      for (int id = 0; id < counts[d]; ++id)
      {
         auto &e = entities[d][id];
         e.high.Init(high,d,id,eh);
         e.low.Init(lor.Space(),d,id,el);
         rows += e.high.interior;
      }
      stages[d] = std::make_unique<Stage>(rows,height,width);
   }
   DenseMatrix copy(1), empty;
   copy(0,0) = 1.0;
   for (const auto &v : entities[0])
   {
      if (v.high.interior) { stages[0]->Add(v,copy,empty); }
   }
   const int cell_dofs = high.FEColl()->DofForGeometry(Geometry::TETRAHEDRON);
   stages[3] = std::make_unique<Stage>(mesh.GetNE()*cell_dofs,height,width);
   if (high.GetFE(0)->GetOrder() > 1)
   {
      for (int k = 0; k < mesh.GetNE(); ++k)
      {
         AddCell(k,high,lor,eh,el,entities,*stages[3]);
      }
   }
   for (int dim : {1,2}) { BuildTraceStage(mesh,dim,entities[dim],*stages[dim]); }
   for (auto &stage : stages) { stage->Finalize(); }
}

} // namespace

std::unique_ptr<Operator> MakeEntityTransfer(FiniteElementSpace &high,
                                             BatchedLOR &lor)
{
   return std::make_unique<EntityTransfer>(high,lor);
}

} // namespace tetlor
