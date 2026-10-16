// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#ifndef MFEM_FLUIDS_ACOUSTICS_HPP
#define MFEM_FLUIDS_ACOUSTICS_HPP

#include "mfem.hpp"
#include <functional>
#include <memory>
#include <vector>

namespace acoustics
{
using namespace mfem;

struct Medium
{
   real_t density = 1.0, speed = 1.0;
};

/// Pressure and velocity flux for a homogeneous, stationary background medium.
class AcousticFlux : public FluxFunction
{
   Medium medium;
public:
   explicit AcousticFlux(const Medium &m) : FluxFunction(3,2), medium(m) { }
   real_t ComputeFlux(const Vector &, ElementTransformation &,
                      DenseMatrix &) const override;
   real_t ComputeFluxDotN(const Vector &, const Vector &,
                          FaceElementTransformations &, Vector &) const override;
   void ComputeFluxJacobian(const Vector &, ElementTransformation &,
                            DenseTensor &) const override;
   void ComputeFluxJacobianDotN(const Vector &, const Vector &,
                                ElementTransformation &, DenseMatrix &) const override;
};

class Problem
{
   struct Mode
   {
      real_t kx, ky, magnitude, phase;
   };
   Medium medium;
   std::string name;
   std::vector<Mode> modes;
public:
   Problem(const Medium &m, const std::string &n);
   void State(const Vector &x, real_t time, Vector &state) const;
};
/// Fixed-seed, unit Euclidean norm vector in global algebraic row ordering.
void RandomRightHandSide(ParFiniteElementSpace &, Vector &);
Mesh PeriodicMesh(int n, bool triangles);
void VerifyCollectively(bool ok, MPI_Comm comm, const char *message);
real_t GlobalNorm(const Vector &x, MPI_Comm comm);
void CheckFinite(const Vector &x, MPI_Comm comm);

/// Forms borrow integrators. Matrix references remain valid for this object's
/// lifetime; the mesh and finite element space must not change underneath it.
class SpatialOperator : public Operator
{
   ParFiniteElementSpace &fes;
   Medium medium;
   AcousticFlux flux;
   RusanovFlux numerical_flux;
   HyperbolicFormIntegrator convection;
   mutable ParNonlinearForm residual;
   ParBilinearForm mass;
   std::unique_ptr<HypreParMatrix> mass_matrix;
   HypreParMatrix *matrix;
public:
   SpatialOperator(ParFiniteElementSpace &, const Medium &);
   void AddIntegrators(ParNonlinearForm &);
   void Mult(const Vector &x, Vector &y) const override
   {
      matrix->Mult(x,y);
   }
   /// Evaluate the original form, useful for verifying the assembled matrix.
   void FormMult(const Vector &x, Vector &y) const
   {
      residual.Mult(x,y);
   }
   Operator &GetGradient(const Vector &) const override
   {
      return *matrix;
   }
   HypreParMatrix &Matrix() const
   {
      return *matrix;
   }
   HypreParMatrix &Mass() const
   {
      return *mass_matrix;
   }
   ParFiniteElementSpace &Space() const
   {
      return fes;
   }
   void Conserved(const Vector &, Vector &) const;
   real_t Energy(const Vector &) const;
   /// Apply the physical energy weights (without mass or the factor one half).
   void EnergyWeights(Vector &) const;
};

/// Permutes processor-local DG rows into one block per element, including all
/// components. Uniform element block sizes are required on each MPI rank.
class ElementBlockPreconditioner : public Solver
{
   Array<int> permutation, inverse_permutation;
   std::unique_ptr<Solver> local_solver;
   mutable Vector rhs, solution;
public:
   ElementBlockPreconditioner(ParFiniteElementSpace &, const std::string &type);
   void SetOperator(const Operator &) override;
   void Mult(const Vector &, Vector &) const override;
};

struct SolverOptions
{
   real_t linear_rtol = 1e-12;
   int linear_max = 300, krylov_dim = 100;
   bool verbose = false;
   std::string preconditioner = "block-ilu";
   std::string history, export_prefix;
};
struct SolverStats
{
   int stages = 0, setups = 0, linear_iterations = 0;
   real_t residual = 0;
   double assembly_time = 0, setup_time = 0, solve_time = 0;
};

/// References are borrowed. The shifted matrix expires on the next rebuild;
/// factory-created solvers are owned by AcousticOperator.
struct PreconditionerContext
{
   SpatialOperator &spatial;
   HypreParMatrix &shifted_matrix;
   real_t mass_shift;
};
using PreconditionerFactory =
   std::function<std::unique_ptr<Solver>(const PreconditionerContext &)>;

class AcousticOperator : public TimeDependentOperator
{
   SpatialOperator &spatial;
   SolverOptions options;
   SolverStats stats;
   ConstantCoefficient shift;
   VectorMassIntegrator mass_integrator;
   ParNonlinearForm stage_form;
   HypreParMatrix *stage_matrix = nullptr;
   real_t cached_a = 0;
   PreconditionerFactory factory;
   Solver *external_solver = nullptr;
   std::unique_ptr<Solver> preconditioner;
   std::unique_ptr<GMRESSolver> gmres;
   mutable HypreSmoother mass_preconditioner;
   mutable CGSolver mass_solver;
   int step_number = 0, step_stage = 0;
   void SolveShifted(real_t a, const Vector &rhs, Vector &state,
                     const Vector *base);
public:
   AcousticOperator(SpatialOperator &, const SolverOptions &);
   void BeginStep(int number, real_t time, real_t dt);
   void SetPreconditionerFactory(PreconditionerFactory f);
   /// Caller retains ownership and must keep the solver alive. Non-iterative
   /// solvers must report their own failures; the true residual is also recorded.
   void SetLinearSolver(Solver &solver);
   void SetExplicitMode()
   {
      SetImplicitVariableType(ImplicitVariableType::SLOPE);
   }
   void Mult(const Vector &, Vector &) const override;
   void ImplicitSolve(real_t a, const Vector &base, Vector &state) override;
   /// Solve (M/a + K) state = rhs directly, using the same setup and diagnostics.
   void SolveSystem(real_t a, const Vector &rhs, Vector &state);
   const SolverStats &Stats() const
   {
      return stats;
   }
};
}
#endif
