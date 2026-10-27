// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// This file is part of MFEM. See LICENSE for details.
#ifndef MFEM_COMPRESSIBLE_FLOW_HPP
#define MFEM_COMPRESSIBLE_FLOW_HPP

#include "mfem.hpp"
#include <functional>
#include <memory>

namespace compressible
{
using namespace mfem;

struct Gas
{
   real_t gamma = 1.4, gas_constant = 1.0, viscosity = 0.0, prandtl = 0.72;
   real_t Cv() const { return gas_constant / (gamma - 1.0); }
   real_t Conductivity() const { return viscosity * gamma * Cv() / prandtl; }
   real_t Pressure(const Vector &u) const;
};

/// Exact away from max/norm switching points; symmetric generalized derivative
/// at equal speeds and zero derivative of the absolute normal velocity at zero normal velocity.
class EulerRusanovFlux : public RusanovFlux
{
   Gas gas;
   real_t Speed(const Vector &u, const Vector &normal, Vector &derivative) const;
public:
   EulerRusanovFlux(const EulerFlux &flux, const Gas &gas_)
      : RusanovFlux(flux), gas(gas_) { }
   void Grad(int side, const Vector &u1, const Vector &u2, const Vector &normal,
             FaceElementTransformations &tr, DenseMatrix &grad) const override;
};

/// Physical viscous flux and its hand-derived directional differential.
void ViscousFlux(const Gas &gas, const Vector &u, const DenseMatrix &q,
                 DenseMatrix &flux);
void ViscousDerivative(const Gas &gas, const Vector &u, const DenseMatrix &q,
                       const Vector &du, const DenseMatrix &dq, DenseMatrix &df);

/// Conservative-variable SIP: Fv = K(U) grad U. Integrates volume and faces.
class ViscousIntegrator : public NonlinearFormIntegrator
{
   Gas gas;
   int offset;
   real_t penalty;
   void Element(const FiniteElement &, ElementTransformation &, const Vector &,
                Vector *, DenseMatrix *);
   void Face(const FiniteElement &, const FiniteElement &,
             FaceElementTransformations &, const Vector &, Vector *, DenseMatrix *);
public:
   ViscousIntegrator(const Gas &g, int offset_, real_t penalty_)
      : gas(g), offset(offset_), penalty(penalty_) { }
   void AssembleElementVector(const FiniteElement &el, ElementTransformation &tr,
                              const Vector &u, Vector &r) override
   { Element(el, tr, u, &r, nullptr); }
   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &tr,
                            const Vector &u, DenseMatrix &j) override
   { Element(el, tr, u, nullptr, &j); }
   void AssembleFaceVector(const FiniteElement &a, const FiniteElement &b,
                           FaceElementTransformations &tr, const Vector &u,
                           Vector &r) override { Face(a, b, tr, u, &r, nullptr); }
   void AssembleFaceGrad(const FiniteElement &a, const FiniteElement &b,
                         FaceElementTransformations &tr, const Vector &u,
                         DenseMatrix &j) override { Face(a, b, tr, u, nullptr, &j); }
};

/// Analytic periodic problem data. All coordinates are in [-2,2]^2.
class Problem
{
   Gas gas;
   std::string name;
public:
   Problem(const Gas &gas_, const std::string &name_) : gas(gas_), name(name_) { }
   void State(const Vector &x, real_t t, Vector &u) const;
   void Source(const Vector &x, real_t t, Vector &s) const;
   bool HasSource() const { return name == "manufactured"; }
};

Mesh PeriodicMesh(int n, bool triangles);

/// Owns spatial integrators; both forms borrow them. No essential DOFs on the
/// periodic mesh. Gradients returned by forms expire at the next assembly.
class SpatialOperator : public Operator
{
   ParFiniteElementSpace &fes;
   Gas gas;
   int offset;
   EulerFlux flux;
   EulerRusanovFlux numerical_flux;
   HyperbolicFormIntegrator convection;
   ViscousIntegrator diffusion;
   ParNonlinearForm residual;
   ParBilinearForm mass;
   std::unique_ptr<HypreParMatrix> mass_matrix;
   const Problem &problem;
   real_t time = 0.0;
public:
   SpatialOperator(ParFiniteElementSpace &, const Gas &, const Problem &,
                   int offset = 3, real_t penalty = 10.0);
   void AddIntegrators(ParNonlinearForm &form);
   void SetTime(real_t t) { time = t; }
   real_t GetTime() const { return time; }
   void Mult(const Vector &u, Vector &r) const override;
   Operator &GetGradient(const Vector &u) const override;
   bool Admissible(const Vector &u) const;
   void Conserved(const Vector &u, Vector &totals) const;
   ParFiniteElementSpace &Space() const { return fes; }
   HypreParMatrix &Mass() const { return *mass_matrix; }
};

struct SolverOptions
{
   real_t newton_rtol = 1e-10, newton_atol = 1e-12, linear_rtol = 1e-10;
   int newton_max = 20, linear_max = 300, krylov_dim = 100;
   bool prec_bool = true; // if true, ILU, if false, Agglom.
   bool verbose = false;
   std::string export_prefix, history;
};

struct SolverStats
{
   int stages = 0, newton_iterations = 0, linear_iterations = 0;
   real_t assembly_time = 0.0, setup_time = 0.0, solve_time = 0.0;
   real_t residual = 0.0;
};

/// Context is valid during factory/SetOperator calls. References must not be
/// retained across gradient assemblies. Spatial Jacobian is available on demand.
struct PreconditionerContext
{
   SpatialOperator &spatial;
   const Vector &state;
   HypreParMatrix &shifted_jacobian;
   real_t mass_shift;
};
using PreconditionerFactory =
   std::function<std::unique_ptr<Solver>(const PreconditionerContext &)>;

class StageOperator : public Operator
{
   SpatialOperator &spatial;
   ConstantCoefficient shift;
   VectorMassIntegrator mass_integrator;
   mutable ParNonlinearForm form;
   Vector base_state;
   mutable const Vector *state = nullptr;
   SolverStats &stats;
public:
   StageOperator(SpatialOperator &, SolverStats &);
   void Configure(real_t a, const Vector &base);
   void Mult(const Vector &u, Vector &r) const override;
   Operator &GetGradient(const Vector &u) const override;
   real_t Shift() const { return shift.constant; }
   const Vector &State() const { return *state; }
};

class FlowOperator : public TimeDependentOperator
{
   SpatialOperator &spatial;
   SolverOptions options;
   SolverStats stats;
   StageOperator stage;
   PreconditionerFactory factory;
   Solver *external_solver = nullptr;
   Vector last_stage;
   int step_stage = 0;
public:
   FlowOperator(SpatialOperator &, const SolverOptions &);
   /// Reset stage numbering and optionally report the step before advancing it.
   void BeginStep(int step_number, real_t time, real_t dt);
   void SetPreconditionerFactory(PreconditionerFactory f) { factory = std::move(f); }
   /// The caller owns the solver, which must report failures itself if it is
   /// not an IterativeSolver. SetOperator is called for every Newton system.
   void SetLinearSolver(Solver &s) { external_solver = &s; }
   /// Explicit ODE solvers require slope mode even though ImplicitSolve is not called.
   void SetExplicitMode() { SetImplicitVariableType(ImplicitVariableType::SLOPE); }
   void Mult(const Vector &u, Vector &du) const override;
   void ImplicitSolve(real_t a, const Vector &base, Vector &state) override;
   const SolverStats &Stats() const { return stats; }
};
}
#endif
