# MPI H1 vertex-patch Poisson solver

`vertex_patch` implements the scalar H1 specialization of Brubeck and Farrell,
*Multigrid solvers for the de Rham complex with optimal complexity in polynomial
degree*, arXiv:2211.14284v2, Sections 2.2, 3, and 4.4–4.5.

## Build and run

Use the existing MFEM make configuration. The miniapp requires MPI, LAPACK,
METIS 5, and hypre. It runs on the CPU; SuiteSparse is not required.

```sh
make -j4
make -C miniapps/solvers vertex_patch
cd miniapps/solvers
mpirun -np 4 ./vertex_patch -m ../../data/inline-hex.mesh -o 4
mpirun -np 4 ./vertex_patch -m ../../data/inline-hex.mesh -o 4 -no-sc
mpirun -np 4 ./vertex_patch -m ../../data/star.mesh -o 7 -c 1
```

Supported spaces are scalar Gauss–Lobatto H1 spaces of uniform degree on
conforming quadrilateral or hexahedral volume meshes, including distorted and
curved cells. Diffusion must be positive and scalar. Essential conditions must
make the problem coercive; pure Neumann problems, hanging nodes, variable degree,
and other element geometries are unsupported. The driver uses homogeneous
Dirichlet data and homogeneous natural data on any unmarked boundary attributes.
For disconnected domains, every connected component must be constrained.

Important options:

| Option | Meaning |
| --- | --- |
| `-o`, `--order` | Polynomial degree; default 3 |
| `-rs`, `-rp` | Serial and parallel uniform refinements; default 0 |
| `-sc`, `-no-sc` | Enable/disable auxiliary static condensation; default enabled |
| `-eb '1 2'` | Essential boundary attributes; omitted means all |
| `-c 0`, `-c 1` | Unit diffusion or `1 + sum(x_i^2)` |
| `-ms` | Manufactured unit-box solution `product(sin(pi*x_i))`; all boundaries essential |
| `-damp` | Smoother damping; default 0 estimates it once during setup |
| `-shift` | Relative ICC diagonal shift; default 0 |
| `-rtol` | Required independently checked true relative residual; default `1e-8` |
| `-maxit` | Maximum CG iterations; default 500 |

The driver returns a nonzero exit status when the solve fails its convergence or
true-residual check. It reports maximum setup/solve times across ranks, the mean
of three timed preconditioner applications (maximum across ranks), global
patch/factor counts, owned factor-array storage including its triangular-solve
workspace, and the number of scalar values exchanged by patch gather/scatter per
smoothing step. A V(1,1) cycle has two such steps. The communication count excludes
basis conversion, matrix applications, coarse-grid communication, and setup.
Factor storage is not a measurement of total or peak process memory.

## Implementation

The physical operator remains MFEM's partially assembled diffusion operator in
the standard nodal basis. The preconditioner converts residuals/corrections with
transpose-consistent sum-factorized FDM basis operators. Reference data are
computed using MFEM's LAPACK eigensolver once per degree and cached in memory.
There is no on-disk basis cache.

Canonical entity orientations use global vertex IDs. Interior eigenfunctions
transform with their reflection parity, rather than with nodal reversal alone.
The existing parallel finite-element space supplies ownership and true-DOF
communication; no new finite-element collection is necessary.

The sparse auxiliary operator uses diagonal weighted mass approximations in the
broken orthonormal basis from equation (3.9). Quadrature is tensor Gauss–Lobatto
of order at least `3*p`. Sparse assembly and condensation never form dense
high-order element matrices. Cell interiors are diagonal. The uncondensed path
retains symbolic zeros for the fill induced by eliminating interiors, instead
of applying ordinary ICC(0) to the original graph.

Each vertex star has one owner, the lowest participating MPI rank. Shared-vertex
neighbors exchange sparse element contributions during setup, including
edge-only and vertex-only contacts. Cached, deduplicated communication schedules
gather residuals and scatter summed patch corrections. Condensation removes
cell-interior unknowns from these exchanges. Essential DOFs are excluded from
patches, but boundary-centered patches are retained when they have free DOFs.
Patches using only locally owned residual DOFs are solved during the gather;
the remaining patches are solved after the remote residuals arrive.

Patch factors use MFEM's prescribed-pattern `IncompleteCholeskySolver`, with
interiors ordered first and a METIS nested-dissection ordering of the interface
graph. A failed pivot reports its patch and rank. No dense or direct fallback is
used. A user-specified shift changes the patch preconditioner only.

MFEM's multigrid composes a symmetric V(1,1) cycle with direct p-coarsening to a
rediscretized order-one problem and one symmetric BoomerAMG cycle. In condensed
mode the cycle acts on the interface Schur complement, surrounded by interior
elimination and recovery. Damping is fixed after setup using `PowerMethod` and
a 1.2 safety factor, unless explicitly supplied. An excessive user-supplied
damping can make the multigrid preconditioner unsuitable for CG. At degree one,
only the coarse solver is used.

The reusable miniapp API is `mfem::vertex_patch::Preconditioner` in
`vertex_patch_solver.hpp`. Its space, mesh, and coefficient must remain alive;
construct a new preconditioner after changing them. Like many MFEM solvers, instances use
mutable work buffers and are not reentrant.

The CPU diffusion application in the library has a dynamically sized,
sum-factorized fallback for orders beyond its tuned kernels' fixed buffers. GPU
kernels and ordinary-order CPU dispatch remain unchanged. This extends operator
application; it does not remove fixed-order limits from every MFEM PA operation
(for example, diagonal assembly).

## Tests and performance validation

```sh
make -C tests/unit -j4 vertex_patch_tests
mpirun -np 1 tests/unit/vertex_patch_tests
mpirun -np 2 tests/unit/vertex_patch_tests
mpirun -np 4 tests/unit/vertex_patch_tests
make -C miniapps/solvers vertex_patch-test-par
```

The focused suite checks reference basis identities and parity, transpose and
inverse operations, Cartesian auxiliary energies, rotated element orientations,
complete distributed patches, preconditioner symmetry and positivity, dense
Schur-complement comparisons, curved cells with variable diffusion, both
condensation modes, and the all-essential-interface case. The reusable ICC
solver and CPU diffusion fallback have independent reference tests in the
library's unit-test directories.

For degree scaling, hold a small mesh fixed and run both modes at degrees
2, 4, 8, 12, 16, and 24. Record factor nonzeros, storage, setup time, solve time,
and iterations separately. The target in 3D is O(p^3) storage and O(p^4) work
for fixed mesh topology; small-degree slopes include lower-order terms. Use
mesh refinement at fixed degree to examine iteration growth. Debug timings are
development measurements; optimized timings and multi-node scaling must be
validated separately on the intended hardware.

### Development validation, 2026-10-29

The existing make configuration (`MFEM_DEBUG=YES`, `-g`, no optimization) was
preserved. The focused suite passed on one, two, and four MPI ranks: 10 cases,
14,574 assertions on one rank and 14,573 on two/four ranks. Both registered
four-rank make smoke tests passed. Additional driver runs covered mixed
boundaries, variable diffusion, distorted cells, and manufactured solutions.
On a 2-by-2-by-2 unit-box mesh with variable diffusion, the manufactured L2 error
decreased from `1.16e-3` at degree 3 to `5.84e-6` at degree 5.

For a fixed 2-by-2-by-2 hexahedral unit-box mesh, unit diffusion, all-Dirichlet
conditions, and two MPI ranks, the degree sweep gave:

| Degree | CG iterations, condensed / full | Factor nonzeros, condensed / full |
| ---: | ---: | ---: |
| 2 | 8 / 10 | 163 / 323 |
| 4 | 13 / 15 | 2,215 / 6,535 |
| 8 | 16 / 18 | 21,295 / 76,175 |
| 12 | 18 / 21 | 75,703 / 288,663 |
| 16 | 20 / 23 | 183,871 / 723,871 |
| 24 | 23 / 26 | 635,215 / 2,581,935 |

The factor counts approach cubic growth in degree. At degree 24, the problem
has 117,649 true DOFs; factor arrays including triangular-solve workspace occupy
7,930,684 bytes with condensation and 40,635,612 bytes without it. Every solve
passed the independent `1e-8` relative-residual check. Four-rank refinement runs
at degree 4 took 18 iterations on 64 cells and 19 on 512 cells, with condensation.
These small development runs do not establish optimized or multi-node scaling.

Final debug timings on the same fixed mesh and two local MPI ranks, in seconds:

| Degree | Mode | Setup | One preconditioner application | CG solve |
| ---: | --- | ---: | ---: | ---: |
| 8 | Condensed | 0.256 | 0.0105 | 0.242 |
| 8 | Full | 0.366 | 0.0170 | 0.394 |
| 16 | Condensed | 1.95 | 0.110 | 3.65 |
| 16 | Full | 3.30 | 0.181 | 5.68 |
| 24 | Condensed | 6.98 | 0.516 | 17.7 |
| 24 | Full | 11.9 | 0.705 | 26.6 |

Application times are means of three calls, taking the maximum across ranks;
setup times cover the preconditioner. The tensor contractions use O(p^4) work
in 3D, and the measured application growth is consistent with that bound over
these degrees. The transition to dynamically sized CPU diffusion kernels at
high order also affects the physical operator and total solve timing. No dense
high-order production matrices or GPU kernels were used.
