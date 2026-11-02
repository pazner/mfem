# MPI H1 vertex-patch Poisson solver

`vertex_patch` implements the scalar H1 specialization of Brubeck and Farrell,
*Multigrid solvers for the de Rham complex with optimal complexity in polynomial
degree*, arXiv:2211.14284v2, Sections 2.2, 3, and 4.4–4.5.

## Build and run

Use the existing MFEM make configuration. The miniapp requires MPI, LAPACK,
METIS 5, and hypre. CPU and CUDA execution use the same make configuration;
CUDA execution also requires a CUDA-enabled MFEM and hypre. SuiteSparse is not
required. The Makefile adds `-O3` to `vertex_patch_solver.o` for its host symbolic
setup; it preserves the configured MFEM library and debug checks. To debug that
object without optimization, remove it and rebuild with
`make VERTEX_PATCH_CXXFLAGS= vertex_patch`. Changing this variable requires
rebuilding the object explicitly.

```sh
make -j4
make -C miniapps/solvers vertex_patch
cd miniapps/solvers
mpirun -np 4 ./vertex_patch -m ../../data/inline-hex.mesh -o 4
mpirun -np 4 ./vertex_patch -m ../../data/inline-hex.mesh -o 4 -no-sc
mpirun -np 4 ./vertex_patch -m ../../data/star.mesh -o 7 -c 1
mpirun -np 4 ./vertex_patch -d cuda -m ../../data/inline-hex.mesh -o 8 -sc
mpirun -np 4 ./vertex_patch -d cuda -m ../../data/inline-hex.mesh -o 8 -no-sc
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
| `-d`, `--device` | MFEM backend, default `cpu`; use `cuda` for GPUs |
| `-did`, `--device-id` | GPU index; default selects node-local rank modulo visible GPU count |
| `-gmpi`, `-no-gmpi` | Device MPI buffers or staged communication buffers; default staged |
| `-profile` | CUDA profiler markers around warmed applications and CG |
| `-oq`, `--overintegrate` | Force auxiliary GLL quadrature for physical diffusion |
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
Factor storage includes numerical factors, shared graph and dependency arrays,
per-patch maps, batched level schedules, and solve workspace. Distinct template
counts are summed across ranks; templates are not shared across MPI ranks.
Factor storage is not total or peak process memory.
Setup timings separately report coarse setup, reference templates, numerical
element assembly, global assembly, patch setup, and damping/transfer setup.
Mesh/space construction is outside these stages. Timed applications are warmed,
and GPU timings synchronize before stopping the clock.

## Implementation

The physical operator remains MFEM's partially assembled diffusion operator in
the standard nodal basis. For unit diffusion on verified affine tensor cells
without mesh nodes, the driver uses Gauss quadrature of order `2*p` (`p+1`
points per direction). Non-affine/curved cells and variable diffusion retain
the auxiliary GLL rule of order `3*p`. The `-oq` flag forces that higher rule
also on affine cells, for comparisons. Auxiliary assembly always keeps its
original quadrature. The preconditioner converts residuals/corrections with
transpose-consistent sum-factorized FDM basis operators. Reference data are
computed using MFEM's LAPACK eigensolver once per degree and cached in memory.
Reference sparsity, condensation maps, and quadrature tables are also cached
by degree, dimension, and condensation mode. There is no on-disk cache.

Canonical entity orientations use global vertex IDs. Interior eigenfunctions
transform with their reflection parity, rather than with nodal reversal alone.
The existing parallel finite-element space supplies ownership and true-DOF
communication; no new finite-element collection is necessary.

The sparse auxiliary operator uses diagonal weighted mass approximations in the
broken orthonormal basis from equation (3.9). Quadrature is tensor Gauss–Lobatto
of order at least `3*p`. Batched device contractions compute the weighted
diagonals; reference sparse contribution maps generate element entries and
Schur updates on device. CUDA sorting and deduplication build the global CSR graph. Patch incidence
maps identify reusable symbolic templates: their CSR graph, ordering, and
assembly contribution maps are computed once, on CPU. One batched device
pass assembles all patch values through those maps, without per-patch GPU
sorts, allocations, or a copy of each patch graph. Sparse assembly and condensation never form dense high-order
element matrices. The standalone `AssembleElement` and `CondenseElement`
functions remain CPU reference implementations. Cell interiors are diagonal. Before factorization,
the uncondensed patch matrix receives structural zeros for the fill induced by
eliminating interiors; this same matrix supplies both the values and the ICC
graph. Trace injection is a local sparse map between owned true DOFs, since the
trace and volume spaces share entity owners.

Each vertex star has one owner, the lowest participating MPI rank. Shared-vertex
neighbors exchange element metadata and numerical element buffers during setup,
including edge-only and vertex-only contacts. Each cell's numerical buffer is
sent once per destination rank; the GPU expands its contributions into patches. Cached, deduplicated communication schedules
gather residuals and scatter summed patch corrections. Condensation removes
cell-interior unknowns from these exchanges. Essential DOFs are excluded from
patches, but boundary-centered patches are retained when they have free DOFs.
Patches using only locally owned residual DOFs are solved during the gather;
the remaining patches are solved after the remote residuals arrive.

Patch factors implement the same prescribed-pattern ICC as MFEM's CPU
`IncompleteCholeskySolver`, with interiors first and a METIS nested-dissection
ordering of the interface graph. Numerical factors and workspaces are packed into device arrays. Matching
patches share their permutation, lower and transpose graphs, source-entry
maps, and dependency schedules. Template reuse includes essential exclusions
and interior flags, and never reuses numerical values, coefficients, or global
DOF maps. These caches are scoped to preconditioner construction, so rebuilding
after changing the space or boundary conditions cannot retain stale ownership.

Large batches with short dependency chains solve one dependency level across
all patches at a time, using eight lanes per sparse row. Smaller batches and
deep chains use one block per patch. Patches of up to 1,024 unknowns use shared
solve workspace in the block path; larger patches use runtime-sized global
workspace. The local and remote MPI batches have separate level schedules.
The transpose solve gathers through the shared transpose graph. A failed pivot reports its original patch-local row, global
vertex, and rank. A user-specified shift changes the patch preconditioner only.

MFEM's multigrid composes a symmetric V(1,1) cycle with direct p-coarsening to a
rediscretized order-one problem and one symmetric BoomerAMG cycle. The coarse
operator uses device full assembly; GPU BoomerAMG uses l1-Jacobi relaxation.
Order-one interpolation is evaluated directly from cell vertices on device.
Its transpose first reduces nodal contributions within each cell, then issues
only one atomic update per cell vertex. In condensed
mode the cycle acts on the interface Schur complement, surrounded by interior
elimination and recovery. Damping is fixed after setup using `PowerMethod` and
a 1.2 safety factor, unless explicitly supplied. An excessive user-supplied
damping can make the multigrid preconditioner unsuitable for CG. At degree one,
only the coarse solver is used.

The reusable miniapp API is `mfem::vertex_patch::Preconditioner` in
`vertex_patch_solver.hpp`. Its space, mesh, and coefficient must remain alive;
construct a new preconditioner after changing them. Like many MFEM solvers, instances use
mutable work buffers and are not reentrant.

GPU execution keeps basis transforms, interior elimination/recovery, patch
factorization/solves, coarse transfer, and vector operations on device. MPI uses
device buffers with `-gmpi`; otherwise only communication buffers are staged.
The one-dimensional reference calculations, mesh/ownership metadata, METIS
ordering, and factor dependency analysis remain on CPU. Their setup costs are
included in the reported timings. CPU execution dispatches the numerical kernels
through MFEM's CPU backend.

Coefficients are evaluated through `Coefficient::Project(QuadratureFunction&)`.
The driver's constant/variable diffusion and manufactured load implement device
projection. General coefficients retain MFEM's host projection fallback;
applications requiring device evaluation should override `Project` or supply a
quadrature-function coefficient. A host callback is not automatically a GPU
function.

The library has runtime-sized device/CPU tensor fallbacks for high-order
diffusion application, value-mapped quadrature evaluation on square/cube volume
meshes, and domain load assembly. These avoid the tuned kernels' fixed buffers
and excessive GPU block sizes. They do not remove limits from unrelated MFEM
operations such as diffusion diagonal assembly.

## Tests and performance validation

```sh
make -C tests/unit -j4 vertex_patch_tests
mpirun -np 1 tests/unit/vertex_patch_tests
mpirun -np 2 tests/unit/vertex_patch_tests
mpirun -np 4 tests/unit/vertex_patch_tests
make -C miniapps/solvers vertex_patch-test-par
make -C miniapps/solvers vertex_patch-test-gpu
MFEM_VERTEX_PATCH_DEVICE=cuda mpirun -np 1 tests/unit/vertex_patch_tests
MFEM_VERTEX_PATCH_DEVICE=cuda mpirun -np 2 tests/unit/vertex_patch_tests
MFEM_VERTEX_PATCH_DEVICE=cuda mpirun -np 4 tests/unit/vertex_patch_tests
```

The focused suite checks reference basis identities and parity, transpose and
inverse operations, Cartesian auxiliary energies, rotated element orientations,
complete distributed patches, preconditioner symmetry and positivity, dense
Schur-complement comparisons, curved cells with variable diffusion, both
condensation modes, and the all-essential-interface case. The reusable ICC
solver and runtime-sized diffusion fallback have independent reference tests in the
library's unit-test directories. The focused executable also checks batched ICC
against the CPU solver (including shifts and overlapping patch scatter), both
quadrature layouts, physical/reference derivatives, determinants, and high-order
load assembly. Set `MFEM_VERTEX_PATCH_DEVICE=debug` to check memory access through
MFEM's debug backend.

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

### GPU development validation, 2026-10-31

The existing CUDA Makefile configuration was preserved (`MFEM_DEBUG=YES`,
`-g`, `sm_80`, CUDA 12.9, GPU-enabled hypre 3.1). CPU and CUDA regression suites
passed on one, two, and four MPI ranks. The four registered CPU/GPU make smoke
tests passed, as did a four-rank GPU-aware MPI symmetry/solve check. Batched
auxiliary matrices and Schur complements are also compared entry by entry with
the CPU reference on curved quadrilateral/hexahedral cells with variable
coefficients. Conforming H1 restriction uses direct device gathers/scatters;
this avoids a cuSPARSE restriction-multiply failure found by the two-rank
basis-inverse regression in this configuration.

Compute Sanitizer reported zero memory errors for degree-eight condensed and
full solves, and zero racecheck hazards for the batched ICC test. The memory
runs used `OMPI_MCA_accelerator=null` with staged MPI to disable Open MPI's
intentional CUDA pointer-probing API failures. This does not disable MFEM CUDA
kernels. Nsight Systems captures bracketed by `-profile` showed no host/device
copies in warmed single-GPU applications and CG. In the two-GPU-aware-MPI
capture, all 394 host/device copies on rank zero occurred inside MPI calls;
the largest was 10,976 bytes. The MPI implementation can stage device buffers
internally even when the application supplies device pointers.

On two NVIDIA A30 GPUs, the fixed 2-by-2-by-2 box sweep passed the independent
`1e-8` true-residual check at degrees 2, 4, 8, 12, 16, and 24 in both modes.
Condensed/full iteration counts were respectively 9/11, 13/15, 16/19, 19/21,
20/22, and 22/25. Factor nonzero counts match the earlier CPU table. At degree
24, factor storage including transpose indices and dependency schedules is
13,013,168 / 61,291,996 bytes; application times were 0.0109 / 0.0273 seconds.

A separate degree-16 comparison using the current code, two MPI ranks,
`OPENBLAS_NUM_THREADS=1`, and the same box gives seconds below:

| Backend | Mode | Setup | One application | CG solve |
| --- | --- | ---: | ---: | ---: |
| CPU | Condensed | 1.938 | 0.0263 | 1.947 |
| CUDA | Condensed | 0.798 | 0.00361 | 0.111 |
| CPU | Full | 1.820 | 0.0385 | 2.434 |
| CUDA | Full | 1.157 | 0.00727 | 0.228 |

These are small local debug-build measurements, not optimized scaling results.
CPU symbolic ordering and cached reference construction still contribute to
setup. Mesh/finite-element-space construction is outside the preconditioner
setup timer; at very high degrees, MFEM's general H1 collection also constructs
unused non-tensor reference elements on CPU. The GPU port does not change that
collection initialization.

### Patch setup and solve optimization, 2026-10-31

The following measurements use one NVIDIA A30, one MPI rank,
`OPENBLAS_NUM_THREADS=1`, and the existing CUDA Makefile configuration. Only the
solver object now adds `-O3` for host setup. Run from `miniapps/solvers`:

```sh
OPENBLAS_NUM_THREADS=1 ./vertex_patch \
  -m ../../data/beam-hex.mesh -o 7 -rs 3 -rp 0 -d cuda
```

This problem has 4,096 cells, 1,458,801 true DOFs, and 5,257 vertex patches.
Two repeated runs of the optimized code gave the ranges below. The baseline
is the preceding GPU implementation with per-patch symbolic setup and assembly.

| Measurement | Before | After |
| --- | ---: | ---: |
| Preconditioner setup | 23.93 s | 3.56–3.68 s |
| One warmed preconditioner application | 14.45 ms | 10.81–11.11 ms |
| CG solve | 0.492 s | 0.234–0.235 s |
| Whole process | 25.44 s | 4.73–4.79 s |
| Factor arrays and solve workspace | 469.3 MB | 215.0 MB |
| CG iterations | 19 | 19 |

True relative residuals are approximately `7.4e-10`. There are only 22 distinct
patch templates, shared within this preconditioner construction; numerical
values, signs, and gather maps remain patch-specific. Factor nonzeros remain
22,396,395. Template reuse removes repeated METIS ordering and symbolic factor
construction. One batched GPU accumulation replaces per-patch sparse assembly,
and device factorization reads these packed values directly.

The solve improvement includes using exact eight-point Gauss quadrature for
the physical operator on this affine, constant-coefficient problem, replacing
the twelve-point auxiliary GLL rule. With `-oq`, setup was 3.52 s, one
application was 11.79 ms, and CG took 0.418 s with the same 19 iterations.
Thus the patch optimizations alone do not account for the full CG speedup.
Curved/non-affine meshes and variable coefficients keep the original rule.

Global auxiliary CSR assembly is now the largest setup stage, at 1.83–1.95 s;
patch setup takes approximately 1.14 s. Sparse triangular solves remain the
largest smoother cost. Batching independent rows by dependency level reduces
block barrier overhead on large patch batches; smaller batches retain the
block-per-patch path to avoid excessive kernel launches. Coarse transpose
interpolation reduces within each cell before updating shared vertices.
These measurements are local development results, not multi-node scaling or
an equal-problem comparison with `plor_solvers`.

After optimization, the vertex-patch regression tests passed on CPU and CUDA
with one, two, and four MPI ranks, including four-rank GPU-aware MPI. Additional
tests verify shared symbolic factors with different numerical values/maps,
both triangular-solve schedules, linear-field interpolation, and the global
coarse-transfer adjoint identity. All four registered CPU/GPU make smoke tests
passed. Degree-24 condensed and full solves passed on two GPUs, as did variable
diffusion on distorted cells and a manufactured-solution run. Compute Sanitizer
reported zero memory errors for the full-size condensed beam case and a smaller
full-space case, and zero racecheck hazards for the batched factor/solve test.
