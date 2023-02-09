#!/bin/sh

# Setup
./bench_dgmassinv --benchmark_filter="FULL_CG_Setup_0_5/.*/.*" --benchmark_out_format=csv --benchmark_out=cg_setup.csv --benchmark_context=device=cuda
./bench_dgmassinv --benchmark_filter="DIRECT_CUBLAS_Setup_0_5/.*/.*" --benchmark_out_format=csv --benchmark_out=inverse_setup.csv --benchmark_context=device=cuda
./bench_dgmassinv --benchmark_filter="DIRECT_CUSOLVER_Setup_0_5/.*/.*" --benchmark_out_format=csv --benchmark_out=cholesky_setup.csv --benchmark_context=device=cuda

# Solve
./bench_dgmassinv --benchmark_filter="FULL_CG_Solve_0_5/.*/.*" --benchmark_out_format=csv --benchmark_out=full_cg_solve.csv --benchmark_context=device=cuda
./bench_dgmassinv --benchmark_filter="LOCAL_CG_LOBATTO_Solve_0_5/.*/.*" --benchmark_out_format=csv --benchmark_out=local_cg_lobatto_solve.csv --benchmark_context=device=cuda
./bench_dgmassinv --benchmark_filter="LOCAL_CG_LEGENDRE_Solve_0_5/.*/.*" --benchmark_out_format=csv --benchmark_out=local_cg_legendre_solve.csv --benchmark_context=device=cuda
./bench_dgmassinv --benchmark_filter="DIRECT_CUBLAS_Solve_0_5/.*/.*" --benchmark_out_format=csv --benchmark_out=inverse_solve.csv --benchmark_context=device=cuda
./bench_dgmassinv --benchmark_filter="DIRECT_CUSOLVER_Solve_0_5/.*/.*" --benchmark_out_format=csv --benchmark_out=cholesky_solve.csv --benchmark_context=device=cuda

sed -i.bak "/MAX_NDOFS/d" ./*.csv
