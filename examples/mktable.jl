using JSON
using DataFrames
using Printf

data = JSON.parsefile("results_full.json")
df = DataFrame(data)

function mk_N_ref_table(mesh_name)
    sub = df[
        (df.mesh .== mesh_name) .&
        (df.solver .== "mumps") .&
        (df.ref .== 0), :]
    sub.N = Int.(sub.N)
    sub = combine(groupby(sub, :N), first)
    sort!(sub, :N)

    sub_amg = df[
        (df.mesh .== mesh_name) .&
        (df.solver .== "amg") .&
        (df.ref .== 0), :]
    sub_amg.N = Int.(sub_amg.N)
    sub_amg = combine(groupby(sub_amg, :N), first)
    sort!(sub_amg, :N)

    println(raw"\begin{tabular}{c c C{6} C{4.2} C{1.2}}")
    println(raw"   \multicolumn{5}{c}{$N$ refinement} \\\\")
    println(raw"   \toprule")
    println(raw"   {$N$} & {\# It.} & {\# DOFs} & {$\nnz(A) / \text{row}$} & {$\nnz(A_{0}) / \text{row}$} \\\\")
    println(raw"   \midrule")

    for (row, row_amg) in zip(eachrow(sub), eachrow(sub_amg))
        @printf("   %d & %d (%d) & %d & %.2f & %.2f & %.3f (%.3f) \\\\\n",
            row.N,
            row.niter,
            row_amg.niter,
            row.ndofs,
            row.nnz / row.ndofs,
            row.nnz_lor / row.ndofs)
    end
    println(raw"   \bottomrule")
    println(raw"\end{tabular}")
end

function mk_h_ref_table(mesh_name)
    sub = df[
        (df.mesh .== mesh_name) .&
        (df.solver .== "mumps") .&
        (df.N .== 8), :]
    sub = combine(groupby(sub, :ref), first)
    sort!(sub, :ref)

    sub_amg = df[
        (df.mesh .== mesh_name) .&
        (df.solver .== "amg") .&
        (df.N .== 8), :]
    sub_amg = combine(groupby(sub_amg, :ref), first)
    sort!(sub_amg, :ref)

    println(raw"\begin{tabular}{c c C{6} C{4.2} C{1.2}}")
    println(raw"   \toprule")
    println(raw"   {Ref.} & {\# It.} & {\# DOFs} & {$\nnz(A) / \text{row}$} & {$\nnz(A_{0}) / \text{row}$}\\\\")
    println(raw"   \midrule")

    for (row, row_amg) in zip(eachrow(sub), eachrow(sub_amg))
        @printf("   %d & %d (%d) & %d & %.2f & %.2f & %.3f (%.3f) \\\\\n",
            row.ref,
            row.niter,
            row_amg.niter,
            row.ndofs,
            row.nnz / row.ndofs,
            row.nnz_lor / row.ndofs)
    end
    println(raw"   \bottomrule")
    println(raw"\end{tabular}")
end
