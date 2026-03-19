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

function sci(x, digits=2)
    if x == 0
        return "0"
    else
        s = @sprintf("%0.*e", digits, x)
        a, b = split(s, 'e')
        return "\$ $(a) \\times 10^{$(parse(Int, b))} \$"
    end
end

function speedup(s)
    return @sprintf "\$ %.2f \\times \$" s
end

function mk_timing_table()
    data = JSON.parsefile("timings_new.json")
    df = DataFrame(data)

    Ns = sort(unique(df.N))

    println("\\begin{tabular}{c cc cc cc}")
    println("   \\toprule")
    println("   \$N\$ & Assembly & Rate & Matvec & Rate & Total & Rate \\\\")
    println("   \\midrule")
    for (i, N) in enumerate(Ns)
        row = df[(df.N .== N) .& (df.pa .== false), :][1, :]
        total = row.assemble_time + row.elapsed

        if i == 1
            a_rate = m_rate = t_rate = "---"
        else
            Nprev = Ns[i-1]
            prev = df[(df.N .== Nprev) .& (df.pa .== false), :][1, :]

            prev_total = prev.assemble_time + prev.elapsed

            a_rate = @sprintf("%.2f", log2(row.assemble_time / prev.assemble_time))
            m_rate = @sprintf("%.2f", log2(row.matvec_time        / prev.matvec_time))
            t_rate = @sprintf("%.2f", log2(total             / prev_total))
        end

        @printf("   %d & %s & %s & %s & %s & %s & %s \\\\\n",
            N,
            sci(row.assemble_time), a_rate,
            sci(row.matvec_time),   m_rate,
            sci(total),             t_rate)
    end
    println("   \\bottomrule")
    println("\\end{tabular}")

    println()

    println("\\begin{tabular}{c ccc ccc ccc}")
    println("   \\toprule")
    println("   \$N\$ & Assembly & Rate & Speedup & Matvec & Rate & Speedup & Total & Rate & Speedup \\\\")
    println("   \\midrule")
    for (i, N) in enumerate(Ns)
        row_fa = df[(df.N .== N) .& (df.pa .== false), :][1, :]
        total_fa = row_fa.assemble_time + row_fa.elapsed

        row = df[(df.N .== N) .& (df.pa .== true), :][1, :]
        total = row.assemble_time + row.elapsed

        a_speedup = row_fa.assemble_time / row.assemble_time
        m_speedup = row_fa.matvec_time / row.matvec_time
        t_speedup = total_fa / total

        if i == 1
            a_rate = m_rate = t_rate = "---"
        else
            Nprev = Ns[i-1]
            prev = df[(df.N .== Nprev) .& (df.pa .== true), :][1, :]

            prev_total = prev.assemble_time + prev.elapsed

            a_rate = @sprintf("%.2f", log2(row.assemble_time / prev.assemble_time))
            m_rate = @sprintf("%.2f", log2(row.matvec_time        / prev.matvec_time))
            t_rate = @sprintf("%.2f", log2(total             / prev_total))
        end

        @printf("   %d & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n",
            N,
            sci(row.assemble_time), a_rate, speedup(a_speedup),
            sci(row.matvec_time),   m_rate, speedup(m_speedup),
            sci(total),             t_rate, speedup(t_speedup))
    end
    println("   \\bottomrule")
    println("\\end{tabular}")
end
