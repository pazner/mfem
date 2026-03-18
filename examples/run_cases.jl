import JSON

if "--full" in ARGS
    println("")
    meshes = [
        (filename="BL3.msh", name="BL3"),
        (filename="naca0012_bl.msh", name="NACA"),
        (filename="naca0012_bl_o3.msh", name="NACA (High-Order)"),
        (filename="visc_BGM45-15.msh", name="BGM45-15"),
        (filename="BL3_surf.msh", name="BGM45-15 (High-Order)"),
        (filename="visc_BGM45-15_reg.msh", name="BGM45-15 (High-Order)"),
    ]
    refs = [0, 1, 2, 3, 4]
    Ns = [2, 4, 8, 16, 32]
    solvers = ["amg", "mumps"]
    outname = "results_full_2.json"
else
    meshes = [
        (filename="BL3.msh", name="BL3"),
    ]
    refs = [0, 1, 2]
    Ns = [2, 4]
    solvers = ["amg"]
    outname = "results.json"
end

function get_results(cmd)
    for line in eachline(IOBuffer(read(cmd, String)))
        startswith(line, "=== ") && return JSON.parse(line[5:end])
    end
    error("No results found")
end

function run_it(m, N, r, s)
    fname = joinpath("meshes", m.filename)
    cmd = `mpirun -np 64 ./poi_duffy -m $fname -r $r -o $N -s $s -d`
    return get_results(cmd)
end

all_results = []

for m in meshes, N in Ns, s in solvers
    results = run_it(m, N, 0, s);
    println(JSON.json(results; pretty=true))
    flush(stdout)
    push!(all_results, results)
end

for m in meshes, r in refs, s in solvers
    results = run_it(m, 8, r, s);
    println(JSON.json(results; pretty=true))
    flush(stdout)
    push!(all_results, results)
end

JSON.json(outname, all_results; pretty=true)
