import JSON

mesh = "naca0012_bl_o3.msh"
r = 1
# Ns = [2, 4, 8, 16, 32]
Ns = [2, 4, 8, 16]
s = "amg"

function get_results(cmd)
    for line in eachline(IOBuffer(read(cmd, String)))
        startswith(line, "=== ") && return JSON.parse(line[5:end])
    end
    error("No results found")
end

function run_it(m, N, r, s, pa)
    fname = joinpath("meshes", m)
    if pa
        cmd = `mpirun -np 64 ./poi_duffy -m $fname -r $r -o $N -s $s -d -pa`
    else
        cmd = `mpirun -np 64 ./poi_duffy -m $fname -r $r -o $N -s $s -d`
    end
    return get_results(cmd)
end

all_results = []
for N in Ns, pa in [true, false]
    results = run_it(mesh, N, r, s, pa);
    println(JSON.json(results; pretty=true))
    flush(stdout)
    push!(all_results, results)
end

JSON.json("timings_new.json", all_results; pretty=true)
