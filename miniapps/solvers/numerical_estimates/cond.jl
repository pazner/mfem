using LinearAlgebra, ReadSparse

readmat(f) = collect(readmatlab(f))
ratio(v) = maximum(v)/minimum(v)

function cond()
    W = readmat("W.txt")
    M = readmat("M.txt")
    B = readmat("B.txt")

    @show size(W) size(M) size(B)

    S = W + B*(M\B')
    D = cat(M, S; dims=(1,2))

    A = [M B'; B -W]

    e = real(eigvals(D\A))
    e_pos = filter(x -> x > 0, e)
    e_neg = filter(x -> x < 0, e)

    ratio(e_pos), ratio(e_neg)
end
