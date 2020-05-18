mutable struct ReplayMemory{S<:Real, A<:Real, R<:Real}
    n::Int64
    maxsize::Int64
    batchsize::Int64
    s::Matrix{S}
    a::Matrix{A}
    r::Vector{R}
    d::Vector{Bool}
    function ReplayMemory{S, A, R}(obssize, actiondim, maxsize, batchsize) where {S<:Real, A<:Real, R<:Real}
        new(
            0, 
            maxsize, 
            batchsize,
            zeros(S, obssize, maxsize), 
            zeros(A, actiondim, maxsize), 
            zeros(R, maxsize), 
            zeros(Bool, maxsize)
        )
    end
end

function append!(memory::ReplayMemory, s, a, r, d)
    ind = memory.n % memory.maxsize + 1
    memory.n += 1
    memory.s[:, ind] .= s
    memory.a[:, ind] .= a
    memory.r[ind] = r
    memory.d[ind] = d
end

function sample(memory::ReplayMemory)
    inds = sample(1:(min(memory.n, memory.maxsize)-1), min(memory.n-1, memory.batchsize), replace=false)
    return (
        memory.s[:, inds],
        memory.a[:, inds],
        memory.r[inds],
        memory.s[:, inds .+ 1],
        memory.d[inds]
    )
end
