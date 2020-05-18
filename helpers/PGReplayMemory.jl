mutable struct PGReplayMemory{S<:Real, A<:Real, O<:Real, R<:Real}
    n::Int64
    maxsize::Int64
    batchsize::Int64
    s::Matrix{S}
    a::Matrix{A}
    oldlogp::Vector{O}
    r::Vector{R}
    adv::Vector{R}
    γ::R
    λ::R
    function PGReplayMemory{S, A, O, R}(obssize, actiondim, maxsize, batchsize;
            γ=0.99, λ=0.97) where {S<:Real, A<:Real, O<:Real, R<:Real}
        new(
            0, 
            maxsize, 
            batchsize,
            zeros(S, obssize, maxsize+1), 
            zeros(A, actiondim, maxsize+1), 
            zeros(O, maxsize+1),
            zeros(R, maxsize+1), 
            zeros(R, maxsize+1),
            γ,
            λ
        )
    end
end

function append!(memory::PGReplayMemory, s, a, oldlogp, r, v)
    memory.n = memory.n % memory.maxsize + 1
    memory.s[:, memory.n] .= s
    memory.a[:, memory.n] .= a
    memory.oldlogp[memory.n] = oldlogp
    memory.r[memory.n] = r
    memory.adv[memory.n] = v
end

function finish!(memory::PGReplayMemory, t, lastval)
    inds = (memory.n - t + 1) : (memory.n + 1)
    memory.r[memory.n+1] = lastval
    memory.adv[memory.n+1] = lastval
    GAEλ!(view(memory.adv, inds), view(memory.r, inds), memory.γ, memory.λ)
    discount!(view(memory.r, inds), memory.γ)
end

function sample(memory::PGReplayMemory, norm=true)
    inds = sample(1:(min(memory.n, memory.maxsize)), memory.batchsize, replace=false)
    adv = memory.adv[inds]
    norm && norm!(adv)
    return (
        memory.s[:, inds],
        memory.a[:, inds],
        memory.oldlogp[inds],
        memory.r[inds],
        adv
    )
end
