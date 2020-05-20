mutable struct PGEpisode{S<:Real, A<:Real, R<:Real}
    n::Int64
    s::Matrix{S}
    a::Matrix{A}
    r::Vector{R}
    v::Vector{R}
    γ::R
    nstep::Int64
    λ::R
    function PGEpisode{S, A, R}(obssize, actiondim, maxsize; 
            γ=0.99, nstep=15, λ=0.97) where {S<:Real, A<:Real, R<:Real}
        new(
            0,
            zeros(S, obssize, maxsize),
            zeros(A, actiondim, maxsize),
            zeros(R, maxsize),
            zeros(R, maxsize),
            γ,
            nstep,
            λ
        )
    end
end

function append!(memory::PGEpisode, s, a, r)
    memory.n += 1
    memory.s[:, memory.n] .= s
    memory.a[:, memory.n] .= a
    memory.r[memory.n] = r
end

function append!(memory::PGEpisode, s, a, r, v)
    append!(memory::PGEpisode, s, a, r)
    memory.v[memory.n] = v
end

discount!(memory::PGEpisode) = discount!(view(memory.r, 1:memory.n), memory.γ)

function nstepr!(memory::PGEpisode)
    nstepr!(view(memory.r, 1:memory.n), view(memory.v, 1:memory.n), memory.γ, memory.nstep)
end

function GAEλ!(memory::PGEpisode, norm=true)
    GAEλ!(view(memory.v, 1:memory.n), view(memory.r, 1:memory.n), memory.γ, memory.λ)
    norm && norm!(view(memory.v, 1:memory.n))
end

function adv!(memory::PGEpisode, norm=false)
    memory.v .= memory.r .- memory.v
    norm && norm!(view(memory.v, 1:memory.n))
end

function batch(memory::PGEpisode, batchsize=0)
    batchsize = (batchsize == 0 || batchsize > memory.n) ? memory.n : batchsize
    inds = sample(1:memory.n, batchsize, replace=false)
    memory.n = 0
    return (
        memory.s[:, inds],
        memory.a[:, inds],
        memory.r[inds],
        memory.v[inds]
    )
end
