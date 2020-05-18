using Flux: onehotbatch
using Distributions: Uniform
import Base.insert!

mutable struct UDRLMemory{S<:Real, A<:Real, R<:Real}
    episodes::Vector{Tuple{R, Vector{Tuple{Vector{S}, A, R}}}}
    obss::Matrix{S}
    dhs::Matrix{S}
    drs::Matrix{S}
    targets::Vector{A}
    maxreward::S
    maxlen::Int64
    minibatchsize::Int64
    function UDRLMemory{S, A, R}(nO, maxreward, maxlen, minibatchsize) where {S, A, R}
        new{S, A, R}(
            Vector{Tuple{R, Vector{Tuple{Vector{S}, Vector{A}, R}}}}(),
            Matrix{S}(undef, nO, minibatchsize),
            Matrix{S}(undef, 1, minibatchsize),
            Matrix{S}(undef, 1, minibatchsize),
            Vector{A}(undef, minibatchsize),
            maxreward,
            maxlen,
            minibatchsize
        )
    end
end

function insert!(memory::UDRLMemory{S, A, R}, r::R, e::Vector{Tuple{Vector{S}, A, R}}) where {S, A, R}
    len = length(memory.episodes)
    i = findfirst(x -> x[1] < r, memory.episodes) |> x -> x === nothing ? len + 1 : x
    if i <= memory.maxlen
        Base.insert!(memory.episodes, i, (r, e))
        len + 1 > memory.maxlen && deleteat!(memory.episodes, memory.maxlen)
    end
end

function command(memory::UDRLMemory{S, A, R}, best) where {S, A, R}
    length(memory.episodes) < best && return (one(S), memory.maxreward)
    h = [length(memory.episodes[i][2]) for i in 1:best]
    r = [memory.episodes[i][1] for i in 1:best]
    rmean = mean(r)
    rstd = std(r, mean=rmean) |> x -> isnan(x) ? zero(R) : x
    return S(mean(h)), S(rand(Uniform(rmean, rmean+rstd+1f-5)))
end

function sample(memory::UDRLMemory, nA)
    @inbounds for i in 1:memory.minibatchsize
        traj = memory.episodes[rand(1:length(memory.episodes))][2]
        t0 = rand(1:length(traj))
        t1 = rand(t0:length(traj))
        memory.obss[:, i] = traj[t0][1]
        memory.dhs[1, i] = t1 + 1 - t0
        memory.drs[1, i] = reduce((x, y) -> x + y[3], traj[t0:t1], init=zero(eltype(memory.drs)))
        memory.targets[i] = traj[t0][2]
    end
    return memory.obss, memory.dhs, memory.drs, onehotbatch(memory.targets, 1:nA)
end
