module RLHelpers
    import Distributions.sample
    import Base.append!
    using Statistics: mean, std
    using Flux: Zygote.@adjoint

    include("discount.jl")
    include("ReplayMemory.jl")
    include("PGReplayMemory.jl")
    include("PGEpisode.jl")
    include("UDRLMemory.jl")

    macro loop(n, ex)
        :(for i = 1:$(esc(n))
            $(esc(ex))
        end)
    end

    gather(y, a) = y[CartesianIndex.(a, axes(a, 1))]
    gather(y, a::AbstractMatrix) = gather(y, vec(a))
    
    @adjoint gather(y, a) = gather(y, a), 
        ȳ -> begin 
            o = zeros(eltype(ȳ), size(y))
            @inbounds for (i, j) in enumerate(a)
                o[j, i] = ȳ[i]
            end
            return (o, nothing)
        end

    function loglikelihood(x, mu, logstd)
        return -5f-1 .* (((x .- mu) ./ (exp.(logstd) .+ 1f-8)) .^ 2f0 .+ 2f0 .* logstd .+ log(2f0*Float32(π)))
    end

    function softupdate!(target::T, model::T, τ=1f-2) where T
        for f in fieldnames(T)
            softupdate!(getfield(target, f), getfield(model, f), τ)
        end
    end

    function softupdate!(dst::A, src::A, τ=T(1f-2)) where {T, A<:AbstractArray{T}}
        dst .= τ .* src .+ (one(T) - τ) .* dst
    end
    
    export @loop, gather, softupdate!, loglikelihood
    export ReplayMemory, PGReplayMemory, PGEpisode, UDRLMemory, command, 
        finish!, sample, discount!, nstepr!, GAEλ!, adv!, batch
end
