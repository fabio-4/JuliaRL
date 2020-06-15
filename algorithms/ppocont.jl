include("../RLAlgorithms.jl")
using .RLAlgorithms
using Plots
using Flux
using Flux: Optimise.update!, Zygote.pullback
using Statistics: mean

struct Policy{S}
    μ::S
    logstd::Vector{Float32}
    Policy(μ, logstd, nA) = new{typeof(μ)}(μ, fill(logstd, nA))
end
(m::Policy)(s) = (m.μ(s), m.logstd)
Flux.@functor Policy

function act(policy, s)
    μ, logstd = policy(s)
    std = exp.(logstd)
    ã = μ .+ std .* randn(eltype(std), size(std))
    logã = sum(loglikelihood(ã, μ, logstd), dims=1)
    return ã, logã
end

approxkl(logp, oldlogp) = mean(oldlogp .- logp)

function ploss(logp, oldlogp, adv, ϵ)
    rat = exp.(logp .- oldlogp)
    return -mean(min.(rat .* adv, min.(max.(rat, one(ϵ)-ϵ), one(ϵ)+ϵ) .* adv))
end

function run!(policy, popt, value, vopt, env; 
        epochs=100, steps=300, maxt=100, ptrainiters=50, vtrainiters=80, 
        ϵ=2f-1, γ=99f-2, λ=97f-2, targetkl=15f-3)
    rewards = zeros(Float32, epochs)
    memory = PGReplayMemory{Float32, Float32, Float32, Float32}(
        length(env.observationspace), length(env.actionspace), steps, steps, γ=γ, λ=λ
    )
    policyps = params(policy)
    valueps = params(value)

    for i in 1:epochs
        episodes!(memory, env, steps, maxt) do s
            a, loga = act(policy, s)
            v = value(s)
            return a, loga[1], v[1]
        end 

        let (s, a, oldlogp, r, adv) = sample(memory)
            for _ in 1:ptrainiters
                loga, back = pullback(policyps) do
                    μ, logstd = policy(s)
                    return vec(sum(loglikelihood(a, μ, logstd), dims=1))
                end
                approxkl(loga, oldlogp) > targetkl && break
                gs = gradient(logp -> ploss(logp, oldlogp, adv, ϵ), loga)[1]
                update!(popt, policyps, back(gs))
            end

            Flux.train!(valueps, Iterators.repeated((s, r), vtrainiters), vopt) do s, r
                return Flux.mse(vec(value(s)), r)
            end
        end
        rewards[i] = test(s -> policy(s)[1], env)
    end
    return rewards
end

env = CartPoleContinuous{Float32}()

policy = Policy(
    Chain(
        Dense(length(env.observationspace), 32, tanh), 
        Dense(32, length(env.actionspace))
    ), 
    5f-1, 
    length(env.actionspace)
)
value = Chain(
    Dense(length(env.observationspace), 32, tanh), 
    Dense(32, 1)
)

r = run!(policy, ADAM(3e-4), value, ADAM(1e-3), env)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "ppocont")
