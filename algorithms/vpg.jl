include("../RLAlgorithms.jl")
using .RLAlgorithms
using Plots
using Flux
using Flux: Optimise.update!
using Statistics: mean
using StatsBase: sample, Weights

function act(policy, s)
    logits = policy(s)
    a = sample(Weights(exp.(logits)))
    loga = gather(logits, a) - log.(sum(exp.(logits), dims=1))
    return a, loga
end

function run!(policy, popt, value, vopt, env; 
        epochs=100, steps=500, maxt=100, vtrainiters=80, γ=99f-2, λ=97f-2)
    rewards = zeros(Float32, epochs)
    memory = PGReplayMemory{Float32, Int64, Float32, Float32}(
        length(env.observationspace), 1, steps, steps, γ=γ, λ=λ
    )
    policyps = params(policy)
    valueps = params(value)

    for i in 1:epochs
        episodes!(memory, env, steps, maxt) do s
            a, loga = act(policy, s)
            v = value(s)
            return a, loga[1], v[1]
        end 

        let (s, a, _, r, adv) = sample(memory)
            pgs = gradient(policyps) do
                logits = policy(s)
                logpa = gather(logits, a) .- vec(log.(sum(exp.(logits), dims=1)))
                return -mean(logpa .* adv)
            end
            update!(popt, policyps, pgs)

            Flux.train!(valueps, Iterators.repeated((s, r), vtrainiters), vopt) do s, r
                return Flux.mse(vec(value(s)), r)
            end
        end
        rewards[i] = test(s -> Flux.onecold(policy(s)), env)
    end
    return rewards
end

env = CartPole{Float32}()

policy = Chain(
    Dense(length(env.observationspace), 32, tanh), 
    Dense(32, 32, tanh),
    Dense(32, length(env.actionspace))
)
value = Chain(
    Dense(length(env.observationspace), 32, tanh), 
    Dense(32, 32, tanh),
    Dense(32, 1)
)

r = run!(policy, ADAM(3e-4), value, ADAM(1e-3), env)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "vpg")
