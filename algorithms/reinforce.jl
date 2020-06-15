include("../RLAlgorithms.jl")
using .RLAlgorithms
using Plots
using Flux
using StatsBase: sample, Weights

function episode!(actor, episode, env, maxt)
    reset!(env)
    for _ in 1:maxt
        s = copy(env.current)
        a = sample(Weights(actor(s)))
        _, r, done = step!(env, a)
        append!(episode, s, a, r)
        done && break
    end
end

function run!(actor, opt, env; epochs=500, maxt=100, γ=99f-2)
    rewards = zeros(Float32, div(epochs, 10))
    episode = PGEpisode{Float32, Int64, Float32}(length(env.observationspace), 1, maxt, γ=γ)
    
    for i in 1:epochs
        episode!(actor, episode, env, maxt)
        discount!(episode)
        
        Flux.train!(params(actor), [batch(episode)], opt) do s, a, r, _
            -sum(log.(gather(actor(s), a)) .* r)
        end
        i % 10 == 0 && (rewards[div(i, 10)] = test(s -> Flux.onecold(actor(s)), env))
    end
    return rewards
end

env = CartPole{Float32}()

actor = Chain(
    Dense(length(env.observationspace), 32, relu), 
    Dense(32, length(env.actionspace)), 
    softmax
)

r = run!(actor, ADAM(0.01, (0.99, 0.999)), env)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "reinforce")
