include("../RLAlgorithms.jl")
using .RLAlgorithms
using Plots
using Flux
using Statistics: mean

struct Policy{S, A, V}
    shared::S
    action::A
    value::V
end
(m::Policy)(s) = (l = m.shared(s); (softmax(m.action(l)), m.value(l)))
Flux.@functor Policy

function run!(policy, opt, env; epochs=500, maxt=100, γ=99f-2)
    rewards = zeros(Float32, div(epochs, 5))
    episode = PGEpisode{Float32, Int64, Float32}(length(env.observationspace), 1, maxt, γ=γ)

    for i in 1:epochs
        episode!(episode, env, maxt) do s
            p, v = policy(s)
            return p, v[1]
        end
        nstepr!(episode) #discount!(episode)
        adv!(episode, true)

        Flux.train!(params(policy), [batch(episode)], opt) do s, a, nr, adv
            p, v = policy(s)
            return -mean(log.(gather(p, a)) .* adv) + 0.5f0 * Flux.mse(vec(v), nr)
        end
        i % 5 == 0 && (rewards[div(i, 5)] = test(s -> Flux.onecold(policy(s)[1]), env))
    end
    return rewards
end

env = CartPole{Float32}()

policy = Policy(
    Dense(length(env.observationspace), 32, relu), 
    Dense(32, length(env.actionspace)),
    Dense(32, 1)
)

r = run!(policy, ADAM(0.005), env)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "a2c")
