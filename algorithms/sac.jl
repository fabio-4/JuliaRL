include("../RLAlgorithms.jl")
using .RLAlgorithms
using Plots
using Flux
using Flux: Zygote.pullback, Optimise.update!
using Statistics: mean

struct Actor{S, A1, A2}
    model::S
    μ::A1
    logstd::A2
end
(m::Actor)(s) = (l = m.model(s); (m.μ(l), m.logstd(l)))
Flux.@functor Actor

struct Critic{C}
    c1::C
    c2::C
end
(m::Critic)(s, a) = (inp = vcat(s, a); (m.c1(inp), m.c2(inp)))
Flux.@functor Critic

function act(actor, s)
    μ, logstd = actor(s)
    std = exp.(logstd)
    ã = μ .+ std .* randn(Float32, size(std))
    logpã = sum(loglikelihood(ã, μ, logstd), dims=1)
    logpã -= sum(2f0 .* (log(2f0) .- ã .- softplus.(-2f0 * ã)), dims=1)
    return tanh.(ã), logpã
end

function run!(actor, aopt, critic, copt, env; 
        epochs=100, steps=300, maxt=100, batchsize=128, α=2f-1, γ=99f-2, τ=5f-3)
    rewards = zeros(Float32, epochs)
    critictar = deepcopy(critic)
    memory = ReplayMemory{Float32, Float32, Float32}(
        length(env.observationspace), length(env.actionspace), 10*steps, batchsize
    )
    actorps = params(actor)
    criticps = params(critic)
    
    for i in 1:epochs
        episodes!(memory, env, steps, maxt) do s
            return act(actor, s)[1]
        end
        
        for _ in 1:steps
            s1, a1, r, s2, done = sample(memory)

            a2, logpa2 = act(actor, s2)
            Qtars2 = vec(min.(critictar(s2, a2)...) .- α .* logpa2)
            Qtar = r .+ (1 .- done) .* γ .* Qtars2
            cgs = gradient(criticps) do
                Q1s1a1, Q2s1a1 = critic(s1, a1)
                Flux.mse(vec(Q1s1a1), Qtar) + Flux.mse(vec(Q2s1a1), Qtar)
            end
            update!(copt, criticps, cgs)

            ags = gradient(actorps) do
                a, logpa = act(actor, s1)
                Qs1a = min.(critic(s1, a)...)
                return mean(α .* logpa .- Qs1a)
            end
            update!(aopt, actorps, ags)

            softupdate!(critictar, critic, τ)
        end
        rewards[i] = test(s -> tanh.(actor(s)[1]), env)
    end
    return rewards
end

env = CartPoleContinuous{Float32}()
aDim = length(env.actionspace)
oDim = length(env.observationspace)

actor = Actor(
    Chain(Dense(oDim, 32, relu), Dense(32, 16, relu)),
    Chain(Dense(16, aDim)),
    Chain(Dense(16, aDim, x -> min(max(x, typeof(x)(-20f0)), typeof(x)(2f0))))
)
critic = Critic(
    Chain(Dense(oDim+aDim, 32, relu), Dense(32, 16, relu), Dense(16, 1)),
    Chain(Dense(oDim+aDim, 32, relu), Dense(32, 16, relu), Dense(16, 1))
)

r = run!(actor, ADAM(), critic, ADAM(), env)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "sac")
