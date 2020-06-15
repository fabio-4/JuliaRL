include("../RLAlgorithms.jl")
using .RLAlgorithms
using Plots
using Flux
using Flux: Optimise.update!
using Statistics: mean

struct Critic{C}
    c1::C
    c2::C
end
(m::Critic)(s, a) = (inp = vcat(s, a); (m.c1(inp), m.c2(inp)))
Flux.@functor Critic

function act(actor, s, noisescale, noiseclamp=false)
    a = actor(s)
    noise = noisescale .* randn(Float32, size(a))
    noise = noiseclamp ? clamp.(noise, -5f-1, 5f-1) : noise
    return clamp.(a .+ noise, -1f0, 1f0)
end

function run!(actor, aopt, critic, copt, env; 
        epochs=100, steps=300, maxt=100,  batchsize=128, 
        noisescale=1f-1, noisescaletar=2f-1, γ=99f-2, τ=1f-3)
    rewards = zeros(Float32, epochs)
    actortar = deepcopy(actor)
    critictar = deepcopy(critic)
    memory = ReplayMemory{Float32, Float32, Float32}(
        length(env.observationspace), length(env.actionspace), 10*steps, batchsize
    )
    actorps = params(actor)
    criticps = params(critic)

    for i in 1:epochs
        episodes!(memory, env, steps, maxt) do s
            return act(actor, s, noisescale)
        end

        for z in 1:steps
            let (s1, a1, r, s2, done) = sample(memory)
                a2 = act(actortar, s2, noisescaletar, true)
                Qtar = r .+ (1 .- done) .* γ .* vec(min.(critictar(s2, a2)...))
                cgs = gradient(criticps) do
                    Q1s1a1, Q2s1a1 = critic(s1, a1)
                    return Flux.mse(vec(Q1s1a1), Qtar) + Flux.mse(vec(Q2s1a1), Qtar)
                end
                update!(copt, criticps, cgs)

                if z % 2 == 0
                    ags = gradient(actorps) do
                        a = actor(s1)
                        return -mean(critic(s1, a)[1])
                    end
                    update!(aopt, actorps, ags)

                    softupdate!(actortar, actor, τ)
                    softupdate!(critictar, critic, τ)
                end
            end
        end
        rewards[i] = test(s -> actor(s), env)
    end
    return rewards
end

env = CartPoleContinuous{Float32}()
aDim = length(env.actionspace)
oDim = length(env.observationspace)

actor = Chain(
    Dense(oDim, 32, relu), 
    Dense(32, aDim, tanh)
)
critic = Critic(
    Chain(Dense(oDim+aDim, 32, relu), Dense(32, 16, relu), Dense(16, 1)),
    Chain(Dense(oDim+aDim, 32, relu), Dense(32, 16, relu), Dense(16, 1))
)

r = run!(actor, ADAM(), critic, ADAM(), env)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "td3")
