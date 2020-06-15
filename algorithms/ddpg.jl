include("../RLAlgorithms.jl")
using .RLAlgorithms
using Plots
using Flux
using Flux: Optimise.update!
using Statistics: mean

function act(actor, s, noisescale)
    a = actor(s)
    return clamp.(a .+ noisescale .* randn(Float32, size(a)), -1f0, 1f0)
end

function run!(actor, aopt, critic, copt, env; 
        epochs=100, steps=300, maxt=100, batchsize=128, noisescale=1f-1, γ=99f-2, τ=1f-3)
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
            act(actor, s, noisescale)
        end
        
        for _ in 1:steps
            s1, a1, r, s2, done = sample(memory)
    
            atar2 = actortar(s2)
            Qtars2 = vec(critictar(vcat(s2, atar2)))
            Qtar = r .+ (1 .- done) .* γ .* Qtars2
            cgs = gradient(criticps) do
                Q = vec(critic(vcat(s1, a1)))
                return Flux.mse(Q, Qtar)
            end
            update!(copt, criticps, cgs)
            
            ags = gradient(actorps) do
                as = actor(s1)
                return -mean(critic(vcat(s1, as)))
            end
            update!(aopt, actorps, ags)
            
            softupdate!(actortar, actor, τ)
            softupdate!(critictar, critic, τ)
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
critic = Chain(
    Dense(oDim+aDim, 32, relu),
    Dense(32, 16, relu),
    Dense(16, 1)
)

r = run!(actor, ADAM(), critic, ADAM(), env)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "ddpg")
