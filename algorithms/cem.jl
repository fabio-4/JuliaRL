include("../envs/RLEnvironments.jl")
using .RLEnvironments
using Plots
using Flux
using Statistics: mean

function episode(model, env, maxt, γ)
    R = zero(γ)
    reset!(env)
    for t in 1:maxt
        a = model(env.current)
        _, r, done = step!(env, a)
        R += γ^t * r
        done && break
    end
    return R
end

function run!(model, env; epochs=100, maxt=100, popsize=50, elitesize=10, σ=5f-1, γ=99f-2)
    rewards = zeros(Float32, epochs)
    p, re = Flux.destructure(model)
    W = σ .* randn(eltype(p), length(p))
    for i in 1:epochs
        Ws = [W + σ .* randn(eltype(p), length(p)) for _ in 1:popsize]
        Rs = map(Wi -> episode(re(Wi), env, maxt, γ), Ws)
        elids = sortperm(Rs, rev=true)[1:elitesize]
        W = mean(Ws[elids])
        
        rewards[i] = mean(Rs[elids])
    end
    return rewards
end

env = Pendulum{Float32}()

model = Chain(
    Dense(length(env.observationspace), 32, relu), 
    Dense(32, length(env.actionspace), tanh)
)

r = run!(model, env)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "cem")

#=
function setweights!(model, W)
    curr = 1
    for layer in model.layers
        Wsize = size(layer.W)
        bsize = size(layer.b)
        Wlen = *(Wsize...)
        blen = *(bsize...)
        layer.W .= reshape(W[curr:curr+Wlen-1], Wsize)
        layer.b .= reshape(W[(curr+Wlen):(curr+Wlen+blen-1)], bsize)
        curr += Wlen + blen
    end
end

numparams = (length(env.observationspace) + 1) * 32 + (32 + 1) * length(env.actionspace)

Rs = map(Wi -> (setweights!(model, Wi); episode(model, env, maxt, γ)), Ws)
=#
