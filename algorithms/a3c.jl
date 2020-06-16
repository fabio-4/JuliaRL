# bash: export JULIA_NUM_THREADS=4
include("../RLAlgorithms.jl")
using .RLAlgorithms
using Plots
using Base.Threads
using Random: MersenneTwister
using Flux
using Flux: Zygote.Grads, Zygote.Params
import Flux.Optimise: update!
using Statistics: mean
using StatsBase: sample, Weights
import.Base: +

function +(x::Grads, y::Grads)
    return Grads(IdDict(p => (get(x.grads, p, nothing) + y.grads[p]) for p in y.params), y.params)
end
+(::Nothing, y) = y

function update!(opt, ps1::Params, ps2::Params, gs::Grads)
    for (p1, p2) in zip(ps1.order, ps2.order)
        gs.grads[p2] !== nothing && update!(opt, p1, gs.grads[p2])
        p2 .= p1
    end
end

struct Policy{S, A, V}
    shared::S
    action::A
    value::V
end
(m::Policy)(s) = (l = m.shared(s); (softmax(m.action(l)), m.value(l)))
Flux.@functor Policy

function episode!(policy, episode, env, maxt, rng)
    reset!(env)
    done = false
    for _ in 1:maxt-1
        s = copy(env.current)
        logits, v = policy(s)
        a = sample(rng, Weights(logits))
        _, r, done = step!(env, a)
        append!(episode, s, a, r, v[1])
        done && break
    end
    if !done
        s = env.current
        p, v = policy(s)
        append!(episode, s, argmax(p), v[1], v[1])
    end
    nothing
end

function grad(policy, ps, s, a, r, adv)
    return gradient(ps) do
        p, v = policy(s)
        logp = log.(p)
        entropy = -mean(sum(p .* logp, dims=1))
        return -mean(gather(logp, a) .* adv) + 0.5f0 * Flux.mse(vec(v), r) - 1f-2 * entropy
    end
end

function run!(sharedpolicy, opt, sharedenv; epochs=500, maxt=100, γ=99f-2)
    rewards = zeros(Float32, div(epochs, 5))
    sharedrng = [MersenneTwister(rand(1:100)) for _ in 1:nthreads()]
    sharedlockobj = ReentrantLock()
    sharedps = params(sharedpolicy)

    @threads for _ in 1:nthreads()
        rng = sharedrng[threadid()]
        env = deepcopy(sharedenv)
        policy = deepcopy(sharedpolicy)
        ps = params(policy)
        gs = Grads(IdDict(), Params())
        episode = PGEpisode{Float32, Int64, Float32}(length(env.observationspace), 1, maxt, γ=γ)

        for i in 1:epochs
            episode!(policy, episode, env, maxt, rng)
            GAEλ!(episode)
            discount!(episode)
            
            s, a, r, adv = batch(episode)
            gs += grad(policy, ps, s, a, r, adv)

            if i % 5 == 0
                lock(sharedlockobj)
                update!(opt, sharedps, ps, gs)
                unlock(sharedlockobj)
                gs = Grads(IdDict(), Params())
            end
            if i % 5 == 0
                ind = div(i, 5)
                if threadid() == (ind % nthreads()) + 1
                    rewards[ind] = test(s -> Flux.onecold(policy(s)[1]), env)
                end
            end
        end
    end
    return rewards
end

env = CartPole{Float32}()

sharedpolicy = Policy(
    Dense(length(env.observationspace), 64, relu),
    Dense(64, length(env.actionspace)),
    Dense(64, 1)
)

r = run!(sharedpolicy, ADAM(0.0015, (0.5, 0.99)), env)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "a3c")

#=
older Zygote version:
gs = Grads(IdDict())
function +(x::Grads, y::Grads)
    return Grads(IdDict(k => (get(x.grads, k, nothing) + y.grads[k]) for k in keys(y.grads)))
end
=#

#=
+(x::Tuple, y::Tuple) = Tuple([(x[k] + y[k]) for k in keys(x)])
function +(x::NamedTuple, y::NamedTuple)
    NamedTuple{keys(x)}([(x[k] === nothing ? nothing : x[k] + y[k]) for k in keys(x)])
end

function update!(opt, m1::T, m2::T, gs) where T <: AbstractArray
    update!(opt, m1, gs)
    m2 .= m1
    gs .= zero(eltype(gs))
end

function update!(opt, m1, m2, gs)
    for k in keys(gs)
        gs[k] === nothing && continue
        if typeof(k) <: Symbol
            update!(opt, getfield(m1, k), getfield(m2, k), gs[k]) 
        else 
            update!(opt, m1[k], m2[k], gs[k])
        end
    end
end

sharedinitgs = (gradient(sharedpolicy) do p
    m = p(zeros(Float32, length(env.observationspace)))
    return Flux.mse(m[1], 0f0) + Flux.mse(m[2], 0f0)
end)[1]

gs = deepcopy(sharedinitgs)

gs += (gradient(policy) do model
    p, v = model(s)
    logp = log.(p)
    entropy = -mean(sum(p .* logp, dims=1))
    return -mean(gather(logp, a) .* adv) + 5f-1 * Flux.mse(vec(v), r) - 1f-2 * entropy
end)[1]

update!(opt, sharedpolicy, policy, gs)
=#
