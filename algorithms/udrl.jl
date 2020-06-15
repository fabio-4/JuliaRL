include("../RLAlgorithms.jl")
using .RLAlgorithms
using Plots
using Flux

struct BehaviorModel{L11, L12, L2}
    l1obs::L11
    l1cmd::L12
    l2out::L2
    scale::Float32
end

(m::BehaviorModel)(s, dh, dr) = m.l2out(m.l1obs(s) .* m.l1cmd(vcat(dh, dr) .* m.scale))
Flux.@functor BehaviorModel

function BehaviorModel(nO, nA)
    l1obs = Dense(nO, 32, σ)
    l1cmd = Dense(2, 32, σ)
    l2out = Chain(
        Dense(32, 32, relu),
        Dense(32, nA)
    )
    return BehaviorModel(l1obs, l1cmd, l2out, 2f-2)
end

function act(model, s, dh, dr, nA, ϵ)
    rand() <= ϵ && return rand(1:nA)
    return Flux.onecold(model(s, dh, dr))
end

function episode(env, model, dh, dr, nA, maxt, maxreward, ϵ)
    traj = Vector{Tuple{Vector{Float32}, Int64, Float32}}()
    totr = 0f0
    reset!(env)
    for _ in 1:maxt
        s = copy(env.current)
        a = act(model, s, dh, dr, nA, ϵ)
        _, r, done = step!(env, a)
        push!(traj, (s, a, r))
        dh = max(dh - 1f0, 1f0)
        dr = min(dr - r, maxreward)
        totr += r
        done && break
    end
    return totr, traj
end

function run!(model, opt, env, maxreward; 
        epochs=100, maxt=100, minibatchsize=32, 
        best=25, trajs=50, trainiters=200, ϵ=1.0)
    rewards = zeros(Float32, epochs)
    memory = UDRLMemory{Float32, Int64, Float32}(
        length(env.observationspace), maxreward, 10*trajs, minibatchsize
    )
    nA = length(env.actionspace)
    ps = params(model)

    for i in 1:epochs
        for j in 1:trajs
            dh, dr = command(memory, best)
            totr, traj = episode(env, model, dh, dr, nA, maxt, maxreward, ϵ)
            insert!(memory, totr, traj)
        end
        
        @loop trainiters Flux.train!(ps, [sample(memory, nA)], opt) do obss, dhs, drs, tars
            Flux.logitcrossentropy(model(obss, dhs, drs), tars)
        end
        ϵ = max(min(ϵ * 0.99, 0.05), 0.01)

        dh, dr = command(memory, 1)
        rewards[i] = episode(env, model, dh, dr, nA, maxt, maxreward, 0.0)[1]
    end
    return rewards
end

env = CartPole{Float32}()

model = BehaviorModel(length(env.observationspace), length(env.actionspace))

r = run!(model, ADAM(0.01), env, 100f0)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "udrl")
