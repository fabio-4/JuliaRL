include("../RLAlgorithms.jl")
using .RLAlgorithms
using Plots
using Flux
using Flux: Optimise.update!

function act(model, s, ϵ, nA)
    rand() <= ϵ && return rand(1:nA)
    return Flux.onecold(model(s))
end

function run!(model, opt, env; 
        epochs=100, steps=300, maxt=100, batchsize=128, 
        trainiters=100, ϵ=0.03, γ=99f-2, τ=1f-2)
    rewards = zeros(Float32, epochs)
    target = deepcopy(model)
    memory = ReplayMemory{Float32, Int64, Float32}(
        length(env.observationspace), 1, 10*steps, batchsize
    )
    nA = length(env.actionspace)
    ps = params(model)

    for i in 1:epochs
        action = let ϵ = ϵ; s -> act(model, s, ϵ, nA) end
        episodes!(action, memory, env, steps, maxt)

        for _ in 1:trainiters
            s1, a, r, s2, d = sample(memory)
            Qs2 = Flux.onecold(model(s2))
            Qtar = r .+ (1 .- d) .* γ .* gather(target(s2), Qs2)
            gs = gradient(ps) do
                Qs1a = gather(model(s1), a)
                return Flux.mse(Qs1a, Qtar)
            end
            update!(opt, ps, gs)
            softupdate!(target, model, τ)
        end

        ϵ = max(ϵ * 0.99, 0.01)
        rewards[i] = test(s -> Flux.onecold(model(s)), env)
    end
    return rewards
end

env = CartPole{Float32}()

model = Chain(
    Dense(length(env.observationspace), 32, relu), 
    Dense(32, length(env.actionspace))
)

r = run!(model, ADAM(), env)
plt = plot(r, labels="Reward")
display(plt)
#savefig(plt, "ddqn")
