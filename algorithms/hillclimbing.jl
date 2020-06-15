include("../envs/RLEnvironments.jl")
using .RLEnvironments
using Plots

act(W, s) = argmax(W * s)

function run!(W, env; epochs=200, maxt=100, noise=1f-2, γ=99f-2)
    rewards = zeros(Float32, epochs)
    steps = zeros(Float32, epochs)

    bestR = typemin(Float32)
    bestW = copy(W)
    for i in 1:epochs
        reset!(env)
        done = false
        Ri = 0f0
        j = 0f0
        while !done && j < maxt
            _, r, done = step!(env, act(W, env.current))
            Ri += γ^j * r
            j += 1f0
        end
        
        if Ri >= bestR
            bestR = Ri
            bestW = copy(W)
            noise = max(1f-3, noise/2f0)
            W += noise .* rand(Float32, size(W))
        else
            noise = min(2f0, noise*2f0)
            W = bestW + noise .* rand(Float32, size(W))
        end
        rewards[i] = Ri
        steps[i] = j
    end
    W .= bestW
    return rewards, steps
end

env = CartPole{Float32}()
W = 1f-4 * randn(Float32, length(env.actionspace), length(env.observationspace))

r, s = run!(W, env)
plt = plot([r s], labels=["Reward" "Steps"], color=[:red :blue], layout=(2, 1))
display(plt)
#savefig(plt, "hillclimbing+adaptivenoisescaling")
