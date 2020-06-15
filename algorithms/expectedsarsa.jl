include("../envs/RLEnvironments.jl")
using .RLEnvironments
using Plots

function act(Q, s, ϵ)
    rand() <= ϵ && return rand(1:size(Q, 1))
    a = @view Q[:, s]
    return rand(findall(a .== maximum(a)))
end

function run!(Q, env; epochs=20, ϵ=0.005, α=0.5, γ=0.99)
    rewards = zeros(epochs)
    steps = zeros(Int64, epochs)
    numactions = length(env.actionspace)

    for i in 1:epochs
        done = false
        s1 = reset!(env)
        while !done
            a = act(Q, s1, ϵ)
            s2, r, done = step!(env, a)
            Qs = view(Q, :, s2)
            r2 = done ? 0.0 : (sum(Qs) * (ϵ / numactions) + maximum(Qs) * (1.0 - ϵ))
            Q[a, s1] += α * (r + γ * r2 - Q[a, s1])
            s1 = s2

            rewards[i] += r
            steps[i] += 1
        end
    end
    return rewards, steps
end

env = SimpleRooms{Int64, Int64, Float64}()
Q = zeros(length(env.actionspace), length(env.observationspace))

r, s = run!(Q, env)
plt = plot([r s], labels=["Reward" "Steps"], color=[:red :blue], layout=(2, 1))
display(plt)
#savefig(plt, "expectedsarsa")
