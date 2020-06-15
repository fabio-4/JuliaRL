include("../envs/RLEnvironments.jl")
using .RLEnvironments
using Plots

function act(Q, s, ϵ)
    rand() <= ϵ && return rand(1:size(Q, 1))
    a = @view Q[:, s]
    return rand(findall(a .== maximum(a)))
end

function episode!(e, env, maxt, Q, ϵ)
    reset!(env)
    done = false
    t = 0
    while !done && t < maxt
        t += 1
        s = env.current
        a = act(Q, s, ϵ)
        _, r, done = step!(env, a)
        e[t] = (s, a, r)
    end
    return t
end

function run!(Q, env; epochs=20, maxt=100, ϵ=0.1, α=0.5, γ=0.99)
    rewards = zeros(epochs)
    steps = zeros(Int64, epochs)
    e = Vector{Tuple{Int64, Int64, Float64}}(undef, maxt)

    for i in 1:epochs
        t = episode!(e, env, maxt, Q, ϵ)
        rdisc = 0.0
        for step in Iterators.reverse(view(e, 1:t))
            rdisc = step[3] + γ * rdisc
            Q[step[2], step[1]] += α * (rdisc - Q[step[2], step[1]])
            
            rewards[i] += step[3]
        end
        steps[i] = t
        ϵ = max(ϵ * 0.995, 0.01)
    end
    return rewards, steps
end

env = SimpleRooms{Int64, Int64, Float64}()
Q = zeros(length(env.actionspace), length(env.observationspace))

r, s = run!(Q, env)
plt = plot([r s], labels=["Reward" "Steps"], color=[:red :blue], layout=(2, 1))
display(plt)
#savefig(plt, "mccontrol")
