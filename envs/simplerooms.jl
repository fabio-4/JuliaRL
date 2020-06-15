mutable struct SimpleRooms{S<:Integer, A<:Integer, R<:Real} <: DiscEnv{S, A, R}
    current::S
    observationspace::ObservationSpace{S}
    actionspace::ActionSpace{Symbol}
    transitions::Dict{S, Vector{S}}
    rewards::Vector{R}
    function SimpleRooms{S, A, R}() where {S<:Integer, A<:Integer, R<:Real}
        observationspace = ObservationSpace{S}(16)
        actionspace = ActionSpace([:up, :down, :left, :right])
        transitions = Dict{S, Vector{S}}(
            1 => [2, 5], 2 => [1, 3, 6], 3 => [2, 4, 7], 4 => [3, 8],
            5 => [1, 6, 9], 6 => [2, 5], 7 => [3, 8], 8 => [4, 7, 12],
            9 => [5, 10, 13], 10 => [9, 14], 11 => [12, 15], 12 => [8, 11, 16],
            13 => [9, 14], 14 => [10, 13, 15], 15 => [11, 14, 16], 16 => [12, 15]
        )
        rewards = zeros(R, 16)
        rewards[16] = one(R)
        new{S, A, R}(1, observationspace, actionspace, transitions, rewards)
    end
end

SimpleRooms() = SimpleRooms{Int64, Int64, Float64}()

function singleStep(s, a, t, r)
    if a == :up && (s-4 in t[s])
        return s-4, r[s-4]
    elseif a == :down && (s+4 in t[s])
        return s+4, r[s+4]
    elseif a == :left && (s-1 in t[s])
        return s-1, r[s-1]
    elseif a == :right && (s+1 in t[s])
        return s+1, r[s+1]
    end
    return s+0, zero(eltype(r))
end

step!(env::SimpleRooms, action::Integer) = step!(env, env.actionspace.actions[action])

function step!(env::SimpleRooms, action::Symbol)
    env.current, r = singleStep(env.current, action, env.transitions, env.rewards)
    return env.current, r, r == 1.0
end

reset!(env::SimpleRooms) = (env.current = 1)
