module RLAlgorithms
    include("envs/RLEnvironments.jl")
    include("helpers/RLHelpers.jl")
    using Reexport
    @reexport using .RLHelpers
    @reexport using .RLEnvironments
    using StatsBase: sample, Weights

    export episode!, episodes!

    function episodes!(f, memory, env, steps, maxt)
        j = 0
        while j < steps
            j += episode!(f, memory, env, min(maxt, steps-j))
        end
    end

    function episode!(f, memory::ReplayMemory{S, A, R}, 
            env::Environment{S, B, R}, maxt) where {S, A, B, R}
        reset!(env)
        done = false
        t = 0
        while t < maxt && !done
            s = copy(env.current)
            a = f(s)
            _, r, done = step!(env, a)
            append!(memory, s, a, r, done)
            t += 1
        end
        return t
    end

    function episode!(f, episode::PGEpisode{S, A, R}, 
            env::Environment{S, B, R}, maxt) where {S, A, B, R}
        reset!(env)
        done = false
        for _ in 1:maxt-1
            s = copy(env.current)
            p, v = f(s)
            a = sample(Weights(p))
            _, r, done = step!(env, a)
            append!(episode, s, a, r, v)
            done && break
        end
        if !done
            s = env.current
            p, v = f(s)
            append!(episode, s, argmax(p), v, v)
        end
        nothing
    end

    function episode!(f, memory::PGReplayMemory{S, A, R}, 
            env::Environment{S, B, R}, maxt) where {S, A, B, R}
        t = 0
        done = false
        reset!(env)
        while t < maxt && !done
            s = copy(env.current)
            a, loga, v = f(s)
            _, r, done = step!(env, a)
            append!(memory, s, a, loga, r, v)
            t += 1
        end
        lastval = done ? zero(R) : f(env.current)[3]
        finish!(memory, t, lastval)
        return t
    end
end
