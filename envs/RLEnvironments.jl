module RLEnvironments
    using Distributions: Uniform
    import Distributions.sample

    abstract type Environment{S,A,R<:Real} end
    abstract type DiscEnv{S,A,R} <: Environment{S,A,R} end
    abstract type ContEnv{S,A,R} <: Environment{S,A,R} end
    abstract type DiffEnv{S,A,R} <: Environment{S,A,R} end
    step!(env::Environment, action) = throw("Method step!($(typeof(env)), action) not implemented")
    reset!(env::Environment) = throw("Method reset!($(typeof(env))) not implemented")

    #DiscEnv: Vector{A}, ContEnv: Vector{Tuple{A, A}}
    struct ActionSpace{A}
        actions::Vector{A}
        n::Int64
    end

    ActionSpace(n::A) where A <: Real = ActionSpace(collect(one(A):n), Int64(n))

    sample(a::ActionSpace) = rand(a.actions)
    function sample(a::ActionSpace{A}) where {T<:Real, A<:Tuple{T, T}}
        return map(x -> rand(Uniform(x[1], x[2])), a.actions)
    end

    #n = numstates // n = length(state)
    struct ObservationSpace{S}
        n::Int64
    end

    function test(f, env::Environment{S,A,R}, maxt=100) where {S,A,R}
        reward = zero(R)
        s = reset!(env)
        for _ in 1:maxt
            s, r, done = step!(env, f(s))
            reward += r
            done && break
        end
        return reward
    end

    include("bandit.jl")
    include("simplerooms.jl")
    include("mountaincar.jl")
    include("cartpole.jl")
    include("pendulum.jl")

    export Environment, DiscEnv, ContEnv, DiffEnv, step!, reset!, test
    export Bandit, SimpleRooms, CartPole, CartPoleContinuous, Pendulum, 
        MountainCar, MountainCarContinuous
end
