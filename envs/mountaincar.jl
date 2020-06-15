struct MountainCar{S, A, R} <: DiscEnv{S, A, R}
    current::Vector{S}
    observationspace::ObservationSpace{Vector{S}}
    actionspace::ActionSpace{A}
    function MountainCar{T}() where T <: AbstractFloat
        observationspace = ObservationSpace{Vector{T}}(2)
        actionspace = ActionSpace(collect(T, -1:1))
        new{T, T, T}(randpos(), observationspace, actionspace)
    end
end

MountainCar() = MountainCar{Float32}()

struct MountainCarContinuous{S, A, R} <: ContEnv{S, A, R}
    current::Vector{S}
    observationspace::ObservationSpace{Vector{S}}
    actionspace::ActionSpace{Tuple{A, A}}
    function MountainCarContinuous{T}() where T <: AbstractFloat
        observationspace = ObservationSpace{Vector{T}}(2)
        actionspace = ActionSpace([(-one(T), one(T))])
        new{T, T, T}(randpos(), observationspace, actionspace)
    end
end

MountainCarContinuous() = MountainCarContinuous{Float32}()

randpos() = [rand(Uniform(-0.6, -0.4)), 0.0]

step!(env::MountainCar, action::Integer) = step!(env, env.actionspace.actions[action])
function step!(env::MountainCar{S, A, R}, action::A) where {S, A, R}
    p, v = env.current
    v += action * 0.001 - 0.0025 * cos(3.0*p)
    v = clamp(v, -0.07, 0.07)
    p = clamp(p + v, -1.2, 0.6)
    if p == -1.2 && v < 0.0
        v = 0.0
    end
    d = p >= 0.5
    r = d ? zero(R) : -one(R)
    env.current .= p, v
    return env.current, r, d
end

step!(env::MountainCarContinuous{S, A, R}, action::Vector{A}) where {S, A, R} = step!(env, action[1])
function step!(env::MountainCarContinuous{S, A, R}, action::A) where {S, A, R}
    p, v = env.current
    v += clamp(action, -1.0, 1.0) * 0.0015 - 0.0025 * cos(3.0*p)
    v = clamp(v, -0.07, 0.07)
    p = clamp(p + v, -1.2, 0.6)
    if p == -1.2 && v < 0.0
        v = 0.0
    end
    d = p >= 0.45
    r = (d ? 100.0 : 0.0) - (action ^ 2.0) * 0.1
    env.current .= p, v
    return env.current, R(r), d
end

reset!(env::Union{MountainCar, MountainCarContinuous}) = (env.current .= randpos())
