struct Pendulum{S, A, R} <: ContEnv{S, A, R}
    state::Vector{S}
    current::Vector{S}
    observationspace::ObservationSpace{Vector{S}}
    actionspace::ActionSpace{Tuple{A, A}}
    function Pendulum{T}() where T <: AbstractFloat
        observationspace = ObservationSpace{Vector{T}}(3)
        actionspace = ActionSpace([(-one(T), one(T))])
        state = [rand(Uniform(-π, π)), rand(Uniform(-1.0, 1.0))]
        current = [cos(state[1]), sin(state[1]), state[2]]
        new{T, T, T}(state, current, observationspace, actionspace)
    end
end

Pendulum() = Pendulum{Float32}()

function obs!(env::Pendulum)
    env.current[1] = cos(env.state[1])
    env.current[2] = sin(env.state[1])
    env.current[3] = env.state[2]
end

step!(env::Pendulum{S, A, R}, action::Vector{A}) where {S, A, R} = step!(env, action[1])
function step!(env::Pendulum{S, A, R}, action::A) where {S, A, R}
    θ, θ̇ = env.state
    u = clamp(2.0 * action, -2.0, 2.0)
    costs = (((θ + π) % (2.0 * π)) - π) ^ 2.0 + 0.1 * (θ̇ ^ 2.0) + 0.001 * (u ^ 2.0)
    θ̇ += (-15.0 * sin(θ + π) + 3.0 * u) * 0.05
    θ += θ̇ * 0.05
    θ̇ = clamp(θ̇, -8.0, 8.0)
    env.state .= θ, θ̇
    obs!(env)
    return env.current, -R(costs), false
end

function reset!(env::Pendulum)
    env.state .= rand(Uniform(-π, π)), rand(Uniform(-1.0, 1.0))
    obs!(env)
    return env.current
end
