struct CartPole{S, A, R} <: DiscEnv{S, A, R}
    current::Vector{S}
    observationspace::ObservationSpace{Vector{S}}
    actionspace::ActionSpace{A}
    function CartPole{T}() where T <: AbstractFloat
        observationspace = ObservationSpace{Vector{T}}(4)
        actionspace = ActionSpace([-one(T), one(T)])
        new{T, T, T}(rand(Uniform(-0.05, 0.05), 4), observationspace, actionspace)
    end
end

CartPole() = CartPole{Float32}()

struct CartPoleContinuous{S, A, R} <: ContEnv{S, A, R}
    current::Vector{S}
    observationspace::ObservationSpace{Vector{S}}
    actionspace::ActionSpace{Tuple{A, A}}
    function CartPoleContinuous{T}() where T <: AbstractFloat
        observationspace = ObservationSpace{Vector{T}}(4)
        actionspace = ActionSpace([(-one(T), one(T))])
        new{T, T, T}(rand(Uniform(-0.05, 0.05), 4), observationspace, actionspace)
    end
end

CartPoleContinuous() = CartPoleContinuous{Float32}()

step!(env::CartPole, action::Integer) = step!(env, env.actionspace.actions[action])
step!(env::CartPoleContinuous{S, A, R}, action::Vector{A}) where {S, A, R} = step!(env, action[1])
function step!(env::Union{CartPole{S, A, R}, CartPoleContinuous{S, A, R}}, action::A) where {S, A, R}
    x, ẋ, θ, θ̇ = env.current
    f = 10.0 * action
    θcos = cos(θ)
    θsin = sin(θ)
    tmp = (f + 0.05 * θ̇ * θ̇ * θsin) / 1.1
    θacc = (9.8 * θsin - θcos * tmp) / (0.5 * (4.0/3.0 - 0.1 * θcos * θcos / 1.1))
    xacc = tmp - 0.05 * θacc * θcos / 1.1

    x += 0.02 * ẋ
    ẋ += 0.02 * xacc
    θ += 0.02 * θ̇
    θ̇ += 0.02 * θacc

    env.current .= x, ẋ, θ, θ̇
    θthr = 2.0 * 12.0 * π / 360.0
    done = !(-2.4 <= x <= 2.4 && -θthr <= θ <= θthr)
    return env.current, R(!done), done
end

reset!(env::Union{CartPole, CartPoleContinuous}) = (env.current .= rand(Uniform(-0.05, 0.05), 4))
