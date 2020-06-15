include("../envs/RLEnvironments.jl")
using .RLEnvironments
using Plots
using Distributions: Beta

abstract type Policy end
act(p::Policy) = throw("Method act($(typeof(p))) not implemented")

struct Greedy{T<:AbstractFloat} <: Policy
    avgrewards::Vector{T}
    counts::Vector{T}
    Greedy{T}(n) where T <: AbstractFloat = new(zeros(T, n), zeros(T, n))
end

struct EpsGreedy{T<:AbstractFloat} <: Policy
    avgrewards::Vector{T}
    counts::Vector{T}
    ϵ::Float64
    EpsGreedy{T}(n, ϵ) where T <: AbstractFloat = new(zeros(T, n), zeros(T, n), ϵ)
end

struct OptGreedy{T<:AbstractFloat} <: Policy
    avgrewards::Vector{T}
    counts::Vector{T}
    OptGreedy{T}(n, initvalue) where T <: AbstractFloat = new(fill(T(initvalue), n), zeros(T, n))
end

mutable struct RoundRobin{T<:AbstractFloat} <: Policy
    avgrewards::Vector{T}
    counts::Vector{T}
    n::Int64
    curraction::Int64
    RoundRobin{T}(n) where T <: AbstractFloat = new(zeros(T, n), zeros(T, n), n, n)
end

mutable struct UCB{T<:AbstractFloat} <: Policy
    avgrewards::Vector{T}
    counts::Vector{T}
    n::Int64
    round::Int64
    UCB{T}(n) where T <: AbstractFloat = new(zeros(T, n), zeros(T, n), n, 0)
end

struct ThompsonBeta <: Policy
    successes::Vector{Int64}
    failures::Vector{Int64}
    ThompsonBeta(n) = new(ones(n), ones(n))
end

function greedyaction(p)
    maxaction = argmax(p.avgrewards)
    if p.avgrewards[maxaction] < 1.0
        unused = findall(p.counts .== 0)
        length(unused) > 0 && return rand(unused)
    end
    return maxaction
end

act(p::Union{Greedy, OptGreedy}) = greedyaction(p)

act(p::EpsGreedy) = rand() <= p.ϵ ? rand(1:length(p.avgrewards)) : greedyaction(p)

act(p::RoundRobin) = (p.curraction = (p.curraction % p.n) + 1)

function act(p::UCB)
    p.round += 1
    p.round <= p.n && return p.round
    return argmax(p.avgrewards .+ sqrt.((2.0 * log(p.round)) ./ p.counts))
end

act(p::ThompsonBeta) = argmax(map(x -> rand(Beta(x[1], x[2])), zip(p.successes, p.failures)))

function improve!(p, action, reward)
    p.counts[action] += 1.0
    p.avgrewards[action] += (reward - p.avgrewards[action]) / p.counts[action]
end

function improve!(p::ThompsonBeta, action, reward)
    if reward > 0.0
        p.successes[action] += 1
    else
        p.failures[action] += 1
    end
end

function run!(p, env::Bandit{A, R}; steps=100) where {A, R}
    rewards = zeros(R, steps)
    regrets = zeros(R, steps)
    for i in 1:steps
        action = act(p)
        reward, regret = step!(env, action)
        improve!(p, action, reward)

        rewards[i] = reward #+ (i > 1 ? rewards[i-1] : 0)
        regrets[i] = regret #+ (i > 1 ? regrets[i-1] : 0)
    end
    return rewards, regrets
end

function getpolicies(env::Bandit{A, R}) where {A, R}
    n = length(env.actionspace)
    return [Greedy{R}(n), EpsGreedy{R}(n, 0.03), OptGreedy{R}(n, 1.0), 
        RoundRobin{R}(n), UCB{R}(n), ThompsonBeta(n)]
end

title(t) = replace(t, r"\{(.*?)\}" => s"")

env = Bandit()
policies = getpolicies(env)
plt = plot(layout=(length(policies), 1), size=(512, 1024))
for (i, p) in enumerate(policies)
    rs, rgs = run!(p, env)
    plot!([rs rgs], labels=["Reward" "Regret"], title=title("$(typeof(p))"), subplot=i)
end
display(plt)
#savefig(plt, "bandits")
