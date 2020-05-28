using Random: seed!

struct Bandit{A<:Integer, R<:AbstractFloat}
    actionspace::ActionSpace{A}
    rwdparams::Matrix{R}
    optsol::R
    function Bandit{A, R}(n=10, seed=1234) where {A<:Integer, R<:AbstractFloat}
        seed!(seed)
        rwdparams = randn(R, n, 2)
        new(ActionSpace(A(n)), rwdparams, maximum(rwdparams[:, 1]))
    end
end

Bandit() = Bandit{Int64, Float64}()

function step!(env::Bandit, action::Integer)
    reward = env.rwdparams[action, 1] + env.rwdparams[action, 2] * randn(eltype(env.rwdparams))
    regret = abs(env.optsol - env.rwdparams[action, 1])
    return reward, regret
end

reset!(::Bandit) = nothing
