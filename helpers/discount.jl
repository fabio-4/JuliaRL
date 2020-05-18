function discount!(rs::AbstractVector{T}, γ::T) where T <: AbstractFloat
    @inbounds for i in length(rs)-1:-1:1
        rs[i] += γ * rs[i+1]
    end
end

function nstepr!(rs::AbstractVector{T}, brs::AbstractVector{T}, γ::T, n::Int64) where T <: AbstractFloat
    @inbounds for t in 1:length(rs)
        runningadd = zero(T)
        l = min(n, length(rs)-t)
        for i in 0:l
            runningadd += γ^(i) * (i == n ? brs[t+i] : rs[t+i])
        end
        rs[t] = runningadd
    end
end

function GAEλ!(vs::AbstractVector{T}, rs::AbstractVector{T}, γ::T, λ::T) where T <: AbstractFloat
    vs[1:end-1] .= rs[1:end-1] .+ γ .* vs[2:end] .- vs[1:end-1]
    vs[end] = rs[end] - vs[end]
    discount!(vs, γ*λ)
end

function norm!(vs::AbstractVector{T}) where T <: AbstractFloat
    θmean = mean(vs)
    θstd = std(vs, mean=θmean)
    vs .= (vs .- θmean) ./ (θstd + T(1f-12))
end
