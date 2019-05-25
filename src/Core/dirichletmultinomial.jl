struct DirichletFast{T<:Real} <:  ContinuousMultivariateDistribution
    alpha::Vector{T}
end

DirichletFast(d::Integer, alpha::T) where {T<:Real} = DirichletFast{T}(d, alpha)
DirichletFast(alpha::Vector{T}) where {T<:Integer} = DirichletFast{Float64}(convert(Vector{Float64},alpha))
DirichletFast(d::Integer, alpha::Integer) = DirichletFast{Float64}(d, Float64(alpha))

convert(::Type{DirichletFast{T}}, alpha::Vector{S}) where {T<:Real, S<:Real} =
    DirichletFast(convert(Vector{T}, alpha))
convert(::Type{DirichletFast{T}}, d::DirichletFast{S}) where {T<:Real, S<:Real} =
    DirichletFast(convert(Vector{T}, d.alpha))

length(d::DirichletFast) = length(d.alpha)
params(d::DirichletFast) = (d.alpha,)
@inline partype(d::DirichletFast{T}) where {T<:Real} = T

function _rand!(d::DirichletFast{T}, x::AbstractVector{<:Real}) where T
    s = T(0)
    n = length(x)
    α = d.alpha
    for i in 1:n
        @inbounds s += (x[i] = rand(Gamma(α[i])))
    end
    MultinomialFast(log.(multiply!(x, inv(s))))
end

@inline rand(d::DirichletCanon) = _rand!(d,similar(d.alpha))

function lmllh(prior::DirichletFast, posterior::DirichletFast, n::Int)
    D = length(prior)
    lgamma(sum(prior.alpha))-lgamma(sum(posterior.alpha)) + sum(lgamma.(posterior.alpha) .- lgamma.(prior.alpha))
end

###
#### Multinomial
###
struct MultinomialFast{T<:Real} <: DiscreteMultivariateDistribution
    logp::Vector{T}
end

# Parameters
ncategories(d::MultinomialFast) = length(d.logp)
length(d::MultinomialFast) = length(d.logp)
probs(d::MultinomialFast) = exp.(d.logp)
params(d::Multinomial) = (d.logp,)
@inline partype(d::Multinomial{T}) where {T<:Real} = T

function logprob(d::MultinomialFast{T}, x::DPSparseVector) where T<:Real
    logp = d.logp
    n    = sum(x)
    s = T(0)
    for (i,index) in enumerate(x.nzind)
        @inbounds s += logp[index]*x.nzval[i]
    end
    return s
end

# FIXME: This is not good!
@inline logprob(d::MultinomialFast, x::AbstractVector{T}) where T<:Real = dot(x, d.logp)
