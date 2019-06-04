"""
    DirichletFast{T<:Real} <:  ContinuousMultivariateDistribution

Dirichlet distribution as a prior to multinomial parameters.

The difference between `DirichletFast` and `Dirichlet` is that `randn`
returns `MultinomialFast` distribution in `DirichletFast`.

It also does not calculate normalization constant at any time,
so it has faster constructor than `Dirichlet`.

see [`MultinomialFast`](@ref)
"""
struct DirichletFast{T<:Real} <:  ContinuousMultivariateDistribution
    α::Vector{T}
end

DirichletFast(d::Integer, α::T) where {T<:Real} = DirichletFast{T}(d, α)
DirichletFast(α::Vector{T}) where {T<:Integer} = DirichletFast{Float64}(convert(Vector{Float64},α))
DirichletFast(d::Integer, α::Integer) = DirichletFast{Float64}(d, Float64(α))

convert(::Type{DirichletFast{T}}, α::Vector{S}) where {T<:Real, S<:Real} =
    DirichletFast(convert(Vector{T}, α))
convert(::Type{DirichletFast{T}}, d::DirichletFast{S}) where {T<:Real, S<:Real} =
    DirichletFast(convert(Vector{T}, d.α))

length(d::DirichletFast) = length(d.α)
params(d::DirichletFast) = (d.α,)
@inline partype(d::DirichletFast{T}) where {T<:Real} = T

function _rand!(d::DirichletFast{T}, x::AbstractVector{<:Real}) where T
    s = T(0)
    n = length(x)
    α = d.α
    for i in 1:n
        @inbounds s += (x[i] = rand(Gamma(α[i])))
    end
    MultinomialFast(log.(multiply!(x, inv(s))))
end

@inline rand(d::DirichletCanon) = _rand!(GLOBAL_RNG,d,similar(d.alpha))

function lmllh(prior::DirichletFast, posterior::DirichletFast, n::Int)
    lgamma(sum(prior.α))-lgamma(sum(posterior.α)) + sum(lgamma.(posterior.α) .- lgamma.(prior.α))
end

###
#### Multinomial
###
"""
    MultionmialFast{T<:Real} <:  ContinuousMultivariateDistribution

Multinomial distribution is redifined for the purpose of fast likelihood calculations
on `DPSparseVector`.

The other difference between `MultinomialFast` and `Multionomial` is that
The `n`: trial numbers is not set. It is calculated by the input vector in the pdf function.
So, it can produce pdf for any discrete `x` vector.
"""
struct MultinomialFast{T<:Real} <: DiscreteMultivariateDistribution
    logp::Vector{T}
end

# Parameters
ncategories(d::MultinomialFast) = length(d.logp)
length(d::MultinomialFast) = length(d.logp)
probs(d::MultinomialFast) = exp.(d.logp)
params(d::MultinomialFast) = (d.logp,)
@inline partype(d::MultinomialFast{T}) where {T<:Real} = T

function logαpdf(d::MultinomialFast{T}, x::DPSparseVector) where T<:Real
    logp = d.logp
    n    = sum(x)
    s = T(0)
    for (i,index) in enumerate(x.nzind)
        @inbounds s += logp[index]*x.nzval[i]
    end
    return s
end

# FIXME: This is not good!
@inline logαpdf(d::MultinomialFast, x::AbstractVector{T}) where T<:Real  = dot(x, d.logp)

function _logpdf(d::MultinomialFast{T}, x::AbstractVector{T}) where T<:Real
    n = sum(x)
    logp = d.logp
    S = partype(d)
    s = S(lgamma(n + 1))
    for i = 1:length(x)
        @inbounds xi = x[i]
        @inbounds p_i = logp[i]
        s -= S(lgamma(S(xi) + 1))
        s += xi * p_i
    end
    return s
end

function _logpdf(d::MultinomialFast{T}, x::DPSparseVector) where T<:Real
    n = sum(x)
    logp = d.logp
    S = partype(d)
    s = S(lgamma(n + 1))
    for (i,index) in enumerate(x.nzind)
        @inbounds xi = x.nzval[i]
        @inbounds p_i = logp[index]
        s -= S(lgamma(S(xi) + 1))
        s += xi * p_i
    end
    return s
end
