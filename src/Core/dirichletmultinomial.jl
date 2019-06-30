
"""
    DirichletFast{T<:Real} <:  ContinuousMultivariateDistribution

Dirichlet distribution as a prior to multinomial parameters.

The difference between `DirichletFast` and `Dirichlet` is that `randn`
returns `MultinomialFast` distribution in `DirichletFast`.

It also does not calculate normalization constant at any time,
so it has faster constructor than `Dirichlet`.

see [`MultinomialFast`](@ref)
"""
struct DirichletFast{T<:Real} <: ContinuousMultivariateDistribution
    α::Vector{T}
end

DirichletFast(α::Vector{<:Integer}) = DirichletFast{Float64}(Float64.(α))

# convert(::Type{DirichletFast{T}}, α::Vector{S}) where {T<:Real, S<:Real} =
#     DirichletFast(convert(Vector{T}, α))
# convert(::Type{DirichletFast{T}}, d::DirichletFast{S}) where {T<:Real, S<:Real} =
#     DirichletFast(convert(Vector{T}, d.α))
@inline length(d::DirichletFast) = length(d.α)
params(d::DirichletFast) = (d.α,)
@inline partype(d::DirichletFast{T}) where {T<:Real} = T
#rand(d::DirichletFast) = _rand!(GLOBAL_RNG,d,Array{Float64,1}(undef,length(d.α)))

function _rand!(rng::Random.MersenneTwister, d::DirichletFast{T}, x::AbstractVector) where T
    s = T(0)
    n = length(d)
    α = d.α
    @simd for i in eachindex(α)
         @inbounds s += (x[i] = rand(rng,Gamma(α[i])))
    end
    lgs = log(s)
    @simd for i in eachindex(α)
         @inbounds x[i] = log(x[i]) - lgs
    end
    MultinomialFast(x)
end

@inline rand(d::DirichletCanon) = _rand!(GLOBAL_RNG,d,similar(d.alpha))

@inline lmllh(prior::DirichletFast, posterior::DirichletFast, n::Int)  = 
    lgamma(sum(prior.α))-lgamma(sum(posterior.α)) + sum(lgamma.(posterior.α) .- lgamma.(prior.α))


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
@inline length(d::MultinomialFast) = length(d.logp)
probs(d::MultinomialFast) = exp.(d.logp)
params(d::MultinomialFast) = (d.logp,)
@inline partype(d::MultinomialFast{T}) where {T<:Real} = T

function logαpdf(d::MultinomialFast{T}, x::DPSparseVector) where T<:Real
    logp  = d.logp
    nzval = nonzeros(x)
    s     = T(0)
    @simd for l in enumerate(x.nzind)
        @inbounds s += logp[last(l)]*nzval[first(l)]
    end
    return s
end

function logαpdf(d::MultinomialFast{T}, x::AbstractVector) where T<:Real
    logp = d.logp
    s = T(0)
    @simd for i in eachindex(logp)
        @inbounds s += logp[i]*T(x[i])
    end
    return s
end

function _logpdf(d::MultinomialFast{T}, x::AbstractVector{T}) where T<:Real
    n = sum(x)
    logp = d.logp
    S = partype(d)
    s = S(lgamma(n + 1))
    for i in eachindex(x)
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

struct DirichletMultPredictive{T<:Real} <:  ContinuousMultivariateDistribution
    α::Vector{T}
    sα::T
    lgsα_slgα::T
    function DirichletMultPredictive{T}(α::Vector{T}) where T
        sα  = sum(α)
        return new{T}(α, sα, lgamma(sα)-sum(lgamma,α))
    end
end
@inline DirichletMultPredictive(α::Vector{T}) where T<:Real = DirichletMultPredictive{T}(α)
@inline DirichletMultPredictive(α::Vector{<:Integer}) = DirichletMultPredictive{Float64}(convert(Vector{Float64},α))

@inline length(d::DirichletMultPredictive) = length(d.α)
params(d::DirichletMultPredictive) = (d.α,)
partype(d::DirichletMultPredictive{T}) where {T<:Real} = T

@inline logαpdf(d::DirichletMultPredictive, x::AbstractVector) = d.lgsα_slgα + sum(lgamma, d.α .+ x) - lgamma(sum(x) + d.sα)

function logαpdf(d::DirichletMultPredictive{T}, x::DPSparseVector) where T
    # log predictive probability of xx given other data items in the component
    # log p(xi|x_1,...,x_n)
    n     = sum(x)
    sα    = sum(d.α)
    dαpx  = add!(copy(d.α),x)
    onepx = add!(ones(T,length(d)),x)

    return lgamma(n+1) -
           sum(lgamma,onepx) +
           lgamma(sα) -
           sum(lgamma,d.α) +
           sum(lgamma,dαpx)-
           lgamma(n + sα)
end
