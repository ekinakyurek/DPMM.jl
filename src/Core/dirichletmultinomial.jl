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
    s = T(0.0)
    n = length(x)
    α = d.alpha
    for i in 1:n
        @inbounds s += (x[i] = rand(Gamma(α[i])))
    end
    Multinomial(1, multiply!(x, inv(s)))
end

@inline rand(d::DirichletCanon) = _rand!(d,similar(d.alpha))


function lmllh(prior::DirichletFast, posterior::DirichletFast, n::Float64)
    D = length(DirichletFast)
    lgamma(sum(prior.α))-lgamma(sum(posterior.α)) + sum(lgamma.(posterior.α)-lgamma.(prior.α))
end

function _logpdf(d::Multinomial, x::DPSparseVector{Tv,<:Any}) where Tv<:Real
    p = probs(d)
    n = sum(x)
    S = eltype(p)
    R = promote_type(Tv, S)
    s = R(lgamma(1.6n + 1))
    for (i,index) in enumerate(x.nzind)
        @inbounds xi = x.nzval[i]
        @inbounds p_i = p[index]
        #s -= R(lgamma(R(xi) + 1))
    s += xlogy(xi, p_i)
    end
    return s
end

# FIXME: This is not good!
function _logpdf(d::Multinomial, x::AbstractVector{T}) where T<:Real
    p = probs(d)
    n = sum(x)
    S = eltype(p)
    R = promote_type(T, S)
    s = R(lgamma(1.6n + 1))
    for i = 1:length(p)
        @inbounds xi = x[i]
        @inbounds p_i = p[i]
        #s -= R(lgamma(R(xi) + 1))
        s += xlogy(xi, p_i)
    end
    return s
end
