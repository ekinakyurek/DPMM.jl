const logpi = log(π)

"""
    NormalWishart{T<:Real,S<:AbstractPDMat} <: ContinuousUnivariateDistribution

Normal Inverse Wishart distribution is prior for `MvNormalFast` distribution.

see [`MvNormalFast`](@ref)
"""
struct NormalWishart{T<:Real,S<:AbstractPDMat} <: ContinuousUnivariateDistribution
    μ::Vector{T}
    λ::T  # This scales precision (inverse covariance)
    Ψ::S
    ν::T
    function NormalWishart{T}(μ::Vector{T}, λ::T, Ψ::AbstractPDMat{T}, ν::T) where T<:Real
        new{T,typeof(Ψ)}(μ, λ, Ψ, ν)
    end
end

@inline length(d::NormalWishart) = length(d.μ)

function NormalWishart(μ::Vector{U}, λ::Real,
                              Ψ::AbstractPDMat{S}, ν::Real) where {S<:Real, U<:Real}
    T = promote_type(eltype(μ), typeof(λ), typeof(ν), S)
    return NormalWishart{T}(Vector{T}(μ), T(λ), AbstractPDMat{T}(Ψ), T(ν))
end

function NormalWishart(μ::Vector{U}, λ::Real,
                              Ψ::Matrix{S}, ν::Real) where {S<:Real, U<:Real}
    T = promote_type(eltype(μ), typeof(λ), typeof(ν), S)
    return NormalWishart{T}(Vector{T}(μ), T(λ), PDMat(Ψ), T(ν))
end

NormalWishart{T}(n::Int) where T<:Real =
    NormalWishart(zeros(T,n),T(1),Matrix{T}(I,n,n),T(n)+3)

NormalWishart{T}(mean::AbstractArray{T}) where T<:Real =
    NormalWishart(mean,T(1),Matrix{T}(I,length(mean),length(mean)),T(length(mean))+3)

NormalWishart{T}(mean::AbstractArray{T}, cov::AbstractMatrix{T}) where T<:Real =
    NormalWishart(mean,T(1),cov,T(length(mean))+3)

# function rand(niw::NormalWishart{T,<:Any}) where T
#     Σ   = rand(InverseWishart(niw.ν, niw.Ψ))
#     μ   = rand(MvNormal(niw.μ, Σ ./ niw.λ))
#     return MvNormal(μ, Σ)
# end

function rand(niw::NormalWishart{T,<:Any}) where T
    J   = PDMat(randWishart(inv(niw.Ψ), niw.ν))
    μ   = randNormal(niw.μ, PDMat(J.chol * niw.λ))
    return MvNormalFast(μ, J)
end

function randWishart(S::AbstractPDMat{T}, df::Real) where T
    p = dim(S)
    A = zeros(T,p,p)
    _wishart_genA!(GLOBAL_RNG,p, df,A)
    unwhiten!(S, A)
    A .= A * A'
end
randInvWishart(Ψ::AbstractPDMat, df::Real) = inv(PDMat(randWishart(inv(Ψ),df)))
params(niw::NormalWishart) = (niw.μ, niw.Ψ, niw.λ, niw.ν)

function _logpdf(d::GenericMvTDist, x::AbstractMatrix{T}) where T<:Real
    r = similar(x,size(x,2))
    return sum(_logpdf!(r,d,x))
end

function mv_lgamma(s::Real, x::Real, D::Int)
    for d=1:D
        s += lgamma(x+(1-d)/2)
    end
    return s
end

function lmllh(prior::NormalWishart{T}, posterior::NormalWishart{T},  n::Int) where T
    D = length(prior)
    s = D*(D-1)/4*logpi
    return -n*D/2*logpi + mv_lgamma(s, posterior.ν/2, D) - mv_lgamma(s, prior.ν/2, D) +
            prior.ν/2*logdet(prior.Ψ) - posterior.ν/2*logdet(posterior.Ψ) +
            D/2*(log(prior.λ)-log(posterior.λ))
end


# pdf(niw::NormalWishart, x::Vector{T}, Σ::Matrix{T}) where T<:Real = exp(logpdf(niw, x, Σ))
#
# function insupport(::Type{NormalWishart}, x::Vector{T}, Σ::Matrix{T}) where T<:Real
#     return (all(isfinite, x) &&
#            size(Σ, 1) == size(Σ, 2) &&
#            isApproxSymmmetric(Σ) &&
#            size(Σ, 1) == length(x) &&
#            hasCholesky(Σ)
# end
#
# function logpdf(niw::NormalWishart, x::Vector{T}, Σ::Matrix{T}) where T<:Real
#     if !insupport(NormalWishart, x, Σ)
#         return -Inf
#     else
#         p = size(x, 1)
#         ν = niw.ν
#         λ = niw.λ
#         μ = niw.μ
#         Ψ = niw.Ψ
#         hnu = 0.5 * ν
#         hp  = 0.5 * p
#
#         # Normalization
#         logp::T = hnu * logdet(Ψ)
#         logp -= hnu * p * log(2.)
#         logp -= logmvgamma(p, hnu)
#         logp -= hp * (log(2.0*pi) - log(λ))
#
#         # Inverse-Wishart
#         logp -= (hnu + hp + 1.) * logdet(Σ)
#         logp -= 0.5 * tr(Σ \ Matrix(Ψ))
#
#         # Normal
#         z = niw.zeromean ? x : x - μ
#         logp -= 0.5 * λ * invquad(PDMat(Σ), z)
#
#         return logp
#
#     end
# end
