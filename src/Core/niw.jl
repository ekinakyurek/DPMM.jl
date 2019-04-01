using LinearAlgebra, PDMats
import Distributions: rand, pdf

struct NormalInverseWishart{T<:Real,S<:AbstractPDMat} <: ContinuousUnivariateDistribution
    μ::Vector{T}
    λ::T  # This scales precision (inverse covariance)
    Ψ::S
    ν::T
    function NormalInverseWishart{T}(μ::Vector{T}, λ::T, Ψ::AbstractPDMat{T}, ν::T) where T<:Real
        new{T,typeof(Ψ)}(μ, λ, Ψ, ν)
    end
end

function NormalInverseWishart(μ::Vector{U}, λ::Real,
                              Ψ::AbstractPDMat{S}, ν::Real) where {S<:Real, U<:Real}
    T = promote_type(eltype(μ), typeof(λ), typeof(ν), S)
    return NormalInverseWishart{T}(Vector{T}(μ), T(λ), AbstractPDMat{T}(Ψ), T(ν))
end

function NormalInverseWishart(μ::Vector{U}, λ::Real,
                              Ψ::Matrix{S}, ν::Real) where {S<:Real, U<:Real}
    T = promote_type(eltype(μ), typeof(λ), typeof(ν), S)
    return NormalInverseWishart{T}(Vector{T}(μ), T(λ), PDMat(Ψ), T(ν))
end

# We sart λ0=1,ν0=D+3. So we in this setting df=ν-D+1 => λ=df-1
# If you change ν0-λ0 in below,
# downdate_posterior_predictive and update_posterior_predictive does't work correctly.
NormalInverseWishart{T}(n::Int) where T<:Real =
    NormalInverseWishart(zeros(T,n),T(1),Matrix{T}(I,n,n),T(n)+3)

NormalInverseWishart{T}(mean::AbstractArray{T}) where T<:Real =
    NormalInverseWishart(mean,T(1),Matrix{T}(I,length(mean),length(mean)),T(length(mean))+3)

NormalInverseWishart{T}(mean::AbstractArray{T}, cov::AbstractMatrix{T}) where T<:Real =
    NormalInverseWishart(mean,T(1),cov,T(length(mean))+3)

params(niw::NormalInverseWishart) = (niw.μ, niw.Ψ, niw.λ, niw.ν)
pdf(niw::NormalInverseWishart, x::Vector{T}, Σ::Matrix{T}) where T<:Real = exp(logpdf(niw, x, Σ))

function logpdf(niw::NormalInverseWishart, x::Vector{T}, Σ::Matrix{T}) where T<:Real
    if !insupport(NormalInverseWishart, x, Σ)
        return -Inf
    else
        p = size(x, 1)
        ν = niw.ν
        λ = niw.λ
        μ = niw.μ
        Ψ = niw.Ψ
        hnu = 0.5 * ν
        hp  = 0.5 * p

        # Normalization
        logp::T = hnu * logdet(Ψ)
        logp -= hnu * p * log(2.)
        logp -= logmvgamma(p, hnu)
        logp -= hp * (log(2.0*pi) - log(λ))

        # Inverse-Wishart
        logp -= (hnu + hp + 1.) * logdet(Σ)
        logp -= 0.5 * tr(Σ \ Matrix(Ψ))

        # Normal
        z = niw.zeromean ? x : x - μ
        logp -= 0.5 * λ * invquad(PDMat(Σ), z)

        return logp

    end
end

function rand(niw::NormalInverseWishart)
    Σ   = rand(InverseWishart(niw.ν, niw.Ψ))
    μ   = rand(MvNormal(niw.μ, Σ ./ niw.λ))
    return MvNormal(μ, Σ)
end

function insupport(::Type{NormalInverseWishart}, x::Vector{T}, Σ::Matrix{T}) where T<:Real
    return (all(isfinite, x) &&
           size(Σ, 1) == size(Σ, 2) &&
           isApproxSymmmetric(Σ) &&
           size(Σ, 1) == length(x) &&
           hasCholesky(Σ))
end
