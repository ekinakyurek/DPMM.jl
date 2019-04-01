using LinearAlgebra
import Distributions: rand, suffstats, length

struct DPDMM{T<:Real,D} <: AbstractDPModel{T,D}
    θprior::Dirichlet{T}
    α::T
end

@inline DPDMM{T,D}(α::T) where {T<:Real,D} = DPDMM{T,dim}(Dirichlet{T}(ones(D),α))

@inline DPDMM{T}(α::T,alphas::AbstractVector{T}) where T<:Real = DPDMM{T,length(alphas)}(Dirichlet{T}(alphas),α)

struct DPDMMStats{T<:Real}
    s::Vector{Int}
    n::Int
end

@inline suffstats(m::DPDMM{T,D}) where {T<:Real,D} =
    DPDMMStats{T}(zeros(Int,D),0)

@inline suffstats(m::DPDMM{T},X::AbstractMatrix{Int}) where T<:Real =
    DPDMMStats{T}(vec(sum(X,dims=2)),size(X,2))

@inline suffstats(m::DPDMM{T},x::AbstractVector{Int}) where T<:Real =
    DPDMMStats{T}(x,1)

@inline function updatestats(m::DPDMMStats{T},x::AbstractVector{Int}) where T<:Real
    DPDMMStats{T}(m.s+x,m.n+1)
end

@inline function updatestats(m::DPDMMStats{T},X::AbstractMatrix{T}) where T<:Real
    DPDMMStats{T}(m.s+vec(sum(X,dims=2)), m.n+size(X,2))
end

@inline function downdatestats(m::DPDMMStats{T}, x::AbstractVector{Int}) where T<:Real
    DPDMMStats{T}(m.s-x,m.n-1)
end

@inline function downdatestats(m::DPDMMStats{T},X::AbstractMatrix{Int}) where T<:Real
    DPDMMStats{T}(m.s-vec(sum(X,dims=2)), m.n-size(X,2))
end

@inline _posterior(m::Dirichlet{V},T::DPDMMStats{V}) where V<:Real = m.alpha + T.s
@inline _posterior(m::DPDMM,T::DPDMMStats) = _posterior(m.θprior,T)
@inline posterior(m::DPDMM,T::DPDMMStats)  =  posterior(m.θprior,T)
@inline posterior(m::Dirichlet{V},T::DPDMMStats{V}) where V<:Real = T.n!=0 ? Dirichlet{T}(_posterior(m,T)...) : m
