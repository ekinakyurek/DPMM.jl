using LinearAlgebra
import Distributions: rand, suffstats, length

struct DPGMM{T<:Real,D} <: ContinuousUnivariateDistribution
    θprior::NormalInverseWishart{T}
    α::T
end

@inline length(::DPGMM{<:Any,D}) where D = D

@inline DPGMM{T,D}(α::T) where {T<:Real,D} =
    DPGMM{T,dim}(NormalInverseWishart{T}(D),α)

@inline DPGMM{T}(α::T,μ0::AbstractVector{T}) where T<:Real =
    DPGMM{T,length(μ0)}(NormalInverseWishart{T}(μ0),α)
