"""
   `DPMNMM{T<:Real,D} <: AbstractDPModel{T,D}`
    Class for DP Multinomial Mixture Models
"""
struct DPMNMM{T<:Real,D} <: AbstractDPModel{T,D}
    θprior::DirichletFast{T}
    α::T
end

function DPMNMM(X::AbstractMatrix{<:Integer}; α::Real=1, eltype=Float64)
    DPMNMM{eltype}(eltype(α), sumcol(X) .+ eltype(1))
end

@inline DPMNMM{T,D}(α::Real) where {T<:Real,D} = DPMNMM{T,dim}(DirichletFast{T}(ones(D),T(α)))

@inline DPMNMM{T}(α::Real, alphas::AbstractVector{T}) where T<:Real = DPMNMM{T,length(alphas)}(DirichletFast{T}(alphas),T(α))

@inline stattype(::DPMNMM{T}) where T = DPMNMMStats{T}


"""
   ` DPMNMMStats{T<:Real} <: SufficientStats`
    Sufficient statistics for Multinomial Models
"""
struct DPMNMMStats{T<:Real} <: SufficientStats
    s::Vector{Int}
    n::Int
end

@inline suffstats(m::DPMNMM{T,D}) where {T<:Real,D} =
    DPMNMMStats{T}(zeros(Int,D),0)

@inline suffstats(m::DPMNMM{T},X::AbstractMatrix{Int}) where T<:Real =
    DPMNMMStats{T}(sumcol(X),size(X,2))

@inline suffstats(m::DPMNMM{T},x::AbstractVector{Int}) where T<:Real =
    DPMNMMStats{T}(x,1)

@inline function updatestats(m::DPMNMMStats{T},x::AbstractVector{Int}) where T<:Real
    DPMNMMStats{T}(add!(m.s,x),m.n+1)
end

@inline function +(s1::DPMNMMStats{T},s2::DPMNMMStats{T}) where T<:Real
    DPMNMMStats{T}(s1.s+s2.s,s1.n+s2.n)
end

@inline function -(s1::DPMNMMStats{T},s2::DPMNMMStats{T}) where T<:Real
    DPMNMMStats{T}(s1.s-s2.s,s1.n-s2.n)
end

@inline function updatestats(m::DPMNMMStats{T},X::AbstractMatrix{T}) where T<:Real
    DPMNMMStats{T}(add!(m.s,sumcol(X)), m.n+size(X,2))
end

@inline function downdatestats(m::DPMNMMStats{T}, x::AbstractVector{Int}) where T<:Real
    DPMNMMStats{T}(substract!(m.s,x),m.n-1)
end

@inline function downdatestats(m::DPMNMMStats{T},X::AbstractMatrix{Int}) where T<:Real
    DPMNMMStats{T}(substract!(m.s,sumcol(X)), m.n-size(X,2))
end

@inline _posterior(m::DirichletFast{V},T::DPMNMMStats{V}) where V<:Real = m.alpha + T.s
@inline _posterior(m::DPMNMM,T::DPMNMMStats) = _posterior(m.θprior,T)
@inline posterior(m::DPMNMM, T::DPMNMMStats)  =  posterior(m.θprior,T)
@inline posterior(m::DirichletFast{V}, T::DPMNMMStats{V}) where V<:Real = T.n!=0 ? DirichletFast{V}(_posterior(m,T)) : m
#
# init(X::AbstractMatrix{<:Integer}, α::V, ninit::Int, T::Type{<:DPMNMM}) where V<:Real =
#     size(X),rand(1:ninit,size(X,2)),T(α,sumcol(X) .+ V(1))
