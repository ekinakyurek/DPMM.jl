"""
   `DPMNMM{T<:Real,D} <: AbstractDPModel{T,D}`
    Class for DP Multinomial Mixture Models
"""
struct DPMNMM{T<:Real,D} <: AbstractDPModel{T,D}
    θprior::DirichletFast{T}
    α::T
end
@inline prior(m::DPMNMM) = m.θprior
function DPMNMM(X::AbstractMatrix{<:Integer}; α::Real=1, eltype=Float64)
    DPMNMM{eltype}(eltype(α), sumcol(X) .+ eltype(1))
end
@inline DPMNMM{T,D}(α::Real) where {T<:Real,D} = DPMNMM{T,dim}(DirichletFast{T}(ones(D),T(α)))
@inline DPMNMM{T}(α::Real, alphas::AbstractVector{T}) where T<:Real = DPMNMM{T,length(alphas)}(DirichletFast{T}(alphas),T(α))

"""
   ` DPMNMMStats{T<:Real} <: SufficientStats`
    Sufficient statistics for Multinomial Models
"""
@inline stattype(::DirichletFast{T}) where T = DPMNMMStats{T}

struct DPMNMMStats{T<:Real} <: SufficientStats
    s::Vector{Int}
    n::Int
end

@inline suffstats(m::DirichletFast{T}) where T<:Real =
    DPMNMMStats{T}(zeros(Int,length(m),0))

@inline suffstats(m::DirichletFast{T},X::AbstractMatrix{Int}) where T<:Real =
    DPMNMMStats{T}(sumcol(X),size(X,2))

@inline suffstats(m::DirichletFast{T},x::AbstractVector{Int}) where T<:Real =
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

@inline function updatestats(m::DPMNMMStats{T},X::AbstractMatrix{Int}) where T<:Real
    DPMNMMStats{T}(add!(m.s,sumcol(X)), m.n+size(X,2))
end

@inline function downdatestats(m::DPMNMMStats{T}, x::AbstractVector{Int}) where T<:Real
    DPMNMMStats{T}(substract!(m.s,x),m.n-1)
end

@inline function downdatestats(m::DPMNMMStats{T},X::AbstractMatrix{Int}) where T<:Real
    DPMNMMStats{T}(substract!(m.s,sumcol(X)), m.n-size(X,2))
end

@inline _posterior(m::DirichletFast{V}, T::DPMNMMStats{V}) where V<:Real = m.α + T.s
@inline posterior(m::DirichletFast) = m.α
@inline posterior_predictive(m::DirichletFast{T}) where T<:Real = DirichletMultPredictive(m.α)

@inline posterior(m::DirichletFast{V}, T::DPMNMMStats{V}) where V<:Real =
    T.n!=0 ? DirichletFast{V}(_posterior(m,T)) : m
@inline posterior_predictive(m::DirichletFast{V},T::DPMNMMStats{V}) where V<:Real =
    T.n != 0 ? DirichletMultPredictive{V}(_posterior(m,T)) : posterior_predictive(m)

@inline downdate_predictive(p::DirichletFast, m::DirichletMultPredictive, x::AbstractVector, n::Int) =
    DirichletMultPredictive(substract!(m.α,x))

@inline update_predictive(p::DirichletFast, m::DirichletMultPredictive, x::AbstractVector, n::Int) =
    DirichletMultPredictive(add!(m.α,x))


#
# init(X::AbstractMatrix{<:Integer}, α::V, ninit::Int, T::Type{<:DPMNMM}) where V<:Real =
#     size(X),rand(1:ninit,size(X,2)),T(α,sumcol(X) .+ V(1))
