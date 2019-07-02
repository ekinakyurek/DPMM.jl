"""
   `DPMNMM{T<:Real,D} <: AbstractDPModel{T,D}`
    Class for DP Multinomial Mixture Models
"""
struct DPMNMM{T<:Real,D,Pr<:DirichletFast} <: AbstractDPModel{T,D}
    θprior::Pr
    α::T
end
@inline prior(m::DPMNMM) = m.θprior
DPMNMM(X::AbstractMatrix{<:Integer}; α::Real=1.0) = DPMNMM(Float64(α), ones(size(X,1)))#Float64.(sumcol(X) .+ 1))
@inline function DPMNMM(α::T, alphas::AbstractVector{<:Real}) where T
    prior = DirichletFast(alphas)
    DPMNMM{T,length(alphas),typeof(prior)}(prior,α)
end

"""
   ` DPMNMMStats <: SufficientStats`
    Sufficient statistics for Multinomial Models
"""
@inline stattype(::DirichletFast) = DPMNMMStats

struct DPMNMMStats <: SufficientStats
    s::Vector{Int}
    n::Int
end

@inline suffstats(m::DirichletFast) =
    DPMNMMStats(zeros(Int,length(m)),0)

@inline suffstats(m::DirichletFast, X::AbstractMatrix{Int}) =
    DPMNMMStats(sumcol(X),size(X,2))

@inline suffstats(m::DirichletFast, x::AbstractVector{Int}) =
    DPMNMMStats(Vector(x),1)

@inline function updatestats(m::DPMNMMStats, x::AbstractVector{Int})
    DPMNMMStats(add!(m.s,x),m.n+1)
end

@inline function +(s1::DPMNMMStats,s2::DPMNMMStats)
    DPMNMMStats(s1.s+s2.s,s1.n+s2.n)
end

@inline function -(s1::DPMNMMStats,s2::DPMNMMStats)
    DPMNMMStats(s1.s-s2.s,s1.n-s2.n)
end

@inline function updatestats(m::DPMNMMStats,X::AbstractMatrix{Int})
    DPMNMMStats(add!(m.s,sumcol(X)), m.n+size(X,2))
end

@inline function downdatestats(m::DPMNMMStats, x::AbstractVector{Int})
    DPMNMMStats(substract!(m.s,x),m.n-1)
end

@inline function downdatestats(m::DPMNMMStats,X::AbstractMatrix{Int})
    DPMNMMStats(substract!(m.s,sumcol(X)), m.n-size(X,2))
end

@inline _posterior(m::DirichletFast, T::DPMNMMStats)  = m.α + T.s
@inline posterior(m::DirichletFast) = m.α
@inline posterior_predictive(m::DirichletFast)= DirichletMultPredictive(m.α)

@inline posterior(m::DirichletFast, T::DPMNMMStats) =
    T.n != 0 ? DirichletFast(_posterior(m,T)) : m

@inline posterior_predictive(m::DirichletFast,T::DPMNMMStats) =
    T.n != 0 ? DirichletMultPredictive(_posterior(m,T)) : posterior_predictive(m)

@inline downdate_predictive(p::DirichletFast, m::DirichletMultPredictive, x::AbstractVector, n::Int) =
    DirichletMultPredictive(substract!(m.α,x))

@inline update_predictive(p::DirichletFast, m::DirichletMultPredictive, x::AbstractVector, n::Int) =
    DirichletMultPredictive(add!(m.α,x))


#
# init(X::AbstractMatrix{<:Integer}, α::V, ninit::Int, T::Type{<:DPMNMM}) where V<:Real =
#     size(X),rand(1:ninit,size(X,2)),T(α,sumcol(X) .+ V(1))
