"""
   AbstractDPModel{T,D}

   Abstract base class for DPMMs

   `T` stands for element type, `D` is for dimensionality of the data
"""
abstract type AbstractDPModel{T,D} end
@inline length(::AbstractDPModel{<:Any,D}) where D = D
@inline stattype(m::AbstractDPModel)  = stattype(prior(m))
@inline suffstats(m::AbstractDPModel) = suffstats(prior(m))
@inline suffstats(m::AbstractDPModel, X) = suffstats(prior(m),X)
@inline posterior(m::AbstractDPModel) =  prior(m)
@inline posterior(m::AbstractDPModel, T::SufficientStats)  = posterior(prior(m),T)
@inline posterior_predictive(m::AbstractDPModel) = posterior_predictive(prior(m))
@inline posterior_predictive(m::AbstractDPModel,  T::SufficientStats) = posterior_predictive(prior(m),T)
@inline _posterior(m::AbstractDPModel, T::SufficientStats) = _posterior(prior(m),T)
