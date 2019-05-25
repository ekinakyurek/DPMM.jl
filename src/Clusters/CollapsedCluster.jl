import Base: +,-, isempty

abstract type AbstractCluster end

struct CollapsedCluster{Pred<:Distribution, Prior<:Distribution} <: AbstractCluster
    n::Int
    predictive::Pred
    prior::Prior
end

@inline isempty(m::CollapsedCluster) = m.n==0

@inline CollapsedCluster(m::AbstractDPModel) = CollapsedCluster(m, suffstats(m))

@inline CollapsedCluster(m::AbstractDPModel,X::AbstractArray) =
    CollapsedCluster(m, suffstats(m,X))

@inline CollapsedCluster(m::AbstractDPModel,s::SufficientStats) =
    CollapsedCluster(s.n, posterior_predictive(m,s), m.θprior)

@inline CollapsedCluster(m::AbstractDPModel,new::Val{true}) =
    CollapsedCluster(floor(Int,m.α),posterior_predictive(m),m.θprior)

@inline -(c::CollapsedCluster{V,P},x::AbstractVector) where {V<:Distribution,P<:Distribution} =
    CollapsedCluster{V,P}(c.n-1, downdate_predictive(c.prior,c.predictive,x,c.n), c.prior)

@inline +(c::CollapsedCluster{V,P},x::AbstractVector) where {V<:Distribution,P<:Distribution} =
    CollapsedCluster{V,P}(c.n+1, update_predictive(c.prior,c.predictive,x,c.n), c.prior)

@inline logprob(m::CollapsedCluster,x) = logprob(m.predictive,x)
@inline (m::CollapsedCluster)(x) = log(m.n) + logprob(m.predictive,x)

CollapsedClusters(model::AbstractDPModel, X::AbstractMatrix, z::AbstractArray{Int}) =
    Dict((k,CollapsedCluster(model,X[:,findall(l->l==k,z)])) for k in unique(z))

#
# function Base.hash(obj::CollapsedCluster, h::UInt)
#     return hash((obj.n, obj.predictive), h)
# end
#
# function Base.:(==)(obj1::CollapsedCluster, obj2::CollapsedCluster)
#     return hash(obj1) == hash(obj2)
# end
