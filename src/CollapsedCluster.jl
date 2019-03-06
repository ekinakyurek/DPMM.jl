import Base: +,-, isempty

struct CollapsedCluster{Pred<:Distribution, Prior<:Distribution}
    n::Int
    predictive::Pred
    prior::Prior
end

@inline isempty(m::CollapsedCluster) = m.n==0

@inline CollapsedCluster(m::DPGMM) = CollapsedCluster(m, suffstats(m))

@inline CollapsedCluster(m::DPGMM,X::AbstractArray) =
    CollapsedCluster(m, suffstats(m,X))

@inline CollapsedCluster(m::DPGMM,s::DPGMMStats) =
    CollapsedCluster(s.n, posterior_predictive(m,s), m.θprior)

function CollapsedCluster(m::DPGMM,new::Val{true})
    c = CollapsedCluster(m, suffstats(m))
    return CollapsedCluster(floor(Int,m.α),c.predictive,c.prior)
end

@inline -(c::CollapsedCluster{V,P},x::AbstractVector) where {V<:Distribution,P<:Distribution} =
    CollapsedCluster{V,P}(c.n-1, downdate_predictive(c.prior,c.predictive,x,c.n), c.prior)

@inline +(c::CollapsedCluster{V,P},x::AbstractVector) where {V<:Distribution,P<:Distribution} =
    CollapsedCluster{V,P}(c.n+1, update_predictive(c.prior,c.predictive,x,c.n), c.prior)

@inline pdf(m::CollapsedCluster,x) = pdf(m.predictive,x)
@inline (m::CollapsedCluster)(x)   = m.n*pdf(m.predictive,x)
#
# function Base.hash(obj::CollapsedCluster, h::UInt)
#     return hash((obj.n, obj.predictive), h)
# end
#
# function Base.:(==)(obj1::CollapsedCluster, obj2::CollapsedCluster)
#     return hash(obj1) == hash(obj2)
# end
