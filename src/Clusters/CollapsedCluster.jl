import Base: +,-, isempty
"""
   Abstract base class for clusters

   Each subtype should provide the following methods:
   - `population(c)` : population of the cluster
   - `logαpdf(c,x)`  : log(∝likelihood) of data
   - `(c::ClusterType)(x)=log(c.n) + logαpdf(c,x)`
   - `ClusterType(m::AbstractDPModel,X::AbstractArray)`: constructor
   - `ClusterType(m::AbstractDPModel,s::SufficientStats)`: constructor

   Other generic functions is implemented on top of these core functions.
"""
abstract type AbstractCluster end
const GenericClusters = Dict{Int, <:AbstractCluster}
@inline isempty(m::AbstractCluster)    = population(m)==0
@inline lognαpdf(m::AbstractCluster,x) = log(population(m)) + logαpdf(m,x)

"""
The CollapsedCluster is designed for Collapsed Gibbs algorithms.

CollapsedCluster is defined by:
    `n` : population
    `predictive` : predictive distribution
    'prior' : prior distribution

A CollapsedCluster are constructed via SufficientStats or data points:
```julia
    CollapsedCluster(m::AbstractDPModel,X::AbstractArray)
    CollapsedCluster(m::AbstractDPModel,s::SufficientStats)
```

There are also specific methods defined for creating clusters for whole data:
```julia
    CollapsedClusters(model::AbstractDPModel, X::AbstractMatrix, labels::AbstractArray{Int})
    CollapsedClusters(model::AbstractDPModel, stats::Dict{Int,<:SufficientStats})
```

There is also generic(not specific to CollapsedCluster) SuffStats method for
getting suffstats for whole data
```julia
    SuffStats(model::AbstractDPModel, X::AbstractMatrix, z::AbstractArray{Int})
```

`-` and `+` operations are defined for data addition and removal:
```julia
    -(c::CollapsedCluster{V,P},x::AbstractVector)
    +(c::CollapsedCluster{V,P},x::AbstractVector)
```

The `logαpdf` function are defined for geting log(∝likelihood) of a data point.
This requires a `logαpdf` function for predictive distribution too.
```julia
logαpdf(m::CollapsedCluster,x)
```

Clusters are callable objects and call to a cluster returns below:
```julia
(m::CollapsedCluster)(x) = log(m.n) + logαpdf(m.predictive,x)
```
"""
struct CollapsedCluster{Pred<:Distribution, Prior<:Distribution} <: AbstractCluster
    n::Int
    predictive::Pred
    prior::Prior
end

@inline population(m::CollapsedCluster) = m.n

@inline CollapsedCluster(m::AbstractDPModel) = CollapsedCluster(m, suffstats(m))

@inline CollapsedCluster(m::AbstractDPModel,X::AbstractArray) =
    CollapsedCluster(m, suffstats(m,X))

@inline CollapsedCluster(m::AbstractDPModel, s::SufficientStats) =
    CollapsedCluster(s.n, posterior_predictive(m,s), m.θprior)

@inline CollapsedCluster(m::AbstractDPModel,new::Val{true}) =
    CollapsedCluster(floor(Int,m.α),posterior_predictive(m),m.θprior)

@inline -(c::CollapsedCluster{V,P},x::AbstractVector) where {V<:Distribution,P<:Distribution} =
    CollapsedCluster{V,P}(c.n-1, downdate_predictive(c.prior,c.predictive,x,c.n), c.prior)

@inline +(c::CollapsedCluster{V,P},x::AbstractVector) where {V<:Distribution,P<:Distribution} =
    CollapsedCluster{V,P}(c.n+1, update_predictive(c.prior,c.predictive,x,c.n), c.prior)

@inline logαpdf(m::CollapsedCluster,x)  = logαpdf(m.predictive,x)

CollapsedClusters(model::AbstractDPModel, X::AbstractMatrix, z::AbstractArray{Int}) =
    Dict((k,CollapsedCluster(model,X[:,findall(l->l==k,z)])) for k in unique(z))

SuffStats(model::AbstractDPModel, X::AbstractMatrix, z::AbstractArray{Int}) =
    Dict((k,suffstats(model,X[:,findall(l->l==k,z)])) for k in unique(z))

CollapsedClusters(model::AbstractDPModel, stats::Dict{Int,<:SufficientStats}) =
    Dict((k,CollapsedCluster(model,stats[k])) for k in keys(stats))
