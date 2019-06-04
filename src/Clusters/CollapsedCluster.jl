import Base: +,-, isempty
"""
    AbstractCluster

Abstract base class for clusters

Each subtype should provide the following methods:
- `population(c)`: population of the cluster
- `isempty(m::AbstractCluster)`: checks whether the cluster is empty?
- `logαpdf(c,x)` : log(∝likelihood) of a data point
- `lognαpdf(c,x)`: log(population) + logαpdf(c,x) for a data point (used in CRP calculations)
- `ClusterType(m::AbstractDPModel,X::AbstractArray)`  : constructor (X is the data as columns)
- `ClusterType(m::AbstractDPModel,s::SufficientStats)`: constructor

Other generic functions are implemented on top of these core functions.
"""
abstract type AbstractCluster end
const GenericClusters = Dict{Int, <:AbstractCluster}

"""
    population(m::AbstractCluster)

Number of data points in a cluster
"""
population(m::AbstractCluster)

"""
    logαpdf(m::AbstractCluster,x::AbstractArray)

log(∝likelihood) of a data point given by a cluster.
"""
logαpdf(m::AbstractCluster,x::AbstractArray)

@inline isempty(m::AbstractCluster)    = population(m)==0

"""
    lognαpdf(m::AbstractCluster,x::AbstractArray)

log(population) + log(∝likelihood) of a data point given by a cluster.
"""
@inline lognαpdf(m::AbstractCluster,x) = log(population(m)) + logαpdf(m,x)

"""

    CollapsedCluster{Pred<:Distribution, Prior<:Distribution} <: AbstractCluster

The CollapsedCluster is designed for Collapsed Gibbs algorithms.

CollapsedCluster has below fields:
    - `n` : population
    - `predictive` : predictive distribution
    - `prior` : prior distribution

A CollapsedCluster are constructed via SufficientStats or data points:
```julia
CollapsedCluster(m::AbstractDPModel, X::AbstractArray) # X is the data as columns
CollapsedCluster(m::AbstractDPModel, s::SufficientStats)
```

There is also generic(not specific to CollapsedCluster) SuffStats method for
getting suffstats for whole data as a dictionary:
```julia
SuffStats(model::AbstractDPModel, X::AbstractMatrix, z::AbstractArray{Int})
```

There are also specific methods defined for creating clusters for whole data as a dictionary:
```julia
CollapsedClusters(model::AbstractDPModel, X::AbstractMatrix, labels::AbstractArray{Int})
CollapsedClusters(model::AbstractDPModel, stats::Dict{Int,<:SufficientStats})
```

`-` and `+` operations are defined for data addition and data removal from the cluster:
```julia
-(c::CollapsedCluster, x::AbstractVector)
+(c::CollapsedCluster, x::AbstractVector)
```

see [`AbstractCluster`](@ref) for generic functions for all Cluster types.
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
