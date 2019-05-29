"""
The DirectCluster is designed for Direct Gibbs algorithms.

DirectCluster is defined by:
    `n` : population
    `sampled` : sampled parameter distribution
    'prior' : prior distribution

A DirectCluster are constructed via SufficientStats or data points:
```julia
    DirectCluster(m::AbstractDPModel,X::AbstractArray)
    DirectCluster(m::AbstractDPModel,s::SufficientStats)
```

There are also specific methods defined for creating clusters for whole data:
```julia
    DirectClusters(model::AbstractDPModel, X::AbstractMatrix, labels::AbstractArray{Int})
    DirectClusters(model::AbstractDPModel, stats::Dict{Int,<:SufficientStats})
```

There is also generic(not specific to DirectCluster) SuffStats method for
getting sufficient stats for whole data
```julia
    SuffStats(model::AbstractDPModel, X::AbstractMatrix, z::AbstractArray{Int})
```

The `logαpdf` function are defined for geting log(∝likelihood) of a data point.
This requires a `logαpdf` function for sampled distribution too.
```julia
logαpdf(m::DirectCluster,x)
```

Clusters are callable objects and call to a cluster returns below:
```julia
(m::DirectCluster)(x) = log(m.n) + logαpdf(m.sampled,x)
```
"""
struct DirectCluster{Pred<:Distribution, Prior<:Distribution} <: AbstractCluster
    n::Int
    sampled::Pred
    prior::Prior
end

@inline population(m::DirectCluster) = m.n

@inline DirectCluster(m::AbstractDPModel) = DirectCluster(m, suffstats(m))

@inline DirectCluster(m::AbstractDPModel, X::AbstractArray) =
    DirectCluster(m, suffstats(m,X))

@inline DirectCluster(m::AbstractDPModel, s::SufficientStats) =
    DirectCluster(s.n, rand(posterior(m,s)), m.θprior)

@inline DirectCluster(m::AbstractDPModel, new::Val{true}) =
    DirectCluster(floor(Int,m.α),rand(m.θprior),m.θprior)

@inline logαpdf(m::DirectCluster,x)  = logαpdf(m.sampled,x)

DirectClusters(model::AbstractDPModel, X::AbstractMatrix, z::AbstractArray{Int}) =
    Dict((k,DirectCluster(model,X[:,findall(l->l==k,z)])) for k in unique(z))

DirectClusters(model::AbstractDPModel, stats::Dict{Int,<:SufficientStats}) =
    Dict((k,DirectCluster(model,stats[k])) for k in keys(stats))
