"""

    DirectCluster{Pred<:Distribution, Prior<:Distribution} <: AbstractCluster

    The DirectCluster is designed for Direct Gibbs algorithms.

    DirectCluster has below fields:
        `n` : population
        `sampled` : sampled parameter distribution
        'prior' : prior distribution

    A DirectCluster are constructed via SufficientStats or data points:
    ```julia
        DirectCluster(m::AbstractDPModel,X::AbstractArray) # X is the data as columns
        DirectCluster(m::AbstractDPModel,s::SufficientStats)
    ```

    There is also generic(not specific to DirectCluster) SuffStats method for
    getting sufficient stats for whole data data as a dictionary
    ```julia
        SuffStats(model::AbstractDPModel, X::AbstractMatrix, z::AbstractArray{Int})
    ```

    There are also specific methods defined for creating clusters for whole data as a dictionary:
    ```julia
        DirectClusters(model::AbstractDPModel, X::AbstractMatrix, labels::AbstractArray{Int})
        DirectClusters(model::AbstractDPModel, stats::Dict{Int,<:SufficientStats})
    ```
    
    see `AbstractCluster` for generic functions for all Cluster types.
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
