"""
    SplitMergeCluster{Pred<:Distribution, Post<:Distribution, Prior<:Distribution} <: AbstractCluster

The SplitMergeCluster is designed for Split-Merge Gibbs algorithm.

SplitMergeCluster has below fields:
    - `n` : population
    - `nr`: right subcluster population
    - `nl`: left subcluster population
    - `sampled` : sampled parameter distribution
    - `right` : right subcluster sampled parameter distribution
    - `left`: left subcluster sampled parameter
    - `post` : posterior distributions
    - `rightpost` : right subcluster posterior distributions
    - `leftpost` : left subcluster posterior distributions
    - 'prior' : prior distribution
    - `llhs` : log marginal likelihoods assigned by cluster, right subcluster, leftsubcluster
    - `llh_hist` : right + left log marginal likelihood history over 4 iteration
    - 'prior' : prior distribution

A SplitMergeCluster are constructed via SufficientStats or data points:
```julia
SplitMergeCluster(m::AbstractDPModel,X::AbstractArray) # X is the data as columns
SplitMergeCluster(m::AbstractDPModel,s::SufficientStats)
```

There is also generic SuffStats method for getting sufficient stats for whole data:
```julia
SuffStats(model::AbstractDPModel, X::AbstractMatrix, z::AbstractVector{Tuple{Int,Bool}})
```

There are also specific methods defined for creating clusters for whole data:
```julia
SplitMergeClusters(model::AbstractDPModel, X::AbstractMatrix, labels::AbstractVector{Tuple{Int,Bool}})
```

see [`AbstractCluster`](@ref) for generic functions for all Cluster types.

The `logαpdf` and `lognαpdf` generic functions are extended for subcluster likelihoods.
```julia
logαpdf(m::SplitMergeCluster,x,::Val{false}) # right subcluster likelihood
logαpdf(m::SplitMergeCluster,x,::Val{true})  # left subcluster likelihood
lognαpdf(m::SplitMergeCluster, x, ::Val{false})  = log(population(m,Val(false))) + logαpdf(m, x, Val(false))
lognαpdf(m::SplitMergeCluster, x, ::Val{true})   = log(population(m,Val(true))) + logαpdf(m, x, Val(true))
```
"""
struct SplitMergeCluster{Pred<:Distribution, Post<:Distribution, Prior<:Distribution} <: AbstractCluster
    n::Int; nr::Int; nl::Int;
    s::SufficientStats;
    sampled::Pred; right::Pred; left::Pred;
    post::Post; rightpost::Post; leftpost::Post;
    llhs::NTuple{3,Float64}; llh_hist::NTuple{4,Float64}; prior::Prior
end

function SplitMergeCluster(m::AbstractDPModel{<:Any,D},
                           s::SufficientStats,
                           sr::SufficientStats,
                           sl::SufficientStats) where D
    prior = m.θprior
    ps, psr, psl  = posterior(prior,s), posterior(prior,sr), posterior(prior,sl)

    llhs = (lmllh(prior, ps, s.n), lmllh(prior, psr, sr.n), lmllh(prior, psl, sl.n))
    SplitMergeCluster(s.n, sr.n, sl.n, s, rand(ps), rand(psr), rand(psl),
                      ps, psr, psl, llhs, (-Inf,-Inf,-Inf,llhs[2]+llhs[3]), prior)
end

# This method is specifically designed for updating an existing cluster
function SplitMergeCluster(c::SplitMergeCluster,
                           s::SufficientStats,
                           sr::SufficientStats,
                           sl::SufficientStats;
                           llh_hist::NTuple{4,Float64}=(-Inf,-Inf,-Inf,-Inf))
    prior = c.prior
    ps, psr, psl  = posterior(prior,s), posterior(prior,sr), posterior(prior,sl)
    llhs = (lmllh(prior, ps,  s.n), lmllh(prior, psr, sr.n), lmllh(prior, psl, sl.n))
    SplitMergeCluster(s.n, sr.n, sl.n, s, c.sampled, c.right, c.left, ps, psr, psl,
                      llhs, (llh_hist[2:end]...,llhs[2]+llhs[3]), prior)
end

function SplitMergeClusters(model::AbstractDPModel, X::AbstractMatrix, z::AbstractVector{Tuple{Int,Bool}})
    uniquez   = unique((l[1] for l in z))
    Dict(map(uniquez) do k
            indices = get_cluster_inds(k,z)
            sr = suffstats(model,X[:,get_right_inds(indices,z)])
            sl = suffstats(model,X[:,get_left_inds(indices,z)])
            (k,SplitMergeCluster(model,sr+sl,sr,sl))
        end)
end

function SuffStats(model::AbstractDPModel, X::AbstractMatrix, z::AbstractVector{Tuple{Int,Bool}})
    uniquez   = unique((l[1] for l in z))
    Dict(map(uniquez) do k
            indices = get_cluster_inds(k,z)
            sr = suffstats(model,X[:,get_right_inds(indices,z)])
            sl = suffstats(model,X[:,get_left_inds(indices,z)])
            (k, (sr,sl))
        end)
end

@inline population(m::SplitMergeCluster) = m.n
@inline population(m::SplitMergeCluster, ::Val{false}) = m.nr
@inline population(m::SplitMergeCluster, ::Val{true}) = m.nl

@inline logαpdf(m::SplitMergeCluster, x)      = logαpdf(m.sampled,x)
@inline logαpdf(m::SplitMergeCluster, x, ::Val{false}) = logαpdf(m.right,x)
@inline logαpdf(m::SplitMergeCluster, x, ::Val{true})  = logαpdf(m.left,x)

@inline isempty(m::SplitMergeCluster, sub::Val) = population(m,side)==0
@inline lognαpdf(m::SplitMergeCluster, x, sub::Val)  = log(population(m, sub)) + logαpdf(m, x, sub)

# Specific `find` functions
@inline get_cluster_inds(key::Int, labels::AbstractVector{Tuple{Int,Bool}}) =
    findall(l->l[1]==key,labels)

@inline get_cluster_inds(k1::Int,k2::Int, labels::AbstractVector{Tuple{Int,Bool}}) =
    findall(l->l[1]==k1 || l[2]==k2,labels)

@inline get_left_inds(indices::Vector{Int}, labels::AbstractVector{Tuple{Int,Bool}}) =
    filter(i->labels[i][2],indices)

@inline get_right_inds(indices::Vector{Int}, labels::AbstractVector{Tuple{Int,Bool}}) =
    filter(i->!labels[i][2],indices)

# Mapping random labels to (cluster, subcluster)
# false -> right cluster
# true  -> left cluster
split_merge_labels(labels::AbstractVector{<:Integer}) =
    map(l->(l,rand()>0.5),labels)
