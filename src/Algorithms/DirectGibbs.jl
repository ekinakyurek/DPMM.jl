###
#### Interface
###

"""
  DirectAlgorithm{P,Q} <: DPMMAlgorithm{P}

  Run it by:
  ```julia
      labels = fit(X; algorithm = DirectAlgorithm, quasi=false, ncpu=1, T=1000, keywords...)
  ```

  `P` stands for parallel, `Q` stands for quasi.
  Quasi algorithm uses cluster population proportions as cluster weights.
  So, it doesn't sample mixture weights from Dirichlet distribution.
  In large `N`, this is very similar to non-quasi sampler.
  The number of workers can passed by `ncpu` keyword argument to `fit` or `run!` functions

   Provides following methods:
   - `DirectAlgorithm(X::AbstractMatrix{T}; modelType=_default_model(T), α=1, ninit=1, parallel=false, quasi=false, o...)`
   - `random_labels(X::AbstractMatrix, algo::DirectAlgorithm) where P`
   - `create_clusters(X::AbstractMatrix, algo::DirectAlgorithm,labels) where P`
   - `empty_cluster(algo::DirectAlgorithm) where P : an empty cluster`
   - `run!(algo::DirectAlgorithm{P,Q}, X, labels, clusters, cluster0; o...) where {P,Q}`

   Other generic functions are implemented on top of these core functions.
"""
struct DirectAlgorithm{P,Q} <: DPMMAlgorithm{P}
    model::AbstractDPModel
    ninit::Int
end

function DirectAlgorithm(X::AbstractMatrix{T};
                              modelType=_default_model(T),
                              α::Real=1, ninit::Int=1,
                              parallel::Bool=false,
                              quasi::Bool=false, o...) where T
    DirectAlgorithm{parallel, quasi}(modelType(X;α=α), ninit)
end

@inline run!(algo::DirectAlgorithm{false,false},X, args...;o...) =
    direct_gibbs!(algo.model,X,args...;o...)

@inline run!(algo::DirectAlgorithm{false,true},X, args...;o...) =
    quasi_direct_gibbs!(algo.model,X,args...;o...)

@inline run!(algo::DirectAlgorithm{true,false},X, args...;o...) =
    quasi_direct_gibbs_parallel!(algo.model,X,args...;o...)

@inline run!(algo::DirectAlgorithm{true,true},X, args...;o...) =
    direct_gibbs_parallel!(algo.model,X,args...;o...)

random_labels(X, algo::DirectAlgorithm) = rand(1:algo.ninit,size(X,2))
create_clusters(X, algo::DirectAlgorithm,labels) = DirectClusters(algo.model,X,labels)
empty_cluster(algo::DirectAlgorithm) = DirectCluster(algo.model,Val(true))

###
#### Serial (Sequential)
###

# Serial Direct Gibbs Algorithm
function direct_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, scene=nothing)
    for t in 1:T
        record!(scene,labels,t)
        logπs        = logmixture_πs(model.α,clusters) # unnormalized weights
        @inbounds for i=1:size(X,2)
            probs     = ClusterProbs(logπs,clusters,empty_cluster,view(X,:,i)) # chinese restraunt process probabilities
            znew      = rand(GLOBAL_RNG,AliasTable(probs))  # new label
            labels[i] = label_x(clusters,znew)
        end
        clusters = DirectClusters(model,X,labels) # TODO handle empty clusters
    end
end

"""
logmixture_πs(α::V, clusters::Dict{<:Integer, <:AbstractCluster}) where V<:Real

Sample log mixture weights from Dirichlet Distribution.
"""
function logmixture_πs(α::V, clusters::Dict{<:Integer, <:AbstractCluster}) where V<:Real
    log.(rand(DirichletCanon([(V(population(c)) for c in values(clusters))...;α])))
end

"""
    `ClusterProbs(πs::AbstractVector{V}, clusters::Dict, cluster0::AbstractCluster, x::AbstractVector) where V<:Real`

Returns normalized probability vector for a data point being any cluster + a new cluster
"""
function ClusterProbs(logπs::AbstractVector{V}, clusters::Dict{Int,<:AbstractCluster}, cluster0::AbstractCluster, x::AbstractVector) where V<:Real
    p = Array{V,1}(undef,length(clusters)+1)
    max = typemin(V)
    for (j,c) in enumerate(values(clusters))
        @inbounds s = p[j] = logπs[j] + logαpdf(c,x)
        max = s>max ? s : max
    end
    @inbounds s = p[end] = logπs[end] + logαpdf(cluster0,x)
    max = s>max ? s : max
    pc = exp.(p .- max)
    return pc ./ sum(pc)
end

# Serial Quasi-Direct Gibbs Algorithm
function quasi_direct_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, scene=nothing)
    for t in 1:T
        record!(scene,labels,t)
        @inbounds for i=1:size(X,2)
            probs     = CRPprobs(model.α,clusters,empty_cluster,X[:,i]) # chinese restraunt process probabilities
            znew      = rand(GLOBAL_RNG,AliasTable(probs)) # new label
            labels[i] = label_x(clusters,znew)
        end
        clusters = DirectClusters(model,X,labels) # TODO handle empty clusters
    end
end


###
#### Parallel
###
# Parallel Direct Gibbs Kernel
function direct_parallel!(logπs, X, range, labels, clusters, empty_cluster)
    for i=1:size(X,2)
        probs      = ClusterProbs(logπs,clusters,empty_cluster,view(X,:,i)) # chinese restraunt process probabilities
        znew       = rand(GLOBAL_RNG,AliasTable(probs))# new label
        labels[range[i]]  = label_x(clusters,znew)
    end
    return SuffStats(Main._model, X, convert(Array,labels[range]))
end

@inline direct_gibbs_parallel!(labels, clusters, πs) =
    direct_parallel!(πs,Main._X,localindices(labels),labels,clusters,Main._cluster0)

# Parallel Direct Gibbs Algorithm
function direct_gibbs_parallel!(model, X, labels::SharedArray, clusters, empty_cluster; scene=nothing, T=10)
    for t=1:T
        record!(scene,labels,t)
        logπs = logmixture_πs(model.α,clusters) # unnormalized weights
        stats = Dict{Int,<:SufficientStats}[]
        @sync begin
            for p in procs(labels)
                @async push!(stats,remotecall_fetch(direct_gibbs_parallel!,p,labels,clusters,logπs))
            end
        end
        clusters = DirectClusters(model,gather_stats(stats))
    end
end

# Parallel Quasi-Direct Gibbs Kernel
function quasi_direct_parallel!(model, X, range, labels, clusters, empty_cluster)
    for i=1:size(X,2)
        probs      = CRPprobs(model.α, clusters,empty_cluster,view(X,:,i)) # chinese restraunt process probabilities
        znew       = rand(GLOBAL_RNG,AliasTable(probs)) # new label
        labels[range[i]] = label_x(clusters,znew)
    end
    return SuffStats(model, X, convert(Array,labels[range]))
end

@inline quasi_direct_gibbs_parallel!(labels, clusters) =
    quasi_direct_parallel!(Main._model,Main._X,localindices(labels),labels,clusters,Main._cluster0)

# Parallel Quasi-Direct Gibbs Algorithm
function quasi_direct_gibbs_parallel!(model, X,  labels::SharedArray, clusters, empty_cluster; scene=nothing, T=10)
    for t=1:T
        record!(scene,labels,t)
        stats = Dict{Int,<:SufficientStats}[]
        @sync begin
            for p in procs(labels)
                @async push!(stats,remotecall_fetch(quasi_direct_gibbs_parallel!,p,labels,clusters))
            end
        end
        clusters = DirectClusters(model,gather_stats(stats))
    end
end
