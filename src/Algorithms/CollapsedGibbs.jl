###
#### Interface
###

"""
  CollapsedAlgorithm{P,Q} <: DPMMAlgorithm{P}

  Run it by:
  ```julia
      labels = fit(X; algorithm = CollapsedAlgorithm, quasi=false, ncpu=1, T=1000, keywords...)
  ```

  `P` stands for parallel, `Q` stands for quasi.
  Quasi algorithm updates the clusters only in the end of each iteration.
  Parallel algorithm is valid for quasi-collapsed algorithm only.
  The number of workers can passed by `ncpu` keyword argument to `fit` or `run!` functions

   Provides following methods:
   - `CollapsedAlgorithm(X::AbstractMatrix{T}; modelType=_default_model(T), α=1, ninit=1, parallel=false, quasi=false, o...)`
   - `random_labels(X::AbstractMatrix, algo::CollapsedAlgorithm) where P`
   - `create_clusters(X::AbstractMatrix, algo::CollapsedAlgorithm,labels) where P`
   - `empty_cluster(algo::CollapsedAlgorithm) where P : an empty cluster`
   - `run!(algo::CollapsedAlgorithm{P,Q}, X, labels, clusters, cluster0; o...) where {P,Q}`

   Other generic functions are implemented on top of these core functions.
"""
struct CollapsedAlgorithm{P,Q} <: DPMMAlgorithm{P}
    model::AbstractDPModel
    ninit::Int
end

function CollapsedAlgorithm(X::AbstractMatrix{T};
                                modelType=_default_model(T),
                                α::Real=1, ninit::Int=1,
                                parallel::Bool=false,
                                quasi::Bool=false, o...) where T

    CollapsedAlgorithm{parallel, quasi}(modelType(X;α=α), ninit)
end

run!(algo::CollapsedAlgorithm{false,false},X,args...;o...) =
    collapsed_gibbs!(algo.model,X,args...;o...)

run!(algo::CollapsedAlgorithm{false,true},X,args...;o...) =
    quasi_collapsed_gibbs!(algo.model,X,args...;o...)

run!(algo::CollapsedAlgorithm{true,true},X,args...;o...) =
    quasi_collapsed_gibbs_parallel!(algo.model,X,args...;o...)

random_labels(X,algo::CollapsedAlgorithm) = rand(1:algo.ninit,size(X,2))
create_clusters(X,algo::CollapsedAlgorithm,labels) = CollapsedClusters(algo.model,X,labels)
empty_cluster(algo::CollapsedAlgorithm) = CollapsedCluster(algo.model,Val(true))

###
#### Serial
###

#Serial Collapsed Gibbs Algorithm
function collapsed_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, scene=nothing)
    for t in 1:T
        record!(scene,labels,t)
        @inbounds for i=1:size(X,2)
            x, z = X[:,i], labels[i]
            clusters[z] -= x # remove xi's statistics
            isempty(clusters[z]) && delete!(clusters,z)
            probs     = CRPprobs(model.α,clusters,empty_cluster,x) # chinese restraunt process probabilities
            znew      = rand(GLOBAL_RNG,AliasTable(probs)) # new label
            labels[i] = place_x!(model,clusters,znew,x)
        end
    end
end

"""
    `CRPprobs(clusters::Dict, cluster0::AbstractCluster, x::AbstractVector) where V<:Real`

Returns Chineese Restraunt Probabilities for a data point being any cluster + a new cluster
"""
function CRPprobs(α::V, clusters::Dict{Int, <:AbstractCluster}, cluster0::AbstractCluster, x::AbstractVector) where V<:Real
    p = Array{V,1}(undef,length(clusters)+1)
    max = typemin(V)
    for (j,c) in enumerate(values(clusters))
        @inbounds s = p[j] = lognαpdf(c,x)
        max = s>max ? s : max
    end
    @inbounds s = p[end] = lognαpdf(cluster0,x)
    max = s>max ? s : max
    pc = exp.(p .- max)
    return pc ./ sum(pc)
end


"""
    `place_x!(model::AbstractDPModel,clusters::Dict,knew::Int,xi::AbstractVector)`

Place a data point to its new cluster. This modifies `clusters`
"""
function place_x!(model::AbstractDPModel,clusters::Dict{Int,<:AbstractCluster},knew::Int,xi::AbstractVector)
    cks = collect(keys(clusters))
    if knew > length(clusters)
        ck = maximum(cks)+1
        clusters[ck] = CollapsedCluster(model,xi)
    else
        ck = cks[knew]
        clusters[ck] += xi
    end
    return ck
end

#Serial Quasi-Collapsed Gibbs Algorithm
function quasi_collapsed_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, scene=nothing)
    for t in 1:T
        record!(scene,labels,t)
        @inbounds for i=1:size(X,2)
            probs     = CRPprobs(model.α,clusters,empty_cluster, view(X,:,i)) # chinese restraunt process probabilities
            znew      = rand(GLOBAL_RNG,AliasTable(probs)) # new label
            labels[i] = label_x(clusters,znew)
        end
        clusters = CollapsedClusters(model,X,labels) # TODO handle empty clusters
    end
end


"""
    `label_x(clusters::Dict,knew::Int)`

Return new cluster number for a data point
"""
function label_x(clusters::Dict{Int,<:AbstractCluster},knew::Int)
    cks = collect(keys(clusters))
    if knew > length(clusters)
        return maximum(cks)+1
    else
        return cks[knew]
    end
end

###
#### Parallel
###

#Parallel Quasi-Collapsed Gibbs Kernel
function quasi_collapsed_parallel!(model, X, range, labels, clusters, empty_cluster)
    for i=1:size(X,2)
        probs      = CRPprobs(model.α, clusters,empty_cluster,view(X,:,i)) # chinese restraunt process probabilities
        znew       = rand(GLOBAL_RNG,AliasTable(probs))# new label
        labels[range[i]]  = label_x(clusters,znew)
    end
    return SuffStats(model,X, convert(Array,labels[range]))
end

@inline quasi_collapsed_gibbs_parallel!(labels, clusters) =
    quasi_collapsed_parallel!(Main._model,Main._X,localindices(labels),labels,clusters,Main._cluster0)

#Parallel Quasi-Collapsed Gibbs Algorithm
function quasi_collapsed_gibbs_parallel!(model, X, labels, clusters, empty_cluster; scene=nothing, T=10)
    for t=1:T
        record!(scene,labels,t)
        stats = Dict{Int,<:SufficientStats}[]
        @sync begin
            for p in procs(labels)
                @async push!(stats,remotecall_fetch(quasi_collapsed_gibbs_parallel!,p,labels,clusters))
            end
        end
        clusters = CollapsedClusters(model,gather_stats(stats))
    end
end

#Gathers parallely collected sufficient stats
function gather_stats(stats::Array{Dict{Int,<: SufficientStats},1})
    gstats = empty(first(stats))
    for s in stats
        for (k,v) in s
            if haskey(gstats,k)
                gstats[k] = gstats[k] + v
            else
                gstats[k] = v
            end
        end
    end
    return gstats
end
