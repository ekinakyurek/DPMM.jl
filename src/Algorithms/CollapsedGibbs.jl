###
#### Interface
###

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

function collapsed_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, scene=nothing)
    for t in 1:T
        record!(scene,labels,t)
        @inbounds for i=1:size(X,2)
            x, z = X[:,i], labels[i]
            clusters[z] -= x # remove xi's statistics
            isempty(clusters[z]) && delete!(clusters,z)
            probs     = CRPprobs(model,clusters,empty_cluster,x) # chinese restraunt process probabilities
            znew      = rand(GLOBAL_RNG,AliasTable(probs)) # new label
            labels[i] = place_x!(model,clusters,znew,x)
        end
    end
end


function CRPprobs(model::AbstractDPModel{V}, clusters::Dict, cluster0::AbstractCluster, x::AbstractVector) where V<:Real
    p = Array{V,1}(undef,length(clusters)+1)
    max = typemin(V)
    for (j,c) in enumerate(values(clusters))
        @inbounds s = p[j] = c(x)
        max = s>max ? s : max
    end
    @inbounds s = p[end] = cluster0(x)
    max = s>max ? s : max
    pc = exp.(p .- max)
    return pc ./ sum(pc)
end

function place_x!(model::AbstractDPModel,clusters::Dict,knew::Int,xi::AbstractVector)
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


function quasi_collapsed_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, scene=nothing)
    for t in 1:T
        record!(scene,labels,t)
        @inbounds for i=1:size(X,2)
            probs     = CRPprobs(model,clusters,empty_cluster, view(X,:,i)) # chinese restraunt process probabilities
            znew      = rand(GLOBAL_RNG,AliasTable(probs)) # new label
            labels[i] = label_x(clusters,znew)
        end
        clusters = CollapsedClusters(model,X,labels) # TODO handle empty clusters
    end
end

function label_x(clusters::Dict,knew::Int)
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

function quasi_collapsed_parallel!(model, X, range, labels, clusters, empty_cluster)
    for i=1:size(X,2)
        probs      = CRPprobs(model,clusters,empty_cluster,view(X,:,i)) # chinese restraunt process probabilities
        znew       = rand(GLOBAL_RNG,AliasTable(probs))# new label
        labels[range[i]]  = label_x(clusters,znew)
    end
    return SuffStats(model,X, convert(Array,labels[range]))
end

@inline quasi_collapsed_gibbs_parallel!(labels, clusters) =
    quasi_collapsed_parallel!(Main._model,Main._X,localindices(labels),labels,clusters,Main._cluster0)

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


function gather_stats(stats::Array{Dict{Int, <: Tuple},1})
    gstats = empty(first(stats))
    for s in stats
        for (k,v) in s
            if haskey(gstats,k)
                gs = gstats[k]
                gstats[k] = (gs[1]+v[1], gs[2]+v[2])
            else
                gstats[k] = (v[1], v[2])
            end
        end
    end
    return gstats
end

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
