###
#### Interface
###

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
#### Serial
###

function direct_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, scene=nothing)
    for t in 1:T
        record!(scene,labels,t)
        πs        = mixture_πs(model,clusters) # unnormalized weights
        @inbounds for i=1:size(X,2)
            probs     = ClusterProbs(πs,clusters,empty_cluster,view(X,:,i)) # chinese restraunt process probabilities
            znew      = rand(GLOBAL_RNG,AliasTable(probs))  # new label
            labels[i] = label_x(clusters,znew)
        end
        clusters = DirectClusters(model,X,labels) # TODO handle empty clusters
    end
end

function mixture_πs(model::AbstractDPModel{V}, clusters::Dict) where V<:Real
    rand(DirichletCanon([(V(c.n) for c in values(clusters))...;model.α]))
end

function ClusterProbs(πs::AbstractVector{V}, clusters::Dict, cluster0::AbstractCluster, x::AbstractVector) where V<:Real
    p = Array{V,1}(undef,length(clusters)+1)
    for (j,c) in enumerate(values(clusters))
        @inbounds p[j] = πs[j]*pdf(c,x)
    end
    @inbounds p[end] = πs[end]*pdf(cluster0,x)
    return p/sum(p)
end


function quasi_direct_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, scene=nothing)
    for t in 1:T
        record!(scene,labels,t)
        @inbounds for i=1:size(X,2)
            probs     = CRPprobs(model,clusters,empty_cluster,X[:,i]) # chinese restraunt process probabilities
            znew      = rand(GLOBAL_RNG,AliasTable(probs)) # new label
            labels[i] = label_x(clusters,znew)
        end
        clusters = DirectClusters(model,X,labels) # TODO handle empty clusters
    end
end


###
#### Parallel
###

function direct_parallel!(πs, X, range, labels, clusters, empty_cluster)
    for i in range
        probs      = ClusterProbs(πs,clusters,empty_cluster,view(X,:,i)) # chinese restraunt process probabilities
        znew       = rand(GLOBAL_RNG,AliasTable(probs))# new label
        labels[i]  = label_x(clusters,znew)
    end
end

@inline direct_gibbs_parallel!(labels, clusters, πs) =
    direct_parallel!(πs,Main.X,localindices(labels),labels,clusters,Main.cluster0)

function direct_gibbs_parallel!(model, X, labels::SharedArray, clusters, empty_cluster; scene=nothing, T=10)
    for t=1:T
        record!(scene,labels,t)
        πs = mixture_πs(model,clusters) # unnormalized weights
        @sync begin
            for p in procs(labels)
                @async remotecall_wait(direct_gibbs_parallel!,p,labels,clusters,πs)
            end
        end
        clusters = DirectClusters(model,X,labels)
    end
end

function printStats(labels)
    unkeys = unique(labels)
    for k in unkeys
       println(k,"=>",count(zi->zi==k,labels))
    end
end


function quasi_direct_parallel!(model, X, range, labels, clusters, empty_cluster)
    for i in range
        probs      = CRPprobs(model,clusters,empty_cluster,X[:,i]) # chinese restraunt process probabilities
        znew       = rand(GLOBAL_RNG,AliasTable(probs)) # new label
        labels[i]  = label_x(clusters,znew)
    end
end

@inline quasi_direct_gibbs_parallel!(labels, clusters) =
    quasi_direct_parallel!(Main.model,Main.X,localindices(labels),labels,clusters,Main.cluster0)


function quasi_direct_gibbs_parallel!(model, X,  labels::SharedArray, clusters, empty_cluster; scene=nothing, T=10)
    for t=1:T
        record!(scene,labels,t)
        @sync begin
            for p in procs(labels)
                @async remotecall_wait(quasi_direct_gibbs_parallel!,p,labels,clusters)
            end
        end
        clusters = DirectClusters(model,X,labels)
    end
end
