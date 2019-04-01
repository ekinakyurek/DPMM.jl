function direct_gibbs(X::AbstractMatrix; T::Int=1000, α::Real=1.0, ninit::Int=6, observables=nothing)
    #Initialization
    (D,N),labels,model = init(X,α,ninit)
    #Get Clusters
    clusters = DirectClusters(model,X,labels) # current clusters
    cluster0 = DirectCluster(model)  # empty cluster
    #Run the gibbs sampler
    direct_gibbs!(model, X, labels, clusters, cluster0; T=T, observables=observables)
    return labels
end

function direct_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, observables=nothing)
    for t in 1:T
        record!(observables,labels,t)
        πs  = mixture_πs(model,clusters) # unnormalized weights
        @inbounds for i=1:size(X,2)
            x = X[:,i]
            probs     = ClusterProbs(πs,clusters,empty_cluster,x) # chinese restraunt process probabilities
            znew      =~ Categorical(probs,NoArgCheck()) # new label
            labels[i] = label_x(clusters,znew)
        end
        clusters = DirectClusters(model,X,labels) # TODO handle empty clusters
    end
end

function mixture_πs(model::DPGMM{V}, clusters::Dict) where V<:Real
    return ~Dirichlet([(c.n for c in values(clusters))...;model.α])
end

function ClusterProbs(πs::AbstractVector{V}, clusters::Dict, cluster0::AbstractCluster, x::AbstractVector{V}) where V<:Real
    probs = Array{V,1}(undef,length(clusters)+1)
    for (j,c) in enumerate(values(clusters))
        @inbounds probs[j] = πs[j]*pdf(c,x)
    end
    probs[end] = πs[end]*pdf(cluster0,x)
    return probs/sum(probs)
end
