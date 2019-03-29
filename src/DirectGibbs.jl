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
        @inbounds for i=1:size(X,2)
            x = X[:,i]
            probs     = CRPprobs(model,clusters,empty_cluster,x) # chinese restraunt process probabilities
            znew      =~ Categorical(probs,NoArgCheck()) # new label
            labels[i] = label_x(clusters,znew)
        end
        clusters = DirectClusters(model,X,labels) # TODO handle empty clusters
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