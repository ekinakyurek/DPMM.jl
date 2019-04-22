function quasi_direct_gibbs(X::AbstractMatrix; T::Int=1000, α::Real=1.0, ninit::Int=6, observables=nothing, modelType=DPGMM{Float64})
    #Initialization
    (D,N),labels,model = init(X,α,ninit,modelType)
    #Get Clusters
    clusters = DirectClusters(model,X,labels) # current clusters
    cluster0 = DirectCluster(model,Val(true))  # empty cluster
    #Run the gibbs sampler
    quasi_direct_gibbs!(model, X, labels, clusters, cluster0; T=T, observables=observables)
    return labels
end

function quasi_direct_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, observables=nothing)
    for t in 1:T
        record!(observables,labels,t)
        @inbounds for i=1:size(X,2)
            probs     = CRPprobs(model,clusters,empty_cluster,X[:,i]) # chinese restraunt process probabilities
            znew      =~ Categorical(probs,NoArgCheck()) # new label
            labels[i] = label_x(clusters,znew)
        end
        clusters = DirectClusters(model,X,labels) # TODO handle empty clusters
    end
end
