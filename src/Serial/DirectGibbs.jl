function direct_gibbs(X::AbstractMatrix; T::Int=1000, α::Real=1.0, ninit::Int=6, observables=nothing, modelType=DPGMM{Float64})
    #Initialization
    (D,N),labels,model = init(X,α,ninit,modelType)
    #Get Clusters
    clusters = DirectClusters(model,X,labels) # current clusters
    cluster0 = DirectCluster(model,Val(true))  # empty cluster
    #Run the gibbs sampler
    direct_gibbs!(model, X, labels, clusters, cluster0; T=T, observables=observables)
    return labels
end

function direct_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, observables=nothing)
    for t in 1:T
        record!(observables,labels,t)
        @inbounds for i=1:size(X,2)
            πs        = mixture_πs(model,clusters) # unnormalized weights
            probs     = ClusterProbs(πs,clusters,empty_cluster,X[:,i]) # chinese restraunt process probabilities
            znew      =~ Categorical(probs,NoArgCheck()) # new label
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
