function collapsed_gibbs(X::AbstractMatrix; T::Int=1000, α::Real=1.0, ninit::Int=6, observables=nothing, modelType=DPGMM{Float64})
    #Initialization
    (D,N),labels,model = init(X,α,ninit,modelType)
    #Get Clusters
    clusters = CollapsedClusters(model,X,labels) # current clusters
    cluster0 = CollapsedCluster(model)  # empty cluster
    #clusters[0] = CollapsedCluster(model,Val(true))
    collapsed_gibbs!(model, X, labels, clusters, cluster0; T=T, observables=observables)
    return labels
end

function collapsed_gibbs!(model, X::AbstractMatrix, labels, clusters, empty_cluster;T=10, observables=nothing)
    for t in 1:T
        record!(observables,labels,t)
        @inbounds for i=1:size(X,2)
            x, z = X[:,i], labels[i]
            clusters[z] -= x # remove xi's statistics
            isempty(clusters[z]) && delete!(clusters,z)
            probs     = CRPprobs(model,clusters,empty_cluster,x) # chinese restraunt process probabilities
            znew      =~ Categorical(probs,NoArgCheck()) # new label
            labels[i] = place_x!(model,clusters,znew,x)
        end
    end
end


function CRPprobs(model::DPGMM{V}, clusters::Dict, cluster0::AbstractCluster, x::AbstractVector) where V<:Real
    probs = Array{V,1}(undef,length(clusters)+1)
    for (j,c) in enumerate(values(clusters))
        @inbounds probs[j] = c(x)
    end
    #probs = map(c->c(x)::V,values(clusters))
    probs[end] = model.α*pdf(cluster0,x)
    return probs/sum(probs)
end

function place_x!(model::DPGMM,clusters::Dict,knew::Int,xi::AbstractVector)
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

record!(observables::Nothing,z,T) = nothing
function record!(observables,z,T)
    K=sort(unique(z))
    colors = map(zi->(findfirst(x->x==zi,K)-1)%12+1,z)
    observables[1][] = colorpalette[colors]
    observables[2][] = ("T=$T","")
    sleep(0.001)
end
