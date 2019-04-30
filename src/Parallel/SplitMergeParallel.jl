function splitmerge_parallel!(πs, sπs, X, range, labels, clusters)
    for i in range
        x = view(X,:,i)
        probs = RestrictedClusterProbs(πs,clusters,x)
        z  = label_x2(clusters,rand(GLOBAL_RNG,AliasTable(probs)))
        labels[i] = (z,SampleSubCluster(sπs[z],clusters[z],x))
    end
end

@inline splitmerge_parallel!(labels, clusters, πs, sπs) =
    splitmerge_parallel!(πs, sπs, Main.X,localindices(labels), labels,clusters)


function splitmerge_parallel_gibbs!(model, X::AbstractMatrix, labels, clusters; T=10, observables=nothing)
    for t in 1:T
        #record!(observables,first.(labels),t)
        πs          = mixture_πsv2(model.α,clusters)
        sπs         = subcluster_πs(model.α/2,clusters)
        maybe_split = maybeSplit(clusters)
        @sync begin
            for p in procs(labels)
                @async remotecall_wait(splitmerge_parallel!,p,labels,clusters,πs,sπs)
            end
        end
        update_clusters!(model,X,clusters,labels)
        will_split = propose_splits!(model, X, labels, clusters, maybe_split)
        materialize_splits!(model, X, labels, clusters, will_split)
        gc_clusters!(clusters, maybe_split)
        will_merge  = propose_merges(model, clusters, X, labels, maybe_split)
        materialize_merges!(model, labels, clusters, will_merge)
    end
end
