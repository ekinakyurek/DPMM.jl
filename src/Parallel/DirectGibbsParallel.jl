function direct_parallel!(πs, X, range, labels, clusters, empty_cluster)
    for i in range
        probs      = ClusterProbs(πs,clusters,empty_cluster,X[:,i]) # chinese restraunt process probabilities
        znew       =~ Categorical(probs,NoArgCheck()) # new label
        labels[i]  = label_x(clusters,znew)
    end
end

@inline direct_gibbs_parallel!(labels, clusters, πs) =
    direct_parallel!(πs,Main.X,localindices(labels),labels,clusters,Main.empty_cluster)

function direct_gibbs_parallel!(model, X, clusters, labels::SharedArray; observables=nothing, T=10)
    for t=1:T
        record!(observables,labels,t)
        πs = mixture_πs(model,clusters) # unnormalized weights
        @sync begin
            for p in procs(labels)
                @async remotecall_wait(direct_gibbs_parallel!,p,labels,clusters,πs)
            end
        end
        clusters = DirectClusters(model,X,labels)
	printStats(labels)
    end
end

function printStats(labels)
    unkeys = unique(labels)
    for k in unkeys
       println(k,"=>",count(zi->zi==k,labels))
    end
end
