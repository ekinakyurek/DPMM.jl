using Distributed, SharedArrays

function direct_parallel!(model, X, range, labels, clusters, empty_cluster)
    for i in range
        probs      = CRPprobs(model,clusters,empty_cluster,X[:,i]) # chinese restraunt process probabilities
        znew       =~ Categorical(probs,NoArgCheck()) # new label
        labels[i]  = label_x(clusters,znew)
    end
end

@inline direct_gibbs_parallel!(labels, clusters) =
    direct_parallel!(Main.model,Main.X,localindices(labels),labels,clusters,Main.empty_cluster)


function direct_gibbs_parallel!(model, X, clusters, labels::SharedArray; observables=nothing, T=10)
    for t=1:T
        record!(observables,labels,t)
        @sync begin
            for p in procs(labels)
                @async remotecall_wait(direct_gibbs_parallel!,p,labels,clusters)
            end
        end
        clusters = DirectClusters(model,X,labels)
    end
end
