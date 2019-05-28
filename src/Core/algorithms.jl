abstract type DPMMAlgorithm{P} end

_default_model(::Type{<:AbstractFloat}) = DPGMM
_default_model(::Type{<:Integer})       = DPDMM

function setup_workers(ncpu)
    if nworkers() != ncpu
        @warn("setting up parallel processes, takes a while for once!")
        addprocs(ncpu; exeflags="--project=$(dir())") # enable threaded blass
        @everywhere @eval Main using DPMM, SharedArrays, Distributed
        @info "workers: $(Main.Distributed.workers()) initialized"
    end
end

function initialize_clusters(X::AbstractMatrix, algo::DPMMAlgorithm{P}) where P
    labels    = random_labels(X,algo)
    clusters  = create_clusters(X,algo,labels)
    cluster0  = empty_cluster(algo)
    if P
        ws = workers()
        labels = SharedArray(labels)
        @everywhere ws _model    = $(algo.model)
        @everywhere ws _cluster0 = $cluster0
        @sync for p in procs(labels)
            inds = remotecall_fetch(localindices,p,labels)
            @async @everywhere [p] _X = $(X[:,inds])
        end
    end
    return labels, clusters, cluster0
end
