"""
    DPMMAlgorithm{P}

   Abstract base class for algorithms

  `P` stands for parallel.

   Each subtype should provide the following methods:
   - `AlgoType(X::AbstractMatrix; o...)`` : constructor
   - `random_labels(X::AbstractMatrix,algo::AlgoType{P}) where P` : random label generator
   - `create_clusters(X::AbstractMatrix,algo::AlgoType{P},labels) where P` : initial clusters
   - `empty_cluster(algo::AlgoType) where P` : an empty cluster (may be nothing)
   - `run!(algo::AlgoType{P}, X, labels, clusters, emptycluster;o...) where P` : run! modifies labels

   Other generic functions is implemented on top of these core functions.
"""
abstract type DPMMAlgorithm{P} end

_default_model(::Type{<:AbstractFloat}) = DPGMM
_default_model(::Type{<:Integer})       = DPMNMM

"""
    run!(algo::DPMMAlgorithm, X, labels, clusters, emptycluster;o...)

    Runs the specified Gibbs Algorithm
"""
run!(algo::DPMMAlgorithm,X,args...;o...)

"""
    setup_workers(ncpu::Integer)

    Setup parallel process, initialize required modules
"""
function setup_workers(ncpu::Integer)
    if nworkers() != ncpu
        @warn("setting up parallel processes, takes a while for once!")
        addprocs(ncpu; exeflags="--project=$(dir())") # enable threaded blass
        @everywhere @eval Main using DPMM, SharedArrays, Distributed
        @info "workers: $(Main.Distributed.workers()) initialized"
    end
end


"""
    initialize_clusters(X::AbstractMatrix, algo::DPMMAlgorithm{P}
Initialize clusters and labels, sends data to workers if algo is parallel
"""
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
