using Pkg; Pkg.activate("..")
using Distributed, DPMM, Distributions, ArgParse
import DPMM: init, DirectClusters, DirectCluster, direct_gibbs_parallel!,
            direct_gibbs!, collapsed_gibbs!, quasi_collapsed_gibbs!,
            quasi_direct_gibbs_parallel!, quasi_direct_gibbs!

function parser(args)
    s = ArgParseSettings()
    @add_arg_table s begin
     "--ncpu"
         help = "number of worker nodes"
         arg_type = Int
         default = 3
     "--alpha"
         help = "DPMM model parameter"
         arg_type = Float64
         default = 1.0
     "--K"
         help = "number of mixtures"
         arg_type = Int
         default = 6
     "--Kinit"
         help = "initial guess for number of clusters"
         arg_type = Int
         default = 1
     "--N"
         help = "number of data points"
         arg_type = Int
         default = 1000000
     "--D"
        help = "dimension of data"
        arg_type = Int
        default = 2
     "--T"
       help = "number of iterations"
       arg_type = Int
       default = 100
    end
    return parse_args(args, s; as_symbols=true)
end

#Hyper Parameters for Experiment
const 𝒪 = parser(ARGS)

# Data Generation and Initialization
## Mixture Model
mixture = RandMixture(𝒪[:K])

## Generating Data
data, mixture_labels = rand_with_label(mixture,𝒪[:N])

## Random Labels and Model
(Dx,Nx),plabels,dpmm = init(data,𝒪[:alpha],𝒪[:Kinit],DPGMM{Float64})

## Empty Cluster
cluster0 = DirectCluster(dpmm,Val(true))

# Initialize worker nodes
addprocs(𝒪[:ncpu])
@everywhere begin
    using Pkg; Pkg.activate("..")
    using DPMM, SharedArrays
end

# Send data to worker nodes
@everywhere begin
    const α     = $𝒪[:alpha]
    const X     = $data
    const D     = $𝒪[:D]
    const N     = $𝒪[:N]
    const model = $dpmm
    const empty_cluster = $cluster0
end

directclusters = DirectClusters(dpmm,data,plabels) # current clusters
collapsedclusters = CollapsedClusters(dpmm,data,plabels)
empty_collapsed_cluster = CollapsedCluster(dpmm,Val(true))

#Run parallel direct gibbs sampler
#Share labels across workers
shared_labels = SharedArray(plabels)

# Cold run: Let Julia to compile all functions
quasi_direct_gibbs_parallel!(model, X, directclusters, shared_labels, T=2)
# Benchmark
shared_labels = SharedArray(plabels)
dgptime = @elapsed quasi_direct_gibbs_parallel!(model, X, directclusters, shared_labels, T=𝒪[:T])

# Comparison

## Direct Sampler
### Cold run
plabels_copy  = copy(plabels)
direct_gibbs!(model, X, plabels_copy, directclusters,empty_cluster,T=2)
### Benchmark
plabels_copy  = copy(plabels)
dgtime = @elapsed direct_gibbs!(model, X, plabels_copy, directclusters,empty_cluster,T=𝒪[:T])


## Quasi-Direct Sampler
### Cold run
plabels_copy  = copy(plabels)
quasi_direct_gibbs!(model, X, plabels_copy, directclusters,empty_cluster,T=2)
### Benchmark
plabels_copy  = copy(plabels)
dgtime = @elapsed quasi_direct_gibbs!(model, X, plabels_copy, directclusters,empty_cluster,T=𝒪[:T])

## Collapsed Sampler
### Cold run
plabels_copy  = copy(plabels)
collapsedclusters = CollapsedClusters(dpmm,data,plabels)
collapsed_gibbs!(model,X,plabels_copy,collapsedclusters,empty_collapsed_cluster,T=2)
### Benchmark
plabels_copy  = copy(plabels)
collapsedclusters = CollapsedClusters(dpmm,data,plabels)
cgtime = @elapsed collapsed_gibbs!(model,X,plabels_copy,collapsedclusters,empty_collapsed_cluster,T=𝒪[:T])


## SplitMerge Sampler
### Cold run
sp_labels = SharedArray(split_merge_labels(plabels))
sp_clusters = SplitMergeClusters(dpmm,data,sp_labels)
splitmerge_parallel_gibbs!(model,X,sp_labels,sp_clusters,T=2)
### Benchmark
plabels_copy  = SharedArray(split_merge_labels(plabels))
sp_clusters = SplitMergeClusters(dpmm,data,sp_labels)
spp_time = @elapsed splitmerge_parallel_gibbs!(model,X,sp_labels,sp_clusters,T=𝒪[:T])


## SplitMerge Sampler
### Cold run
sp_labels = SharedArray(split_merge_labels(plabels))
sp_clusters = SplitMergeClusters(dpmm,data,sp_labels)
DPMM.split_merge_gibbs!(model,X,sp_labels,sp_clusters,T=2)
### Benchmark
plabels_copy  = SharedArray(split_merge_labels(plabels))
sp_clusters = SplitMergeClusters(dpmm,data,sp_labels)
sps_time = @elapsed split_merge_gibbs!(model,X,sp_labels,sp_clusters,T=𝒪[:T])





println("$(𝒪[:N])\t$(𝒪[:K])\t$(𝒪[:Kinit])\t$(𝒪[:alpha])\t$(𝒪[:D])\t
$(dgptime/𝒪[:T])\t
$(dgtime/𝒪[:T])\t
$(cgtime/𝒪[:T])\t
$(spp_time/𝒪[:T])\t
$(sps_time/𝒪[:T])\t")
