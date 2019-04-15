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
         default = 13
     "--alpha"
         help = "DPMM model parameter"
         arg_type = Float64
         default = 1000.0
     "--K"
         help = "number of mixtures"
         arg_type = Int
         default = 5
     "--Kinit"
         help = "initial guess for number of clusters"
         arg_type = Int
         default = 50
     "--N"
         help = "number of data points"
         arg_type = Int
         default = 300000
     "--D"
        help = "dimension of data"
        arg_type = Int
        default = 102660
     "--T"
       help = "number of iterations"
       arg_type = Int
       default = 1000
    end
    return parse_args(args, s; as_symbols=true)
end

#Hyper Parameters for Experiment
const 𝒪 = parser(ARGS)

data = readNYTimes("../data/docword.nytimes.txt")

## Random Labels and Model
(Dx,Nx),plabels,dpmm = init(data,𝒪[:alpha],𝒪[:Kinit], DPDMM{Float64})

## Empty Cluster
cluster0 = DirectCluster(dpmm)

# Initialize worker nodes
addprocs(𝒪[:ncpu])
@everywhere begin
    using Pkg; Pkg.activate("..")
    using DPMM, SharedArrays
end

# Send data to worker nodes
@everywhere begin
    const α     = $𝒪[:alpha]
    const D     = $Dx
    const N     = $Nx
end

@everywhere begin
    const model = $dpmm
    const empty_cluster = $cluster0
end

@everywhere const X  = $data

directclusters = DirectClusters(dpmm,data,plabels) # current clusters
shared_labels = SharedArray(plabels)
direct_gibbs_parallel!(model, X, directclusters, shared_labels, T=𝒪[:T])
