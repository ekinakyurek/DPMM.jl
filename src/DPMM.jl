module DPMM
using Distributions, ColorBrewer, Colors, Distributed, SharedArrays, SparseArrays, LinearAlgebra, PDMats #Makie

import Base: length, convert, size, *, +, -, getindex, sum, length, rand,~,@propagate_inbounds, @fastmath
@inline ~(x::Distribution) = rand(x)

const colorpalette  = RGBA.(palette("Set3", 12))

import SparseArrays: AbstractSparseMatrix, AbstractSparseVector, nonzeroinds, nonzeros


import Distributions: _rand!, partype, AbstractRNG, multiply!, DirichletCanon,
                      _logpdf!, rand, pdf, params, _wishart_genA, var,
                      mean, cov, params, invcov, logdetcov, sqmahal, sqmahal!,
                      partype, unwhiten_winv!,log2π, mvnormal_c0, _logpdf, lgamma,
                      xlogy, NoArgCheck, suffstats, SufficientStats, GenericMvTDist,
                      AliasTable, GLOBAL_RNG

import PDMats: unwhiten!, add!, quad, quad!

using TimerOutputs
const to = TimerOutput()

dir(path...) = joinpath(dirname(@__DIR__),path...)

include("Core/linearalgebra.jl")
include("Core/mvnormal.jl"); export MvNormalFast
include("Core/niw.jl"); export NormalInverseWishart
include("Core/sparse.jl"); export DPSparseMatrix, DPSparseVector
include("Core/dirichletmultinomial.jl"); export DirichletFast
include("Core/algorithms.jl"); export run!, setup_workers, initialize_clusters
include("Data/data.jl");  export rand_with_label, RandMixture, GridMixture
include("Data/nytimes.jl"); export readNYTimes
include("Data/visualize.jl"); export setup_scene
include("Models/model.jl")
include("Models/dpgmm.jl"); export DPGMM, DPGMMStats #, suffstats, updatestats, downdatestats, posterior, posterior_predictive
include("Models/dpmnmm.jl"); export DPMNMM, DPMNMMStats
include("Clusters/CollapsedCluster.jl"); export CollapsedCluster, CollapsedClusters
include("Clusters/DirectCluster.jl"); export DirectCluster, DirectClusters
include("Clusters/SplitMergeCluster.jl"); export SplitMergeCluster, SplitMergeClusters
include("Algorithms/CollapsedGibbs.jl"); export  CollapsedAlgorithm
include("Algorithms/DirectGibbs.jl"); export DirectAlgorithm
include("Algorithms/SplitMerge.jl"); export SplitMergeAlgorithm

"""
    fit(X::AbstractMatrix; algorithm=DEFAULT_ALGO, ncpu=1, T=3000, benchmark=false, scene=nothing, o...)

`fit` is the main function of DPMM.jl which clusters given data matrix where columns are data points.

The output is the labels for each data point.

Default clustering algorithm is `SplitMergeAlgorithm`

Keywords:

- `ncpu=1` : the number of parallel workers.

- `T=3000` : iteration count

- `benchmarks=false` : if true returns elapsed time

- `scene=nothing`: plot scene for visualization. see `setup_scene`

- o... : other keyword argument specific to `algorithm`
"""
function fit(X::AbstractMatrix; algorithm=DEFAULT_ALGO, ncpu=1, T=3000, benchmark=false, scene=nothing, o...)
    if ncpu>1
         setup_workers(ncpu)
    end
    algo = algorithm(X; parallel=ncpu>1, o...)
    labels, clusters, cluster0 = initialize_clusters(X,algo)
    tres = @elapsed run!(algo, X, labels, clusters, cluster0; T=T, scene=scene)
    @info "$tres second passed"
    if benchmark
        return labels, tres
    end
    return labels
end
export fit

end # module

# include("Serial/CollapsedGibbs.jl"); export collapsed_gibbs
# include("Serial/QuasiCollapsedGibbs.jl");export quasi_collapsed_gibbs
# include("Serial/DirectGibbs.jl"); export direct_gibbs
# include("Serial/QuasiDirectGibbs.jl"); export quasi_direct_gibbs
# include("Serial/SplitMerge.jl"); export split_merge_gibbs, split_merge_gibbs!, split_merge_labels
# include("Parallel/DirectGibbsParallel.jl"); export direct_parallel!, direct_gibbs_parallel!
# include("Parallel/QuasiDirectParallel.jl"); export quasi_direct_parallel!,  quasi_direct_gibbs_parallel!
# include("Parallel/QuasiCollapsedParallel.jl");export quasi_collapsed_parallel!,  quasi_collapsed_gibbs_parallel!
# include("Parallel/SplitMergeParallel.jl");export splitmerge_parallel!, splitmerge_parallel_gibbs!
