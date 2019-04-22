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

include("Core/linearalgebra.jl")
include("Core/mvnormal.jl"); export MvNormalFast
include("Core/niw.jl"); export NormalInverseWishart
include("Core/dirichletmultinomial.jl"); export DirMul
include("Core/sparse.jl"); export DPSparseMatrix, DPSparseVector
include("Data/data.jl"); export rand_with_label, RandMixture, GridMixture
include("Data/nytimes.jl"); export readNYTimes
include("Models/model.jl")
include("Models/dpgmm.jl"); export DPGMM, DPGMMStats, suffstats, updatestats, downdatestats, posterior, posterior_predictive
include("Models/dpdmm.jl"); export DPDMM, DPDMMStats
include("Clusters/CollapsedCluster.jl"); export CollapsedCluster, CollapsedClusters
include("Clusters/DirectCluster.jl"); export DirectCluster, DirectClusters
include("Clusters/SplitMergeCluster.jl"); export SplitMergeCluster, SplitMergeClusters
include("Serial/CollapsedGibbs.jl"); export collapsed_gibbs
include("Serial/QuasiCollapsedGibbs.jl");export quasi_collapsed_gibbs
include("Serial/DirectGibbs.jl"); export direct_gibbs
include("Serial/QuasiDirectGibbs.jl"); export quasi_direct_gibbs
include("Serial/SplitMerge.jl")
include("Parallel/DirectGibbsParallel.jl"); export direct_parallel!, direct_gibbs_parallel!
include("Parallel/QuasiDirectParallel.jl"); export quasi_direct_parallel!,  quasi_direct_gibbs_parallel!
include("Parallel/QuasiCollapsedParallel.jl");export quasi_collapsed_parallel!,  quasi_collapsed_gibbs_parallel!

end # module
