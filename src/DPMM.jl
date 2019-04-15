module DPMM

using Distributions, ColorBrewer, Colors, SharedArrays #Makie
import Distributions: NoArgCheck
import Base: ~
@inline ~(x::Distribution) = rand(x)
const colorpalette  = RGBA.(palette("Set3", 12))


include("Core/linearalgebra.jl")
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
include("Serial/CollapsedGibbs.jl"); export collapsed_gibbs
include("Serial/QuasiCollapsedGibbs.jl");export quasi_collapsed_gibbs
include("Serial/DirectGibbs.jl"); export direct_gibbs
include("Serial/QuasiDirectGibbs.jl"); export quasi_direct_gibbs
include("Parallel/DirectGibbsParallel.jl"); export direct_parallel!, direct_gibbs_parallel!
include("Parallel/QuasiDirectParallel.jl"); export quasi_direct_parallel!,  quasi_direct_gibbs_parallel!
include("Parallel/QuasiCollapsedParallel.jl");export quasi_collapsed_parallel!,  quasi_collapsed_gibbs_parallel!

end # module
