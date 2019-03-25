module DPMM

using Distributions, ColorBrewer, Colors, SharedArrays #Makie
import Distributions: NoArgCheck
import Base: ~
@inline ~(x::Distribution) = rand(x)
const colorpalette  = RGBA.(palette("Set3", 12))

include("pdmat.jl")
include("data.jl"); export rand_with_label, RandMixture
include("niw.jl"); export NormalInverseWishart
include("dpgmm.jl"); export DPGMM
include("suffstats.jl"); export DPGMMStats, suffstats, updatestats, downdatestats, posterior, posterior_predictive
include("CollapsedCluster.jl"); export CollapsedCluster, CollapsedClusters
include("DirectCluster.jl"); export DirectCluster, DirectClusters
include("CollapsedGibbs.jl"); export collapsed_gibbs
include("DirectGibbs.jl"); export direct_gibbs
include("DirectParallel.jl");

end # module
