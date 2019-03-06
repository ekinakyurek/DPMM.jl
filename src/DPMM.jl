module DPMM

using Distributions
import Distributions: NoArgCheck

import Base: ~
@inline ~(x::Distribution) = rand(x)

using ColorBrewer, Colors
const colorpalette  = RGBA.(palette("Set3", 12))

include("pdmat.jl")
include("data.jl"); export rand_w_label, RandMixture
include("niw.jl"); export NormalInverseWishart
include("dpgmm.jl"); export DPGMM
include("suffstats.jl"); export DPGMMStats, suffstats, updatestats, downdatestats, posterior, posterior_predictive
# include("collapsedgibbs.jl"); export collapsed_gibbs
# include("collapsedgibbs2.jl"); export collapsed_gibbs2
include("CollapsedCluster.jl"); export CollapsedCluster
include("CollapsedGibbs.jl"); export collapsed_gibbs


end # module
