using DPMM, ArgParse

function parser(args)
    s = ArgParseSettings()
    @add_arg_table s begin
     "--ncpu"
         help = "number of worker nodes"
         arg_type = Int
         default = 2
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
𝒪[:α] = 𝒪[:alpha]
gmodel = RandMixture(𝒪[:K])
X,clabels = rand_with_label(gmodel,𝒪[:N])

println("Benchmarking Quasi-Collapsed Algorithm")
fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=10, ninit=𝒪[:K], benchmark=true)
_,qct = fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true)
println("Benchmarking Direct Algorithm")
fit(X; algorithm=DirectAlgorithm, T=10, ninit=𝒪[:K], benchmark=true)
_,dt = fit(X; algorithm=DirectAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true)
println("Benchmarking Split Merge Algorithm")
fit(X; algorithm=SplitMergeAlgorithm,T=10, ninit=1)
_,smt = fit(X; algorithm=SplitMergeAlgorithm,T=𝒪[:T],ninit=𝒪[:Kinit], benchmark=true)


println("Benchmarking Quasi-Collapsed Algorithm with $(𝒪[:ncpu]) workers")
fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=10, ninit=𝒪[:K], ncpu=𝒪[:ncpu], benchmark=true)
fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=𝒪[:T], ninit=𝒪[:Kinit], ncpu=𝒪[:ncpu], benchmark=true)
_,qcpt= fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=𝒪[:T], ninit=𝒪[:Kinit], ncpu=𝒪[:ncpu], benchmark=true)

println("Benchmarking Direct Algorithm with with $(𝒪[:ncpu]) workers")
fit(X; algorithm=DirectAlgorithm, T=10, ninit=𝒪[:K], ncpu=𝒪[:ncpu], benchmark=true)
fit(X; algorithm=DirectAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], ncpu=𝒪[:ncpu], benchmark=true)
_,dpt = fit(X; algorithm=DirectAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], ncpu=𝒪[:ncpu], benchmark=true)

println("Benchmarking Split-Merge Algorithm with $(𝒪[:ncpu]) workers")
fit(X; algorithm=SplitMergeAlgorithm, T=10, ninit=1, ncpu=𝒪[:ncpu], benchmark=true)
fit(X; algorithm=SplitMergeAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], ncpu=𝒪[:ncpu], benchmark=true)
_,smpt = fit(X; algorithm=DirectAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], ncpu=𝒪[:ncpu], benchmark=true)

println("N\tD\tα\tK\tKinit\tCollapsed\tCollapsed-P\tDirect\tDirect-P\tS-M\tS-M-P\t")
print("$(𝒪[:N])\t$(𝒪[:D])\t$(𝒪[:alpha])\t$(𝒪[:K])\t$(𝒪[:Kinit])\t")
println("$(qct)\t$(qcpt)\t$(dt)\t$(dpt)\t$(smt)\t$(smpt)")
