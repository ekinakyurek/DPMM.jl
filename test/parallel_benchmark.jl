using DPMM, ArgParse, Random

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
      "--seed"
       help = "seed for random number generator"
       arg_type = Int
       default = 11131994
      "--discrete"
        action = :store_true
        help = "test on multinomial data"
      "--runserials"
        action = :store_true
        help = "run non parallel versions of the algorithm for comparison"
 
    end
    return parse_args(args, s; as_symbols=true)
end

#Hyper Parameters for Experiment
const 𝒪 = parser(ARGS)
𝒪[:α] = 𝒪[:alpha]

if 𝒪[:discrete]
    gmodel = RandDiscreteMixture(𝒪[:K]; D=𝒪[:D])
else
    gmodel = RandMixture(𝒪[:K]; D=𝒪[:D])
end

Random.seed!(11131994)
X = rand(gmodel,𝒪[:N])

ti1=ti2=ti3=0.0
if  𝒪[:runserials]
    println("Benchmarking: ", CollapsedAlgorithm)
    fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=10, ninit=𝒪[:K], benchmark=true)
    Random.seed!(11131994)
    labels,ti1 = fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true)
    println(length(unique(labels)))
    println(ti1)
    
    println("Benchmarking: ", DirectAlgorithm)
    fit(X; algorithm=DirectAlgorithm, T=10, ninit=𝒪[:K], benchmark=true)
    Random.seed!(11131994)
    _,ti2 = fit(X; algorithm=DirectAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true)
    Random.seed!(11131994)
    labels2,ti2 = fit(X; algorithm=DirectAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true)
    println(length(unique(labels2)))
    println(ti2)

    println("Benchmarking: ", SplitMergeAlgorithm)
    fit(X; algorithm=SplitMergeAlgorithm, T=10, ninit=𝒪[:K], benchmark=true)
    Random.seed!(11131994)
    _,ti3 = fit(X; algorithm=SplitMergeAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true)
    Random.seed!(11131994)
    labels3,ti3 = fit(X; algorithm=SplitMergeAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true)
    println(ti3)
    println(length(unique(labels3)))
end


println("Benchmarking: ", CollapsedAlgorithm)
fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=10, ninit=𝒪[:K], benchmark=true, ncpu=𝒪[:ncpu])
Random.seed!(11131994)
_,pti1 = fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true, ncpu=𝒪[:ncpu])
Random.seed!(11131994)
labelsp1,pti1 = fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true, ncpu=𝒪[:ncpu])
println(pti1)
println(length(unique(labelsp1)))

println("Benchmarking: ", DirectAlgorithm)
fit(X; algorithm=DirectAlgorithm, T=10, ninit=𝒪[:K], benchmark=true, ncpu=𝒪[:ncpu])
Random.seed!(11131994)
_,pti2 = fit(X; algorithm=DirectAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true, ncpu=𝒪[:ncpu])
Random.seed!(11131994)
labelsp2,pti2 = fit(X; algorithm=DirectAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true, ncpu=𝒪[:ncpu])
println(pti2)
println(length(unique(labelsp2)))

println("Benchmarking: ", SplitMergeAlgorithm)
fit(X; algorithm=SplitMergeAlgorithm, T=10, ninit=𝒪[:K], benchmark=true, ncpu=𝒪[:ncpu])
Random.seed!(11131994)
_,pti3 = fit(X; algorithm=SplitMergeAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true, ncpu=𝒪[:ncpu])
Random.seed!(11131994)
labelsp3,pti3 = fit(X; algorithm=SplitMergeAlgorithm, T=𝒪[:T], ninit=𝒪[:Kinit], benchmark=true, ncpu=𝒪[:ncpu])
println(pti3)
println(length(unique(labelsp3)))

println("N\tD\tα\tK\tKinit\tCollapsed\tCollapsed-P\tDirect\tDirect-P\tS-M\tS-M-P\t")
print("$(𝒪[:N])\t$(𝒪[:D])\t$(𝒪[:alpha])\t$(𝒪[:K])\t$(𝒪[:Kinit])\t")
println("$(ti1)\t$(pti1)\t$(ti2)\t$(pti2)\t$(ti3)\t$(pti3)")
