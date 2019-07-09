julia --project test/parallel_benchmark.jl  --N 1000000 --K 6 --D 100 --Kinit 1 --ncpu 2 --discrete --runserials
julia --project test/parallel_benchmark.jl  --N 1000000 --K 6 --D 100 --Kinit 1 --ncpu 4 --discrete
julia --project test/parallel_benchmark.jl  --N 1000000 --K 6 --D 100 --Kinit 1 --ncpu 8 --discrete

julia --project test/parallel_benchmark.jl  --N 1000000 --K 6 --D 2 --Kinit 1 --ncpu 2 --runserials
julia --project test/parallel_benchmark.jl  --N 1000000 --K 6 --D 2 --Kinit 1 --ncpu 4
julia --project test/parallel_benchmark.jl  --N 1000000 --K 6 --D 2 --Kinit 1 --ncpu 8


