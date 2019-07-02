julia --project test/utilizations.jl  --N 100000 --K 6 --D 100 --Kinit 1 --ncpu 2 --discrete --runserials
julia --project test/utilizations.jl  --N 100000 --K 6 --D 100 --Kinit 1 --ncpu 4 --discrete
julia --project test/utilizations.jl  --N 100000 --K 6 --D 100 --Kinit 1 --ncpu 8 --discrete


julia --project test/utilizations.jl  --N 100000 --K 6 --D 2 --Kinit 1 --ncpu 2 --runserials
julia --project test/utilizations.jl  --N 100000 --K 6 --D 2 --Kinit 1 --ncpu 4
julia --project test/utilizations.jl  --N 100000 --K 6 --D 2 --Kinit 1 --ncpu 8


julia --project test/utilizations.jl  --N 10000 --K 6 --D 100 --Kinit 1 --ncpu 2 --discrete --runserials
julia --project test/utilizations.jl  --N 10000 --K 6 --D 100 --Kinit 1 --ncpu 4 --discrete
julia --project test/utilizations.jl  --N 10000 --K 6 --D 100 --Kinit 1 --ncpu 8 --discrete


julia --project test/utilizations.jl  --N 10000 --K 6 --D 2 --Kinit 1 --ncpu 2 --runserials
julia --project test/utilizations.jl  --N 10000 --K 6 --D 2 --Kinit 1 --ncpu 4
julia --project test/utilizations.jl  --N 10000 --K 6 --D 2 --Kinit 1 --ncpu 8


julia --project test/utilizations.jl  --N 1000000 --K 60 --D 100 --Kinit 1 --ncpu 2 --discrete --runserials
julia --project test/utilizations.jl  --N 1000000 --K 60 --D 100 --Kinit 1 --ncpu 4 --discrete
julia --project test/utilizations.jl  --N 1000000 --K 60 --D 100 --Kinit 1 --ncpu 8 --discrete


julia --project test/utilizations.jl  --N 1000000 --K 6 --D 30 --Kinit 1 --ncpu 2 --runserials
julia --project test/utilizations.jl  --N 1000000 --K 6 --D 30 --Kinit 1 --ncpu 4
julia --project test/utilizations.jl  --N 1000000 --K 6 --D 30 --Kinit 1 --ncpu 8





