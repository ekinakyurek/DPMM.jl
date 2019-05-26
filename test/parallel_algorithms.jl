@testset "parallel algorithms" begin
    gmodel = GridMixture(2)
    X,labels = rand_with_label(gmodel,1000)
    #labels = fit(X; algorithm=CollapsedAlgorithm, T=100)
    @info "Testing Collapsed Algorithm with 2 worker"
    labels = fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=100, ncpu=2)
    @info "Testing Direct Algorithm with 2 worker"
    labels = fit(X; algorithm=DirectAlgorithm, T=100, ncpu=2)
    @info "Testing Quasi-Direct Algorithm with 2 worker"
    labels = fit(X; algorithm=DirectAlgorithm, quasi=true, T=100, ncpu=2)
    @info "Testing Split-Merge Algorithm with 2 workers"
    labels = fit(X; algorithm=SplitMergeAlgorithm,T=100, ncpu=2)
    @info "Testing Split-Merge, without merge, Algorithm with 2 workers"
    labels = fit(X; algorithm=SplitMergeAlgorithm, merge=true, T=100, ncpu=2)
    @test true
end
