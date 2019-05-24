@testset "serial algorithms" begin
    gmodel = GridMixture(2)
    X,labels = rand_with_label(gmodel,1000)
    labels = fit(X; algorithm=CollapsedAlgorithm, T=100)
    labels = fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=100)
    labels = fit(X; algorithm=DirectAlgorithm, T=100)
    labels = fit(X; algorithm=DirectAlgorithm, quasi=true, T=100)
    labels = fit(X; algorithm=SplitMergeAlgorithm,T=100)
    labels = fit(X; algorithm=SplitMergeAlgorithm, merge=true, T=100)
    @test true
end
