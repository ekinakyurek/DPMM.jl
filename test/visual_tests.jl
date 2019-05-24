@testset "visual tests" begin
    gmodel = GridMixture(2)
    X,labels = rand_with_label(gmodel,1000)
    scene = setup_scene(X)
    labels = fit(X; algorithm=CollapsedAlgorithm, T=1000, scene=scene)
    labels = fit(X; algorithm=CollapsedAlgorithm, quasi=true, T=1000, scene=scene)
    labels = fit(X; algorithm=DirectAlgorithm, T=1000, scene=scene)
    labels = fit(X; algorithm=DirectAlgorithm, quasi=true, T=1000, scene=scene)
    labels = fit(X; algorithm=SplitMergeAlgorithm,T=1000, scene=scene)
    labels = fit(X; algorithm=SplitMergeAlgorithm, merge=true, T=1000, scene=scene)
    @test true
end
