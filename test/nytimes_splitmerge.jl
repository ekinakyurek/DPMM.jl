X = DPMM.readNYTimes(DPMM.dir("data/docword.nytimes.txt"))

@testset "nytimes" begin
    labels = fit(X; ncpu=3, T=1000)
    @show unique(labels)
    @test true
end
