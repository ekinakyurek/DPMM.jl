X = DPMM.readNYTimes(DPMM.dir("data/docword.nytimes.txt"))

@testset "nytimes" begin
    labels = fit(X; T=2)
    @show unique(labels)
    @test true
end
