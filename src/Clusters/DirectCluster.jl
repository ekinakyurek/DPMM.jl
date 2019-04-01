import Base: +,-, isempty

struct DirectCluster{Pred<:Distribution, Prior<:Distribution} <: AbstractCluster
    n::Int
    sampled::Pred
    prior::Prior
end

@inline isempty(m::DirectCluster) = m.n==0

@inline DirectCluster(m::DPGMM) = DirectCluster(m, suffstats(m))

@inline DirectCluster(m::DPGMM,X::AbstractArray) =
    DirectCluster(m, suffstats(m,X))

@inline DirectCluster(m::DPGMM,s::DPGMMStats) =
    DirectCluster(s.n, rand(posterior(m,s)), m.θprior)

function DirectCluster(m::DPGMM,new::Val{true})
    c = DirectCluster(m, suffstats(m))
    return DirectCluster(floor(Int,m.α),c.sampled,c.prior)
end

@inline pdf(m::DirectCluster,x) = pdf(m.sampled,x)
@inline (m::DirectCluster)(x)   = m.n*pdf(m.sampled,x)

DirectClusters(model::DPGMM, X::AbstractMatrix, z::AbstractArray{Int}) =
    Dict((k,DirectCluster(model,X[:,findall(l->l==k,z)])) for k in unique(z))
