import Base: +,-, isempty

struct DirectCluster{Pred<:Distribution, Prior<:Distribution} <: AbstractCluster
    n::Int
    sampled::Pred
    prior::Prior
end

@inline isempty(m::DirectCluster) = m.n==0

@inline DirectCluster(m::AbstractDPModel) = DirectCluster(m, suffstats(m))

@inline DirectCluster(m::AbstractDPModel, X::AbstractArray) =
    DirectCluster(m, suffstats(m,X))

@inline DirectCluster(m::AbstractDPModel, s::SufficientStats) =
    DirectCluster(s.n, rand(posterior(m,s)), m.θprior)

function DirectCluster(m::AbstractDPModel, new::Val{true})
    c = DirectCluster(m, suffstats(m))
    return DirectCluster(floor(Int,m.α),c.sampled,c.prior)
end

@inline pdf(m::DirectCluster,x) = pdf(m.sampled,x)
@inline (m::DirectCluster)(x)   = m.n*pdf(m.sampled,x)

DirectClusters(model::AbstractDPModel, X::AbstractMatrix, z::AbstractArray{Int}) =
    Dict((k,DirectCluster(model,X[:,findall(l->l==k,z)])) for k in unique(z))
