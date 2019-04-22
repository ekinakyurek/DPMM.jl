struct SplitMergeCluster{Pred<:Distribution, Post<:Distribution, Prior<:Distribution} <: AbstractCluster
    n::Int; nr::Int; nl::Int;
    s::SufficientStats;
    sampled::Pred; right::Pred; left::Pred;
    post::Post; rightpost::Post; leftpost::Post;
    llhs::NTuple{3,Float64}; llh_hist::NTuple{4,Float64}; prior::Prior
end

@inline isempty(m::SplitMergeCluster) = m.n==0
@inline is_right_empty(m::SplitMergeCluster) = m.nr==0
@inline is_left_empty(m::SplitMergeCluster) = m.nl==0

#@inline SplitMergeCluster(m::AbstractDPModel) = SplitMergeCluster(m, suffstats(m))

function SplitMergeCluster(m::AbstractDPModel{<:Any,D},
                           s::SufficientStats,
                           sr::SufficientStats,
                           sl::SufficientStats) where D
    prior = m.θprior
    ps, psr, psl  = posterior(prior,s), posterior(prior,sr), posterior(prior,sl)

    llhs = (lmllh(prior, ps, s.n), lmllh(prior, psr, sr.n), lmllh(prior, psl, sl.n))
    SplitMergeCluster(s.n, sr.n, sl.n, s, rand(ps), rand(psr), rand(psl),
                      ps, psr, psl, llhs, (-Inf,-Inf,-Inf,llhs[2]+llhs[3]), prior)
end

function SplitMergeCluster(c::SplitMergeCluster,
                           s::SufficientStats,
                           sr::SufficientStats,
                           sl::SufficientStats;
                           llh_hist::NTuple{4,Float64}=(-Inf,-Inf,-Inf,-Inf))
    prior = c.prior
    ps, psr, psl  = posterior(prior,s), posterior(prior,sr), posterior(prior,sl)
    llhs = (lmllh(prior, ps,  s.n), lmllh(prior, psr, sr.n), lmllh(prior, psl, sl.n))
    SplitMergeCluster(s.n, sr.n, sl.n, s, c.sampled, c.right, c.left, ps, psr, psl,
                      llhs, (llh_hist[2:end]...,llhs[2]+llhs[3]), prior)
end


#@inline SplitMergeCluster(m::AbstractDPModel, new::Val{true}) = #uninitialized cluster
#     SplitMergeCluster(floor(Int,m.α),0,0,rand(m.θprior),rand(m.θprior),rand(m.θprior),m.θprior)

function SplitMergeClusters(model::AbstractDPModel, X::AbstractMatrix, z::AbstractVector{Tuple{Int,Bool}})
    uniquez   = unique((l[1] for l in z))
    Dict(map(uniquez) do k
            indices = get_cluster_inds(k,z)
            sr = suffstats(model,X[:,get_right_inds(indices,z)])
            sl = suffstats(model,X[:,get_left_inds(indices,z)])
            (k,SplitMergeCluster(model,sr+sl,sr,sl))
        end)
end

@inline pdf(m::SplitMergeCluster,x)         = pdf(m.sampled,x)
@inline rightpdf(m::SplitMergeCluster,x)    = pdf(m.right,x)
@inline leftpdf(m::SplitMergeCluster,x)     = pdf(m.left,x)
@inline (m::SplitMergeCluster)(x)           = m.n*pdf(m.sampled,x)
@inline (m::SplitMergeCluster)(x, ::Val{1}) = m.nr*pdf(m.rightx)
@inline (m::SplitMergeCluster)(x, ::Val{2}) = m.nl*pdf(m.left,x)

@inline get_cluster_inds(key::Int, labels::AbstractVector{Tuple{Int,Bool}}) =
    findall(l->l[1]==key,labels)

@inline get_cluster_inds(k1::Int,k2::Int, labels::AbstractVector{Tuple{Int,Bool}}) =
    findall(l->l[1]==k1 || l[2]==k2,labels)

@inline get_left_inds(indices::Vector{Int}, labels::AbstractVector{Tuple{Int,Bool}}) =
    filter(i->labels[i][2],indices)

@inline get_right_inds(indices::Vector{Int}, labels::AbstractVector{Tuple{Int,Bool}}) =
    filter(i->!labels[i][2],indices)

split_merge_labels(labels::AbstractVector{Int}) =
    map(l->(l,rand()>0.5),labels)
