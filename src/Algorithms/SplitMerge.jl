###
#### Interface
###
"""
    SplitMergeAlgorithm{P,Q} <: DPMMAlgorithm{P}


Run it by:
```julia
labels = fit(X; algorithm = SplitMergeAlgorithm, quasi=false, ncpu=1, T=1000, keywords...)
```

`P` stands for parallel, `Q` stands for quasi.
`M=false` algorithm doesn't do merge moves at all, so it is not exact
However, emprical results shows that merge moves very less likely.
The number of workers can passed by `ncpu` keyword argument to `fit` or `run!` functions

Provides following methods:
- `SplitMergeAlgorithm(X::AbstractMatrix{T}; modelType=_default_model(T), α=1, ninit=1, parallel=false, quasi=false, o...)`
- `random_labels(X::AbstractMatrix, algo::SplitMergeAlgorithm) where P`
- `create_clusters(X::AbstractMatrix, algo::SplitMergeAlgorithm,labels) where P`
- `empty_cluster(algo::SplitMergeAlgorithm) where P : an empty cluster`
- `run!(algo::SplitMergeAlgorithm{P,Q}, X, labels, clusters, cluster0; o...) where {P,Q}`

Other generic functions are implemented on top of these core functions.
"""
struct SplitMergeAlgorithm{P, M} <: DPMMAlgorithm{P}
    model::AbstractDPModel
    ninit::Int
end

function SplitMergeAlgorithm(X::AbstractMatrix{T};
                                modelType=_default_model(T),
                                α::Real=1, ninit::Int=1,
                                parallel::Bool=false,
                                merge::Bool=true,
                                o...) where T
    SplitMergeAlgorithm{parallel, merge}(modelType(X;α=α), ninit)
end

run!(algo::SplitMergeAlgorithm{false,M}, X, args...; o...) where M =
    splitmerge_gibbs!(algo.model,X,args...;merge=M, o...)

run!(algo::SplitMergeAlgorithm{true,M}, X, args...;o...) where M =
    splitmerge_gibbs_parallel!(algo.model,X,args...;merge=M, o...)

random_labels(X,algo::SplitMergeAlgorithm)  =
    map(l->(l,rand()>0.5),rand(1:algo.ninit,size(X,2)))

create_clusters(X, algo::SplitMergeAlgorithm, labels) =
    SplitMergeClusters(algo.model,X,labels)

empty_cluster(algo::SplitMergeAlgorithm) = nothing # undefined

# Default algorithm for DPMM.jl is SplitMergeAlgorithm.
const DEFAULT_ALGO = SplitMergeAlgorithm

###
#### Serial
###
function splitmerge_gibbs!(model::AbstractDPModel{V},
                           X::AbstractMatrix,
                           labels::AbstractVector,
                           clusters::Dict{Int,<:SplitMergeCluster},
                           cluster0; merge::Bool=true, T=10, scene=nothing) where V<:Real
    maybe_split = maybeSplit(clusters)
    for t in 1:T
        #record!(scene,labels,t)
        logπs  = logmixture_πs(model.α,clusters)
        logsπs = logsubcluster_πs(model.α/2,clusters)
        @inbounds for i=1:length(labels)
            x = X[:,i]
            probs = RestrictedClusterProbs(logπs,clusters,x)
            z  = label_x2(clusters,rand(AliasTable(probs)))
            labels[i] = (z,SampleSubCluster(logsπs[z],clusters[z],x))
        end
        update_clusters!(model,X,clusters,labels)
        will_split = propose_splits!(model, X, labels, clusters, maybe_split)
        materialize_splits!(model, X, labels, clusters, will_split, maybe_split)
        gc_clusters!(clusters)
        if merge
            will_merge  = propose_merges(model, clusters, X, labels, maybe_split)
            materialize_merges!(model, labels, clusters, will_merge)
        end
        maybeSplit(clusters,maybe_split)
    end
end

logsubcluster_πs(Δ::V, clusters::Dict{Int,<:SplitMergeCluster}) where V<:Real =
    Dict{Int,Tuple{V,V}}((k,randlogdir(GLOBAL_RNG,V(c.nr)+Δ,V(c.nl)+Δ)) for (k,c) in clusters)

"""

    RestrictedClusterProbs(πs::AbstractVector{V}, clusters::Dict,  x::AbstractVector) where V<:Real
    
Returns normalized probability vector for a data point being any cluster
    """
function RestrictedClusterProbs(logπs::AbstractVector{V}, clusters::GenericClusters,
                                x::AbstractVector) where V<:Real
    p = Vector{V}(undef,length(clusters))
    max = typemin(V)
    for (j,c) in enumerate(values(clusters))
        @inbounds s = p[j] = logπs[j] + logαpdf(c,x)
        max = ifelse(s > max,s,max)
    end
    s = zero(V)
    @simd for i in eachindex(p)
        @inbounds s += p[i] = exp(p[i]-max)
    end
    return p ./ s
end

function maybeSplit(clusters::Dict{Int,<:SplitMergeCluster})
    s = Dict{Int,Bool}()
    for (k,c) in clusters
        hist = c.llh_hist
        s[k]  = first(hist) != -Inf && sum(hist)/length(hist) - last(hist) < 1e-2
    end
    return s
end

function maybeSplit(clusters::Dict{Int,<:SplitMergeCluster}, maybe_split::Dict{Int,Bool})
    for (k,c) in clusters
        if !get!(maybe_split,k,false)
            hist = c.llh_hist
            if first(hist) != -Inf
                maybe_split[k]  = sum(hist)/length(hist) - last(hist) < 1e-2
            end
        end
    end
    return maybe_split
end

"""
     SampleSubCluster(πs::Vector{V}, cluster::SplitMergeCluster, x::AbstractVector) where V<:Real

Returns normalized probability vector for a data point being right or left subcluster
"""
function SampleSubCluster(logπs::Tuple{V,V}, cluster::SplitMergeCluster,
                          x::AbstractVector) where V<:Real
    p1 = first(logπs) + logαpdf(cluster,x,Val(false))
    p2 = last(logπs)  + logαpdf(cluster,x,Val(true))
    s  = rand()
    if p1>p2
        return s > one(V)/(one(V)+exp(p2-p1))
    else
        p = exp(p1-p2)
        return s > p/(p+one(V))
    end
end


function update_clusters!(m::AbstractDPModel, X::AbstractMatrix, clusters::GenericClusters,
                            labels::AbstractVector{Tuple{Int,Bool}})
    for (k,c) in clusters
        indices = get_cluster_inds(k,labels)
        @inbounds right   = suffstats(m,X[:,get_right_inds(indices,labels)])
        @inbounds left    = suffstats(m,X[:,get_left_inds(indices,labels)])
        clusters[k] = SplitMergeCluster(c, right+left, right, left; llh_hist=c.llh_hist)
    end
    return clusters
end

function propose_merges(m::AbstractDPModel{T}, clusters::GenericClusters,
                        X::AbstractMatrix, labels::AbstractVector{Tuple{Int,Bool}},
                        maySplit::Dict{Int,Bool}) where T
    α = m.α
    cnstnt = -log(α)+lgamma(α)-2*lgamma(0.5*α)-log(100)
    merge_with = Dict{Int,Int}()
    ckeys = collect(keys(clusters))
    for i=1:length(ckeys)
        k1 = ckeys[i]
        (haskey(merge_with,k1) || !maySplit[k1]) && continue
        for j=i+1:length(ckeys)
            k2 = ckeys[j]
            (haskey(merge_with,k2) || !maySplit[k2]) && continue
            c1, c2 = clusters[k1], clusters[k2]
            s1, s2 = c1.s, c2.s
            s = s1+s2
            p  = posterior(m,s)
            prior = m.θprior
            logH = cnstnt +
                lgamma(s.n)-lgamma(s.n+α)+
            lgamma(s1.n+0.5*α)-lgamma(s1.n) +
                lgamma(s2.n+0.5*α)-lgamma(s2.n)+
            lmllh(prior,p,s.n)-lmllh(prior,c1.post,s1.n)
            -lmllh(prior,c2.post,s2.n)
            
            if logH>0 || logH > log(rand())
                merge_with[k2] = k1
                break
            end
        end
    end
    return merge_with
end
function propose_splits!(m::AbstractDPModel, X::AbstractMatrix, labels::AbstractVector{Tuple{Int,Bool}},
                         clusters::Dict{Int,<:SplitMergeCluster}, maybe_split::Dict{Int,Bool})
    logα = log(m.α)
    will_split = Dict{Int,Bool}()
    for (k,c) in clusters
        if maybe_split[k]
            if c.nr == 0 || c.nl == 0
                reset_cluster!(m,X,labels,clusters,k)
                will_split[k] = false
                continue
            end
            logH = logα+lgamma(c.nr)+
                   lgamma(c.nl)-
                   lgamma(c.n)+
            c.llhs[2]+c.llhs[3]-c.llhs[1]
            
            will_split[k] = (logH>0 || logH>log(rand()))
        else
            will_split[k] = false
        end
    end
    return will_split
end

function reset_cluster!(model::AbstractDPModel{V,D},
                        X::AbstractMatrix,
                        labels::AbstractVector{Tuple{Int,Bool}},
                        clusters::Dict{Int,<:SplitMergeCluster},
                        key::Int) where {V<:Real,D}
    indices = get_cluster_inds(key,labels)
    for i in indices
        @inbounds labels[i] = (key,rand()>0.5)
    end
    sr = suffstats(model,X[:,get_right_inds(indices,labels)])
    sl = suffstats(model,X[:,get_left_inds(indices,labels)])
    c  = clusters[key]
    clusters[key] = SplitMergeCluster(c,c.s,sr,sl)
end


function get_new_keys(will_split::Dict{Int,Bool})
    kmax  = maximum(keys(will_split))
    nkeys = Dict{Int,Int}()
    for (k,ws) in will_split
        if ws
            nkeys[k] = kmax+=1
        else
            nkeys[k] = k
        end
    end
    return nkeys
end

function materialize_splits!(model::AbstractDPModel,
                             X::AbstractMatrix,
                             labels::AbstractVector,
                             clusters::Dict{Int,<:SplitMergeCluster}, will_split::Dict{Int,Bool}, maybe_split::Dict{Int,Bool})
    new_keys  = get_new_keys(will_split)
    for k in collect(keys(clusters))
        c = clusters[k]
        if will_split[k]
            newkey = new_keys[k]
            maybe_split[k] = false
            maybe_split[newkey] = false
            clusters[k], clusters[newkey] = split_cluster!(model, X, labels, c , k, newkey)
        elseif population(c) != 0
            clusters[k] = SplitMergeCluster(population(c), population(c,Val(false)), population(c,Val(true)), c.s,
                                            rand(c.post),
                                            rand(c.rightpost),
                                            rand(c.leftpost),
                                            c.post,
                                            c.rightpost,
                                            c.leftpost,
                                            c.llhs, c.llh_hist, c.prior)
        end
    end
end

function materialize_merges!(model, labels, clusters, will_merge)
    for (k1,k2) in will_merge
        indices = get_cluster_inds(k1,k2,labels)
        for i in indices
            @inbounds labels[i] = (k1,first(labels[i])==k2) # assign k2 to left
        end
        sr, sl = clusters[k1].s, clusters[k2].s
        clusters[k1] = SplitMergeCluster(model, sr+sl, sr, sl)
        delete!(clusters,k2)
    end
end

function gc_clusters!(clusters)
    for (k,c) in clusters
        if population(c) == 0
            delete!(clusters,k)
        end
    end
end


function split_cluster!(model::AbstractDPModel,
                        X::AbstractMatrix,
                        labels::AbstractVector{Tuple{Int,Bool}},
                        cluster::SplitMergeCluster,
                        kold::Int,knew::Int)
    split_indices!(kold,knew,labels)
    return ntuple(2) do j
        k = j==1 ? kold : knew
        indices = get_cluster_inds(k, labels)
        @inbounds sr = suffstats(model,X[:,get_right_inds(indices,labels)])
        @inbounds sl = suffstats(model,X[:,get_left_inds(indices,labels)])
        SplitMergeCluster(model, sr+sl, sr, sl) #FIXME: we know sr+sl from $cluster$
    end
end

function split_indices!(kold::Int, knew::Int, labels::AbstractVector{Tuple{Int,Bool}})
    for i in get_cluster_inds(kold, labels)
        @inbounds  labels[i] = ((last(labels[i]) ? knew : kold), rand()>0.5)
    end
end

function label_x2(clusters::Dict{<:Int, <:Any}, knew::Int)
    for (i,k) in enumerate(keys(clusters))
        if i==knew
            return k
        end
    end
    return 0
end

###
#### Parallel
###

function splitmerge_parallel!(logπs, logsπs, X::AbstractMatrix, range::AbstractRange, labels, clusters)
    @inbounds for i=1:length(range)
        x = X[:,i]
        probs = RestrictedClusterProbs(logπs,clusters,x)
        z  = label_x2(clusters,rand(AliasTable(probs)))
        labels[range[i]] = (z,SampleSubCluster(logsπs[z],clusters[z],x))
    end
    return SuffStats(Main._model, X, convert(Array,labels[range]))
end

@inline splitmerge_parallel!(labels, clusters, πs, sπs) =
    splitmerge_parallel!(πs, sπs, Main._X, localindices(labels), labels,clusters)

function update_clusters!(m::AbstractDPModel, clusters::Dict, stats::Dict{Int,<:Tuple})
    for (k,c) in clusters
        if haskey(stats,k)
            sr, sl = stats[k]
            clusters[k] = SplitMergeCluster(c, sr + sl, sr, sl; llh_hist=c.llh_hist)
        else
            s = suffstats(m)
            clusters[k] = SplitMergeCluster(c, s, s, s; llh_hist=c.llh_hist)
        end
    end
    return clusters
end

function splitmerge_gibbs_parallel!(model, X::AbstractMatrix, labels::SharedArray, clusters, empty_cluster; merge=true, T=10, scene=nothing)
    maybe_split = maybeSplit(clusters)
    stats = Vector{Dict{Int,Tuple{stattype(model),stattype(model)}}}(undef,length(procs(labels)))
    @inbounds for t in 1:T
        #record!(scene,labels,t)
        logπs   = logmixture_πs(model.α,clusters)
        logsπs  = logsubcluster_πs(model.α/2,clusters)
        @sync begin
            for (i,p) in enumerate(procs(labels))
                @async stats[i] = remotecall_fetch(splitmerge_parallel!,p,labels,clusters,logπs,logsπs)
            end
        end
        update_clusters!(model, clusters, gather_stats(stats))
        will_split = propose_splits!(model, X, labels, clusters, maybe_split)
        materialize_splits!(model, X, labels, clusters, will_split, maybe_split)
        gc_clusters!(clusters)
        if merge
            will_merge  = propose_merges(model, clusters, X, labels, maybe_split)
            materialize_merges!(model, labels, clusters, will_merge)
        end
        maybeSplit(clusters, maybe_split)
    end
end

#Gathers parallely collected sufficient stats
function gather_stats(stats::Vector{Dict{Int,Tuple{T,T}}}) where T<:SufficientStats
    gstats = empty(first(stats))
    for s in stats
        for (k,v) in s
            if haskey(gstats,k)
                gs = gstats[k]
                gstats[k] = (first(gs)+first(v),last(gs)+last(v))
            else
                gstats[k] = v
            end
        end
    end
    return gstats
end

# function update_cluster_parallel!(cluster::Pair{Int,<:SplitMergeCluster}, labels::AbstractVector{Tuple{Int,Bool}})
#     k,c = cluster
#     indices = get_cluster_inds(k,labels)
#     right   = suffstats(Main.model,Main._X[:,get_right_inds(indices,labels)])
#     left    = suffstats(Main.model,Main._X[:,get_left_inds(indices,labels)])
#     return Pair(k,SplitMergeCluster(c, right+left, right, left; llh_hist=c.llh_hist))
# end

# function update_clusters_parallel!(clusters::Dict, labels::AbstractVector{Tuple{Int,Bool}})
#     return Dict(pmap(c->update_cluster_parallel!(c,labels),collect(clusters)))
# end
