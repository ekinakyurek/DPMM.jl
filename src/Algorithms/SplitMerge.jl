###
#### Interface
###

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

random_labels(X,algo::SplitMergeAlgorithm) =
    map(l->(l,rand()>0.5),rand(1:algo.ninit,size(X,2)))
create_clusters(X, algo::SplitMergeAlgorithm, labels) =
    SplitMergeClusters(algo.model,X,labels)
empty_cluster(algo::SplitMergeAlgorithm) = nothing # undefined

const DEFAULT_ALGO = SplitMergeAlgorithm

###
#### Serial
###
function splitmerge_gibbs!(model, X::AbstractMatrix, labels, clusters, cluster0; merge=true, T=10, scene=nothing)
    for t in 1:T
        record!(scene,labels,t)
        πs          = mixture_πsv2(model.α,clusters)
        sπs         = subcluster_πs(model.α/2,clusters)
        maybe_split = maybeSplit(clusters)

        @inbounds for i=1:size(X,2) # make parallel
            x = view(X,:,i)
            probs = RestrictedClusterProbs(πs,clusters,x)
            z  = label_x2(clusters,rand(GLOBAL_RNG,AliasTable(probs)))
            labels[i] = (z,SampleSubCluster(sπs[z],clusters[z],x))
        end

        update_clusters!(model,X,clusters,labels)
        will_split = propose_splits!(model, X, labels, clusters, maybe_split)
        materialize_splits!(model, X, labels, clusters, will_split)
        gc_clusters!(clusters, maybe_split)
        if merge
            will_merge  = propose_merges(model, clusters, X, labels, maybe_split)
            materialize_merges!(model, labels, clusters, will_merge)
        end
    end
end

subcluster_πs(Δ::V, clusters::Dict) where V<:Real  =
    Dict((k,rand(DirichletCanon([V(c.nr)+Δ,V(c.nl)+Δ]))) for (k,c) in clusters)

function mixture_πsv2(α::V, clusters::Dict) where V<:Real
    rand(DirichletCanon([(V(c.n) for c in values(clusters))...;α]))
end

function RestrictedClusterProbs(πs::AbstractVector{V}, clusters::Dict,  x::AbstractVector) where V<:Real
    p = Array{V,1}(undef,length(clusters))
    for (j,c) in enumerate(values(clusters))
        @inbounds p[j] = πs[j]*pdf(c,x)
    end
    return p/sum(p)
end

function maybeSplit(clusters)
    s = Dict{Int,Bool}()
    for (k,c) in clusters
        llavg = sum(c.llh_hist)/length(c.llh_hist)
        s[k]  = llavg != -Inf && llavg-last(c.llh_hist)<1e-2
    end
    return s
end

@inline function SampleSubCluster(πs::Vector{V}, cluster::SplitMergeCluster, x::AbstractVector) where V<:Real
    p1 = πs[1]*rightpdf(cluster,x)
    p2 = πs[2]*leftpdf(cluster,x)
    return rand(GLOBAL_RNG) > (p1/(p1+p2))
end


function update_clusters!(m::AbstractDPModel, X::AbstractMatrix, clusters::Dict, labels::AbstractVector{Tuple{Int,Bool}})
    for (k,c) in clusters
        indices = get_cluster_inds(k,labels)
        right   = suffstats(m,X[:,get_right_inds(indices,labels)])
        left    = suffstats(m,X[:,get_left_inds(indices,labels)])
        clusters[k] = SplitMergeCluster(c, right+left, right, left; llh_hist=c.llh_hist)
    end
    return clusters
end

function propose_merges(m::AbstractDPModel{T}, clusters::Dict{Int,<:SplitMergeCluster},
                        X::AbstractMatrix, labels::AbstractVector{Tuple{Int,Bool}}, maySplit::Dict{Int,Bool}) where T
    α    = m.α
    logα = log(α)
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
            logH = -logα+lgamma(m.α)-2*lgamma(0.5*m.α)-log(100)+
                    lgamma(s.n)-lgamma(s.n+m.α)+
                    lgamma(s1.n+0.5*m.α)-lgamma(s1.n) +
                    lgamma(s2.n+0.5*m.α)-lgamma(s2.n)+
                    lmllh(prior,p,s.n)-lmllh(prior,c1.post,s1.n)-
                    lmllh(prior,c2.post,s2.n)

            if logH>0 || logH > log(rand())
                merge_with[k2] = k1
                break;
            end
        end
    end
    return merge_with
end
function propose_splits!(m::AbstractDPModel, X::AbstractMatrix, labels::AbstractVector{Tuple{Int,Bool}},  clusters::Dict, maybe_split::Dict{Int,Bool})
    logα = log(m.α)
    will_split = Dict{Int,Bool}()
    for (k,c) in clusters
        if maybe_split[k]
            if c.nr == 0 || c.nl == 0
                c = reset_cluster!(m,X,labels,clusters,k)
            end
            logH = logα+lgamma(c.nr)+lgamma(c.nl)-lgamma(c.n)+c.llhs[2]+c.llhs[3]-c.llhs[1]
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
                        clusters::Dict,
                        key::Int) where {V<:Real,D}
    indices = get_cluster_inds(key,labels)
    for i in indices
        @inbounds labels[i] = (key,rand()>0.5)
    end
    sr = suffstats(model,view(X,:,get_right_inds(indices,labels)))
    sl = suffstats(model,view(X,:,get_left_inds(indices,labels)))
    clusters[key] = SplitMergeCluster(clusters[key], sr+sl, sr, sl)
end


function get_new_keys(will_split::Dict{Int,Bool})
    kmax  = maximum(keys(will_split))
    nkeys = Dict{Int,Int}()
    for (k,ws) in will_split
        nkeys[k] = ws ? (kmax+=1) : k
    end
    return nkeys
end

function materialize_splits!(model, X, labels, clusters, will_split)
    new_keys  = get_new_keys(will_split)
    old_keys  = collect(keys(clusters))
    for k in old_keys
        if will_split[k]
            newkey = new_keys[k]
            clusters[k], clusters[newkey] = split_cluster!(model, X, labels, clusters[k], k, newkey)
        else
            c = clusters[k]
            if c.n != 0
                clusters[k] = SplitMergeCluster(c.n, c.nr, c.nl,c.s,
                                                rand(c.post), rand(c.rightpost), rand(c.leftpost),
                                                c.post, c.rightpost, c.leftpost,
                                                c.llhs, c.llh_hist, c.prior)
            end
        end
    end
end

function materialize_merges!(model, labels, clusters, will_merge)
    for (k1,k2) in will_merge
        indices = get_cluster_inds(k1,k2,labels)
        for i in indices
            @inbounds labels[i] = (k1,labels[i][1]==k2) # assign k2 to left
        end
        sr = clusters[k1].s
        sl = clusters[k2].s
        clusters[k1] = SplitMergeCluster(model, sr+sl, sr, sl)
        delete!(clusters,k2)
    end
end

function gc_clusters!(clusters, maybe_split)
    for (k,c) in clusters
        if c.n == 0
            delete!(clusters,k)
        else
            if !haskey(maybe_split,k)
                maybe_split[k] = false
            end
        end
    end
end


function split_cluster!(model::AbstractDPModel{V,D}, X::AbstractMatrix,
                        labels::AbstractVector{Tuple{Int,Bool}}, cluster::SplitMergeCluster,
                        kold::Int, knew::Int) where {V,D}
    split_indices!(kold,knew,labels)
    return ntuple(2) do j
        k = j==1 ? kold : knew
        indices = get_cluster_inds(k, labels)
        sr = suffstats(model,view(X,:,get_right_inds(indices,labels)))
        sl = suffstats(model,view(X,:,get_left_inds(indices,labels)))
        SplitMergeCluster(model, sr+sl, sr, sl) #FIXME: we know sr+sl from $cluster$
    end
end

function split_indices!(kold::Int, knew::Int,labels::AbstractVector{Tuple{Int,Bool}})
    indices = get_cluster_inds(kold, labels)
    @inbounds for i in indices
        if labels[i][2]
            labels[i] = (knew,rand()>0.5)
        else
            labels[i] = (kold,rand()>0.5)
        end
    end
end

function label_x2(clusters::Dict,knew::Int)
    for (i,k) in enumerate(keys(clusters))
        i==knew && return k
    end
    return 0
end

###
#### Parallel
###

function splitmerge_parallel!(πs, sπs, X, range, labels, clusters)
    for i in range
        x = view(X,:,i)
        probs = RestrictedClusterProbs(πs,clusters,x)
        z  = label_x2(clusters,rand(GLOBAL_RNG,AliasTable(probs)))
        labels[i] = (z,SampleSubCluster(sπs[z],clusters[z],x))
    end
end

@inline splitmerge_parallel!(labels, clusters, πs, sπs) =
    splitmerge_parallel!(πs, sπs, Main.X, localindices(labels), labels,clusters)

function splitmerge_gibbs_parallel!(model, X::AbstractMatrix, labels::SharedArray, clusters, empty_cluster; merge=true, T=10, scene=nothing)
    for t in 1:T
        record!(scene,labels,t)
        πs          = mixture_πsv2(model.α,clusters)
        sπs         = subcluster_πs(model.α/2,clusters)
        maybe_split = maybeSplit(clusters)

        @sync begin
            for p in procs(labels)
                @async remotecall_wait(splitmerge_parallel!,p,labels,clusters,πs,sπs)
            end
        end

        update_clusters!(model, X, clusters, labels)
        will_split = propose_splits!(model, X, labels, clusters, maybe_split)
        materialize_splits!(model, X, labels, clusters, will_split)
        gc_clusters!(clusters, maybe_split)
        if merge
            will_merge  = propose_merges(model, clusters, X, labels, maybe_split)
            materialize_merges!(model, labels, clusters, will_merge)
        end
    end
end

# function update_cluster_parallel!(cluster::Pair{Int,<:SplitMergeCluster}, labels::AbstractVector{Tuple{Int,Bool}})
#     k,c = cluster
#     indices = get_cluster_inds(k,labels)
#     right   = suffstats(Main.model,Main.X[:,get_right_inds(indices,labels)])
#     left    = suffstats(Main.model,Main.X[:,get_left_inds(indices,labels)])
#     return Pair(k,SplitMergeCluster(c, right+left, right, left; llh_hist=c.llh_hist))
# end

# function update_clusters_parallel!(clusters::Dict, labels::AbstractVector{Tuple{Int,Bool}})
#     return Dict(pmap(c->update_cluster_parallel!(c,labels),collect(clusters)))
# end
