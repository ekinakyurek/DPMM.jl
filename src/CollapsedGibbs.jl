function collapsed_gibbs(X::AbstractMatrix; T::Int=1000, α::Real=1.0, ninit::Int=6, observables=nothing)
    #Initialization
    (D,N),labels,model = init(X,α,ninit)
    #Get Clusters
    clusters = Clusters(model,X,labels) # current clusters
    cluster0 = CollapsedCluster(model)  # empty cluster
    #clusters[0] = CollapsedCluster(model,Val(true))
    for t in 1:T
        record!(observables,labels,t)
        @inbounds for i=1:N
            x, z = X[:,i], labels[i]
            clusters[z] -= x # remove xi's statistics
            isempty(clusters[z]) && delete!(clusters,z)
            probs     = CRPprobs(model,clusters,cluster0,x) # chinese restraunt process probabilities
            znew      =~ Categorical(probs,NoArgCheck()) # new label
            labels[i] = place_x!(model,clusters,znew,x)
        end
    end
    return labels
end

init(X::AbstractMatrix{V}, α::Real, ninit::Int) where V<:Real =
    size(X),rand(1:ninit,size(X,2)),DPGMM{V}(V(α), vec(mean(X,dims=2)))

Clusters(model::DPGMM, X::AbstractMatrix, z::Array{Int}) =
    Dict((k,CollapsedCluster(model,X[:,findall(l->l==k,z)])) for k in unique(z))

function CRPprobs(model::DPGMM{V}, clusters::Dict, cluster0::CollapsedCluster, x::AbstractVector) where V<:Real
    probs = Array{V,1}(undef,length(clusters)+1)
    for (j,c) in enumerate(values(clusters))
        @inbounds probs[j] = c.n*pdf(c,x)
    end
    #probs = map(c->c(x)::V,values(clusters))
    probs[end] = model.α*pdf(cluster0,x)
    return probs/sum(probs)
end

function place_x!(model::DPGMM,clusters::Dict,knew::Int,xi::AbstractVector)
    cks = collect(keys(clusters))
    if knew > length(clusters)
        ck = maximum(cks)+1
        clusters[ck] = CollapsedCluster(model,xi)
    else
        ck = cks[knew]
        clusters[ck] += xi
    end
    return ck
end

record!(observables::Nothing,z,T) = nothing
function record!(observables,z,T)
    K=sort(unique(z))
    colors = map(zi->(findfirst(x->x==zi,K)-1)%12+1,z)
    observables[1][] = colorpalette[colors]
    observables[2][] = ("T=$T","")
    sleep(0.001)
end

function main()
    #using DPMM,Makie
    gmodel = RandMixture(6)
    X,labels = rand_w_label(gmodel,1000)
    scene  = scatter(X[1,:],X[2,:],color=DPMM.colorpalette[0labels.+1],markersize=1.0)
    colors = scene[end][:color]
    axisnames = scene[Axis][:names][:axisnames]
    collapsed_gibbs(X,T=1000,observables=(colors,axisnames))
end



# @inline crp_probs4(model::DPGMM{V},clusters::Dict,x::AbstractVector) where V<:Real =
#     [c(x) for (k,c) in clusters]
#
# function place_x4!(model::DPGMM,clusters::Dict,knew::Int,xi::AbstractVector)
#     cks = collect(keys(clusters))
#     if knew == 1
#         ck = maximum(cks)+1
#         clusters[ck] = CollapsedCluster(model,xi)
#     else
#         ck = cks[knew]
#         clusters[ck] += xi
#     end
#     return ck
# end
