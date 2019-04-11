import Distributions:  _rand!

function createΣ(T,d...;α=0.1)
     σ = α*randn(T,d...)
     return σ*σ'
end

function RandMixture(K::Integer;D::Int=2,πs::Vector{T}=ones(K)/K) where T<:Real
    comps = [MvNormal(2K*(rand(T,D) .+ 2i),createΣ(T,D,D)) for i=1:K]
    return MixtureModel(comps,πs)
end

function GridMixture(L::Integer; πs::Vector{T}=ones(L*L)/(L*L)) where T<:Real
    r = L÷2
    if isodd(L)
        range = collect(-r:r)
    else
        range = union(-r:-1,1:r)
    end
    comps = [MvNormal([T(i),T(j)],createΣ(T,2,2)) for i in range, j in range]
    return MixtureModel(vec(comps),πs)
end

@inline function _rand_with_label!(d::MixtureModel{<:Any,<:Any,<:Any,T},v::AbstractVector{T}) where T<:Real
    c  = rand(d.prior)
    _rand!(component(d,c),v)
    return c
end

@inline _rand_with_label!(d::MixtureModel{<:Any,<:Any,<:Any,T},v::AbstractMatrix{T}) where T<:Real =
     [_rand_with_label!(d,col) for col in eachcol(v)]

@inline function rand_with_label(d::MixtureModel{<:Any,<:Any,<:Any,T}) where T<:Real
    v = Vector{T}(undef,length(d))
    c = _rand_with_label!(d,v)
    return v,c
end

@inline function rand_with_label(d::MixtureModel{<:Any,<:Any,<:Any,T},n::Integer) where T<:Real
    w = Matrix{T}(undef,length(d),n)
    c = _rand_with_label!(d,w)
    return w,c
end