import Distributions:  _rand!

function RandMixture(K::Integer;D::Int=2,πs::Vector{T}=ones(K)/K) where T<:Real
    comps = map(1:K) do i
                μ = 2K*(rand(T,D) .+ 2i)
                σ = rand(1:0.5:3)*randn(T,D,D)
                Σ = σ*σ'
                MvNormal(μ,Σ)
            end
    return MixtureModel(comps,πs)
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
