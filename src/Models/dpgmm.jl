"""
   DPGMM{T<:Real,D} <: AbstractDPModel{T,D}

   Class for DP Gaussian Mixture Models
"""
struct DPGMM{T<:Real,D} <: AbstractDPModel{T,D}
    θprior::NormalWishart{T}
    α::T
end

@inline prior(m::DPGMM) = m.θprior

function DPGMM(X::AbstractMatrix{T}; α::Real=1) where T<:Real
    DPGMM{T}(T(α), vec(mean(X,dims=2)),Matrix{T}(I,size(X,1),size(X,1)))
end

@inline DPGMM{T,D}(α::Real) where {T<:Real,D} =
    DPGMM{T,dim}(NormalWishart{T}(D),T(α))

@inline DPGMM{T}(α::Real, μ0::AbstractVector{T}) where T<:Real =
    DPGMM{T,length(μ0)}(NormalWishart{T}(μ0),T(α))

@inline DPGMM{T}(α::Real, μ0::AbstractVector{T}, Σ0::AbstractMatrix{T}) where T<:Real =
    DPGMM{T,length(μ0)}(NormalWishart{T}(μ0,Σ0), T(α))

@inline stattype(::NormalWishart{T}) where T = DPGMMStats{T}


"""
    DPGMMStats{T<:Real} <: SufficientStats

    Sufficient statistics for Gaussian Models
"""
struct DPGMMStats{T<:Real} <: SufficientStats
    nμ::Vector{T}
    S::Matrix{T}
    n::Int
end

@inline function suffstats(m::NormalWishart{T}) where T
    D = length(m)
    DPGMMStats{T}(zeros(T,D),zeros(T,D,D),0)
end

@inline suffstats(m::NormalWishart{T},X::AbstractMatrix{T}) where T<:Real =
    DPGMMStats{T}(vec(sum(X,dims=2)),X*X',size(X,2))

@inline suffstats(m::NormalWishart{T},x::AbstractVector{T}) where T<:Real =
    DPGMMStats{T}(Vector(x),x*x',1)

@inline function +(s1::DPGMMStats{T},s2::DPGMMStats{T}) where T<:Real
    DPGMMStats{T}(s1.nμ .+ s2.nμ,s1.S .+ s2.S,s1.n+s2.n)
end

@inline function -(s1::DPGMMStats{T},s2::DPGMMStats{T}) where T<:Real
    DPGMMStats{T}(s1.nμ .- s2.nμ,s1.S .- s2.S,s1.n-s2.n)
end

@inline function updatestats(m::DPGMMStats{T},x::AbstractVector{T}) where T<:Real
    m.nμ .+= x
    m.S  .+= x*x'
    DPGMMStats{T}(m.nμ .+ x, m.S .+  x*x',m.n+1)
end

@inline function updatestats(m::DPGMMStats{T},X::AbstractMatrix{T}) where T<:Real
    m.nμ .+= vec(sum(X,dims=2))
    m.S  .+= X*X'
    DPGMMStats{T}(m.nμ,m.S,m.n+size(X,2))
end

@inline function downdatestats(m::DPGMMStats{T},x::AbstractVector{T}) where T<:Real
    m.nμ .-= x
    m.S  .-= x*x'
    DPGMMStats{T}(m.nμ,m.S,m.n-1)
end

@inline function downdatestats(m::DPGMMStats{T},X::AbstractMatrix{T}) where T<:Real
    m.nμ .-= vec(sum(X,dims=2))
    m.S  .-= X*X'
    DPGMMStats{T}(m.nμ,m.S,m.n-size(X,2))
end

function _posterior(m::NormalWishart{V},T::DPGMMStats{V}) where V<:Real
    λn   = m.λ + V(T.n)
    νn   = m.ν + V(T.n)
    μn   = (m.λ * m.μ .+ T.nμ)/λn
    Ψn   = m.Ψ + T.S .+ m.λ * (m.μ * m.μ') - λn * (μn * μn')
    return (μn,λn,Ψn,νn)
end

@inline _posterior(m::NormalWishart,T::DPGMMStats) = _posterior(m.θprior,T)

@inline function posterior_predictive(m::NormalWishart{T}) where T<:Real
    df = m.ν-length(m.μ)+1
    MvTDist(df, m.μ, ((m.λ+1)/(m.λ*df)) * m.Ψ)
end

@inline function posterior_predictive(m::NormalWishart{V},T::DPGMMStats{V}) where V
    if T.n!=0
        (μn,λn,Ψn,νn) = _posterior(m,T)
        df = νn-length(m)+1
        MvTDist(df, μn, ((λn+1)/(λn*df)) * Ψn)
    else
        posterior_predictive(m)
    end
end


@inline posterior(m::NormalWishart{V},T::DPGMMStats{V}) where V<:Real =
    T.n!=0 ? NormalWishart(_posterior(m,T)...) : m

@inline function downdate_predictive(p::NormalWishart, m::MvTDist, x::AbstractVector{V}, n::Int) where {V<:Real}
    λ   = p.λ+n
    dfn = m.df-1
    λn  = λ - 1
    μn  = (λ * m.μ - x)/λn
    MvTDist(dfn, μn, PDMat(((λn+1)/(λn*dfn)) * lowrankdowndate(((λ*m.df)/(λ+1))*m.Σ.chol,sqrt(λ/λn) * (x-m.μ))))
end

@inline function update_predictive(p::NormalWishart, m::MvTDist, x::AbstractVector{V}, n::Int) where {V<:Real}
    λ   = p.λ+n # We sart λ0=1,ν0=D+3.So df=ν-D+1 => λ=df
    dfn = m.df+1
    λn  = λ + 1
    μn  = (λ * m.μ + x)/λn
    MvTDist(dfn, μn, PDMat(((λn+1)/(λn*dfn)) * lowrankupdate(((λ*m.df)/(λ+1))*m.Σ.chol, sqrt(λ/λn) * (x-m.μ))))
end
#
# init(X::AbstractMatrix{V}, α::Real, ninit::Int, T::Type{<:DPGMM}) where V<:Real =
#     size(X),rand(1:ninit,size(X,2)),T(V(α), vec(mean(X,dims=2)),(X*X')/size(X,2))
