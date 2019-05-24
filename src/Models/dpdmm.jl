struct DPDMM{T<:Real,D} <: AbstractDPModel{T,D}
    θprior::DirichletFast{T}
    α::T
end

function DPDMM(X::AbstractMatrix{<:Integer}; α::Real=1, eltype=Float64)
    DPDMM{eltype}(eltype(α), sumcol(X) .+ Float64(1))
end

@inline DPDMM{T,D}(α::Real) where {T<:Real,D} = DPDMM{T,dim}(DirichletFast{T}(ones(D),T(α)))

@inline DPDMM{T}(α::Real, alphas::AbstractVector{T}) where T<:Real = DPDMM{T,length(alphas)}(DirichletFast{T}(alphas),T(α))

@inline stattype(::DPDMM{T}) where T = DPDMMStats{T}

struct DPDMMStats{T<:Real} <: SufficientStats
    s::Vector{Int}
    n::Int
end

@inline suffstats(m::DPDMM{T,D}) where {T<:Real,D} =
    DPDMMStats{T}(zeros(Int,D),0)

@inline suffstats(m::DPDMM{T},X::AbstractMatrix{Int}) where T<:Real =
    DPDMMStats{T}(sumcol(X),size(X,2))

@inline suffstats(m::DPDMM{T},x::AbstractVector{Int}) where T<:Real =
    DPDMMStats{T}(x,1)

@inline function updatestats(m::DPDMMStats{T},x::AbstractVector{Int}) where T<:Real
    DPDMMStats{T}(add!(m.s,x),m.n+1)
end

@inline function +(s1::DPDMMStats{T},s2::DPDMMStats{T}) where T<:Real
    DPDMMStats{T}(s1.s+s2.s,s1.n+s2.n)
end

@inline function -(s1::DPDMMStats{T},s2::DPDMMStats{T}) where T<:Real
    DPDMMStats{T}(s1.s-s2.s,s1.n-s2.n)
end

@inline function updatestats(m::DPDMMStats{T},X::AbstractMatrix{T}) where T<:Real
    DPDMMStats{T}(add!(m.s,sumcol(X)), m.n+size(X,2))
end

@inline function downdatestats(m::DPDMMStats{T}, x::AbstractVector{Int}) where T<:Real
    DPDMMStats{T}(substract!(m.s,x),m.n-1)
end

@inline function downdatestats(m::DPDMMStats{T},X::AbstractMatrix{Int}) where T<:Real
    DPDMMStats{T}(substract!(m.s,sumcol(X)), m.n-size(X,2))
end

@inline _posterior(m::DirichletFast{V},T::DPDMMStats{V}) where V<:Real = m.alpha + T.s
@inline _posterior(m::DPDMM,T::DPDMMStats) = _posterior(m.θprior,T)
@inline posterior(m::DPDMM, T::DPDMMStats)  =  posterior(m.θprior,T)
@inline posterior(m::DirichletFast{V}, T::DPDMMStats{V}) where V<:Real = T.n!=0 ? DirichletFast{V}(_posterior(m,T)) : m
#
# init(X::AbstractMatrix{<:Integer}, α::V, ninit::Int, T::Type{<:DPDMM}) where V<:Real =
#     size(X),rand(1:ninit,size(X,2)),T(α,sumcol(X) .+ V(1))
