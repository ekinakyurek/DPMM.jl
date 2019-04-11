import Distributions.length

abstract type AbstractDPModel{T,D} end
@inline length(::AbstractDPModel{<:Any,D}) where D = D

init(X::AbstractMatrix{V}, α::Real, ninit::Int, T::Type{<:DPGMM}) where V<:Real =
    size(X),rand(1:ninit,size(X,2)),T(V(α), vec(mean(X,dims=2)), (X*X')/size(X,2))

init(X::AbstractMatrix{<:Integer}, α::V, ninit::Int, T::Type{<:DPDMM}) where V<:Real =
    size(X),rand(1:ninit,size(X,2)),T(α, V.(sumcol(X)) .+ 1)
