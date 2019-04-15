import Distributions.length

abstract type AbstractDPModel{T,D} end
@inline length(::AbstractDPModel{<:Any,D}) where D = D

