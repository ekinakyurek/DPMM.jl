"""
   AbstractDPModel{T,D}

   Abstract base class for DPMMs

   `T` stands for element type, `D` is for dimensionality of the data
"""
abstract type AbstractDPModel{T,D} end
@inline length(::AbstractDPModel{<:Any,D}) where D = D
