struct DirMul{T<:Real} <:  ContinuousMultivariateDistribution
    alpha::Vector{T}
end

DirMul(d::Integer, alpha::T) where {T<:Real} = DirMul{T}(d, alpha)
DirMul(alpha::Vector{T}) where {T<:Integer} = DirMul{Float64}(convert(Vector{Float64},alpha))
DirMul(d::Integer, alpha::Integer) = DirMul{Float64}(d, Float64(alpha))

convert(::Type{DirMul{T}}, alpha::Vector{S}) where {T<:Real, S<:Real} =
    DirMul(convert(Vector{T}, alpha))
convert(::Type{DirMul{T}}, d::DirMul{S}) where {T<:Real, S<:Real} =
    DirMul(convert(Vector{T}, d.alpha))

length(d::DirMul) = length(d.alpha)
params(d::DirMul) = (d.alpha,)
@inline partype(d::DirMul{T}) where {T<:Real} = T

function _rand!(d::DirMul{T}, x::AbstractVector{<:Real}) where T
    s = T(0.0)
    n = length(x)
    α = d.alpha
    for i in 1:n
        @inbounds s += (x[i] = rand(Gamma(α[i])))
    end
    Multinomial(1, multiply!(x, inv(s)))
end

@inline rand(d::DirichletCanon) = _rand!(d,similar(d.alpha))
