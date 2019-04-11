import Distributions: _rand!, partype, AbstractRNG, multiply!, DirichletCanon
import Base: length, convert

struct DirMul{T<:Real} <:  ContinuousMultivariateDistribution
    alpha::Vector{T}
    # alpha0::T
    # lmnB::T

    # function DirMul{T}(alpha::Vector{T}) where T
    #     alpha0::T = zero(T)
    #     lmnB::T = zero(T)
    #     for i in 1:length(alpha)
    #         ai = alpha[i]
    #         ai > 0 ||
    #             throw(ArgumentError("DirMul: alpha must be a positive vector."))
    #         alpha0 += ai
    #         lmnB += lgamma(ai)
    #     end
    #     lmnB -= lgamma(alpha0)
    #     new{T}(alpha, alpha0, lmnB)
    # end

    # function DirMul{T}(d::Integer, alpha::T) where T
    #     alpha0 = alpha * d
    #     new{T}(fill(alpha, d), alpha0, lgamma(alpha) * d - lgamma(alpha0))
    # end
end


#DirMul(alpha::Vector{T}) where {T<:Real} = DirMul{T}(alpha)
DirMul(d::Integer, alpha::T) where {T<:Real} = DirMul{T}(d, alpha)
DirMul(alpha::Vector{T}) where {T<:Integer} =
    DirMul{Float64}(convert(Vector{Float64},alpha))
DirMul(d::Integer, alpha::Integer) = DirMul{Float64}(d, Float64(alpha))
convert(::Type{DirMul{T}}, alpha::Vector{S}) where {T<:Real, S<:Real} =
    DirMul(convert(Vector{T}, alpha))
convert(::Type{DirMul{T}}, d::DirMul{S}) where {T<:Real, S<:Real} =
    DirMul(convert(Vector{T}, d.alpha))


Base.show(io::IO, d::DirMul) = show(io, d, (:alpha,))

# Properties

length(d::DirMul) = length(d.alpha)
params(d::DirMul) = (d.alpha,)
@inline partype(d::DirMul{T}) where {T<:Real} = T



# function _logpdf(d::DirMul, x::AbstractVector{T}) where T<:Real
#     a = d.alpha
#     s = 0.
#     for i in 1:length(a)
#         @inbounds s += (a[i] - 1.0) * log(x[i])
#     end
#     return s - d.lmnB
# end

# sampling
function rand(d::DirichletCanon)
    _rand!(d,similar(d.alpha))
end

function _rand!(d::DirMul,
                x::AbstractVector{<:Real})
    s = 0.0
    n = length(x)
    α = d.alpha
    for i in 1:n
        @inbounds s += (x[i] = rand(Gamma(α[i])))
    end
    Multinomial(1, multiply!(x, inv(s)))
end
