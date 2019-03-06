using LinearAlgebra, PDMats
#import LinearAlgebra: lowrankupdate!, lowrankupdate, lowrankdowndate, lowrankdowndate!
import Base: *

# @inline lowrankupdate(A::AbstractPDMat{T},x::AbstractVector{T}) where T<:Real =
#     PDMat(lowrankupdate(A.chol,x))
#
# @inline lowrankdowndate(A::AbstractPDMat{T},x::AbstractVector{T}) where T<:Real =
#     PDMat(lowrankdowndate(A.chol,x))

*(c::Cholesky,a::Real) = Cholesky(sqrt(a)*getfield(c,:factors),getfield(c,:uplo),getfield(c,:info))
*(a::Real,c::Cholesky) =  c * a

# @inline function lowrankupdate!(A::AbstractPDMat{T},x::AbstractVector{T}) where T<:Real
#     lowrankupdate!(A.chol,x)
#     A.mat .= Matrix(A.chol)
# end
#
# @inline function lowrankdowndate!(A::AbstractPDMat{T},x::AbstractVector{T}) where T<:Real
#     lowrankdowndate(A.chol,x)
#     A.mat .= Matrix(A.chol)
# end
