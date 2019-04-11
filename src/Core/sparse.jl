using SparseArrays, Distributions
import Base: size, +, -, *, getindex, sum, length
import SparseArrays: AbstractSparseMatrix, AbstractSparseVector, nonzeroinds, nonzeros
import Distributions: _logpdf, lgamma, xlogy

struct DPSparseVector{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    n::Int              # Length of the sparse vector
    nzind::Vector{Ti}   # Indices of stored values
    nzval::Vector{Tv}   # Stored values, typically nonzeros

    function DPSparseVector{Tv,Ti}(n::Integer, nzind::Vector{Ti}, nzval::Vector{Tv}) where {Tv,Ti<:Integer}
        n >= 0 || throw(ArgumentError("The number of elements must be non-negative."))
        length(nzind) == length(nzval) ||
            throw(ArgumentError("index and value vectors must be the same length"))
        new(convert(Int, n), nzind, nzval)
    end
end

DPSparseVector(n::Integer, nzind::Vector{Ti}, nzval::Vector{Tv}) where {Tv,Ti} =
    DPSparseVector{Tv,Ti}(n, nzind, nzval)

DPSparseVector(x::SparseVector{Tv,Ti}) where {Tv,Ti} =
    DPSparseVector{Tv,Ti}(x.n, x.nzind, x.nzval)

struct DPSparseMatrix{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int                               # Number of rows
    n::Int                               # Number of columns
    data::Vector{DPSparseVector{Tv,Ti}}       # Stored columns as sparse vector
end

function DPSparseMatrix(X::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    DPSparseMatrix{Tv,Ti}(X.m,X.n,map(i->DPSparseVector(X[:,i]),1:size(X,2)))
end

@inline size(X::DPSparseMatrix) = (X.m,X.n)
@inline length(x::DPSparseVector) = x.n
@inline getindex(X::DPSparseMatrix, ::Colon, inds::Vector{<:Integer}) = DPSparseMatrix(X.m,length(inds),X.data[inds])
@inline getindex(X::DPSparseMatrix, ::Colon, ind::Integer) = X.data[ind]
@inline getindex(X::DPSparseMatrix, ind1::Integer, ind2::Integer) = X.data[ind2][ind1]
@inline nonzeroinds(x::DPSparseVector) = x.nzind
@inline nonzeros(x::DPSparseVector)    = x.nzval

function getindex(x::DPSparseVector{Tv,<:Any}, ind::Integer) where Tv
    inds = x.nzind
    i = findfirst(x->x==ind,inds)
    if i===nothing
        return Tv(0)
    else
        return x.nzval[i]
    end
end

function sumcol(X::DPSparseMatrix{Tv,<:Any}) where Tv
    y = zeros(Tv,X.m)
    for i=1:X.n
        @inbounds add!(y,X[:,i])
    end
    return y
end

@inline sumcol(X) = vec(sum(X,dims=2))
@inline sum(x::DPSparseVector) = sum(x.nzval)
@inline +(x::DPSparseVector, y::DPSparseVector) = add!(Vector(x),y)
@inline -(x::DPSparseVector, y::DPSparseVector) = substract!(Vector(x),y)

function add!(x::AbstractVector, y::DPSparseVector)
    for (i,index) in enumerate(y.nzind)
        @inbounds x[index] += y.nzval[i]
    end
    return x
end

function substract!(x::AbstractVector, y::DPSparseVector)
    for (i,index) in enumerate(y.nzind)
      @inbounds x[index] -= y.nzval[i]
    end
    return x
end

@inline add!(x,y) = x+y
@inline substract!(x,y) = x-y

function _logpdf(d::Multinomial, x::DPSparseVector{Tv,<:Any}) where Tv<:Real
    p = probs(d)
    n = sum(x)
    S = eltype(p)
    R = promote_type(Tv, S)
    s = R(lgamma(n + 1))
    for (i,index) in enumerate(x.nzind)
        @inbounds xi = x.nzval[i]
        @inbounds p_i = p[index]
        s -= R(lgamma(R(xi) + 1))
        s += xlogy(xi, p_i)
    end
    return s
end

# FIXME: This is not good!
function _logpdf(d::Multinomial, x::AbstractVector{T}) where T<:Real
    p = probs(d)
    n = sum(x)
    S = eltype(p)
    R = promote_type(T, S)
    s = R(lgamma(n + 1))
    for i = 1:length(p)
        @inbounds xi = x[i]
        @inbounds p_i = p[i]
        s -= R(lgamma(R(xi) + 1))
        s += xlogy(xi, p_i)
    end
    return s
end
