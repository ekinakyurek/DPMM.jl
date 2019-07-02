import SparseArrays: nnz, nonzeroinds, nonzeros
"""
    DPSparseVector{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}

`DPSparseVector` is almost same with `SparseArrays.SparseVector`

The only difference is summation between `DPSparseVector`s results with a `Vector`.
"""
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

function Vector(s::DPSparseVector{Tv,<:Any}) where Tv
    nzval = nonzeros(s)
    nzind = nonzeroinds(s)
    x = zeros(Tv,nnz(s))
    for i in eachindex(nzval)
        @inbounds x[nzind[i]] =nzval[i]
    end
    return x
end

DPSparseVector(n::Integer, nzind::Vector{Ti}, nzval::Vector{Tv}) where {Tv,Ti} =
    DPSparseVector{Tv,Ti}(n, nzind, nzval)

DPSparseVector(x::SparseVector{Tv,Ti}) where {Tv,Ti} =
    DPSparseVector{Tv,Ti}(x.n, x.nzind, x.nzval)

Base.size(m::DPSparseVector{Int64,Int64}) = (m.n,)
Base.length(m::DPSparseVector{Int64,Int64}) = m.n
struct DPSparseMatrix{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int                               # Number of rows
    n::Int                               # Number of columns
    data::Vector{DPSparseVector{Tv,Ti}}       # Stored columns as sparse vector
end

"""
    DPSparseMatrix(X::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}

DPSparseMatrix has fast getindex methods for column indexing (i.e X[:,i])
It also doesn't copy column and return `DPSparseVector` for a column indexing.

see [`DPSparseVector`](@ref)
"""
function DPSparseMatrix(X::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    DPSparseMatrix{Tv,Ti}(X.m,X.n,map(i->DPSparseVector(X[:,i]),1:size(X,2)))
end

@inline size(X::DPSparseMatrix) = (X.m,X.n)
@inline length(x::DPSparseVector) = x.n
@inline getindex(X::DPSparseMatrix, ::Colon, inds::Vector{<:Integer}) = DPSparseMatrix(X.m,length(inds),X.data[inds])
@inline getindex(X::DPSparseMatrix, ::Colon, inds::AbstractRange) = DPSparseMatrix(X.m,length(inds),X.data[inds])
@inline getindex(X::DPSparseMatrix, ::Colon, ind::Integer) = X.data[ind]
@inline Base.view(X::DPSparseMatrix, ::Colon, ind::Integer) = X[:,ind]
@inline Base.view(X::DPSparseMatrix, ::Colon, inds::Vector{<:Integer}) = X[:,inds]
@inline Base.view(X::DPSparseMatrix, ::Colon, inds::AbstractRange) = X[:,inds]
@inline getindex(X::DPSparseMatrix, ind1::Integer, ind2::Integer) = X.data[ind2][ind1]
@inline nonzeroinds(x::DPSparseVector) = x.nzind
@inline nonzeros(x::DPSparseVector)    = x.nzval
@inline nnz(x::DPSparseVector)    = x.n

function getindex(x::DPSparseVector{Tv,<:Any}, ind::Integer) where Tv
    inds = x.nzind
    i = findfirst(x->x==ind,inds)
    if i===nothing
        return Tv(0)
    else
        return x.nzval[i]
    end
end


function Base.setindex!(x::DPSparseVector{Tv,Ti}, v::Tv, i::Ti) where {Tv,Ti<:Integer}
    checkbounds(x, i)
    nzind = nonzeroinds(x)
    nzval = nonzeros(x)

    m = length(nzind)
    k = searchsortedfirst(nzind, i)
    if 1 <= k <= m && nzind[k] == i  # i found
        nzval[k] = v
    else  # i not found
        if v != 0
            insert!(nzind, k, i)
            insert!(nzval, k, v)
        end
    end
    x
end

Base.setindex!(x::DPSparseVector{Tv,Ti}, v, i::Integer) where {Tv,Ti<:Integer} =
    setindex!(x, convert(Tv, v), convert(Ti, i))

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
