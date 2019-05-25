struct MvNormalFast{T<:Real,Prec<:AbstractPDMat,Mean<:AbstractVector} <: AbstractMvNormal
    μ::Mean
    J::Prec
    c0::T
end

MvNormalFast(μ::AbstractVector{<:Real}, J::Matrix{<:Real}) = MvNormalFast(μ, PDMat(J))
MvNormalFast(μ::AbstractVector{<:Real}, J::Union{Symmetric{<:Real}, Hermitian{<:Real}}) = MvNormalFast(μ, PDMat(J))
MvNormalFast(μ::AbstractVector{<:Real}, J::Diagonal{<:Real}) = MvNormalFast(μ, PDiagMat(diag(J)))

function MvNormalFast(μ::AbstractVector, J::AbstractPDMat)
    R = Base.promote_eltype(μ, J)
    MvNormalFast(convert(AbstractArray{R}, μ), convert(AbstractArray{R}, J))
end

function MvNormalFast(μ::AbstractVector{T}, J::AbstractPDMat{T}) where T<: Real
    MvNormalFast(μ,J,mvnormal_c0(μ,J))
end

function MvNormalFast(J::Cov) where {T, Cov<:AbstractPDMat{T}}
    MvNormalFast(ZeroVector(T, dim(J)), J)
end

MvNormalFast(J::Matrix{<:Real}) = MvNormalFast(PDMat(J))

@inline length(d::MvNormalFast) = length(d.μ)
@inline mean(d::MvNormalFast)   = d.μ
@inline params(d::MvNormalFast) = (d.μ, d.J)
@inline partype(d::MvNormalFast{T}) where {T<:Real} = T

var(d::MvNormalFast) = diag(inv(d.J))
cov(d::MvNormalFast) = Matrix(inv(d.J))

invcov(d::MvNormalFast) = Matrix(d.J)
logdetcov(d::MvNormalFast) = -logdet(d.J)
mvnormal_c0(d::MvNormalFast) = d.c0

sqmahal(d::MvNormalFast, x::AbstractVector) = quad(d.J, broadcast(-, x, d.μ))
sqmahal!(r::AbstractVector, d::MvNormalFast, x::AbstractMatrix) = quad!(r, d.J, broadcast(-, x, d.μ))

@inline logprob(d::GenericMvTDist{T}, x::AbstractVector{T}) where T = _logpdf(d,x)

function logprob(d::MvNormalFast{T}, x::AbstractVector{T}) where T
    D = length(x)
    μ = d.μ
    J = d.J.mat
    s = zero(T)

    y = Vector{T}(undef,D)
    @fastmath @simd for i=1:D
        @inbounds y[i] = x[i]-μ[i]
    end

    @fastmath @simd for i=1:D
        for j=1:D
            @inbounds s += y[i] * y[j] * J[i, j]
        end
    end
    return d.c0 - s/T(2)
end
####
##### Helper Functions
####

mvnormal_c0(μ::AbstractVector, J::AbstractPDMat) =
    -(length(μ) * Float64(log2π) - logdet(J))/2

## Sample normal with mean and precision matrix

randNormal(μ::AbstractVector{T}, J::AbstractPDMat{T}) where T =
    randNormal!(μ, J, similar(μ))


function randNormal!(μ::AbstractVector{T}, J::AbstractPDMat{T}, x::AbstractVector{T}) where T
    for i in eachindex(x)
        @inbounds x[i] = randn()
    end
    return add!(unwhiten_winv!(J, x), μ)
end
