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
@inline mean(d::MvNormalFast) = d.μ
@inline params(d::MvNormalFast) = (d.μ, d.J)
@inline partype(d::MvNormalFast{T}) where {T<:Real} = T

var(d::MvNormalFast) = diag(inv(d.J))
cov(d::MvNormalFast) = Matrix(inv(d.J))

invcov(d::MvNormalFast) = Matrix(d.J)
logdetcov(d::MvNormalFast) = -logdet(d.J)
@inline mvnormal_c0(d::MvNormalFast) = d.c0

sqmahal(d::MvNormalFast, x::AbstractVector) = quad(d.J, broadcast(-, x, d.μ))
sqmahal!(r::AbstractVector, d::MvNormalFast, x::AbstractMatrix) = quad!(r, d.J, broadcast(-, x, d.μ))

####
##### Helper Functions
####

mvnormal_c0(μ::AbstractVector, J::AbstractPDMat) = -(length(μ) * Float64(log2π) - logdet(J))/2

## Sample normal with mean and precision matrix

@inline function randNormal(μ::AbstractVector{T}, J::AbstractPDMat{T}) where T
    randNormal!(μ, J, similar(μ))
end

function randNormal!(μ::AbstractVector{T}, J::AbstractPDMat{T}, x::AbstractVector{T}) where T
    for i in eachindex(x)
        @inbounds x[i] = randn()
    end
    add!(unwhiten_winv!(J, x), μ)
end
