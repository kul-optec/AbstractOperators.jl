export SoftMax

"""
	SoftMax([domain_type=Float64::Type,] dim_in::Tuple)

Creates the softmax non-linear operator with input dimensions `dim_in`.
```math
\\sigma(\\mathbf{x}) = \\frac{e^{\\mathbf{x} }}{ \\sum e^{\\mathbf{x} } }
```

"""
struct SoftMax{T, N, B <: AbstractArray{T, N}} <: NonLinearOperator
    dim::NTuple{N, Int}
    buf::B
end

function SoftMax(x::AbstractArray{T, N}) where {T, N}
    buf = similar(x)
    return SoftMax{T, N, typeof(buf)}(size(x), buf)
end

function SoftMax(domain_type::Type, DomainDim::NTuple{N, Int}) where {N}
    buf = zeros(domain_type, DomainDim)
    return SoftMax{domain_type, N, typeof(buf)}(DomainDim, buf)
end

SoftMax(DomainDim::NTuple{N, Int}) where {N} = SoftMax(Float64, DomainDim)

function mul!(y::AbstractArray, L::SoftMax, x::AbstractArray)
    check(y, L, x)
    y .= exp.(x .- maximum(x))
    return y ./= sum(y)
end

function mul!(y::AbstractArray, J::AdjointOperator{<:Jacobian{<:SoftMax}}, b::AbstractArray)
    check(y, J, b)
    L = J.A
    fill!(y, zero(eltype(y)))
    L.A.buf .= exp.(L.x .- maximum(L.x))
    L.A.buf ./= sum(L.A.buf)
    for i in eachindex(y)
        y[i] = -L.A.buf[i] * dot(L.A.buf, b)
        y[i] += L.A.buf[i] * b[i]
    end
    return y
end

fun_name(L::SoftMax) = "σ"

size(L::SoftMax) = (L.dim, L.dim)

domain_type(::SoftMax{T}) where {T} = T
codomain_type(::SoftMax{T}) where {T} = T
is_thread_safe(::SoftMax) = true
