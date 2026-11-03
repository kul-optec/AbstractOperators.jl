module AcceleratedDCTsExt

using AbstractFFTs
using AcceleratedDCTs
using FFTWOperators
using GPUArrays

import Base: eltype, inv, ndims, size
import LinearAlgebra: mul!
import FFTWOperators: DCT, IDCT
import AbstractOperators: AdjointOperator, check

struct OrthoDCTPlan{T, N, P, S, B} <: AbstractFFTs.Plan{T}
    raw::P
    scale::S
    temp::B
    pinv::Base.RefValue{Any}
end

struct OrthoIDCTPlan{T, N, P, S, B} <: AbstractFFTs.Plan{T}
    raw::P
    scale::S
    temp::B
    pinv::Base.RefValue{Any}
end

# `num_threads = 1`: a GPU DCT plan has no CPU thread count, and the device kernel is
# already parallel, so `is_threaded` correctly reports false for GPU-backed transforms.
# `kwargs...` swallows the CPU-only `num_threads`/`threaded` keywords.
function FFTWOperators.DCT(x::AbstractGPUArray{T, N}; kwargs...) where {T <: Real, N}
    A, At = _ortho_plan_pair(x)
    buf = similar(x, 0)  # zero-length marker for storage type inference
    return DCT{N, T, typeof(A), typeof(At), typeof(buf)}(size(x), A, At, buf, 1)
end

function FFTWOperators.IDCT(x::AbstractGPUArray{T, N}; kwargs...) where {T <: Real, N}
    At, A = _ortho_plan_pair(x)
    buf = similar(x, 0)  # zero-length marker for storage type inference
    return IDCT{N, T, typeof(A), typeof(At), typeof(buf)}(size(x), A, At, buf, 1)
end

Base.size(p::OrthoDCTPlan) = size(p.scale)
Base.size(p::OrthoIDCTPlan) = size(p.scale)
Base.ndims(::OrthoDCTPlan{T, N}) where {T, N} = N
Base.ndims(::OrthoIDCTPlan{T, N}) where {T, N} = N
Base.eltype(::OrthoDCTPlan{T}) where {T} = T
Base.eltype(::OrthoIDCTPlan{T}) where {T} = T

Base.inv(p::OrthoDCTPlan) = p.pinv[]
Base.inv(p::OrthoIDCTPlan) = p.pinv[]

function mul!(y::AbstractGPUArray{T}, p::OrthoDCTPlan{T}, x::AbstractGPUArray{T}) where {T}
    mul!(y, p.raw, x)
    y ./= p.scale
    return y
end

function mul!(y::AbstractGPUArray{T}, p::OrthoIDCTPlan{T}, x::AbstractGPUArray{T}) where {T}
    p.temp .= x .* p.scale
    mul!(y, p.raw, p.temp)
    return y
end

# AcceleratedDCTs plans do not modify input, so no scratch copy is needed.
# These overrides bypass the copyto!(buf, b) in the CPU mul! methods.
function mul!(y::AbstractGPUArray, A::IDCT, b::AbstractGPUArray)
    check(y, A, b)
    return mul!(y, A.A, b)
end

function mul!(y::AbstractGPUArray, A::AdjointOperator{<:DCT}, b::AbstractGPUArray)
    check(y, A, b)
    return mul!(y, A.A.At, b)
end

function _backend_vector(x::AbstractGPUArray, values::Vector{T}) where {T}
    ArrayT = Base.typename(typeof(x)).wrapper
    return ArrayT(values)
end

function _ortho_scale(x::AbstractGPUArray{T, N}) where {T <: Real, N}
    scale = similar(x)
    fill!(scale, one(T))
    for d in 1:N
        axis = fill(sqrt(T(size(x, d)) / T(2)), size(x, d))
        axis[1] = sqrt(T(size(x, d)))
        axis_gpu = _backend_vector(x, axis)
        shape = ntuple(i -> i == d ? size(x, d) : 1, N)
        scale .*= reshape(axis_gpu, shape)
    end
    return scale
end

function _ortho_plan_pair(x::AbstractGPUArray{T, N}) where {T <: Real, N}
    scale = _ortho_scale(x)
    dct_raw = AcceleratedDCTs.plan_dct(x)
    idct_raw = AcceleratedDCTs.plan_idct(x)
    fwd = OrthoDCTPlan{T, N, typeof(dct_raw), typeof(scale), typeof(similar(x))}(
        dct_raw, scale, similar(x), Ref{Any}(nothing),
    )
    inv_plan = OrthoIDCTPlan{T, N, typeof(idct_raw), typeof(scale), typeof(similar(x))}(
        idct_raw, scale, similar(x), Ref{Any}(fwd),
    )
    fwd.pinv[] = inv_plan
    return fwd, inv_plan
end

end # module AcceleratedDCTsExt
