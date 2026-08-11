module GpuExt

using GPUArrays
using DSPOperators
import LinearAlgebra: mul!
import DSPOperators: Filt, MIMOFilt, AbstractFilt, AbstractMIMOFilt
import AbstractFFTs: plan_rfft, plan_irfft
import AbstractOperators: AdjointOperator, check

_wrapper_type(x::AbstractArray) = Base.typename(typeof(x)).wrapper

struct GpuFilt{T, N, S <: AbstractArray{T}, B <: AbstractGPUArray{T}, C <: AbstractGPUArray, P1, P2} <:
    AbstractFilt{T, N, S}
    dim_in::NTuple{N, Int}
    b::Vector{T}
    a::Vector{T}
    si::Vector{T}
    h_fft::C
    h_fft_conj::C
    buf::B
    buf_fft::C
    buf_out::B
    plan_fwd::P1
    plan_inv::P2
end

struct GpuMIMOFilt{T, S <: AbstractArray{T}, F <: GpuFilt} <: AbstractMIMOFilt{T, S}
    dim_out::Tuple{Int, Int}
    dim_in::Tuple{Int, Int}
    filters::Vector{F}
end

function GpuMIMOFilt(dim_out, dim_in, filters::Vector{F}) where {F <: GpuFilt{T, N, S}} where {T, N, S}
    return GpuMIMOFilt{T, S, F}(dim_out, dim_in, filters)
end

function _gpu_complex_buffer(ref::AbstractGPUArray{T}, fftlen::Int) where {T}
    return similar(ref, Complex{T}, fftlen ÷ 2 + 1)
end

function _make_gpu_filt(ref::AbstractGPUArray{T}, cpu_op::AbstractFilt{T, N}) where {T <: Real, N}
    fftlen = nextpow(2, cpu_op.dim_in[1] + length(cpu_op.b) - 1)
    storage_type = _wrapper_type(ref){T}
    buf = similar(ref, T, fftlen)
    buf_fft = _gpu_complex_buffer(ref, fftlen)
    buf_out = similar(ref, T, fftlen)
    plan_fwd = plan_rfft(buf)
    plan_inv = plan_irfft(buf_fft, fftlen)

    h_pad_cpu = zeros(T, fftlen)
    h_pad_cpu[1:length(cpu_op.b)] .= cpu_op.b
    h_pad = similar(ref, T, fftlen)
    copyto!(h_pad, h_pad_cpu)

    h_fft = similar(buf_fft)
    mul!(h_fft, plan_fwd, h_pad)
    h_fft_conj = similar(h_fft)
    h_fft_conj .= conj.(h_fft)

    return GpuFilt{T, N, storage_type, typeof(buf), typeof(buf_fft), typeof(plan_fwd), typeof(plan_inv)}(
        cpu_op.dim_in,
        collect(cpu_op.b),
        collect(cpu_op.a),
        collect(cpu_op.si),
        h_fft,
        h_fft_conj,
        buf,
        buf_fft,
        buf_out,
        plan_fwd,
        plan_inv,
    )
end

function Filt(x::AbstractGPUArray{T}, b::AbstractVector{T}, a::AbstractVector{T}) where {T <: Real}
    cpu_op = Filt(T, size(x), copy(b), copy(a))
    return _make_gpu_filt(x, cpu_op)
end

Filt(x::AbstractGPUArray{T}, b::AbstractVector{T}) where {T <: Real} = Filt(x, b, T[one(T)])

function _load_signal!(buf::AbstractGPUArray{T}, x::AbstractGPUArray{T}) where {T}
    fill!(buf, zero(T))
    copyto!(view(buf, 1:length(x)), x)
    return buf
end

function _copy_head!(y::AbstractGPUArray, buf_out::AbstractGPUArray, n::Int)
    copyto!(y, view(buf_out, 1:n))
    return y
end

function _mul_fir_forward!(y::AbstractGPUArray{T}, A::GpuFilt{T}, b::AbstractGPUArray{T}) where {T}
    _load_signal!(A.buf, b)
    mul!(A.buf_fft, A.plan_fwd, A.buf)
    A.buf_fft .*= A.h_fft
    mul!(A.buf_out, A.plan_inv, A.buf_fft)
    return _copy_head!(y, A.buf_out, length(y))
end

function _mul_fir_adjoint!(y::AbstractGPUArray{T}, A::GpuFilt{T}, b::AbstractGPUArray{T}) where {T}
    _load_signal!(A.buf, b)
    mul!(A.buf_fft, A.plan_fwd, A.buf)
    A.buf_fft .*= A.h_fft_conj
    mul!(A.buf_out, A.plan_inv, A.buf_fft)
    return _copy_head!(y, A.buf_out, length(y))
end

function _add_fir_forward!(y::AbstractGPUArray{T}, A::GpuFilt{T}, b::AbstractGPUArray{T}) where {T}
    _load_signal!(A.buf, b)
    mul!(A.buf_fft, A.plan_fwd, A.buf)
    A.buf_fft .*= A.h_fft
    mul!(A.buf_out, A.plan_inv, A.buf_fft)
    y .+= view(A.buf_out, 1:length(y))
    return y
end

function _add_fir_adjoint!(y::AbstractGPUArray{T}, A::GpuFilt{T}, b::AbstractGPUArray{T}) where {T}
    _load_signal!(A.buf, b)
    mul!(A.buf_fft, A.plan_fwd, A.buf)
    A.buf_fft .*= A.h_fft_conj
    mul!(A.buf_out, A.plan_inv, A.buf_fft)
    y .+= view(A.buf_out, 1:length(y))
    return y
end

# ─── Filt GPU mul! ────────────────────────────────────────────────────────────

function mul!(y::AbstractGPUArray{T}, A::GpuFilt{T}, b::AbstractGPUArray{T}) where {T <: Real}
    check(y, A, b)
    length(A.a) == 1 ||
        throw(ArgumentError("IIR Filt is not supported on GPU arrays. Use CpuOperatorWrapper to run on CPU."))
    for col in 1:size(b, 2)
        _mul_fir_forward!(view(y, :, col), A, view(b, :, col))
    end
    return y
end

function mul!(
        y::AbstractGPUArray{T}, L::AdjointOperator{<:GpuFilt{T}}, b::AbstractGPUArray{T}
    ) where {T <: Real}
    check(y, L, b)
    A = L.A
    length(A.a) == 1 ||
        throw(ArgumentError("IIR Filt adjoint is not supported on GPU arrays. Use CpuOperatorWrapper."))
    for col in 1:size(b, 2)
        _mul_fir_adjoint!(view(y, :, col), A, view(b, :, col))
    end
    return y
end

# ─── MIMOFilt GPU mul! ────────────────────────────────────────────────────────

function _make_gpu_mimofilt(ref::AbstractGPUArray{T}, cpu_op::AbstractMIMOFilt{T}) where {T <: Real}
    filters = [
        _make_gpu_filt(ref, Filt(T, (cpu_op.dim_in[1],), copy(cpu_op.B[i]), copy(cpu_op.A[i]))) for
            i in eachindex(cpu_op.B)
    ]
    return GpuMIMOFilt(cpu_op.dim_out, cpu_op.dim_in, filters)
end

function MIMOFilt(
        x::AbstractGPUArray{T}, b::Vector{<:AbstractVector{T}}, a::Vector{<:AbstractVector{T}}
    ) where {T <: Real}
    cpu_op = MIMOFilt(T, size(x), copy(b), copy(a))
    return _make_gpu_mimofilt(x, cpu_op)
end

function MIMOFilt(x::AbstractGPUArray{T}, b::Vector{<:AbstractVector{T}}) where {T <: Real}
    return MIMOFilt(x, b, [T[one(T)] for _ in eachindex(b)])
end

function mul!(y::AbstractGPUArray{T}, L::GpuMIMOFilt{T}, b::AbstractGPUArray{T}) where {T <: Real}
    check(y, L, b)
    for filter in L.filters
        length(filter.a) == 1 || throw(
            ArgumentError("IIR MIMOFilt is not supported on GPU arrays. Use CpuOperatorWrapper."),
        )
    end
    fill!(y, zero(T))
    cnt = 0
    for cy in 1:L.dim_out[2]
        for cx in 1:L.dim_in[2]
            cnt += 1
            _add_fir_forward!(view(y, :, cy), L.filters[cnt], view(b, :, cx))
        end
    end
    return y
end

function mul!(
        y::AbstractGPUArray{T}, M::AdjointOperator{<:GpuMIMOFilt{T}}, b::AbstractGPUArray{T}
    ) where {T <: Real}
    check(y, M, b)
    L = M.A
    for filter in L.filters
        length(filter.a) == 1 || throw(
            ArgumentError(
                "IIR MIMOFilt adjoint is not supported on GPU arrays. Use CpuOperatorWrapper."
            ),
        )
    end
    fill!(y, zero(T))
    cnt = 0
    for cy in 1:L.dim_out[2]
        for cx in 1:L.dim_in[2]
            cnt += 1
            _add_fir_adjoint!(view(y, :, cx), L.filters[cnt], view(b, :, cy))
        end
    end
    return y
end

end # module GpuExt
