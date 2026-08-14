# The _mul! and _get_normal_op function are based on the implementation of the
# NFFTToeplitzNormalOp from LinearOperatorCollection.jl. That's the reason why
# the license is included here. The original code can be found here:
# https://github.com/JuliaImageRecon/LinearOperatorCollection.jl/blob/main/ext/LinearOperatorNFFTExt/NFFTOp.jl

# MIT License
#
# Copyright (c) 2023 Tobias Knopp <tobias.knopp@tuhh.de> and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

struct NfftNormalOp{N, T, A <: AbstractArray, F, I} <: AbstractOperators.LinearOperator
    array_type::Type{T}
    shape::NTuple{N, Int}
    λ::A
    fftplan::F
    ifftplan::I
    buf::A
    threaded::Bool
end

function mul!(y::AbstractArray, op::NfftNormalOp, x::AbstractArray)
    AbstractOperators.check(y, op, x)
    return with_nfft_threading(op.threaded) do
        _mul!(y, op, x)
    end
end

function _mul!(y, op::NfftNormalOp, x)
    op.buf .= 0
    op.buf[CartesianIndices(x)] .= x
    op.fftplan * op.buf # in-place FFT
    op.buf .*= op.λ
    op.ifftplan * op.buf # in-place IFFT
    y .= @view op.buf[CartesianIndices(x)]
    return y
end

AbstractOperators.has_optimized_normalop(::NFFTOp) = true
function AbstractOperators.get_normal_op(op::NFFTOp)
    return with_nfft_threading(op.threaded) do
        _get_normal_op(op)
    end
end

function _get_normal_op(op::NFFTOp)
    shape = op.plan.N
    shape_ext = 2 .* shape

    buf = allocate_in_domain(op, shape_ext...)
    fill!(buf, 0)
    tmp = allocate_in_codomain(op, size(op.plan.k, 2))
    tmp .= vec(op.dcf)

    fftplan = FFTW.plan_fft!(buf)
    p = NFFT.plan_nfft(
        op.plan.k,
        shape_ext;
        m = op.plan.params.m,
        σ = op.plan.params.σ,
        precompute = NFFT.POLYNOMIAL,
        fftflags = FFTW.ESTIMATE,
        blocking = true,
    )

    mul!(buf, adjoint(p), tmp)
    λ = fftplan * FFTW.fftshift(buf) # create a new array by fftshift and apply in-place FFT

    return NfftNormalOp(domain_array_type(op), shape, λ, fftplan, inv(fftplan), buf, op.threaded)
end

# properties

Base.size(op::NfftNormalOp) = op.shape, op.shape
AbstractOperators.fun_name(::NfftNormalOp) = "(𝒩ᵃ𝒩)"
domain_type(::NfftNormalOp{N, T}) where {N, T} = eltype(T)
codomain_type(::NfftNormalOp{N, T}) where {N, T} = eltype(T)
domain_array_type(op::NfftNormalOp{N, T}) where {N, T} = op.array_type
codomain_array_type(op::NfftNormalOp{N, T}) where {N, T} = op.array_type
is_symmetric(::NfftNormalOp) = true
AdjointOperator(op::NfftNormalOp) = op
