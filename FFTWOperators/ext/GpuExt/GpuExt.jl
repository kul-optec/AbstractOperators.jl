module GpuExt

using AbstractFFTs: plan_fft, plan_bfft
using GPUArrays
using AbstractOperators: check
using FFTWOperators
import FFTWOperators: DFT, Normalization, UNNORMALIZED, _dft_scaling, SignAlternation
import LinearAlgebra: mul!

# GPU constructor for DFT with real input — avoids FFTW-specific `flags`/`timelimit`/`num_threads` kwargs
# that are not accepted by GPU FFT backends (e.g. CUDA.CUFFT, AMDGPU.rocFFT).
function FFTWOperators.DFT(
        x::AbstractGPUArray{D, N},
        dims = 1:ndims(x);
        normalization::Normalization = UNNORMALIZED,
        kwargs...,  # silently ignore FFTW-specific kwargs (flags, timelimit, num_threads)
    ) where {N, D <: Real}
    xc = similar(x, Complex{D})
    A = plan_fft(xc, dims)
    At = plan_bfft(xc, dims)
    S = typeof(x isa SubArray ? parent(x) : x).name.wrapper
    dims_t = tuple(dims...)
    scaling = _dft_scaling(size(x), dims_t, normalization)
    # `num_threads = 1`: a GPU FFT plan has no CPU thread count, and the device kernel is
    # already parallel, so `is_threaded` correctly reports false for GPU-backed DFTs.
    return DFT{N, Complex{D}, D, dims_t, S, typeof(A), typeof(At), D}(
        size(x), A, At, normalization, scaling, 1,
    )
end

# GPU constructor for DFT with complex input
function FFTWOperators.DFT(
        x::AbstractGPUArray{D, N},
        dims = 1:ndims(x);
        normalization::Normalization = UNNORMALIZED,
        kwargs...,
    ) where {N, D <: Complex}
    A = plan_fft(x, dims)
    At = plan_bfft(x, dims)
    S = typeof(x isa SubArray ? parent(x) : x).name.wrapper
    dims_t = tuple(dims...)
    scaling = _dft_scaling(size(x), dims_t, normalization)
    return DFT{N, D, D, dims_t, S, typeof(A), typeof(At), real(D)}(
        size(x), A, At, normalization, scaling, 1,
    )
end

# GPU mul! for SignAlternation: uses broadcast to avoid scalar indexing
function mul!(y::AbstractGPUArray, L::SignAlternation{T, N}, b::AbstractGPUArray) where {T, N}
    check(y, L, b)
    y .= b
    for d in L.dirs
        nd = size(b, d)
        sv_shape = ntuple(i -> i == d ? nd : 1, N)
        sv = similar(b, T, sv_shape)
        copyto!(sv, T[isodd(i - 1) ? -one(T) : one(T) for i in 1:nd])
        y .*= sv
    end
    return y
end

end # module GpuExt
