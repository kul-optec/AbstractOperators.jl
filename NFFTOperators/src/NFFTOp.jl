struct NFFTOp{
        T,
        D,
        P <: NFFT.AbstractNFFTPlan{T, D},
        K <: AbstractMatrix{Complex{T}},
        DC <: AbstractMatrix{T},
    } <: AbstractOperators.LinearOperator
    plan::P
    ksp_buffer::K
    dcf::DC
    threaded::Bool
end

"""
	NFFTOp(image_size::NTuple{D,Int}, trajectory::AbstractArray{T}, dcf::AbstractArray; threaded::Bool=true, kwargs...)

Create a non-uniform fast Fourier transform operator [1]. The operator is created with a given image 
size, trajectory, and density compensation function (dcf). The dcf is used to correct for the 
non-uniform sample density of the trajectory. The operator can be used to transform images to 
k-space and back.

<em>To use the operator, the NFFT package must be explicitly imported!</em>

# Arguments
- `image_size::NTuple{D,Int}`: The size of the image to transform.
- `trajectory::AbstractArray{T}`: The trajectory of the samples in k-space. The first dimension
  of the trajectory must match the number of image dimensions. The trajectory must have at least
  two dimensions.
- `dcf::AbstractArray`: The density compensation function. The shape of the trajectory from the
  second dimension must match the shape of the dcf array. The element type of the trajectory must
  match the element type of the dcf array. This argument is optional and defaults to `nothing`.
  If `nothing` is passed, the dcf will be estimated using the sample density compensation method [2].
- `threaded::Bool=true`: Whether to use threading when applying the operator. Defaults to `true`.
- `dcf_estimation_iterations::Union{Nothing,Int}=nothing`: The number of iterations to use when
  estimating the dcf. Defaults to `20`. This argument is only used if `dcf` is not provided.
- `dcf_correction_function::Function=identity`: A correction function to apply to the estimated dcf.
  Defaults to the identity function. This argument is only used if `dcf` is not provided.
- `kwargs...`: Additional keyword arguments to pass to the NFFTPlan constructor.

# References
1. Fessler, J. A., & Sutton, B. P. (2003). Nonuniform fast Fourier transforms using min-max interpolation.
IEEE Transactions on Signal Processing, 51(2), 560-574.
2. Pipe, J. G., & Menon, P. (1999). Sampling density compensation in MRI: rationale and an iterative numerical solution.

# Examples
```jldoctest
julia> using NFFTOperators

julia> image_size = (128, 128);

julia> trajectory = rand(2, 128, 50) .- 0.5;

julia> dcf = rand(128, 50);

julia> op = NFFTOp(image_size, trajectory, dcf)
𝒩  ℂ^(128, 128) -> ℂ^(128, 50)

julia> image = rand(ComplexF64, image_size);

julia> ksp = op * image;

julia> image_reconstructed = op' * ksp;

```
"""
function NFFTOp(
        image_size::NTuple{D, Int},
        trajectory::AbstractArray{T},
        dcf::AbstractArray;
        threaded::Bool = true,
        array_type::Type = Array{T},
        kwargs...,
    ) where {T, D}
    check_traj_and_dcf(trajectory, dcf, D)
    arr_wrapper = _array_wrapper_type(array_type)
    plan = _nfft_plan(arr_wrapper, trajectory, image_size, threaded; kwargs...)
    ksp_shape = size(trajectory)[2:end]
    ksp_buffer = _nfft_adapt(arr_wrapper, zeros(complex(T), ksp_shape...))
    adapted_dcf = _nfft_adapt(arr_wrapper, collect(dcf))
    threaded_flag = threaded && arr_wrapper === Array
    return NFFTOp{T, D, typeof(plan), typeof(ksp_buffer), typeof(adapted_dcf)}(plan, ksp_buffer, adapted_dcf, threaded_flag)
end

function NFFTOp(
        image_size::NTuple{D, Int},
        trajectory::AbstractArray{T};
        threaded::Bool = true,
        array_type::Type = Array{T},
        dcf_estimation_iterations::Int = 20,
        dcf_correction_function::Function = identity,
        kwargs...,
    ) where {T, D}
    check_traj(trajectory, D)
    arr_wrapper = _array_wrapper_type(array_type)
    plan = _nfft_plan(arr_wrapper, trajectory, image_size, threaded; kwargs...)
    ksp_shape = size(trajectory)[2:end]
    ksp_buffer = _nfft_adapt(arr_wrapper, zeros(complex(T), ksp_shape...))
    raw_dcf = NFFTTools.sdc(plan; iters = dcf_estimation_iterations)
    dcf_cpu = dcf_correction_function(reshape(raw_dcf, ksp_shape))
    adapted_dcf = _nfft_adapt(arr_wrapper, collect(dcf_cpu))
    threaded_flag = threaded && arr_wrapper === Array
    return NFFTOp{T, D, typeof(plan), typeof(ksp_buffer), typeof(adapted_dcf)}(plan, ksp_buffer, adapted_dcf, threaded_flag)
end

# Default (CPU) implementations — overridden by NFFTOperatorsGPUArraysExt for GPU types
function _nfft_plan(::Type{Array}, trajectory, image_size, threaded; kwargs...)
    return create_plan(trajectory, image_size, threaded; kwargs...)
end
_nfft_adapt(::Type{Array}, arr::AbstractArray) = collect(arr)

"""
    with_nfft_threading(f, threaded::Bool)

Run `f()` with NFFT's, BLAS's and FFTW's threading turned on or off together.

`threaded = true` actively *enables* them rather than merely leaving them alone, because
`NFFT._use_threads[]` defaults to off and is what the NFFT plan consults. Both directions
go through `NestedThreading`, so the request is clamped by any outer budget: an operator
constructed with `threaded = true` that ends up being called from inside a saturated batch
loop stays single-threaded, and the save/restore bookkeeping is refcounted rather than
per-call.
"""
with_nfft_threading(f::F, threaded::Bool) where {F} =
    threaded ? with_full_threads(f) : with_restricted_threads(f)

function mul!(ksp::AbstractArray, op::NFFTOp, img::AbstractArray)
    AbstractOperators.check(ksp, op, img)
    with_nfft_threading(op.threaded) do
        mul!(vec(ksp), op.plan, img)
    end
    return ksp
end

function mul!(
        img::AbstractArray,
        adjop::AbstractOperators.AdjointOperator{<:NFFTOp},
        ksp::AbstractArray,
    )
    AbstractOperators.check(img, adjop, ksp)
    op = adjop.A
    if op.threaded
        @.. thread = true op.ksp_buffer = ksp * op.dcf
    else
        @.. op.ksp_buffer = ksp * op.dcf
    end
    with_nfft_threading(op.threaded) do
        mul!(img, op.plan', vec(op.ksp_buffer))
    end
    return img
end

# Properties

size(L::NFFTOp) = size(L.ksp_buffer), NFFT.size_in(L.plan)
fun_name(::NFFTOp) = "𝒩"
domain_type(::NFFTOp{T}) where {T} = complex(T)
codomain_type(::NFFTOp{T}) where {T} = complex(T)
domain_array_type(op::NFFTOp) = typeof(op.plan.tmpVec)
codomain_array_type(op::NFFTOp{T, D, P, K}) where {T, D, P, K} = K

# Utility

function check_traj(traj, D)
    @assert size(traj, 1) == D "The first dimension of the trajectory must match the number of image dimensions"
    return @assert ndims(traj) > 1 "The trajectory must have at least two dimensions"
end

function check_traj_and_dcf(traj, dcf, D)
    check_traj(traj, D)
    @assert tuple(size(traj)[2:end]...) == size(dcf) "Shape of the trajectory from the second dimension must match the shape of the dcf array"
    return @assert eltype(traj) == eltype(dcf) "The element type of the trajectory must match the element type of the dcf array"
end

function create_plan(trajectory, image_size, threaded; kwargs...)
    traj = reshape(trajectory, size(trajectory, 1), :)
    return with_nfft_threading(threaded) do
        NFFTPlan(traj, image_size; kwargs...)
    end
end

# Helper to create matched forward/backward FFT plans so that JET can track
# that both plans have the same element type T and dimension D.
struct _MatchedFFTPlans{T, D}
    forward::FFTW.cFFTWPlan{Complex{T}, -1, true, D, UnitRange{Int64}}
    backward::FFTW.cFFTWPlan{Complex{T}, 1, true, D, UnitRange{Int64}}
end

function _make_matched_fft_plans(tmpVec::Array{Complex{T}, D}, dims_; kwargs...) where {T, D}
    FP = FFTW.plan_fft!(tmpVec, dims_; kwargs...)::FFTW.cFFTWPlan{Complex{T}, -1, true, D, UnitRange{Int64}}
    BP = FFTW.plan_bfft!(tmpVec, dims_; kwargs...)::FFTW.cFFTWPlan{Complex{T}, 1, true, D, UnitRange{Int64}}
    return _MatchedFFTPlans{T, D}(FP, BP)
end

function NFFTPlan(
        k::Matrix{T},
        N::NTuple{D, Int};
        dims::Union{Integer, UnitRange{Int64}} = 1:D,
        fftflags = nothing,
        kwargs...,
    ) where {T, D}
    NFFT.checkNodes(k)

    params, N, NOut, J, Ñ, dims_ = NFFT.initParams(k, N, dims; kwargs...)

    if length(NOut) > 1
        params.precompute = NFFT.LINEAR
    end

    tmpVec = Array{Complex{T}, D}(undef, Ñ)

    fftflags_ = (fftflags !== nothing) ? (flags = fftflags,) : NamedTuple()
    plans = _make_matched_fft_plans(tmpVec, dims_; num_threads = FFTW.get_num_threads(), fftflags_...)
    FP = plans.forward
    BP = plans.backward

    calcBlocks =
        (
        params.precompute == NFFT.LINEAR ||
            params.precompute == NFFT.TENSOR ||
            params.precompute == NFFT.POLYNOMIAL
    ) &&
        params.blocking &&
        length(dims_) == D

    blocks, nodesInBlocks, blockOffsets, idxInBlock, windowTensor = NFFT.precomputeBlocks(
        k, Ñ, params, calcBlocks
    )

    windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B = NFFT.precomputation(
        k, N[dims_], Ñ[dims_], params
    )

    U = params.storeDeconvolutionIdx ? N : ntuple(d -> 0, D)
    tmpVecHat = Array{Complex{T}, D}(undef, U)

    return NFFT.NFFTPlan(
        N,
        NOut,
        J,
        k,
        Ñ,
        dims_,
        params,
        FP,
        BP,
        tmpVec,
        tmpVecHat,
        deconvolveIdx,
        windowHatInvLUT,
        windowLinInterp,
        windowPolyInterp,
        blocks,
        nodesInBlocks,
        blockOffsets,
        idxInBlock,
        windowTensor,
        B,
    )
end

# ─── Threading ────────────────────────────────────────────────────────────────
#
# NFFT is a *counted* pool in NestedThreading: the plan itself is built for a thread count
# and `mul!` runs under `with_full_threads`/`with_restricted_threads` (see `_nfft_run`).
# So `threaded` here selects a plan-time thread count, not a Julia loop, and it cannot be
# flipped after construction -- switching it rebuilds the plan.
is_threaded(op::NFFTOp) = op.threaded
supports_threading(::NFFTOp) = true

# Not thread-safe: `ksp_buffer` is operator-owned scratch written by every `mul!`.
is_thread_safe(::NFFTOp) = false

function _copy_operator_impl(
        op::NFFTOp{T, D, P, K, DC}; storage_type = nothing, threaded = nothing
    ) where {T, D, P, K, DC}
    new_threaded = threaded === nothing ? op.threaded : threaded
    # The plan is immutable and thread-count-specific: it can be shared only when the
    # thread count is unchanged *and* the storage backend is unchanged. Otherwise the
    # caller must rebuild the operator from its trajectory, which this operator no longer
    # retains -- so say so rather than return something that silently ignores the request.
    if storage_type !== nothing
        throw(
            ArgumentError(
                "NFFTOp cannot change storage_type after construction: the NFFT plan is " *
                    "built for a specific array backend. Rebuild with " *
                    "NFFTOp(image_size, trajectory, dcf; array_type = ...) instead."
            ),
        )
    end
    if new_threaded != op.threaded
        throw(
            ArgumentError(
                "NFFTOp cannot change `threaded` after construction: the NFFT plan is " *
                    "built for a fixed thread count. Rebuild with " *
                    "NFFTOp(image_size, trajectory, dcf; threaded = $(new_threaded)) instead."
            ),
        )
    end
    # Same constraints: share the (immutable) plan and dcf, give the copy its own scratch.
    return NFFTOp{T, D, P, K, DC}(op.plan, similar(op.ksp_buffer), op.dcf, op.threaded)
end
