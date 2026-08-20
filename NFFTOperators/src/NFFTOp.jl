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
- `threaded`: `false` disables threading outright; `true` (or the default `nothing`) enables it subject to the threading policy, which also requires more than one Julia thread and CPU storage. Fixed at construction, since the NFFT plan is built for a thread count.
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
    # Resolved before planning: the plan itself is built for this thread count, so the
    # policy has to have had its say by now (a `nothing` reaching `create_plan` would not
    # even dispatch).
    threaded_flag = _nfft_threaded(threaded, arr_wrapper)
    plan = _nfft_plan(arr_wrapper, trajectory, image_size, threaded_flag; kwargs...)
    ksp_shape = size(trajectory)[2:end]
    ksp_buffer = _nfft_adapt(arr_wrapper, zeros(complex(T), ksp_shape...))
    adapted_dcf = _nfft_adapt(arr_wrapper, collect(dcf))
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
    # Resolved before planning: the plan itself is built for this thread count, so the
    # policy has to have had its say by now (a `nothing` reaching `create_plan` would not
    # even dispatch).
    threaded_flag = _nfft_threaded(threaded, arr_wrapper)
    plan = _nfft_plan(arr_wrapper, trajectory, image_size, threaded_flag; kwargs...)
    ksp_shape = size(trajectory)[2:end]
    ksp_buffer = _nfft_adapt(arr_wrapper, zeros(complex(T), ksp_shape...))
    raw_dcf = NFFTTools.sdc(plan; iters = dcf_estimation_iterations)
    dcf_cpu = dcf_correction_function(reshape(raw_dcf, ksp_shape))
    adapted_dcf = _nfft_adapt(arr_wrapper, collect(dcf_cpu))
    return NFFTOp{T, D, typeof(plan), typeof(ksp_buffer), typeof(adapted_dcf)}(plan, ksp_buffer, adapted_dcf, threaded_flag)
end

"""
	_nfft_threaded(threaded, arr_wrapper) -> Bool

Resolve NFFT's `threaded` keyword under the package-wide rule: `false` vetoes, `true` and
`nothing` enable subject to policy (see `AbstractOperators._resolve_threaded`).

The policy here is the CPU/thread-count check only. NFFT has **no measured size gate**: the
transform is planned for a specific trajectory rather than a plain array length, so there is
no single element count to threshold on, and unlike FFTW it is not cheap to sweep. If a size
gate is wanted later it belongs here.
"""
function _nfft_threaded(threaded::Bool, arr_wrapper)
    return _resolve_threaded(threaded) do
        Threads.nthreads() > 1 && arr_wrapper === Array
    end
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
    # The plan is immutable and thread-count-specific: it can be shared only when neither
    # the storage backend nor the thread count changes. Otherwise the whole operator has to
    # be replanned, which requires the trajectory back -- this operator does not retain it
    # as a separate field, but the plan itself does (`plan.k`, flattened to the 2D form
    # `create_plan` already reshapes every trajectory into), so it can be recovered from
    # there rather than genuinely refusing the request.
    if storage_type === nothing && new_threaded == op.threaded
        # Same constraints: share the (immutable) plan and dcf, give the copy its own scratch.
        return NFFTOp{T, D, P, K, DC}(op.plan, similar(op.ksp_buffer), op.dcf, op.threaded)
    end
    image_size = NFFT.size_in(op.plan)
    ksp_shape = size(op.dcf)
    trajectory = reshape(collect(op.plan.k), D, ksp_shape...)
    dcf = collect(op.dcf)
    new_array_type = storage_type === nothing ? _array_wrapper_type(K){T} : storage_type{T}
    return NFFTOp(image_size, trajectory, dcf; threaded = new_threaded, array_type = new_array_type)
end
