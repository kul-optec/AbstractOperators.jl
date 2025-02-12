module NfftExt

export NfftOp

using LinearAlgebra
using AbstractOperators
using FastBroadcast
import LinearAlgebra: mul!
import Base: size
import NFFT: NFFT
import AbstractOperators:
	NfftOp,
	set_thread_counts_expr,
	thread_count_functions,
	domainType,
	codomainType,
	fun_name
import Base.Threads: nthreads
using FFTW

struct NfftOpData{T,D} <: AbstractOperators.LinearOperator
	plan::NFFT.AbstractNFFTPlan{T,D}
	ksp_buffer::AbstractMatrix{Complex{T}}
	dcf::AbstractMatrix{T}
	threaded::Bool
end

"""
	NfftOp(image_size::NTuple{D,Int}, trajectory::AbstractArray{T}, dcf::AbstractArray; threaded::Bool=true, kwargs...)

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
julia> using NFFT # import NFFT to trigger loading NfftOp implementation

julia> image_size = (128, 128)

julia> trajectory = rand(2, 128, 50) .- 0.5;

julia> dcf = rand(128, 50);

julia> op = NfftOp(image_size, trajectory, dcf)

julia> image = rand(ComplexF64, image_size);

julia> ksp = op * image;

julia> image_reconstructed = op' * ksp;

```
"""
function NfftOp(
	image_size::NTuple{D,Int},
	trajectory::AbstractArray{T},
	dcf::AbstractArray;
	threaded::Bool=true,
	kwargs...,
) where {T,D}
	check_traj_and_dcf(trajectory, dcf, D)
	plan = create_plan(trajectory, image_size, threaded; kwargs...)
	ksp_buffer = similar(
		trajectory, complex(eltype(trajectory)), size(trajectory)[2:end]...
	)
	return NfftOpData{T,D}(plan, ksp_buffer, dcf, threaded)
end

function NfftOp(
	image_size::NTuple{D,Int},
	trajectory::AbstractArray{T};
	threaded::Bool=true,
	dcf_estimation_iterations::Int=20,
	dcf_correction_function::Function=identity,
	kwargs...,
) where {T,D}
	check_traj(trajectory, D)
	plan = create_plan(trajectory, image_size, threaded; kwargs...)
	ksp_buffer = similar(
		trajectory, complex(eltype(trajectory)), size(trajectory)[2:end]...
	)
	raw_dcf = sdc(plan; iters=dcf_estimation_iterations)
	dcf = dcf_correction_function(reshape(raw_dcf, size(ksp_buffer)))
	return NfftOpData{T,D}(plan, ksp_buffer, dcf, threaded)
end

function set_nfft_threading_expr(threading_state_expr, thread_count_expr, body_expr)
	quote
		local prev_nfft_threading_state = NFFT._use_threads[]
		NFFT._use_threads[] = $threading_state_expr
		local res = $(set_thread_counts_expr(thread_count_expr, body_expr))
		NFFT._use_threads[] = prev_nfft_threading_state
		res
	end
end

macro enable_nfft_threading(expr)
	return set_nfft_threading_expr(true, nthreads(), expr)
end

macro disable_nfft_threading(expr)
	return set_nfft_threading_expr(false, 1, expr)
end

function mul!(ksp::AbstractArray, op::NfftOpData, img::AbstractArray)
	AbstractOperators.check(img, ksp, op)
	if op.threaded
		@enable_nfft_threading mul!(vec(ksp), op.plan, img)
	else
		@disable_nfft_threading mul!(vec(ksp), op.plan, img)
	end
	return ksp
end

function mul!(
	img::AbstractArray,
	op::AbstractOperators.AdjointOperator{<:NfftOpData},
	ksp::AbstractArray,
)
	op = op.A
	AbstractOperators.check(img, ksp, op)
	if op.threaded
		@.. thread = true op.ksp_buffer = ksp * op.dcf
		@enable_nfft_threading mul!(img, op.plan', vec(op.ksp_buffer))
	else
		@.. op.ksp_buffer = ksp * op.dcf
		@disable_nfft_threading mul!(img, op.plan', vec(op.ksp_buffer))
	end
end

# Properties

size(L::NfftOpData) = size(L.ksp_buffer), NFFT.size_in(L.plan)
fun_name(::NfftOpData) = "𝒩"
domainType(::NfftOpData{T}) where {T} = complex(T)
codomainType(::NfftOpData{T}) where {T} = complex(T)

# Utility

function check_traj(traj, D)
	@assert size(traj, 1) == D "The first dimension of the trajectory must match the number of image dimensions"
	@assert ndims(traj) > 1 "The trajectory must have at least two dimensions"
end

function check_traj_and_dcf(traj, dcf, D)
	check_traj(traj, D)
	@assert tuple(size(traj)[2:end]...) == size(dcf) "Shape of the trajectory from the second dimension must match the shape of the dcf array"
	@assert eltype(traj) == eltype(dcf) "The element type of the trajectory must match the element type of the dcf array"
end

function create_plan(trajectory, image_size, threaded; kwargs...)
	traj = reshape(trajectory, size(trajectory, 1), :)
	return if threaded
		return @enable_nfft_threading NFFTPlan(traj, image_size; kwargs...)
	else
		return @disable_nfft_threading NFFTPlan(traj, image_size; kwargs...)
	end
end

function sdc(p::NFFT.AbstractNFFTPlan{T,D,1}; iters=20) where {T,D}
	# Weights for sample density compensation.
	# Uses method of Pipe & Menon, 1999. Mag Reson Med, 186, 179.
	weights = similar(p.tmpVec, Complex{T}, p.J)
	weights .= one(Complex{T})
	weights_tmp = similar(weights)
	scaling_factor = zero(T)

	# Pre-weighting to correct non-uniform sample density
	for i in 1:iters
		NFFT.convolve_transpose!(p, weights, p.tmpVec)
		if i == 1
			scaling_factor = maximum(abs.(p.tmpVec))
		end

		p.tmpVec ./= scaling_factor
		NFFT.convolve!(p, p.tmpVec, weights_tmp)
		weights_tmp ./= scaling_factor
		weights ./= (abs.(weights_tmp) .+ eps(T))
	end
	# Post weights to correct image scaling
	# This finds c, where ||u - c*v||_2^2 = 0 and then uses
	# c to scale all weights by a scalar factor.
	u = similar(weights, Complex{T}, p.N)
	u .= one(Complex{T})

	# conversion to Array is a workaround for CuNFFT. Without it we get strange
	# results that indicate some synchronization issue
	#f = Array( p * u ) 
	#b = f .* Array(weights) # apply weights from above
	#v = Array( adjoint(p) * convert(typeof(weights), b) )
	#c = vec(v) \ vec(Array(u))  # least squares diff
	#return abs.(convert(typeof(weights), c * Array(weights))) 

	# non converting version
	f = similar(p.tmpVec, Complex{T}, p.J)
	mul!(f, p, u)
	f .*= weights # apply weights from above
	v = similar(p.tmpVec, Complex{T}, p.N)
	mul!(v, adjoint(p), f)
	c = vec(v) \ vec(u)  # least squares diff
	return abs.(c * weights)
end

function NFFTPlan(
	k::Matrix{T},
	N::NTuple{D,Int};
	dims::Union{Integer,UnitRange{Int64}}=1:D,
	fftflags=nothing,
	kwargs...,
) where {T,D}
	NFFT.checkNodes(k)

	params, N, NOut, J, Ñ, dims_ = NFFT.initParams(k, N, dims; kwargs...)

	if length(NOut) > 1
		params.precompute = NFFT.LINEAR
	end

	tmpVec = Array{Complex{T},D}(undef, Ñ)

	fftflags_ = (fftflags !== nothing) ? (flags=fftflags,) : NamedTuple()
	FP = plan_fft!(tmpVec, dims_; num_threads=FFTW.get_num_threads(), fftflags_...)
	BP = plan_bfft!(tmpVec, dims_; num_threads=FFTW.get_num_threads(), fftflags_...)

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
	tmpVecHat = Array{Complex{T},D}(undef, U)

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

end # module NfftExt