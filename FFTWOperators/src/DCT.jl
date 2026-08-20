export DCT, IDCT

abstract type CosineTransform{N, C, T1, T2} <: LinearOperator end

"""
	DCT([domain_type=Float64::Type,] dim_in::Tuple)
	DCT(dim_in...)
	DCT(x::AbstractArray)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the `N`-dimensional Inverse Discrete Cosine Transform of `x`.

```jldoctest
julia> using FFTWOperators

julia> DCT(Complex{Float64},(10,10))
ℱc  ℂ^(10, 10) -> ℂ^(10, 10)

julia> DCT(10,10)
ℱc  ℝ^(10, 10) -> ℝ^(10, 10)

julia> A = DCT(ones(3))
ℱc  ℝ^3 -> ℝ^3

julia> A*ones(3)
3-element Vector{Float64}:
 1.7320508075688772
 0.0
 0.0
	
```
"""
struct DCT{N, C, T1 <: AbstractFFTs.Plan, T2 <: AbstractFFTs.Plan, B} <: CosineTransform{N, C, T1, T2}
    dim_in::NTuple{N, Int}
    A::T1
    At::T2
    buf::B
    # Plan-time FFTW thread count. FFTW threads r2r transforms (measured 2.25x forward,
    # 1.91x inverse at n=2^22), so this is a real choice, fixed when the plan is built.
    num_threads::Int
end

"""
	IDCT([domain_type=Float64::Type,] dim_in::Tuple)
	IDCT(dim_in...)
	IDCT(x::AbstractArray)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the `N`-dimensional inverse Discrete Cosine Transform of `x`.

```jldoctest
julia> using FFTWOperators

julia> IDCT(Complex{Float64},(10,10))
ℱc⁻¹  ℂ^(10, 10) -> ℂ^(10, 10)

julia> IDCT(10,10)
ℱc⁻¹  ℝ^(10, 10) -> ℝ^(10, 10)

julia> A = IDCT(ones(3))
ℱc⁻¹  ℝ^3 -> ℝ^3

julia> A*[1.;0.;0.]
3-element Vector{Float64}:
 0.5773502691896258
 0.5773502691896258
 0.5773502691896258

```
"""
struct IDCT{N, C, T1 <: AbstractFFTs.Plan, T2 <: AbstractFFTs.Plan, B} <: CosineTransform{N, C, T1, T2}
    dim_in::NTuple{N, Int}
    A::T1
    At::T2
    buf::B
    # Plan-time FFTW thread count; see the note on DCT.
    num_threads::Int
end

# Constructors
#standard constructor
function DCT(T::Type, dim_in::NTuple{N, Int}; kwargs...) where {N}
    return DCT(zeros(T, dim_in); kwargs...)
end
DCT(dim_in::NTuple{N, Int}; kwargs...) where {N} = DCT(zeros(dim_in); kwargs...)
DCT(dim_in::Vararg{Int64}; kwargs...) = DCT(dim_in; kwargs...)
DCT(T::Type, dim_in::Vararg{Int64}; kwargs...) = DCT(T, dim_in; kwargs...)

function DCT(x::AbstractArray{C, N}; num_threads = nothing, threaded::Bool = true) where {N, C}
    nthr = _fftw_num_threads(:r2r, num_threads, threaded, length(x))
    A, At = _with_fftw_threads(nthr) do
        plan_dct(x), plan_idct(x)
    end
    buf = similar(x)
    return DCT{N, C, typeof(A), typeof(At), typeof(buf)}(size(x), A, At, buf, nthr)
end

#standard constructor
function IDCT(T::Type, dim_in::NTuple{N, Int}; kwargs...) where {N}
    return IDCT(zeros(T, dim_in); kwargs...)
end
IDCT(dim_in::NTuple{N, Int}; kwargs...) where {N} = IDCT(zeros(dim_in); kwargs...)
IDCT(dim_in::Vararg{Int64}; kwargs...) = IDCT(dim_in; kwargs...)
IDCT(T::Type, dim_in::Vararg{Int64}; kwargs...) = IDCT(T, dim_in; kwargs...)

function IDCT(x::AbstractArray{C, N}; num_threads = nothing, threaded::Bool = true) where {N, C}
    nthr = _fftw_num_threads(:r2r, num_threads, threaded, length(x))
    A, At = _with_fftw_threads(nthr) do
        plan_idct(x), plan_dct(x)
    end
    buf = similar(x)
    return IDCT{N, C, typeof(A), typeof(At), typeof(buf)}(size(x), A, At, buf, nthr)
end

# Mappings

function mul!(y::AbstractArray, A::DCT, b::AbstractArray)
    check(y, A, b)
    mul!(y, A.A, b)  # DCT plan (REDFT10): non-destructive to input
    return y
end

function mul!(y::AbstractArray, A::AdjointOperator{<:DCT}, b::AbstractArray)
    check(y, A, b)
    # IDCT plan (REDFT01) modifies its input in-place; use scratch buffer
    copyto!(A.A.buf, b)
    mul!(y, A.A.At, A.A.buf)
    return y
end

function mul!(y::AbstractArray, A::IDCT, b::AbstractArray)
    check(y, A, b)
    # IDCT plan (REDFT01) modifies its input in-place; use scratch buffer
    copyto!(A.buf, b)
    mul!(y, A.A, A.buf)
    return y
end

function mul!(y::AbstractArray, A::AdjointOperator{<:IDCT}, b::AbstractArray)
    check(y, A, b)
    mul!(y, A.A.At, b)  # DCT plan (REDFT10): non-destructive to input
    return y
end

# Properties

size(L::CosineTransform) = (L.dim_in, L.dim_in)

fun_name(A::DCT) = "ℱc"
fun_name(A::IDCT) = "ℱc⁻¹"

domain_type(::CosineTransform{N, C}) where {N, C} = C
codomain_type(::CosineTransform{N, C}) where {N, C} = C
domain_array_type(::DCT{N, C, T1, T2, B}) where {N, C, T1, T2, B} = Base.typename(B).wrapper{C}
domain_array_type(::IDCT{N, C, T1, T2, B}) where {N, C, T1, T2, B} = Base.typename(B).wrapper{C}
codomain_array_type(::DCT{N, C, T1, T2, B}) where {N, C, T1, T2, B} = Base.typename(B).wrapper{C}
codomain_array_type(::IDCT{N, C, T1, T2, B}) where {N, C, T1, T2, B} = Base.typename(B).wrapper{C}
is_thread_safe(::CosineTransform) = false

is_AcA_diagonal(L::CosineTransform) = true
is_AAc_diagonal(L::CosineTransform) = true
is_orthogonal(L::CosineTransform) = true
is_invertible(L::CosineTransform) = true
is_full_row_rank(L::CosineTransform) = true
is_full_column_rank(L::CosineTransform) = true

diag_AcA(L::CosineTransform) = 1.0
diag_AAc(L::CosineTransform) = 1.0

has_optimized_normalop(L::CosineTransform) = true
get_normal_op(L::CosineTransform) = Eye(allocate_in_domain(L))

has_fast_opnorm(::CosineTransform) = true
LinearAlgebra.opnorm(L::CosineTransform) = one(real(domain_type(L)))

# ─── Threading ────────────────────────────────────────────────────────────────
#
# FFTW is a counted thread pool and it does thread r2r transforms: measured 2.25x (DCT) and
# 1.91x (IDCT) at n = 2^22 with 8 threads. The count is baked into the plan, so `threaded`
# is a construction-time choice that `is_threaded` reads back rather than a switchable loop.
is_threaded(op::CosineTransform) = op.num_threads > 1
supports_threading(::CosineTransform) = true

function _copy_operator_impl(
        op::DCT{N, C}; storage_type = nothing, threaded = nothing
    ) where {N, C}
    return _copy_cosine_transform(DCT, op, C, storage_type, threaded)
end

function _copy_operator_impl(
        op::IDCT{N, C}; storage_type = nothing, threaded = nothing
    ) where {N, C}
    return _copy_cosine_transform(IDCT, op, C, storage_type, threaded)
end

function _copy_cosine_transform(ctor, op, ::Type{C}, storage_type, threaded) where {C}
    new_threaded = threaded === nothing ? is_threaded(op) : threaded
    # `buf` is per-call scratch, so a copy always needs its own -- sharing it is what makes
    # these operators unsafe to run from two threads. Replanning is only needed when the
    # storage type or thread count actually changes.
    if storage_type === nothing && new_threaded == is_threaded(op)
        return typeof(op)(op.dim_in, op.A, op.At, similar(op.buf), op.num_threads)
    end
    # No persistent data to carry over (a cosine transform holds only plans and scratch),
    # so the prototype can be uninitialized.
    new_storage = storage_type === nothing ? _array_wrapper_type(typeof(op.buf)) : storage_type
    return ctor(similar(new_storage{C}, op.dim_in); threaded = new_threaded)
end
