export RDFT

"""
	RDFT([domain_type=Float64::Type,] dim_in::Tuple [,dims=1])
	RDFT(dim_in...)
	RDFT(x::AbstractArray [,dims=1])

Creates a `LinearOperator` which, when multiplied with a real array `x`, returns the DFT over the dimension `dims`, exploiting Hermitian symmetry.

```jldoctest
julia> using FFTWOperators

julia> RDFT(Float64,(10,10))
ℱ  ℝ^(10, 10) -> ℂ^(6, 10)

julia> RDFT((10,10,10),2)
ℱ  ℝ^(10, 10, 10) -> ℂ^(10, 6, 10)
	
```
"""
struct RDFT{
        T <: Number, N, T1 <: AbstractFFTs.Plan, T2 <: AbstractFFTs.Plan, T3 <: AbstractArray{Complex{T}, N},
    } <: LinearOperator
    dim_in::NTuple{N, Int}
    dim_out::NTuple{N, Int}
    A::T1
    At::T2
    b2::T3
    y2::T3
    Zp::ZeroPad{N, Complex{T}}
    # Plan-time FFTW thread count. FFTW threads r2c transforms (measured 3.08x at n=2^22).
    num_threads::Int
end

# Constructors
#standard constructor

function RDFT(
        x::AbstractArray{T, N}, dims::Int = 1; num_threads = nothing, threaded::Bool = true
    ) where {T <: Real, N}
    nthr = _fftw_num_threads(:r2c, num_threads, threaded, length(x))
    b2 = similar(x, complex(T), size(x))
    y2 = similar(x, complex(T), size(x))
    A, At = _with_fftw_threads(nthr) do
        plan_rfft(x, dims), plan_bfft(y2, dims)
    end
    dim_in = size(x)
    dim_out = ()
    for i in 1:N
        dim_out = i == dims ? (dim_out..., div(dim_in[i], 2) + 1) : (dim_out..., dim_in[i])
    end
    Z = ZeroPad(Complex{T}, dim_out, size(b2) .- dim_out)
    return RDFT{T, N, typeof(A), typeof(At), typeof(b2)}(dim_in, dim_out, A, At, b2, y2, Z, nthr)
end

function RDFT(T::Type, dim_in::NTuple{N, Int}, dims::Int = 1; kwargs...) where {N}
    return RDFT(zeros(T, dim_in), dims; kwargs...)
end
function RDFT(dim_in::NTuple{N, Int}, dims::Int = 1; kwargs...) where {N}
    return RDFT(zeros(dim_in), dims; kwargs...)
end
RDFT(dim_in::Vararg{Int}; kwargs...) = RDFT(dim_in; kwargs...)
RDFT(T::Type, dim_in::Vararg{Int}; kwargs...) = RDFT(T, dim_in; kwargs...)

# Mappings

function mul!(
        y::T3, L::RDFT{T, N, T1, T2, T3}, b::T4
    ) where {N, T, T1, T2, T3, T4 <: AbstractArray{T, N}}
    check(y, L, b)
    mul!(y, L.A, b)
    return y
end

function mul!(
        y::T4, L::AdjointOperator{RDFT{T, N, T1, T2, T3}}, b::T3
    ) where {N, T, T1, T2, T3, T4 <: AbstractArray{T, N}}
    check(y, L, b)
    A = L.A
    fill!(A.b2, zero(eltype(A.b2)))
    copyto!(view(A.b2, ntuple(i -> Base.OneTo(size(b, i)), N)...), b)
    mul!(A.y2, A.At, A.b2)
    y .= real.(A.y2)
    return y
end

# Properties

size(L::RDFT) = (L.dim_out, L.dim_in)

fun_name(A::RDFT) = "ℱ"

domain_type(::RDFT{T}) where {T} = T
codomain_type(::RDFT{T}) where {T} = Complex{T}
is_thread_safe(::RDFT) = false

function domain_array_type(L::RDFT{T, N, T1, T2, T3}) where {T, N, T1, T2, T3}
    return T3.name.wrapper{T}
end
function codomain_array_type(L::RDFT{T, N, T1, T2, T3}) where {T, N, T1, T2, T3}
    return T3.name.wrapper{Complex{T}}
end

is_AAc_diagonal(L::RDFT) = false #TODO but might be true?
is_invertible(L::RDFT) = true
is_full_row_rank(L::RDFT) = true

# ─── Threading ────────────────────────────────────────────────────────────────
#
# FFTW threads r2c transforms; measured 3.08x at n = 2^22 with 8 threads. Plan-time, so
# `is_threaded` reads the recorded count back rather than switching a loop.
is_threaded(op::RDFT) = op.num_threads > 1
supports_threading(::RDFT) = true

function _copy_operator_impl(op::RDFT{T, N}; storage_type = nothing, threaded = nothing) where {T, N}
    if storage_type !== nothing
        throw(
            ArgumentError(
                "RDFT cannot change storage_type after construction: the FFTW plan is " *
                    "built for a specific array backend. Rebuild the operator instead."
            ),
        )
    end
    new_threaded = threaded === nothing ? is_threaded(op) : threaded
    dims = findfirst(i -> op.dim_out[i] != op.dim_in[i], 1:N)
    dims = dims === nothing ? 1 : dims
    # b2/y2 are per-call scratch and must not be shared with the copy.
    if new_threaded == is_threaded(op)
        return RDFT{T, N, typeof(op.A), typeof(op.At), typeof(op.b2)}(
            op.dim_in, op.dim_out, op.A, op.At, similar(op.b2), similar(op.y2), op.Zp, op.num_threads
        )
    end
    return RDFT(zeros(T, op.dim_in), dims; threaded = new_threaded)
end
