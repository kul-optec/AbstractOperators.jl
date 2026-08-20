export DFT, IDFT

@enum Normalization begin
    UNNORMALIZED
    ORTHO
    FORWARD
    BACKWARD
end

"""
DFT([domain_type=Float64::Type,] dim_in::Tuple [,dims]; [normalization, flags, timelimit, num_threads, threaded])
DFT(dim_in...; [normalization, flags, timelimit, num_threads, threaded])
DFT(x::AbstractArray [,dims]; [normalization, flags, timelimit, num_threads, threaded])

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the `N`-dimensional Discrete
Fourier Transform over dimensions `dims` of `x`.

Arguments:
- `domain_type`: The type of the input array. Defaults to `Float64`.
- `dim_in`: The dimensions of the input array. If `dim_in` is a tuple, it specifies the size of the input array.
  If `dim_in` is a type, it specifies the type of the input array.
- `x`: An input array. If provided, the dimensions of `x` will be used as `dim_in`.
- `dims`: The dimensions over which to perform the Fourier Transform. Defaults to all dimensions of `x`.
- `flags`: FFTW flags for the plan. Defaults to `FFTW.ESTIMATE`.
- `normalization`: The normalization scheme to use. The options are:
    - `UNNORMALIZED`: No normalization is applied (default).
    - `ORTHO`: Orthogonal normalization, scaling by `1/sqrt(N)` both forward and backward transforms.
    - `FORWARD`: Forward normalization, scaling by `1/N`.
    - `BACKWARD`: Backward normalization, scaling by `1/N`.
- `timelimit`: The maximum time in seconds to spend on planning the FFT. Defaults to `Inf`, which means no time limit.
  If set to a finite value, the plan will be created within that time limit, potentially resulting in a less optimal plan.
- `num_threads`: The number of threads to use for the FFT. Defaults to the number of threads available in Julia.
  It should be set to 1 for single-threaded execution (e.g., when using within a `@threads` block).
- `threaded`: The package-wide spelling of the same choice: `true` uses the available Julia
  threads, `false` uses one. `num_threads` wins if both are given. FFTW is a counted thread
  pool, so this is fixed when the plan is built and reported back by `is_threaded`.

```jldoctest
julia> using FFTW, FFTWOperators

julia> DFT(Complex{Float64},(10,10))
ℱ  ℂ^(10, 10) -> ℂ^(10, 10)

julia> DFT(10,10)
ℱ  ℝ^(10, 10) -> ℂ^(10, 10)

julia> op = DFT(ones(3))
ℱ  ℝ^3 -> ℂ^3

julia> op*ones(3) ≈ FFTW.fft(ones(3))
true
```
"""
struct DFT{N, C, D, Dir, S, T1 <: AbstractFFTs.Plan, T2 <: AbstractFFTs.Plan, R} <: LinearOperator
    dim_in::NTuple{N, Int}
    A::T1
    At::T2
    normalization::Normalization
    scale::R
    # The thread count the FFTW plans were built with. FFTW is a *counted* pool: threading
    # is a property of the plan, decided at construction, not of the `mul!` loop -- so it
    # is recorded rather than switchable, and `is_threaded` reads it back.
    num_threads::Int
end

"""
IDFT([domain_type=Float64::Type,] dim_in::Tuple [,dims]; [flags, timelimit, num_threads, threaded])
IDFT(dim_in...; [flags, timelimit, num_threads, threaded])
IDFT(x::AbstractArray [,dims]; [flags, timelimit, num_threads, threaded])

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the `N`-dimensional
Inverse Discrete Fourier Transform over dimensions `dims` of `x`.

Arguments:
- `domain_type`: The type of the input array. Defaults to `Float64`.
- `dim_in`: The dimensions of the input array. If `dim_in` is a tuple, it specifies the size of the input array.
  If `dim_in` is a type, it specifies the type of the input array.
- `x`: An input array. If provided, the dimensions of `x` will be used as `dim_in`.
- `dims`: The dimensions over which to perform the Inverse Fourier Transform. Defaults to all dimensions of `x`.
- `normalization`: The normalization scheme to use. The options are:
    - `UNNORMALIZED`: No normalization is applied.
    - `ORTHO`: Orthogonal normalization, scaling by `1/sqrt(N)` both forward and backward transforms.
    - `FORWARD`: Forward normalization, scaling by `1/N`.
    - `BACKWARD`: Backward normalization, scaling by `1/N` (default).
- `flags`: FFTW flags for the plan. Defaults to `FFTW.ESTIMATE`.
- `timelimit`: The maximum time in seconds to spend on planning the FFT. Defaults to `Inf`, which means no time limit.
  If set to a finite value, the plan will be created within that time limit, potentially resulting in a less optimal plan.
- `num_threads`: The number of threads to use for the FFT. Defaults to the number of threads available in Julia.
  It should be set to 1 for single-threaded execution (e.g., when using within a `@threads` block).
- `threaded`: The package-wide spelling of the same choice: `true` uses the available Julia
  threads, `false` uses one. `num_threads` wins if both are given. FFTW is a counted thread
  pool, so this is fixed when the plan is built and reported back by `is_threaded`.

```jldoctest
julia> using FFTW, FFTWOperators

julia> IDFT(Complex{Float64},(10,10))
ℱ⁻¹  ℂ^(10, 10) -> ℂ^(10, 10)

julia> IDFT(10,10)
ℱ⁻¹  ℂ^(10, 10) -> ℂ^(10, 10)

julia> op = IDFT(ones(ComplexF64, 3))
ℱ⁻¹  ℂ^3 -> ℂ^3

julia> op*ones(ComplexF64, 3) ≈ FFTW.ifft(ones(ComplexF64, 3))
true

```
"""
function IDFT end

# Constructors
#standard constructor
function DFT(
        dim_in::NTuple{N, Int},
        dims = 1:N;
        normalization::Normalization = UNNORMALIZED,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    ) where {N}
    return DFT(zeros(dim_in), dims; normalization, flags, timelimit, num_threads, threaded)
end

function DFT(
        x::AbstractArray{D, N},
        dims = 1:ndims(x);
        normalization::Normalization = UNNORMALIZED,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    ) where {N, D <: Real}
    x = similar(x, Complex{D})
    num_threads = _fftw_num_threads(:c2c, num_threads, threaded, length(x))
    prev_fftw_threads = FFTW.get_num_threads()
    FFTW.set_num_threads(num_threads)
    A = plan_fft(x, dims; flags, timelimit)
    At = plan_bfft(x, dims; flags, timelimit)
    FFTW.set_num_threads(prev_fftw_threads)
    S = typeof(x isa SubArray ? parent(x) : x).name.wrapper
    dims = tuple(dims...)
    scaling = _dft_scaling(size(x), dims, normalization)
    return DFT{N, Complex{D}, D, dims, S, typeof(A), typeof(At), Float64}(
        size(x), A, At, normalization, scaling, num_threads
    )
end

function DFT(
        x::AbstractArray{D, N},
        dims = 1:ndims(x);
        normalization::Normalization = UNNORMALIZED,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    ) where {N, D <: Complex}
    if x != FFTW.ESTIMATE
        x = similar(x) # FFTW.MEASURE and FFTW.PATIENT may cause the input array to be modified
    end
    num_threads = _fftw_num_threads(:c2c, num_threads, threaded, length(x))
    prev_fftw_threads = FFTW.get_num_threads()
    FFTW.set_num_threads(num_threads)
    A = plan_fft(x, dims; flags, timelimit)
    At = plan_bfft(x, dims; flags, timelimit)
    FFTW.set_num_threads(prev_fftw_threads)
    S = typeof(x isa SubArray ? parent(x) : x).name.wrapper
    dims = tuple(dims...)
    scaling = _dft_scaling(size(x), dims, normalization)
    return DFT{N, D, D, dims, S, typeof(A), typeof(At), Float64}(
        size(x), A, At, normalization, scaling, num_threads
    )
end

function DFT(
        T::Type,
        dim_in::NTuple{N, Int},
        dims = 1:N;
        normalization::Normalization = UNNORMALIZED,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    ) where {N}
    return DFT(zeros(T, dim_in), dims; normalization, flags, timelimit, num_threads, threaded)
end
function DFT(
        dim_in::Vararg{Int};
        normalization::Normalization = UNNORMALIZED,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    )
    return DFT(dim_in; normalization, flags, timelimit, num_threads, threaded)
end
function DFT(
        T::Type,
        dim_in::Vararg{Int};
        normalization::Normalization = UNNORMALIZED,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    )
    return DFT(T, dim_in; normalization, flags, timelimit, num_threads, threaded)
end

#standard constructor
function IDFT(
        T::Type,
        dim_in::NTuple{N, Int},
        dims = 1:N;
        normalization::Normalization = BACKWARD,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    ) where {N}
    @assert T <: Complex "Input type for IDFT must be a complex type"
    return DFT(T, dim_in, dims; normalization, flags, timelimit, num_threads, threaded)'
end

function IDFT(
        x::AbstractArray{D, N},
        dims = 1:ndims(x);
        normalization::Normalization = BACKWARD,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    ) where {N, D}
    @assert D <: Complex "Input array for IDFT must have complex element type"
    return DFT(x, dims; normalization, flags, timelimit, num_threads, threaded)'
end

function IDFT(
        dim_in::NTuple{N, Int},
        dims = 1:N;
        normalization::Normalization = BACKWARD,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    ) where {N}
    return DFT(ComplexF64, dim_in, dims; normalization, flags, timelimit, num_threads, threaded)'
end
function IDFT(
        dim_in::Vararg{Int};
        normalization::Normalization = BACKWARD,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    )
    return DFT(ComplexF64, dim_in; normalization, flags, timelimit, num_threads, threaded)'
end
function IDFT(
        T::Type,
        dim_in::Vararg{Int};
        normalization::Normalization = BACKWARD,
        flags = FFTW.ESTIMATE,
        timelimit = Inf,
        num_threads = nothing,
        threaded::Bool = true,
    )
    @assert T <: Complex "Input type for IDFT must be a complex type"
    return DFT(T, dim_in; normalization, flags, timelimit, num_threads, threaded)'
end

# Mappings

function mul!(
        y::AbstractArray, L::DFT{N, C, D}, b::AbstractArray
    ) where {N, C, D <: Complex}
    check(y, L, b)
    mul!(y, L.A, b)
    return scale_output!(y, L)
end

function mul!(
        y::AbstractArray, L::DFT{N, C, D}, b::AbstractArray
    ) where {N, C, D <: Real}
    check(y, L, b)
    mul!(y, L.A, complex(b))
    return scale_output!(y, L)
end

function mul!(
        y::AbstractArray, L::AdjointOperator{<:DFT{N, C, D}}, b::AbstractArray
    ) where {N, C, D <: Complex}
    check(y, L, b)
    mul!(y, L.A.At, b)
    return scale_output!(y, L)
end

function mul!(
        y::AbstractArray, L::AdjointOperator{<:DFT{N, C, D}}, b::AbstractArray
    ) where {N, C, D <: Real}
    check(y, L, b)
    y2 = complex(y)
    mul!(y2, L.A.At, b)
    y .= real.(y2)
    return scale_output!(y, L)
end

# Properties

size(L::DFT) = (L.dim_in, L.dim_in)
function domain_array_type(
        ::DFT{N, C, D, Dir, S}
    ) where {N, C, D, Dir, S}
    return S{D}
end
function codomain_array_type(
        ::DFT{N, C, D, Dir, S}
    ) where {N, C, D, Dir, S}
    return S{C}
end

fun_name(A::DFT) = "ℱ"
fun_name(A::AdjointOperator{<:DFT}) = A.A.normalization == UNNORMALIZED ? "ℱᵃ" : "ℱ⁻¹"

domain_type(::DFT{N, C, D}) where {N, C, D} = D
codomain_type(::DFT{N, C, D}) where {N, C, D} = C
is_thread_safe(::DFT) = true

is_AcA_diagonal(L::DFT) = true
is_AAc_diagonal(L::DFT) = true
is_invertible(L::DFT) = true
is_full_row_rank(L::DFT) = true
is_full_column_rank(L::DFT) = true

function diag_AcA(L::DFT{N, C, D, Dir, S}) where {N, C, D, Dir, S}
    return if L.normalization == UNNORMALIZED
        _dft_scaling(size(L, 1), Dir, FORWARD)
    else
        one(real(C))
    end
end
function diag_AAc(L::DFT{N, C, D, Dir, S}) where {N, C, D, Dir, S}
    return if L.normalization == UNNORMALIZED
        _dft_scaling(size(L, 2), Dir, FORWARD)
    else
        one(real(C))
    end
end

has_optimized_normalop(L::DFT) = true
has_optimized_normalop(L::AdjointOperator{<:DFT}) = true
get_normal_op(L::DFT) = diag_AcA(L) * Eye(size(L, 1))
get_normal_op(L::AdjointOperator{<:DFT}) = diag_AAc(L) * Eye(size(L, 1))

AbstractOperators.has_fast_opnorm(::DFT) = true
AbstractOperators.has_fast_opnorm(::AdjointOperator{<:DFT}) = true
LinearAlgebra.opnorm(L::DFT) = sqrt(diag_AcA(L))
LinearAlgebra.opnorm(L::AdjointOperator{<:DFT}) = sqrt(diag_AAc(L))

# Utils

function _dft_scaling(dim_in, dirs, normalization::Normalization)::Float64
    if normalization == UNNORMALIZED
        return 1.0
    elseif normalization == ORTHO
        return float(sqrt(prod(dim_in[collect(dirs)])))
    elseif normalization == FORWARD || normalization == BACKWARD
        return float(prod(dim_in[collect(dirs)]))
    else
        throw(ArgumentError("Invalid normalization type"))
    end
end

function scale_output!(y, L::DFT)
    if L.normalization == FORWARD || L.normalization == ORTHO
        y ./= L.scale
    end
    return y
end

function scale_output!(y, L::AdjointOperator{<:DFT})
    if L.A.normalization == BACKWARD || L.A.normalization == ORTHO
        y ./= L.A.scale
    end
    return y
end

# ─── Threading ────────────────────────────────────────────────────────────────

is_threaded(op::DFT) = op.num_threads > 1
supports_threading(::DFT) = true

function _copy_operator_impl(
        op::DFT{N, C, D, Dir, S, T1, T2, R}; storage_type = nothing, threaded = nothing
    ) where {N, C, D, Dir, S, T1, T2, R}
    new_threaded = threaded === nothing ? is_threaded(op) : threaded
    if storage_type === nothing && new_threaded == is_threaded(op)
        # Plans are immutable and hold no per-call scratch, so a copy shares them.
        return DFT{N, C, D, Dir, S, T1, T2, R}(
            op.dim_in, op.A, op.At, op.normalization, op.scale, op.num_threads
        )
    end
    # Changing the storage type or thread count means replanning. The prototype must use
    # the *domain* element type -- type parameter `D`, not the codomain `C`: for a
    # real-input DFT they differ (`C == Complex{D}`), and planning from `C` would silently
    # produce an operator with a complex domain. There is no persistent data to carry over
    # (a DFT holds only plans, not an array of values), so the prototype can be
    # uninitialized -- unlike, say, `Conv`'s `h`.
    new_storage = storage_type === nothing ? S : storage_type
    return DFT(
        similar(new_storage{D}, op.dim_in), Dir;
        normalization = op.normalization, threaded = new_threaded,
    )
end
