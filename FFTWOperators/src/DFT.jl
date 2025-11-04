export DFT, IDFT

@enum Normalization begin
    UNNORMALIZED
    ORTHO
    FORWARD
    BACKWARD
end

"""
DFT([domain_type=Float64::Type,] dim_in::Tuple [,dims]; [normalization, flags, timelimit, num_threads])
DFT(dim_in...; [normalization, flags, timelimit, num_threads])
DFT(x::AbstractArray [,dims]; [normalization, flags, timelimit, num_threads])

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

```jldoctest
julia> using MriReconstructionToolbox, FFTW

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
struct DFT{N,C,D,Dir,S,T1<:AbstractFFTs.Plan,T2<:AbstractFFTs.Plan,R} <: LinearOperator
    dim_in::NTuple{N,Int}
    A::T1
    At::T2
    normalization::Normalization
    scale::R
end

"""
IDFT([domain_type=Float64::Type,] dim_in::Tuple [,dims]; [flags, timelimit, num_threads])
IDFT(dim_in...; [flags, timelimit, num_threads])
IDFT(x::AbstractArray [,dims]; [flags, timelimit, num_threads])

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

```jldoctest
julia> IDFT(Complex{Float64},(10,10))
ℱ⁻¹  ℂ^(10, 10) -> ℂ^(10, 10)

julia> IDFT(10,10)
ℱ⁻¹  ℂ^(10, 10) -> ℂ^(10, 10)

julia> op = IDFT(ones(ComplexF64, 3))
ℱ⁻¹  ℂ^3 -> ℂ^3

julia> op*ones(3) ≈ FFTW.ifft(ones(3))
true

```
"""
function IDFT end

# Constructors
#standard constructor
function DFT(
    dim_in::NTuple{N,Int},
    dims=1:N;
    normalization::Normalization=UNNORMALIZED,
    flags=FFTW.ESTIMATE,
    timelimit=Inf,
    num_threads=Threads.nthreads(),
) where {N}
    DFT(zeros(dim_in), dims; normalization, flags, timelimit, num_threads)
end

function DFT(
    x::AbstractArray{D,N},
    dims=1:ndims(x);
    normalization::Normalization=UNNORMALIZED,
    flags=FFTW.ESTIMATE,
    timelimit=Inf,
    num_threads=Threads.nthreads(),
) where {N,D<:Real}
    x = similar(x, Complex{D})
    prev_fftw_threads = FFTW.get_num_threads()
    FFTW.set_num_threads(num_threads)
    A = plan_fft(x, dims; flags, timelimit)
    At = plan_bfft(x, dims; flags, timelimit)
    FFTW.set_num_threads(prev_fftw_threads)
    S = typeof(x isa SubArray ? parent(x) : x).name.wrapper
    dims = tuple(dims...)
    scaling = get_scaling(size(x), dims, normalization)
    return DFT{N,Complex{D},D,dims,S,typeof(A),typeof(At),D}(
        size(x), A, At, normalization, scaling
    )
end

function DFT(
    x::AbstractArray{D,N},
    dims=1:ndims(x);
    normalization::Normalization=UNNORMALIZED,
    flags=FFTW.ESTIMATE,
    timelimit=Inf,
    num_threads=Threads.nthreads(),
) where {N,D<:Complex}
    if x != FFTW.ESTIMATE
        x = similar(x) # FFTW.MEASURE and FFTW.PATIENT may cause the input array to be modified
    end
    prev_fftw_threads = FFTW.get_num_threads()
    FFTW.set_num_threads(num_threads)
    A = plan_fft(x, dims; flags, timelimit)
    At = plan_bfft(x, dims; flags, timelimit)
    FFTW.set_num_threads(prev_fftw_threads)
    S = typeof(x isa SubArray ? parent(x) : x).name.wrapper
    dims = tuple(dims...)
    scaling = get_scaling(size(x), dims, normalization)
    return DFT{N,D,D,dims,S,typeof(A),typeof(At),real(D)}(size(x), A, At, normalization, scaling)
end

function DFT(
    T::Type,
    dim_in::NTuple{N,Int},
    dims=1:N;
    normalization::Normalization=UNNORMALIZED,
    flags=FFTW.ESTIMATE,
    timelimit=Inf,
    num_threads=Threads.nthreads(),
) where {N}
    DFT(zeros(T, dim_in), dims; normalization, flags, timelimit, num_threads)
end
function DFT(
    dim_in::Vararg{Int};
    normalization::Normalization=UNNORMALIZED,
    flags=FFTW.ESTIMATE,
    timelimit=Inf,
    num_threads=Threads.nthreads(),
)
    DFT(dim_in; normalization, flags, timelimit, num_threads)
end
function DFT(
    T::Type,
    dim_in::Vararg{Int};
    normalization::Normalization=UNNORMALIZED,
    flags=FFTW.ESTIMATE,
    timelimit=Inf,
    num_threads=Threads.nthreads(),
)
    DFT(T, dim_in; normalization, flags, timelimit, num_threads)
end

#standard constructor
function IDFT(
    T::Type,
    dim_in::NTuple{N,Int},
    dims=1:N;
    normalization::Normalization=BACKWARD,
    flags=FFTW.ESTIMATE,
    timelimit=Inf,
    num_threads=Threads.nthreads(),
) where {N}
    @assert T <: Complex "Input type for IDFT must be a complex type"
    return DFT(T, dim_in, dims; normalization, flags, timelimit, num_threads)'
end

function IDFT(
    x::AbstractArray{D,N},
    dims=1:ndims(x);
    normalization::Normalization=BACKWARD,
    flags=FFTW.ESTIMATE,
    timelimit=Inf,
    num_threads=Threads.nthreads(),
) where {N,D}
    @assert D <: Complex "Input array for IDFT must have complex element type"
    return DFT(x, dims; normalization, flags, timelimit, num_threads)'
end

function IDFT(
    dim_in::NTuple{N,Int},
    dims=1:N;
    normalization::Normalization=BACKWARD,
    flags=FFTW.ESTIMATE,
    timelimit=Inf,
    num_threads=Threads.nthreads(),
) where {N}
    return DFT(ComplexF64, dim_in, dims; normalization, flags, timelimit, num_threads)'
end
function IDFT(
    dim_in::Vararg{Int}; normalization::Normalization=BACKWARD, flags=FFTW.ESTIMATE, timelimit=Inf, num_threads=Threads.nthreads()
)
    return DFT(ComplexF64, dim_in; normalization, flags, timelimit, num_threads)'
end
function IDFT(
    T::Type,
    dim_in::Vararg{Int};
    normalization::Normalization=BACKWARD,
    flags=FFTW.ESTIMATE,
    timelimit=Inf,
    num_threads=Threads.nthreads(),
)
    @assert T <: Complex "Input type for IDFT must be a complex type"
    return DFT(T, dim_in; normalization, flags, timelimit, num_threads)'
end

# Mappings

function mul!(
    y::AbstractArray, L::DFT{N,C,D}, b::AbstractArray
) where {N,C,D<:Complex}
    check(y, L, b)
    mul!(y, L.A, b)
    return scale_output!(y, L)
end

function mul!(
    y::AbstractArray, L::DFT{N,C,D}, b::AbstractArray
) where {N,C,D<:Real}
    check(y, L, b)
    mul!(y, L.A, complex(b))
    return scale_output!(y, L)
end

function mul!(
    y::AbstractArray, L::AdjointOperator{<:DFT{N,C,D}}, b::AbstractArray
) where {N,C,D<:Complex}
    check(y, L, b)
    mul!(y, L.A.At, b)
    return scale_output!(y, L)
end

function mul!(
    y::AbstractArray, L::AdjointOperator{<:DFT{N,C,D}}, b::AbstractArray
) where {N,C,D<:Real}
    check(y, L, b)
    y2 = complex(y)
    mul!(y2, L.A.At, b)
    y .= real.(y2)
    return scale_output!(y, L)
end

# Properties

size(L::DFT) = (L.dim_in, L.dim_in)
function domain_storage_type(
    ::DFT{N,C,D,Dir,S}
) where {N,C,D,Dir,S}
    S{D}
end
function codomain_storage_type(
    ::DFT{N,C,D,Dir,S}
) where {N,C,D,Dir,S}
    S{C}
end

fun_name(A::DFT) = "ℱ"
fun_name(A::AdjointOperator{<:DFT}) = A.A.normalization == UNNORMALIZED ? "ℱᵃ" : "ℱ⁻¹"

domain_type(::DFT{N,C,D}) where {N,C,D} = D
codomain_type(::DFT{N,C,D}) where {N,C,D} = C
is_thread_safe(::DFT) = true

is_AcA_diagonal(L::DFT) = true
is_AAc_diagonal(L::DFT) = true
is_invertible(L::DFT) = true
is_full_row_rank(L::DFT) = true
is_full_column_rank(L::DFT) = true

function diag_AcA(L::DFT{N,C,D,Dir,S}) where {N,C,D,Dir,S}
    if L.normalization == UNNORMALIZED
        get_scaling(size(L, 1), Dir, FORWARD)
    else
        one(real(C))
    end
end
function diag_AAc(L::DFT{N,C,D,Dir,S}) where {N,C,D,Dir,S}
    if L.normalization == UNNORMALIZED
        get_scaling(size(L, 2), Dir, FORWARD)
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

function get_scaling(dim_in, dirs, normalization)
    if normalization == UNNORMALIZED
        return one(Float64)
    elseif normalization == ORTHO
        return sqrt(prod(dim_in[collect(dirs)]))
    elseif normalization == FORWARD || normalization == BACKWARD
        return prod(dim_in[collect(dirs)])
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
