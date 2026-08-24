export FFTShift,
    IFFTShift, SignAlternation, fftshift_op, ifftshift_op, alternate_sign, alternate_sign!

abstract type ShiftOp <: LinearOperator end

"""
    FFTShift([T::Type=Float64,] dim_in::Tuple, [dirs])
    FFTShift(dim_in...)

Creates a `LinearOperator` that permutes the array like `FFTW.fftshift` over the given `dirs`.
`dirs` must contain at least one dimension; each must be within `1:length(dim_in)`.

```jldoctest
julia> using FFTWOperators

julia> A = FFTShift((4,)); x = collect(Float64, 1:4);

julia> A * x
4-element Vector{Float64}:
 3.0
 4.0
 1.0
 2.0

julia> B = IFFTShift((4,)); B * (A * x) == x
true

julia> A2 = FFTShift((2,2), (1,2)); A2 * collect(reshape(1.0:4.0, 2, 2))
2×2 Matrix{Float64}:
 4.0  2.0
 3.0  1.0
```
"""
struct FFTShift{T, N, M, S <: AbstractArray{T}} <: ShiftOp
    dim_in::NTuple{N, Int}
    dirs::NTuple{M, Int}
end

"""
    IFFTShift([T::Type=Float64,] dim_in::Tuple, [dirs])
    IFFTShift(dim_in...)

Creates a `LinearOperator` that permutes the array like `FFTW.ifftshift` over `dirs`.
`dirs` must contain at least one dimension; each must be within `1:length(dim_in)`.

```jldoctest
julia> using FFTWOperators

julia> A = FFTShift((4,)); B = IFFTShift((4,)); x = collect(1.0:4.0);

julia> B * (A * x) == x
true

julia> B2 = IFFTShift((2,2), (1,2)); B2 * [4.0 3.0; 2.0 1.0] == [1.0 2.0; 3.0 4.0]
true
```
"""
struct IFFTShift{T, N, M, S <: AbstractArray{T}} <: ShiftOp
    dim_in::NTuple{N, Int}
    dirs::NTuple{M, Int}
end

"""
    SignAlternation([T::Type=Float64,] dim_in::Tuple, dirs)

Creates a `LinearOperator` that multiplies entries by -1 on indices where the parity sum across `dirs` is odd.
`dirs` must contain at least one dimension; each must be within `1:length(dim_in)`.

# Why is it useful?
Due to the properties of the discrete Fourier transform, when there is an even number of points along a dimension,
alternating the sign of the entries along the domain of the Fourier transform (i.e., multiplying by -1 at every
other index) is equivalent to a half-sample shift in the frequency domain.
One can see this by applying the shift theorem of the Fourier transform:  
if ``\\mathcal{F}({x_n})_k = X_k``  
then ``\\mathcal{F}({x_n \\cdot e^{\\frac{i 2\\pi}{N}n m}})_k = \\mathcal{F}({x_n \\cdot (-1)^n })_k = X_{k - N/2}``  
where `N` is the number of samples along that dimension.

# Examples
```jldoctest
julia> using FFTWOperators

julia> S = SignAlternation((4,), 1); S * collect(1.0:4.0)
4-element Vector{Float64}:
  1.0
 -2.0
  3.0
 -4.0

julia> S2 = SignAlternation((2,2), (1,2)); S2 * ones(2,2)
2×2 Matrix{Float64}:
  1.0  -1.0
 -1.0   1.0
```
"""
struct SignAlternation{T, N, M, Th, S <: AbstractArray{T}} <: LinearOperator
    dim_in::NTuple{N, Int}
    dirs::NTuple{M, Int}
end

function _normalize_dirs(::NTuple{N, Int}, dirs) where {N}
    d = Tuple(dirs)
    isempty(d) && throw(ArgumentError("dirs must contain at least one dimension"))
    all(1 <= x <= N for x in d) || throw(ArgumentError("dirs must be in 1:$N"))
    return d
end

function _make_shift(::Type{Op}, ::Type{T}, dim_in::NTuple{N, Int}, dirs, array_type::Type) where {Op, T <: Number, N}
    d = _normalize_dirs(dim_in, dirs)
    S = _normalize_array_type(array_type, T)
    return Op{T, N, length(d), S}(dim_in, d)
end

function FFTShift(
        ::Type{T}, dim_in::NTuple{N, Int}, dirs;
        array_type::Type{<:AbstractArray} = Array{T},
    ) where {T <: Number, N}
    return _make_shift(FFTShift, T, dim_in, dirs, array_type)
end
function FFTShift(
        ::Type{T}, dim_in::NTuple{N, Int};
        array_type::Type{<:AbstractArray} = Array{T},
    ) where {T <: Number, N}
    return FFTShift(T, dim_in, Tuple(1:N); array_type)
end
function FFTShift(dim_in::NTuple{N, Int}, dirs; array_type::Type{<:AbstractArray} = Array{Float64}) where {N}
    return FFTShift(Float64, dim_in, dirs; array_type)
end
function FFTShift(dim_in::NTuple{N, Int}; array_type::Type{<:AbstractArray} = Array{Float64}) where {N}
    return FFTShift(Float64, dim_in, Tuple(1:N); array_type)
end
FFTShift(dim_in::Vararg{Int}) = FFTShift(dim_in)
function FFTShift(x::A) where {A <: AbstractArray}
    return FFTShift(eltype(x), size(x); array_type = typeof(x isa SubArray ? parent(x) : x))
end

function IFFTShift(
        ::Type{T}, dim_in::NTuple{N, Int}, dirs;
        array_type::Type{<:AbstractArray} = Array{T},
    ) where {T <: Number, N}
    return _make_shift(IFFTShift, T, dim_in, dirs, array_type)
end
function IFFTShift(
        ::Type{T}, dim_in::NTuple{N, Int};
        array_type::Type{<:AbstractArray} = Array{T},
    ) where {T <: Number, N}
    return IFFTShift(T, dim_in, Tuple(1:N); array_type)
end
function IFFTShift(dim_in::NTuple{N, Int}, dirs; array_type::Type{<:AbstractArray} = Array{Float64}) where {N}
    return IFFTShift(Float64, dim_in, dirs; array_type)
end
function IFFTShift(dim_in::NTuple{N, Int}; array_type::Type{<:AbstractArray} = Array{Float64}) where {N}
    return IFFTShift(Float64, dim_in, Tuple(1:N); array_type)
end
IFFTShift(dim_in::Vararg{Int}) = IFFTShift(dim_in)
function IFFTShift(x::A) where {A <: AbstractArray}
    return IFFTShift(eltype(x), size(x); array_type = typeof(x isa SubArray ? parent(x) : x))
end

function SignAlternation(
        ::Type{T}, dim_in::NTuple{N, Int}, dirs;
        threaded::Bool = true, array_type::Type{<:AbstractArray} = Array{T},
    ) where {T <: Number, N}
    d = _normalize_dirs(dim_in, dirs)
    S = _normalize_array_type(array_type, T)
    # Routed through the shared resolver like every other operator: `false` vetoes, `true`
    # enables subject to the policy, which also covers the thread count and GPU storage.
    th = _elementwise_threaded(SignAlternation, threaded, T, dim_in, S)
    return SignAlternation{T, N, length(d), th, S}(dim_in, d)
end
function SignAlternation(
        dim_in::NTuple{N, Int}, dirs;
        threaded::Bool = true, array_type::Type{<:AbstractArray} = Array{Float64},
    ) where {N}
    return SignAlternation(Float64, dim_in, dirs; threaded, array_type)
end
function SignAlternation(x::A, dirs; threaded::Bool = true) where {A <: AbstractArray}
    return SignAlternation(
        eltype(x), size(x), dirs;
        threaded, array_type = typeof(x isa SubArray ? parent(x) : x),
    )
end

domain_type(::FFTShift{T}) where {T} = T
codomain_type(::FFTShift{T}) where {T} = T
domain_array_type(::FFTShift{T, N, M, S}) where {T, N, M, S} = S
codomain_array_type(::FFTShift{T, N, M, S}) where {T, N, M, S} = S
domain_type(::IFFTShift{T}) where {T} = T
codomain_type(::IFFTShift{T}) where {T} = T
domain_array_type(::IFFTShift{T, N, M, S}) where {T, N, M, S} = S
codomain_array_type(::IFFTShift{T, N, M, S}) where {T, N, M, S} = S
domain_type(::SignAlternation{T}) where {T} = T
codomain_type(::SignAlternation{T}) where {T} = T
domain_array_type(::SignAlternation{T, N, M, Th, S}) where {T, N, M, Th, S} = S
codomain_array_type(::SignAlternation{T, N, M, Th, S}) where {T, N, M, Th, S} = S

size(L::FFTShift) = (L.dim_in, L.dim_in)
size(L::IFFTShift) = (L.dim_in, L.dim_in)
size(L::SignAlternation) = (L.dim_in, L.dim_in)

fun_name(::FFTShift) = "⇉"
fun_name(::IFFTShift) = "⇇"
fun_name(::SignAlternation) = "±"

is_thread_safe(::FFTShift) = true
is_thread_safe(::IFFTShift) = true
is_thread_safe(::SignAlternation) = true

is_AcA_diagonal(::FFTShift) = true
is_AAc_diagonal(::FFTShift) = true
diag_AcA(L::FFTShift) = one(real(domain_type(L)))
diag_AAc(L::FFTShift) = one(real(domain_type(L)))
is_symmetric(L::FFTShift) = all(d -> iseven(L.dim_in[d]), L.dirs)
is_orthogonal(::FFTShift) = true
is_invertible(::FFTShift) = true
is_full_row_rank(::FFTShift) = true
is_full_column_rank(::FFTShift) = true

is_AcA_diagonal(::IFFTShift) = true
is_AAc_diagonal(::IFFTShift) = true
diag_AcA(L::IFFTShift) = one(real(domain_type(L)))
diag_AAc(L::IFFTShift) = one(real(domain_type(L)))
is_symmetric(L::IFFTShift) = all(d -> iseven(L.dim_in[d]), L.dirs)
is_orthogonal(::IFFTShift) = true
is_invertible(::IFFTShift) = true
is_full_row_rank(::IFFTShift) = true
is_full_column_rank(::IFFTShift) = true

is_AcA_diagonal(::SignAlternation) = true
is_AAc_diagonal(::SignAlternation) = true
diag_AcA(L::SignAlternation) = one(real(domain_type(L)))
diag_AAc(L::SignAlternation) = one(real(domain_type(L)))
is_symmetric(::SignAlternation) = true
is_orthogonal(::SignAlternation) = true
is_invertible(::SignAlternation) = true
is_full_row_rank(::SignAlternation) = true
is_full_column_rank(::SignAlternation) = true

has_fast_opnorm(::Union{FFTShift, IFFTShift, SignAlternation}) = true
function LinearAlgebra.opnorm(L::Union{FFTShift, IFFTShift, SignAlternation})
    return one(real(domain_type(L)))
end

function mul!(y::AbstractArray, L::FFTShift, b::AbstractArray)
    check(y, L, b)
    return FFTW.fftshift!(y, b, L.dirs)
end

function mul!(y::AbstractArray, L::IFFTShift, b::AbstractArray)
    check(y, L, b)
    return FFTW.ifftshift!(y, b, L.dirs)
end

function mul!(y::AbstractArray, L::AdjointOperator{<:FFTShift}, b::AbstractArray)
    check(y, L, b)
    return FFTW.ifftshift!(y, b, L.A.dirs)
end

function mul!(y::AbstractArray, L::AdjointOperator{<:IFFTShift}, b::AbstractArray)
    check(y, L, b)
    return FFTW.fftshift!(y, b, L.A.dirs)
end

"""
        alternate_sign!(x[, dirs...])
In-place sign alternation across specified dimensions. Flips the sign where the parity sum across `dirs` is odd.
The provided `dirs` must be within `1:ndims(x)`, and sorted in ascending order. They can also be provided as a tuple.

```jldoctest
julia> using FFTWOperators

julia> v = collect(1.0:4.0);

julia> alternate_sign!(v, 1)
4-element Vector{Float64}:
  1.0
 -2.0
  3.0
 -4.0

julia> M = ones(2,2); alternate_sign!(M, 1, 2)
2×2 Matrix{Float64}:
  1.0  -1.0
 -1.0   1.0

julia> M = ones(2,2); alternate_sign!(M, (1, 2))
2×2 Matrix{Float64}:
  1.0  -1.0
 -1.0   1.0
```
"""
function alternate_sign!(x::AbstractArray, dirs::Int...; threaded::Bool = true)
    return alternate_sign!(x, tuple(dirs...); threaded)
end

function alternate_sign!(
        x::AbstractArray, dirs::NTuple{M, Int}; threaded::Bool = true
    ) where {M}
    _check_shift_dirs(size(x), dirs)
    return _alternate_sign!(x, dirs; threaded)
end

function _alternate_sign!(
        x::AbstractArray, dirs::NTuple{M, Int}; threaded::Bool = true
    ) where {M}
    if isempty(dirs)
        return x
    end
    sz = size(x)
    if threaded && Threads.nthreads() > 1
        @inbounds @batch for I in CartesianIndices(sz)
            flips = sum(iseven(I[d]) ? 1 : 0 for d in dirs)
            if isodd(flips)
                x[I] = -x[I]
            end
        end
        return x
    else
        @inbounds for I in CartesianIndices(sz)
            flips = sum(iseven(I[d]) ? 1 : 0 for d in dirs)
            if isodd(flips)
                x[I] = -x[I]
            end
        end
        return x
    end
end

"""
    alternate_sign!(y, x[, dirs...])
Out-of-place variant: apply sign alternation of `x` across `dirs` and store the result in `y`.
`y` and `x` must have the same size. The provided `dirs` must be within `1:ndims(x)`, and
sorted in ascending order. They can also be provided as a tuple.

```jldoctest
julia> using FFTWOperators

julia> x = reshape(1.0:4.0, 2, 2); y = similar(x);

julia> alternate_sign!(y, x, 1, 2)
2×2 Matrix{Float64}:
  1.0  -3.0
 -2.0   4.0

julia> alternate_sign!(y, x, (1, 2))
2×2 Matrix{Float64}:
  1.0  -3.0
 -2.0   4.0
```
"""
function alternate_sign!(
        y::AbstractArray, x::AbstractArray, dirs::Int...; threaded::Bool = true
    )
    return alternate_sign!(y, x, tuple(dirs...); threaded)
end
function alternate_sign!(
        y::AbstractArray, x::AbstractArray, dirs::NTuple{M, Int}; threaded::Bool = true
    ) where {M}
    _check_shift_dirs(size(x), dirs)
    return _alternate_sign!(y, x, dirs; threaded)
end

function _alternate_sign!(
        y::AbstractArray, x::AbstractArray, dirs::NTuple{M, Int}; threaded::Bool = true
    ) where {M}
    size(y) == size(x) || throw(ArgumentError("y and x must have the same size"))
    if isempty(dirs)
        y .= x
        return y
    end
    sz = size(x)
    if threaded && Threads.nthreads() > 1
        @inbounds @batch for I in CartesianIndices(sz)
            flips = sum(iseven(I[d]) ? 1 : 0 for d in dirs)
            y[I] = isodd(flips) ? -x[I] : x[I]
        end
        return y
    else
        @inbounds for I in CartesianIndices(sz)
            flips = sum(iseven(I[d]) ? 1 : 0 for d in dirs)
            y[I] = isodd(flips) ? -x[I] : x[I]
        end
    end
    return y
end

"""
    alternate_sign(x[, dirs...])
Returns a copy with sign alternation across specified dimensions.

```jldoctest
julia> using FFTWOperators

julia> alternate_sign(collect(1.0:4.0), 1)
4-element Vector{Float64}:
  1.0
 -2.0
  3.0
 -4.0
```
"""
function alternate_sign(x::AbstractArray, dirs::Int...; threaded::Bool = true)
    return alternate_sign!(copy(x), dirs...; threaded)
end

function mul!(
        y::AbstractArray, L::SignAlternation{T, N, M, Th}, b::AbstractArray
    ) where {T, N, M, Th}
    check(y, L, b)
    return _alternate_sign!(y, b, L.dirs; threaded = Th)
end

has_optimized_normalop(::Union{FFTShift, IFFTShift, SignAlternation}) = true
function get_normal_op(L::Union{FFTShift, IFFTShift, SignAlternation})
    return Eye(domain_type(L), size(L, 1); array_type = domain_array_type(L))
end

LinearAlgebra.adjoint(L::SignAlternation) = L

# Utility

function _check_shift_dirs(::NTuple{N, Int}, dirs::NTuple{M, Int}) where {N, M}
    if M == 0
        throw(ArgumentError("dirs must contain at least one dimension"))
    end
    if N < M
        throw(ArgumentError("Number of dirs exceeds number of dimensions of x"))
    end
    if dirs[1] < 1 || dirs[end] > N
        throw(ArgumentError("dirs must be in 1:$N"))
    end
    for i in 2:M
        if dirs[i] < dirs[i - 1]
            throw(ArgumentError("dirs must be sorted in ascending order"))
        end
    end
    return nothing
end

function _is_dft_op(op, side)
    if op isa DFT || op isa AdjointOperator{<:DFT}
        return true
    elseif op isa Compose
        subops = AbstractOperators.get_operators(op)
        if side == :domain
            # Domain shift: innermost (first) op must be DFT-like; all outer ops must be
            # diagonal so they commute with SignAlternation.
            return _is_dft_op(first(subops), side) && all(is_diagonal, subops[2:end])
        else
            # Codomain shift: outermost (last) op must be DFT-like; all inner ops must be
            # diagonal so they commute with SignAlternation.
            return _is_dft_op(last(subops), side) && all(is_diagonal, subops[1:(end - 1)])
        end
    else
        return false
    end
end

function _shift_op(
        shift_op_type, op::AbstractOperator, domain_shifts::Tuple = (), codomain_shifts::Tuple = ()
    )
    if !isempty(domain_shifts)
        _check_shift_dirs(size(op, 2), domain_shifts)
        shifted_domain_dims_shape = size(op, 2)[collect(domain_shifts)]
        if all(iseven, shifted_domain_dims_shape) && _is_dft_op(op, :domain)
            domain_op = SignAlternation(codomain_type(op), size(op, 1), domain_shifts; array_type = codomain_array_type(op))
            op = domain_op * op
        else
            domain_op = shift_op_type(domain_type(op), size(op, 2), domain_shifts; array_type = domain_array_type(op))
            op = op * domain_op
        end
    end
    if !isempty(codomain_shifts)
        _check_shift_dirs(size(op, 1), codomain_shifts)
        shifted_codomain_dims_shape = size(op, 1)[collect(codomain_shifts)]
        if all(iseven, shifted_codomain_dims_shape) && _is_dft_op(op, :codomain)
            codomain_op = SignAlternation(domain_type(op), size(op, 2), codomain_shifts; array_type = domain_array_type(op))
            op = op * codomain_op
        else
            codomain_op = shift_op_type(codomain_type(op), size(op, 1), codomain_shifts; array_type = codomain_array_type(op))
            op = codomain_op * op
        end
    end
    return op
end

"""
    fftshift_op(op::AbstractOperator; domain_shifts::Tuple=(), codomain_shifts::Tuple=())

Applies `FFTShift` to the domain and/or codomain of `op`, depending on `domain_shifts` and `codomain_shifts`.
If the shifted dimensions all have even length and `op` is a (possibly modified) DFT/IDFT, the `FFTShift` is replaced by a `SignAlternation`
on the other side of the DFT/IDFT.

# Examples
```jldoctest
julia> using FFTWOperators, FFTW

julia> x = rand(15);

julia> F = fftshift_op(DFT(15); domain_shifts=(1,))
ℱ*⇉  ℝ^15 -> ℂ^15

julia> F * x ≈ FFTW.fft(FFTW.fftshift(x))
true

julia> F = fftshift_op(DFT(15); codomain_shifts=(1,))
⇉*ℱ  ℝ^15 -> ℂ^15

julia> F * x ≈ FFTW.fftshift(FFTW.fft(x))
true

julia> F = fftshift_op(DFT(15); domain_shifts=(1,), codomain_shifts=(1,))
Π  ℝ^15 -> ℂ^15

julia> F * x ≈ FFTW.fftshift(FFTW.fft(FFTW.fftshift(x)))
true

julia> F = fftshift_op(DFT(16); codomain_shifts=(1,)) # note that 16 is even, so we get a SignAlternation (±)
ℱ*±  ℝ^16 -> ℂ^16


```
"""
function fftshift_op(
        op::AbstractOperator; domain_shifts::Tuple = (), codomain_shifts::Tuple = ()
    )
    return _shift_op(FFTShift, op, domain_shifts, codomain_shifts)
end

"""
    ifftshift_op(op::AbstractOperator; domain_shifts::Tuple=(), codomain_shifts::Tuple=())

Applies `IFFTShift` to the domain and/or codomain of `op`, depending on `domain_shifts` and `codomain_shifts`.
If the shifted dimensions all have even length and `op` is a (possibly modified) DFT/IDFT, the `IFFTShift` is replaced by a `SignAlternation`
on the other side of the DFT/IDFT.

# Examples
```jldoctest
julia> using FFTWOperators, FFTW

julia> x = rand(ComplexF64, 15);

julia> F = ifftshift_op(IDFT(15); domain_shifts=(1,))
ℱ⁻¹*⇇  ℂ^15 -> ℂ^15

julia> F * x ≈ FFTW.ifft(FFTW.ifftshift(x))
true

julia> F = ifftshift_op(IDFT(15); codomain_shifts=(1,))
⇇*ℱ⁻¹  ℂ^15 -> ℂ^15

julia> F * x ≈ FFTW.ifftshift(FFTW.ifft(x))
true

julia> F = ifftshift_op(IDFT(15); domain_shifts=(1,), codomain_shifts=(1,))
Π  ℂ^15 -> ℂ^15

julia> F * x ≈ FFTW.ifftshift(FFTW.ifft(FFTW.ifftshift(x)))
true

julia> F = ifftshift_op(IDFT(16); codomain_shifts=(1,)) # note that 16 is even, so we get a SignAlternation (±)
ℱ⁻¹*±  ℂ^16 -> ℂ^16

```
"""
function ifftshift_op(
        op::AbstractOperator; domain_shifts::Tuple = (), codomain_shifts::Tuple = ()
    )
    return _shift_op(IFFTShift, op, domain_shifts, codomain_shifts)
end

# Sign alternation is a strided in-place multiply by ±1 -- pure data movement, so it sits at
# the memory-bound threshold.
threading_threshold(::Type{<:SignAlternation}) = THRESHOLD_MEMORY_BOUND
is_threaded(::SignAlternation{T, N, M, Th}) where {T, N, M, Th} = Th
supports_threading(::SignAlternation) = true

# FFTShift/IFFTShift have no threaded path of their own.
is_threaded(::ShiftOp) = false
supports_threading(::ShiftOp) = false
