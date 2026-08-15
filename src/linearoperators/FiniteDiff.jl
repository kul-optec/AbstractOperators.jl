export FiniteDiff

#TODO add boundary conditions

"""
	FiniteDiff([domain_type=Float64::Type,] dim_in::Tuple, direction = 1)
	FiniteDiff(x::AbstractArray, direction = 1)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the discretized gradient over the specified `direction` obtained using forward finite differences.

```jldoctest
julia> FiniteDiff(Float64,(3,))
δx  ℝ^3 -> ℝ^2

julia> FiniteDiff((3,4),2)
δy  ℝ^(3, 4) -> ℝ^(3, 3)

julia> all(FiniteDiff(ones(2,2,2,3),1)*ones(2,2,2,3) .== 0)
true
	
```
"""
struct FiniteDiff{N, D, T, S <: AbstractArray{T}, Th} <: LinearOperator
    dim_in::NTuple{N, Int}
    function FiniteDiff{N, D, T, S, Th}(dim_in) where {N, D, T, S <: AbstractArray{T}, Th}
        D > N && error("direction is bigger the number of dimension $N")
        Th isa Bool || throw(ArgumentError("FiniteDiff threading parameter must be a Bool"))
        return new{N, D, T, S, Th}(dim_in)
    end
end

# Thin alias for the shared resolver, kept because the constructors below read better with
# a short name. `threaded = false` vetoes; anything else defers to the policy.
function _finitediff_threaded(threaded::Bool, ::Type{T}, dim_in, ::Type{S}) where {T, S <: AbstractArray}
    return _elementwise_threaded(FiniteDiff, threaded, T, dim_in, S)
end

# Constructors
# Val-dispatch constructor — fully type-stable (D is known at compile time)
function FiniteDiff(
        ::Type{T}, dim_in::NTuple{N, Int}, ::Val{D};
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N, D}
    S = _normalize_array_type(array_type, T)
    return FiniteDiff{N, D, T, S, _finitediff_threaded(threaded, T, dim_in, S)}(dim_in)
end

# Specialized no-direction constructor: D=1 is a compile-time literal — fully type-stable
function FiniteDiff(
        dim_in::NTuple{N, Int}; array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N}
    S = _normalize_array_type(array_type, Float64)
    return FiniteDiff{N, 1, Float64, S, _finitediff_threaded(threaded, Float64, dim_in, S)}(dim_in)
end

# Specialized no-direction constructor: D=1 is a compile-time literal, so this stays fully
# type-stable. Without it the two-argument call would fall through to the `dir::Int` method
# below and pay a runtime dispatch on `Val(dir)` — which JET's `@test_opt` flags.
function FiniteDiff(
        domain_type::Type{T}, dim_in::NTuple{N, Int};
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N}
    S = _normalize_array_type(array_type, T)
    return FiniteDiff{N, 1, T, S, _finitediff_threaded(threaded, T, dim_in, S)}(dim_in)
end

# Direction as a runtime Int — necessarily delegates through `Val`, so this path is
# dynamically dispatched by construction. Call the `Val{D}` method directly from
# performance-sensitive code.
function FiniteDiff(
        domain_type::Type{T}, dim_in::NTuple{N, Int}, dir::Int;
        array_type::Type = Array{T}, threaded::Bool = true
    ) where {T, N}
    return FiniteDiff(domain_type, dim_in, Val(dir); array_type, threaded)
end

function FiniteDiff(
        dim_in::NTuple{N, Int}, dir::Int; array_type::Type = Array{Float64}, threaded::Bool = true
    ) where {N}
    return FiniteDiff(Float64, dim_in, Val(dir); array_type, threaded)
end

function FiniteDiff(x::AbstractArray{T, N}, dir::Int = 1; threaded::Bool = true) where {T, N}
    S = _normalize_array_type(_array_wrapper(x), T)
    return FiniteDiff{N, dir, T, S, _finitediff_threaded(threaded, T, size(x), S)}(size(x))
end

# Mappings

# `b[idx]` would materialise a temporary for each side of the subtraction; `@views` keeps
# the whole forward difference allocation-free, which is also what makes it worth threading.
function _finitediff_indices(dim_in::NTuple{N, Int}, ::Val{D}) where {N, D}
    idx_1 = CartesianIndices(ntuple(i -> i == D ? (2:dim_in[i]) : (1:dim_in[i]), Val(N)))
    idx_2 = CartesianIndices(ntuple(i -> i == D ? (1:(dim_in[i] - 1)) : (1:dim_in[i]), Val(N)))
    return idx_1, idx_2
end

function mul!(y::AbstractArray, L::FiniteDiff{N, D, T, S, false}, b::AbstractArray) where {N, D, T, S}
    check(y, L, b)
    idx_1, idx_2 = _finitediff_indices(L.dim_in, Val(D))
    @views @. y = b[idx_1] - b[idx_2]
    return y
end

function mul!(y::AbstractArray, L::FiniteDiff{N, D, T, S, true}, b::AbstractArray) where {N, D, T, S}
    check(y, L, b)
    idx_1, idx_2 = _finitediff_indices(L.dim_in, Val(D))
    @views @.. thread = true y = b[idx_1] - b[idx_2]
    return y
end

function mul!(
        y::AbstractArray, L::AdjointOperator{<:FiniteDiff{N, D, T, S, Th}}, b::AbstractArray
    ) where {N, D, T, S, Th}
    check(y, L, b)
    dim_in = L.A.dim_in
    idx_start = CartesianIndices(ntuple(i -> i == D ? (1:1) : (1:dim_in[i]), Val(N)))
    idx_between_1 = CartesianIndices(ntuple(i -> i == D ? (1:(dim_in[i] - 2)) : (1:dim_in[i]), Val(N)))
    idx_between_2 = CartesianIndices(ntuple(i -> i == D ? (2:(dim_in[i] - 1)) : (1:dim_in[i]), Val(N)))
    idx_end_1 = CartesianIndices(ntuple(i -> i == D ? ((dim_in[i] - 1):(dim_in[i] - 1)) : (1:dim_in[i]), Val(N)))
    idx_end_2 = CartesianIndices(ntuple(i -> i == D ? (dim_in[i]:dim_in[i]) : (1:dim_in[i]), Val(N)))
    # Same story as the forward pass: `@views` removes the temporaries, and the middle
    # block -- the only one whose size grows with `dim_in` -- is the part worth threading.
    @views @. y[idx_start] = -b[idx_start]
    if Th
        @views @.. thread = true y[idx_between_2] = b[idx_between_1] - b[idx_between_2]
    else
        @views @. y[idx_between_2] = b[idx_between_1] - b[idx_between_2]
    end
    @views @. y[idx_end_2] = b[idx_end_1]
    return y
end

# Properties

domain_type(::FiniteDiff{<:Any, <:Any, T}) where {T} = T
codomain_type(::FiniteDiff{<:Any, <:Any, T}) where {T} = T
domain_array_type(::FiniteDiff{N, D, T, S}) where {N, D, T, S} = S
codomain_array_type(::FiniteDiff{N, D, T, S}) where {N, D, T, S} = S
is_thread_safe(::FiniteDiff) = true
is_threaded(::FiniteDiff{N, D, T, S, Th}) where {N, D, T, S, Th} = Th

# PROVENANCE: measured per-operator, benchmark/operator_thresholds.jl.
# Crossover of this operator's real `mul!`: Float64 2^15, Float32 2^16.
threading_threshold(::Type{<:FiniteDiff}) = 2^16

function _copy_operator_impl(
        op::FiniteDiff{N, D, T, S, Th}; storage_type = nothing, threaded = nothing
    ) where {N, D, T, S, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return FiniteDiff(T, op.dim_in, Val(D); array_type = new_at, threaded = new_threaded)
end

function size(L::FiniteDiff{N, D}) where {N, D}
    dim_out = ntuple(i -> i == D ? L.dim_in[i] - 1 : L.dim_in[i], Val(N))
    return dim_out, L.dim_in
end

fun_name(::FiniteDiff{<:Any, 1}) = "δx"
fun_name(::FiniteDiff{<:Any, 2}) = "δy"
fun_name(::FiniteDiff{<:Any, 3}) = "δz"
fun_name(::FiniteDiff{<:Any, D}) where {D} = "δx$D"

is_full_row_rank(::FiniteDiff) = true
supports_threading(::FiniteDiff) = true
