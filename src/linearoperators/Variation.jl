export Variation

"""
Variation([domain_type=Float64::Type,] dim_in::Tuple)
Variation(dims...)
Variation(x::AbstractArray)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns a matrix with its `i`th column consisting of the vectorized discretized gradient over the `i`th `direction obtained using forward finite differences.

```jldoctest
julia> Variation(Float64,(10,2))
Ʋ  ℝ^(10, 2) -> ℝ^(20, 2)

julia> Variation(2,2,2)
Ʋ  ℝ^(2, 2, 2) -> ℝ^(8, 3)

julia> Variation(ones(2,2))*[1. 2.; 1. 2.]
4×2 Matrix{Float64}:
 0.0  1.0
 0.0  1.0
 0.0  1.0
 0.0  1.0

```
"""
struct Variation{T, N, Th, S <: AbstractArray{T}} <: LinearOperator
    dim_in::NTuple{N, Int}
end

# Constructors
#default constructor
function Variation(
        domain_type::Type{T}, dim_in::NTuple{N, Int};
        threaded = true, array_type::Type = Array{T}
    ) where {T, N}
    N == 1 && error("use FiniteDiff instead!")
    threaded = threaded && Threads.nthreads() > 1 && prod(dim_in) * sizeof(domain_type) > 2^16
    S = _normalize_array_type(array_type, domain_type)
    return Variation{domain_type, N, threaded, S}(dim_in)
end

function Variation(
        domain_type::Type{T}, dim_in::Vararg{Int}; threaded = true, array_type::Type = Array{T}
    ) where {T}
    return Variation(domain_type, dim_in; threaded, array_type)
end
function Variation(
        dim_in::NTuple{N, Int}; threaded = true, array_type::Type = Array{Float64}
    ) where {N}
    return Variation(Float64, dim_in; threaded, array_type)
end
function Variation(dim_in::Vararg{Int}; threaded = true, array_type::Type = Array{Float64})
    return Variation(dim_in; threaded, array_type)
end
function Variation(x::AbstractArray; threaded = true)
    S = _array_wrapper(x){eltype(x)}
    T = eltype(x)
    N = ndims(x)
    threaded = threaded && _should_thread(x)
    N == 1 && error("use FiniteDiff instead!")
    return Variation{T, N, threaded, S}(size(x))
end

# Mappings
# Non-threaded forward
@inbounds function LinearAlgebra.mul!(
        y::AbstractArray, A::Variation{T, N, false, <:Any}, b::AbstractArray
    ) where {T, N}
    check(y, A, b)
    @assert firstindex(b) == 1 "Only support 1-based arrays"
    @assert firstindex(y) == 1 "Only support 1-based arrays"

    # First dimension -- special case
    batch_length = size(b, 1)
    @.. y[2:end, 1] = b[2:end] - b[1:(end - 1)] # finite difference along the first dimension, but incorrect for boundaries
    @.. y[1:batch_length:end, 1] = b[2:batch_length:end] - b[1:batch_length:end] # correct boundaries with mirrored boundary conditions

    # Other dimensions
    batch_count = length(b) ÷ batch_length
    for d in 2:N
        for k in 0:(batch_count - 1)
            slice_start = k * batch_length + 1
            slice_end = (k + 1) * batch_length
            slicing = slice_start:slice_end
            if k % size(b, d) == 0
                next_slice_start = (k + 1) * batch_length + 1
                next_slice_end = (k + 2) * batch_length
                next_slicing = next_slice_start:next_slice_end
                @views y[slicing, d] .= b[next_slicing] .- b[slicing]
            else
                prev_slice_start = (k - 1) * batch_length + 1
                prev_slice_end = k * batch_length
                prev_slicing = prev_slice_start:prev_slice_end
                @views y[slicing, d] .= b[slicing] .- b[prev_slicing]
            end
        end
        batch_count ÷= size(b, d)
        batch_length *= size(b, d)
    end
    return y
end

@inbounds function LinearAlgebra.mul!(
        y::AbstractArray, A::Variation{T, N, true, <:Any}, b::AbstractArray
    ) where {T, N}
    check(y, A, b)
    @assert firstindex(b) == 1 "Only support 1-based arrays"
    @assert firstindex(y) == 1 "Only support 1-based arrays"

    # First dimension -- special case
    batch_length = size(b, 1)
    @.. thread = true y[2:end, 1] = b[2:end] - b[1:(end - 1)] # finite difference along the first dimension, but incorrect for boundaries
    @.. thread = true y[1:batch_length:end, 1] = b[2:batch_length:end] - b[1:batch_length:end] # correct boundaries with mirrored boundary conditions

    # Other dimensions
    batch_count = length(b) ÷ batch_length
    for d in 2:N
        @batch for k in 0:(batch_count - 1)
            slice_start = k * batch_length + 1
            slice_end = (k + 1) * batch_length
            slicing = slice_start:slice_end
            if k % size(b, d) == 0
                next_slice_start = (k + 1) * batch_length + 1
                next_slice_end = (k + 2) * batch_length
                next_slicing = next_slice_start:next_slice_end
                @views y[slicing, d] .= b[next_slicing] .- b[slicing]
            else
                prev_slice_start = (k - 1) * batch_length + 1
                prev_slice_end = k * batch_length
                prev_slicing = prev_slice_start:prev_slice_end
                @views y[slicing, d] .= b[slicing] .- b[prev_slicing]
            end
        end
        batch_count ÷= size(b, d)
        batch_length *= size(b, d)
    end
    return y
end

# Non-threaded adjoint
@inbounds function LinearAlgebra.mul!(
        y::AbstractArray, A::AdjointOperator{<:Variation{T, N, false}}, b::AbstractArray
    ) where {T, N}
    check(y, A, b)
    for cnt in LinearIndices(size(y))
        i_1 = (cnt - 1) % size(y, 1) + 1
        y[cnt] = if i_1 == 1
            -(b[cnt, 1] + b[cnt + 1, 1])
        elseif i_1 == 2
            b[cnt, 1] + b[cnt - 1, 1] - b[cnt + 1, 1]
        elseif i_1 == size(y, 1)
            b[cnt, 1]
        else
            b[cnt, 1] - b[cnt + 1, 1]
        end
        stride = size(y, 1)
        for d in 2:N
            i_d = ((cnt - 1) ÷ stride) % size(y, d) + 1
            y[cnt] += if i_d == 1
                -(b[cnt, d] + b[cnt + stride, d])
            elseif i_d == 2
                b[cnt, d] + b[cnt - stride, d] - b[cnt + stride, d]
            elseif i_d == size(y, d)
                b[cnt, d]
            else
                b[cnt, d] - b[cnt + stride, d]
            end
            stride *= size(y, d)
        end
    end
    return y
end

# Threaded adjoint
@inbounds function LinearAlgebra.mul!(
        y::AbstractArray, A::AdjointOperator{<:Variation{T, N, true}}, b::AbstractArray
    ) where {T, N}
    check(y, A, b)
    @batch for cnt in LinearIndices(size(y))
        i_1 = (cnt - 1) % size(y, 1) + 1
        y[cnt] = if i_1 == 1
            -(b[cnt, 1] + b[cnt + 1, 1])
        elseif i_1 == 2
            b[cnt, 1] + b[cnt - 1, 1] - b[cnt + 1, 1]
        elseif i_1 == size(y, 1)
            b[cnt, 1]
        else
            b[cnt, 1] - b[cnt + 1, 1]
        end
        stride = size(y, 1)
        for d in 2:N
            i_d = ((cnt - 1) ÷ stride) % size(y, d) + 1
            y[cnt] += if i_d == 1
                -(b[cnt, d] + b[cnt + stride, d])
            elseif i_d == 2
                b[cnt, d] + b[cnt - stride, d] - b[cnt + stride, d]
            elseif i_d == size(y, d)
                b[cnt, d]
            else
                b[cnt, d] - b[cnt + stride, d]
            end
            stride *= size(y, d)
        end
    end
    return y
end

# Properties

domain_type(::Variation{T}) where {T} = T
codomain_type(::Variation{T}) where {T} = T
domain_array_type(::Variation{T, N, Th, S}) where {T, N, Th, S} = S
codomain_array_type(::Variation{T, N, Th, S}) where {T, N, Th, S} = S
is_thread_safe(::Variation) = true

function _copy_operator_impl(
        op::Variation{T, N, Th, S}; storage_type = nothing, threaded = nothing
    ) where {T, N, Th, S}
    new_threaded = threaded === nothing ? Th : threaded
    new_at = storage_type === nothing ? _array_wrapper_type(S) : storage_type
    return Variation(T, op.dim_in; threaded = new_threaded, array_type = new_at)
end

size(L::Variation{T, N}) where {T, N} = ((prod(L.dim_in), N), L.dim_in)

fun_name(L::Variation) = "Ʋ"

is_threaded(::Variation{T, N, Th, S}) where {T, N, Th, S} = Th
threading_threshold(::Type{<:Variation}) = THRESHOLD_ELEMENTWISE_ARITHMETIC
supports_threading(::Variation) = true
