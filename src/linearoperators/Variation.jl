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
struct Variation{T,N,Th} <: LinearOperator
    dim_in::NTuple{N,Int}
end

# Constructors
#default constructor
function Variation(domain_type::Type, dim_in::NTuple{N,Int}; threaded=true) where {N}
    N == 1 && error("use FiniteDiff instead!")
    threaded = threaded && Threads.nthreads() > 1 && prod(dim_in) * sizeof(domain_type) > 2^16
    return Variation{domain_type,N,threaded}(dim_in)
end

function Variation(domain_type::Type, dim_in::Vararg{Int}; threaded=true)
    Variation(domain_type, dim_in; threaded)
end
function Variation(dim_in::NTuple{N,Int}; threaded=true) where {N}
    Variation(Float64, dim_in; threaded)
end
Variation(dim_in::Vararg{Int}; threaded=true) = Variation(dim_in; threaded)
Variation(x::AbstractArray; threaded=true) = Variation(eltype(x), size(x); threaded)

# Mappings
# Non-threaded forward
@inbounds function LinearAlgebra.mul!(
    y::AbstractArray, A::Variation{T,N,false}, b::AbstractArray
) where {T,N}
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
                y[slicing, d] = b[next_slicing] - b[slicing]
            else
                prev_slice_start = (k - 1) * batch_length + 1
                prev_slice_end = k * batch_length
                prev_slicing = prev_slice_start:prev_slice_end
                y[slicing, d] = b[slicing] - b[prev_slicing]
            end
        end
        batch_count ÷= size(b, d)
        batch_length *= size(b, d)
    end
    return y
end

@inbounds function LinearAlgebra.mul!(
    y::AbstractArray, A::Variation{T,N,true}, b::AbstractArray
) where {T,N}
    check(y, A, b)
    @assert firstindex(b) == 1 "Only support 1-based arrays"
    @assert firstindex(y) == 1 "Only support 1-based arrays"

    # First dimension -- special case
    batch_length = size(b, 1)
    @.. thread=true y[2:end, 1] = b[2:end] - b[1:(end - 1)] # finite difference along the first dimension, but incorrect for boundaries
    @.. thread=true y[1:batch_length:end, 1] = b[2:batch_length:end] - b[1:batch_length:end] # correct boundaries with mirrored boundary conditions

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
                y[slicing, d] = b[next_slicing] - b[slicing]
            else
                prev_slice_start = (k - 1) * batch_length + 1
                prev_slice_end = k * batch_length
                prev_slicing = prev_slice_start:prev_slice_end
                y[slicing, d] = b[slicing] - b[prev_slicing]
            end
        end
        batch_count ÷= size(b, d)
        batch_length *= size(b, d)
    end
    return y
end

# Non-threaded adjoint
@inbounds function LinearAlgebra.mul!(
    y::AbstractArray, A::AdjointOperator{Variation{T,N,false}}, b::AbstractArray
) where {T,N}
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
    y::AbstractArray, A::AdjointOperator{Variation{T,N,true}}, b::AbstractArray
) where {T,N}
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
is_thread_safe(::Variation) = true

size(L::Variation{T,N}) where {T,N} = ((prod(L.dim_in), N), L.dim_in)

fun_name(L::Variation) = "Ʋ"
