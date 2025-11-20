export GetIndex

"""
GetIndex([domain_type=Float64::Type,] dim_in::Tuple, idx...)
GetIndex(x::AbstractArray, idx::Tuple)

Creates a `LinearOperator` which, when multiplied with `x`, returns `x[idx]`.
Supported indices are:
- `Colon()`: selects the entire dimension.
- `BitArray` or `AbstractArray{Bool}`: selects elements where the mask is `true`.
- `AbstractVector{Int}`: selects elements by their indices.
- `AbstractVector{CartesianIndex}`: selects elements by their Cartesian indices.
- `Tuple`: a tuple of indices, where each element can be one of the above types.

```jldoctest
julia> x = (1:10) .* 1.0;

julia> G = GetIndex(Float64,(10,), 1:3)
↓  ℝ^10 -> ℝ^3

julia> G*x
3-element Vector{Float64}:
 1.0
 2.0
 3.0

julia> GetIndex(randn(10,20,30),(1:2,1:4,1))
↓  ℝ^(10, 20, 30) -> ℝ^(2, 4)

```
"""
struct GetIndex{I,N,M,T,S} <: LinearOperator
    dim_out::NTuple{N,Int}
    dim_in::NTuple{M,Int}
    idx::I
    function GetIndex(
        T::Type, S::Type, dim_out::NTuple{N,Int}, dim_in::NTuple{M,Int}, idx
    ) where {N,M}
        if !(idx isa Tuple)
            idx = (idx,)
        end
        checkbounds(CartesianIndices(dim_in), idx...)
        return new{typeof(idx),N,M,T,S}(dim_out, dim_in, idx)
    end
end

# Constructors
# default
function GetIndex(domain_type::Type, dim_in::Dims, idx::Tuple)
    dim_out = get_dim_out(dim_in, idx...)
    if dim_out == dim_in
        return Eye(domain_type, dim_in)
    else
        return GetIndex(domain_type, Array{domain_type}, dim_out, dim_in, idx)
    end
end

function GetIndex(
    domain_type::Type,
    dim_in::Dims,
    idx::Union{AbstractVector{Int},AbstractVector{<:CartesianIndex}},
)
    dim_out = (length(idx),)
    return GetIndex(domain_type, Array{domain_type}, dim_out, dim_in, idx)
end

function GetIndex(domain_type::Type, dim_in::Dims, mask::AbstractArray{Bool})
    dim_out = (sum(mask),)
    if dim_out[1] == prod(dim_in)
        return reshape(Eye(domain_type, dim_in), dim_out)
    else
        return GetIndex(domain_type, Array{domain_type}, dim_out, dim_in, mask)
    end
end

GetIndex(domain_type::Type, dim_in::Dims, idx...) = GetIndex(domain_type, dim_in, idx)
GetIndex(dim_in::Dims, idx...) = GetIndex(Float64, dim_in, idx)
GetIndex(dim_in::Dims, idx::Tuple) = GetIndex(Float64, dim_in, idx)
GetIndex(dim_in::Dims, idx) = GetIndex(Float64, dim_in, idx)
function GetIndex(x::AbstractArray, idx::Tuple)
    dim_in = size(x)
    dim_out = get_dim_out(dim_in, idx...)
    if dim_out == dim_in
        return Eye(eltype(x), dim_in)
    else
        S = typeof(x isa SubArray ? parent(x) : x).name.wrapper{eltype(x)}
        return GetIndex(eltype(x), S, dim_out, dim_in, idx)
    end
end
function GetIndex(
    x::AbstractArray,
    idx::Union{AbstractVector{Int},AbstractVector{<:CartesianIndex},AbstractArray{Bool}},
)
    dim_in = size(x)
    dim_out = get_dim_out(dim_in, idx)
    if dim_out == dim_in
        return Eye(eltype(x), dim_in)
    else
        S = typeof(x isa SubArray ? parent(x) : x).name.wrapper{eltype(x)}
        return GetIndex(eltype(x), S, dim_out, dim_in, idx)
    end
end

# Mappings

@generated function mul!(
    y::AbstractArray, L::GetIndex{I}, b::AbstractArray
) where {K,I<:Tuple{Vararg{Any,K}}}
    if K == 1
        if I.parameters[1] <: AbstractArray{Bool}
            indices = :(to_indices(b, L.idx)[1])
        else
            indices = :(L.idx[1])
        end
        quote
            check(y, L, b)
            for (i, j) in enumerate($indices)
                @inbounds y[i] = b[j]
            end
            return y
        end
    else
        quote
            check(y, L, b)
            indices = to_indices(b, L.idx)
            j = 1
            @nloops $K i d -> indices[d] begin
                @inbounds y[j] = @nref $K b i
                j += 1
            end
            return y
        end
    end
end

function mul!(y::AbstractArray, L::AdjointOperator{<:GetIndex}, b::AbstractArray)
    check(y, L, b)
    fill!(y, 0)
    @inbounds setindex!(y, b, L.A.idx...)
    return y
end

"""
NormalGetIndex([domain_type=Float64::Type,] dim_in::Tuple, idx...)

Optimized implementation of Grammian operator for `GetIndex`, i.e., A' * A, where A is a `GetIndex` operator.
This operator is used internally by `GetIndex` to provide an optimized normal operator.
"""
struct NormalGetIndex{I,N,T,S} <: LinearOperator
    dim_in::NTuple{N,Int}
    idx::I
    function NormalGetIndex(T, S, dim_in::NTuple{N,Int}, idx) where {N}
        if !(idx isa Tuple)
            idx = (idx,)
        end
        checkbounds(CartesianIndices(dim_in), idx...)
        return new{typeof(idx),N,T,S}(dim_in, idx)
    end
end

AdjointOperator(L::NormalGetIndex) = L

@inbounds function mul!(y::AbstractArray, L::NormalGetIndex, b::AbstractArray)
    check(y, L, b)
    fill!(y, 0)
    setindex!(y, view(b, L.idx...), L.idx...)
    return y
end

# Properties
domain_type(::GetIndex{I,N,M,T}) where {I,N,M,T} = T
domain_type(::NormalGetIndex{I,N,T}) where {I,N,T} = T
domain_storage_type(::GetIndex{I,N,M,T,S}) where {I,N,M,T,S} = S
domain_storage_type(::NormalGetIndex{I,N,T,S}) where {I,N,T,S} = S
codomain_type(::GetIndex{I,N,M,T}) where {I,N,M,T} = T
codomain_type(::NormalGetIndex{I,N,T}) where {I,N,T} = T
codomain_storage_type(::GetIndex{I,N,M,T,S}) where {I,N,M,T,S} = S
codomain_storage_type(::NormalGetIndex{I,N,T,S}) where {I,N,T,S} = S
is_thread_safe(L::GetIndex) = true
is_thread_safe(L::NormalGetIndex) = true

size(L::GetIndex) = (L.dim_out, L.dim_in)
size(L::NormalGetIndex) = (L.dim_in, L.dim_in)

fun_name(L::GetIndex) = "↓"
fun_name(L::NormalGetIndex) = "↓ᵃ↓"

is_diagonal(L::NormalGetIndex) = true
function diag(L::NormalGetIndex)
    x = allocate_in_domain(L)
    fill!(x, 0)
    x[L.idx...] .= 1
    return x
end

is_AcA_diagonal(L::GetIndex) = true
diag_AcA(L::GetIndex) = diag(L' * L)

is_AAc_diagonal(L::GetIndex) = true
diag_AAc(L::GetIndex) = one(real(domain_type(L)))

is_full_row_rank(L::GetIndex) = true
is_full_row_rank(L::NormalGetIndex) = true
is_symmetric(::NormalGetIndex) = true
is_sliced(L::GetIndex) = true
get_slicing_expr(L::GetIndex) = L.idx
get_slicing_mask(L::GetIndex{<:BitArray}) = L.idx
function get_slicing_mask(L::GetIndex{<:Tuple})
    mask = falses(L.dim_in)
    mask[L.idx...] .= true
    return mask
end
remove_slicing(L::GetIndex{I,N,M,T,S}) where {I,N,M,T,S} = Eye{T,M,S}(L.dim_out)

has_optimized_normalop(L::GetIndex) = true
function get_normal_op(L::GetIndex{I,N,M,T,S}) where {I,N,M,T,S}
    NormalGetIndex(T, S, L.dim_in, L.idx)
end
has_optimized_normalop(L::AdjointOperator{<:GetIndex}) = true
get_normal_op(L::AdjointOperator{<:GetIndex}) = Eye(domain_type(L), size(L, 2))

has_fast_opnorm(::GetIndex) = true
LinearAlgebra.opnorm(L::GetIndex) = one(real(domain_type(L)))

# Utils

get_idx(L::GetIndex) = L.idx

get_dim_out(::Dims) = error("get_dim_out requires indices to be provided")
get_dim_out(::Dims, idx::Int...) = (1,)

function get_dim_out(in_dim::Dims, idxs...)
    dim2 = ()
    i = 1
    for idx in idxs
        if idx == Colon()
            dim2 = (dim2..., in_dim[i])
        elseif idx isa BitArray || idx isa AbstractArray{Bool}
            dim2 = (dim2..., sum(idx))
            i = i + ndims(idx) - 1
        elseif idx isa AbstractVector{<:CartesianIndex}
            dim2 = (dim2..., length(idx))
            i += ndims(idx) - 1
        elseif idx isa AbstractVector{Int} || idx isa OrdinalRange{Int}
            dim2 = (dim2..., length(idx))
        elseif idx isa AbstractArray{<:Union{Integer,CartesianIndex}}
            dim2 = (dim2..., size(idx)...)
            i += ndims(idx) - 1
        elseif idx isa Int
            # nothing to do
        else
            throw(ArgumentError("Unsupported index type: $(typeof(idx))"))
        end
        i += 1
    end
    return dim2
end
