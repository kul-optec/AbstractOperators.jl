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
struct FiniteDiff{N, D, T} <: LinearOperator
    dim_in::NTuple{N, Int}
    function FiniteDiff{N, D, T}(dim_in) where {N, D, T}
        D > N && error("direction is bigger the number of dimension $N")
        return new{N, D, T}(dim_in)
    end
end

# Constructors
# Val-dispatch constructor — fully type-stable (D is known at compile time)
function FiniteDiff(::Type{T}, dim_in::NTuple{N, Int}, ::Val{D}) where {T, N, D}
    return FiniteDiff{N, D, T}(dim_in)
end

# Specialized no-direction constructor: D=1 is a compile-time literal — fully type-stable
FiniteDiff(dim_in::NTuple{N, Int}) where {N} = FiniteDiff{N, 1, Float64}(dim_in)

#default constructor (direction as runtime Int — delegates to Val)
function FiniteDiff(domain_type::Type, dim_in::NTuple{N, Int}, dir::Int = 1) where {N}
    return FiniteDiff(domain_type, dim_in, Val(dir))
end

FiniteDiff(dim_in::NTuple{N, Int}, dir::Int) where {N} = FiniteDiff(Float64, dim_in, Val(dir))

function FiniteDiff(x::AbstractArray{T, N}, dir::Int = 1) where {T, N}
    return FiniteDiff(T, size(x), Val(dir))
end

# Mappings

function mul!(y::AbstractArray, L::FiniteDiff{N, D}, b::AbstractArray) where {N, D}
    check(y, L, b)
    idx_1 = CartesianIndices(ntuple(i -> i == D ? (2:L.dim_in[i]) : (1:L.dim_in[i]), Val(N)))
    idx_2 = CartesianIndices(ntuple(i -> i == D ? (1:(L.dim_in[i] - 1)) : (1:L.dim_in[i]), Val(N)))
    y .= b[idx_1] .- b[idx_2]
    return y
end

function mul!(y::AbstractArray, L::AdjointOperator{<:FiniteDiff{N, D}}, b::AbstractArray) where {N, D}
    check(y, L, b)
    dim_in = L.A.dim_in
    idx_start = CartesianIndices(ntuple(i -> i == D ? (1:1) : (1:dim_in[i]), Val(N)))
    idx_between_1 = CartesianIndices(ntuple(i -> i == D ? (1:(dim_in[i] - 2)) : (1:dim_in[i]), Val(N)))
    idx_between_2 = CartesianIndices(ntuple(i -> i == D ? (2:(dim_in[i] - 1)) : (1:dim_in[i]), Val(N)))
    idx_end_1 = CartesianIndices(ntuple(i -> i == D ? ((dim_in[i] - 1):(dim_in[i] - 1)) : (1:dim_in[i]), Val(N)))
    idx_end_2 = CartesianIndices(ntuple(i -> i == D ? (dim_in[i]:dim_in[i]) : (1:dim_in[i]), Val(N)))
    y[idx_start] .= -b[idx_start]
    y[idx_between_2] .= b[idx_between_1] .- b[idx_between_2]
    y[idx_end_2] .= b[idx_end_1]
    return y
end

# Properties

domain_type(::FiniteDiff{<:Any, <:Any, T}) where {T} = T
codomain_type(::FiniteDiff{<:Any, <:Any, T}) where {T} = T
is_thread_safe(::FiniteDiff) = true

function size(L::FiniteDiff{N, D}) where {N, D}
    dim_out = ntuple(i -> i == D ? L.dim_in[i] - 1 : L.dim_in[i], Val(N))
    return dim_out, L.dim_in
end

fun_name(::FiniteDiff{<:Any, 1}) = "δx"
fun_name(::FiniteDiff{<:Any, 2}) = "δy"
fun_name(::FiniteDiff{<:Any, 3}) = "δz"
fun_name(::FiniteDiff{<:Any, D}) where {D} = "δx$D"

is_full_row_rank(::FiniteDiff) = true
