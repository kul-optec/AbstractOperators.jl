export FiniteDiff

#TODO add boundary conditions

"""
`FiniteDiff([domainType=Float64::Type,] dim_in::Tuple, direction = 1)`

`FiniteDiff(x::AbstractArray, direction = 1)`

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the discretized gradient over the specified `direction` obtained using forward finite differences. 

```julia
julia> FiniteDiff(Float64,(3,))
δx  ℝ^3 -> ℝ^2

julia> FiniteDiff((3,4),2)
δy  ℝ^(3, 4) -> ℝ^(3, 3)

julia> all(FiniteDiff(ones(2,2,2,3),1)*ones(2,2,2,3) .== 0)
true

```

"""
struct FiniteDiff{T,N,D,C <: CartesianIndices{N}} <: LinearOperator
	dim_in::NTuple{N,Int}
    idx::C
	function FiniteDiff{T,N,D}(dim_in) where {T,N,D}
		D > N && error("direction is bigger the number of dimension $N")
        idx = CartesianIndices(([i == D ? (2:d) : (1:d) for (i,d) in enumerate(dim_in)]...,))
        new{T,N,D,typeof(idx)}(dim_in,idx)
	end
end

# Constructors
#default constructor
FiniteDiff(domainType::Type, dim_in::NTuple{N,Int}, dir::Int64 = 1) where {N} =
FiniteDiff{domainType,N,dir}(dim_in)

FiniteDiff(dim_in::NTuple{N,Int}, dir::Int64 = 1) where {N} =
FiniteDiff(Float64, dim_in, dir)

FiniteDiff(x::AbstractArray{T,N}, dir::Int64 = 1) where {T,N}  = FiniteDiff(eltype(x), size(x), dir)

# Mappings

@generated function mul!(y::AbstractArray{T,N},
                         L::FiniteDiff{T,N,D},
                         b::AbstractArray{T,N}) where {T,N,D}
	z = zeros(Int,N)
	z[D] = 1
	idx = CartesianIndex(z...)
	ex = quote
        for I in L.idx
			y[I-$idx] = b[I]-b[I-$idx]
		end
		return y
	end
end

@generated function mul!(y::AbstractArray{T,N},
                         L::AdjointOperator{FiniteDiff{T,N,D,C}},
                         b::AbstractArray{T,N}) where {T,N,D,C}
	z = zeros(Int,N)
	z[D] = 1
	idx = CartesianIndex(z...)
	ex = quote
		for I in CartesianIndices(size(y))
			y[I] = 
			I[$D] == 1 ? -b[I]  :
			I[$D] == size(y,$D) ?   b[I-$idx]  : -b[I]+b[I-$idx]
		end
		return y
	end
end

# Properties

domainType(L::FiniteDiff{T, N}) where {T, N} = T
codomainType(L::FiniteDiff{T, N}) where {T, N} = T

function size(L::FiniteDiff{T,N,D}) where {T,N,D} 
	dim_out = [L.dim_in...]
	dim_out[D] = dim_out[D]-1
	return ((dim_out...,), L.dim_in)
end

fun_name(L::FiniteDiff{T,N,1}) where  {T,N} = "δx"
fun_name(L::FiniteDiff{T,N,2}) where  {T,N} = "δy"
fun_name(L::FiniteDiff{T,N,3}) where  {T,N} = "δz"
fun_name(L::FiniteDiff{T,N,D}) where {T,N,D}  = "δx$D"


is_full_row_rank(L::FiniteDiff) = true


