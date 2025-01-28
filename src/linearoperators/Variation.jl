export Variation

"""
	Variation([domainType=Float64::Type,] dim_in::Tuple)
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
struct Variation{T,N} <: LinearOperator
	dim_in::NTuple{N,Int}
end

# Constructors
#default constructor
function Variation(domainType::Type, dim_in::NTuple{N,Int}) where {N}
	N == 1 && error("use FiniteDiff instead!")
	return Variation{domainType,N}(dim_in)
end

Variation(domainType::Type, dim_in::Vararg{Int}) = Variation(domainType, dim_in)
Variation(dim_in::NTuple{N,Int}) where {N} = Variation(Float64, dim_in)
Variation(dim_in::Vararg{Int}) = Variation(dim_in)
Variation(x::AbstractArray) = Variation(eltype(x), size(x))

# Mappings

@generated function mul!(
	y::AbstractArray{T,2}, A::Variation{T,N}, b::AbstractArray{T,N}
) where {T,N}
	ex = :()

	for i in 1:N
		z = zeros(Int, N)
		z[i] = 1
		z = (z...,)
		ex = :($ex;
		y[cnt, $i] = if I[$i] == 1
			b[I + CartesianIndex($z)] - b[I]
		else
			b[I] - b[I - CartesianIndex($z)]
		end)
	end

	ex2 = quote
		cnt = 0
		for I in CartesianIndices(size(b))
			cnt += 1
			$ex
		end
		return y
	end
end

@generated function mul!(
	y::AbstractArray{T,N}, A::AdjointOperator{Variation{T,N}}, b::AbstractArray{T,2}
) where {T,N}
	ex = :(y[I] = if I[1] == 1
		-(b[cnt, 1] + b[cnt + 1, 1])
	elseif I[1] == 2
		b[cnt, 1] + b[cnt - 1, 1] - b[cnt + 1, 1]
	elseif I[1] == size(y, 1)
		b[cnt, 1]
	else
		b[cnt, 1] - b[cnt + 1, 1]
	end)

	Nx = :(size(y, 1))
	for i in 2:N
		ex = quote
			$ex
			y[I] += if I[$i] == 1
				-(b[cnt, $i] + b[cnt + $Nx, $i])
			elseif I[$i] == 2
				b[cnt, $i] + b[cnt - $Nx, $i] - b[cnt + $Nx, $i]
			elseif I[$i] == size(y, $i)
				b[cnt, $i]
			else
				b[cnt, $i] - b[cnt + $Nx, $i]
			end
		end
		Nx = :($Nx * size(y, $i))
	end

	ex2 = quote
		cnt = 0
		for I in CartesianIndices(size(y))
			cnt += 1
			$ex
		end
		return y
	end
end

# Properties

domainType(::Variation{T}) where {T} = T
codomainType(::Variation{T}) where {T} = T
is_thread_safe(::Variation) = true

size(L::Variation{T,N}) where {T,N} = ((prod(L.dim_in), N), L.dim_in)

fun_name(L::Variation) = "Ʋ"
