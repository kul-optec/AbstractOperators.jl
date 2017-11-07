export Variation

"""
`Variation([domainType=Float64::Type,] dim_in::Tuple)`

`Variation(dims...)`

`Variation(x::AbstractArray)`

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns a matrix with its `i`th column consisting of the vectorized discretized gradient over the `i`th `direction obtained using forward finite differences. 

```julia
julia> Variation(Float64,(10,2))
Ʋ  ℝ^(10, 2) -> ℝ^(20, 2)

julia> Variation(2,2,2)
Ʋ  ℝ^(2, 2, 2) -> ℝ^(8, 3)

julia> Variation(ones(2,2))*[1. 2.; 1. 2.]
4×2 Array{Float64,2}:
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
	Variation{domainType,N}(dim_in)
end

Variation(domainType::Type, dim_in::Vararg{Int}) = Variation(domainType, dim_in)
Variation(dim_in::NTuple{N,Int}) where {N} = Variation(Float64, dim_in)
Variation(dim_in::Vararg{Int}) = Variation(dim_in)
Variation(x::AbstractArray)  = Variation(eltype(x), size(x))

# Mappings

@generated function A_mul_B!(y::AbstractArray{T,2}, 
			     A::Variation{T,N}, b::AbstractArray{T,N}) where {T,N}

	ex = :()

	for i = 1:N
		z = zeros(Int,N)
		z[i] = 1
		z = (z...)
		ex = :($ex; y[cnt,$i] = I[$i] == 1 ? b[I+CartesianIndex($z)]-b[I] : 
		       b[I]-b[I-CartesianIndex($z)])
	end

	ex2 = quote 
		cnt = 0
		for I in CartesianRange(size(b))
			cnt += 1
			$ex
		end
		return y
	end
end

@generated function Ac_mul_B!(y::AbstractArray{T,N}, 
			      A::Variation{T,N}, b::AbstractArray{T,2}) where {T,N}

	ex = :(y[I] = I[1] == 1  ? -(b[cnt,1] + b[cnt+1,1]) :
	              I[1] == 2  ?   b[cnt,1] + b[cnt-1,1] - b[cnt+1,1] :
	       I[1] == size(y,1) ?   b[cnt,1] : b[cnt,  1] - b[cnt+1,1]
	       )

	Nx = :(size(y,1))
	for i = 2:N
		ex = quote 
			$ex 
			y[I] += I[$i] == 1  ? -(b[cnt,$i] + b[cnt+$Nx,$i]) :
			        I[$i] == 2  ?   b[cnt,$i] + b[cnt-$Nx,$i] - b[cnt+$Nx,$i] :
			        I[$i] == size(y,$i) ?   b[cnt,$i] : b[cnt,  $i]   - b[cnt+$Nx,$i]
		end
		Nx = :($Nx*size(y,$i))
	end

	ex2 = quote 
		cnt = 0
		for I in CartesianRange(size(y))
			cnt += 1
			$ex
		end
		return y
	end
end

# Properties

  domainType(L::Variation{T,N}) where {T,N} = T
codomainType(L::Variation{T,N}) where {T,N} = T

size(L::Variation{T,N}) where {T,N} = ((prod(L.dim_in), N), L.dim_in)

fun_name(L::Variation)  = "Ʋ"
