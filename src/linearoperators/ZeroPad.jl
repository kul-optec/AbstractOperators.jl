export ZeroPad


"""
`ZeroPad([domainType::Type,] dim_in::Tuple, zp::Tuple)`

`ZeroPad(x::AbstractArray, zp::Tuple)`

Create a `LinearOperator` which, when multiplied to an array `x` of size `dim_in`, returns an expanded array `y` of size `dim_in .+ zp` where `y[1:dim_in[1], 1:dim_in[2] ... ] = x` and zero elsewhere.  

```julia
julia> Z = ZeroPad((2,2),(0,2))
[I;0]  ℝ^(2, 2) -> ℝ^(2, 4)

julia> Z*ones(2,2)
2×4 Array{Float64,2}:
 1.0  1.0  0.0  0.0
 1.0  1.0  0.0  0.0

```

"""
struct ZeroPad{T,N} <: LinearOperator
	dim_in::NTuple{N,Int}
	zp::NTuple{N,Int}
end

# Constructors
#standard constructor
function ZeroPad(domainType::Type, dim_in::NTuple{N,Int}, zp::NTuple{M,Int}) where {N,M}
	M != N && error("dim_in and zp must have the same length")
	any([zp...].<0) && error("zero padding cannot be negative")
	ZeroPad{domainType,N}(dim_in,zp)
end

ZeroPad(dim_in::Tuple, zp::NTuple{N,Int})                   where {N} = ZeroPad(Float64, dim_in, zp)
ZeroPad(domainType::Type, dim_in::Tuple, zp::Vararg{Int,N}) where {N} = ZeroPad(domainType, dim_in, zp)
ZeroPad(dim_in::Tuple, zp::Vararg{Int,N})                   where {N} = ZeroPad(Float64, dim_in, zp)
ZeroPad(x::AbstractArray, zp::NTuple{N,Int})                where {N} = ZeroPad(eltype(x), size(x), zp)
ZeroPad(x::AbstractArray, zp::Vararg{Int,N})                where {N} = ZeroPad(eltype(x), size(x), zp)

# Mappings
@generated function mul!(y::AbstractArray{T,N},L::ZeroPad{T,N},b::AbstractArray{T,N}) where {T,N}


	# builds
	#for i1 =1:size(y,1), i2 =1:size(y,2) 
	#	y[i1,i2] = i1 <= size(b,1) && i2 <= size(b,2)  ?  b[i1,i2] : 0. 
	#end

	ex = "for "
	for i = 1:N ex *= "i$i =1:size(y,$i)," end

	ex = ex[1:end-1] #remove ,

	ex *= " y[" 
	for i = 1:N ex *= "i$i," end

	ex = ex[1:end-1] #remove ,

	ex *= "] = " 
	for i = 1:N ex *= " i$i <= size(b,$i) &&" end

	ex = ex[1:end-2] #remove &&

	ex *= " ?  b[" 
	for i = 1:N ex *= "i$i," end

	ex = ex[1:end-1] #remove ,
	ex *= "] : 0. end" 

	ex = Meta.parse(ex)

end

@generated function mul!(y::AbstractArray{T,N},L::AdjointOperator{ZeroPad{T,N}},b::AbstractArray{T,N}) where {T,N}

	#builds
	#for l = 1:size(y,1), m = 1:size(y,2)
	#	y[l,m] = b[l,m]
	#end

	ex = "for "
	for i = 1:N ex *= "i$i =1:size(y,$i)," end

	ex *= " y[" 
	idx = ""
	for i = 1:N idx *= "i$i," end

	idx = idx[1:end-1] #remove ,

	ex *= idx 

	ex *= "] = b[" 
	ex *= idx 
	ex *= "] end" 

	ex = Meta.parse(ex)

end

# Properties

  domainType(L::ZeroPad{T}) where {T} = T
codomainType(L::ZeroPad{T}) where {T} = T

size(L::ZeroPad) = L.dim_in .+ L.zp, L.dim_in

fun_name(L::ZeroPad)       = "[I;0]"
is_AcA_diagonal(L::ZeroPad) = true
diag_AcA(L::ZeroPad) = 1

is_full_column_rank(L::ZeroPad) = true
