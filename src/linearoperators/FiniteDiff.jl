export FiniteDiff

#TODO add boundary conditions
immutable FiniteDiff{T,N,D} <: LinearOperator
	dim_in::NTuple{N,Int}
	function FiniteDiff{T,N,D}(dim_in) where {T,N,D}
		N > 3 && error("currently FiniteDiff is supported only for Arrays with ndims <= 3 ")
		D > N && error("dir > $N")
		new{T,N,D}(dim_in)
	end
end

# Constructors
#default constructor
FiniteDiff{N}(domainType::Type, dim_in::NTuple{N,Int}, dir::Int64 = 1) =
FiniteDiff{domainType,N,dir}(dim_in)

FiniteDiff{N}(dim_in::NTuple{N,Int}, dir::Int64 = 1) =
FiniteDiff(Float64, dim_in, dir)

FiniteDiff{T,N}(x::AbstractArray{T,N}, dir::Int64 = 1)  = FiniteDiff(eltype(x), size(x), dir)

# Mappings

@generated function A_mul_B!{T,N,D}(y::AbstractArray{T,N},L::FiniteDiff{T,N,D},b::AbstractArray{T,N})
	o = ones(Int,N)
	o[D] = 2
	I1 = CartesianIndex(o...)
	z = zeros(Int,N)
	z[D] = 1
	idx = CartesianIndex(z...)
	ex = quote
		I2 = CartesianIndex(size(b))
		for I in CartesianRange($I1,I2)
			y[I-$idx] = b[I]-b[I-$idx]
		end
		return y
	end
end

@generated function Ac_mul_B!{T,N,D}(y::AbstractArray{T,N},L::FiniteDiff{T,N,D},b::AbstractArray{T,N})
	z = zeros(Int,N)
	z[D] = 1
	idx = CartesianIndex(z...)
	ex = quote
		for I in CartesianRange(size(y))
			y[I] = 
			I[$D] == 1 ? -b[I]  :
			I[$D] == size(y,$D) ?   b[I-$idx]  : -b[I]+b[I-$idx]
		end
		return y
	end
end

# Properties

domainType{T, N}(L::FiniteDiff{T, N}) = T
codomainType{T, N}(L::FiniteDiff{T, N}) = T

function size{T,N,D}(L::FiniteDiff{T,N,D}) 
	dim_out = [L.dim_in...]
	dim_out[D] = dim_out[D]-1
	return ((dim_out...), L.dim_in)
end

fun_name{T,N}(L::FiniteDiff{T,N,1})  = "δx"
fun_name{T,N}(L::FiniteDiff{T,N,2})  = "δy"
fun_name{T,N}(L::FiniteDiff{T,N,3})  = "δz"


is_full_row_rank(L::FiniteDiff) = true


