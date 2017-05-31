export FiniteDiff

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
# TODO use @generated functions ?
function A_mul_B!{T}(y::AbstractArray{T,1},L::FiniteDiff{T,1,1},b::AbstractArray{T,1})
	for l = 1:length(b)
		y[l] = l == 1 ? b[l+1]-b[l] : b[l]-b[l-1]
	end
end

function Ac_mul_B!{T}(y::AbstractArray{T,1},L::FiniteDiff{T,1,1},b::AbstractArray{T,1})
	for l = 1:length(b)
		y[l] =
		l == 1 ? -(b[l] + b[l+1]) :
		l == 2 ?   b[l] + b[l-1] - b[l+1] :
		l == length(b) ? b[l] : b[l]-b[l+1]

	end
end

function A_mul_B!{T}(y::AbstractArray{T,2}, L::FiniteDiff{T,2,1}, b::AbstractArray{T,2})
	for l = 1:size(b,1), m = 1:size(b,2)
		y[l,m] = l == 1 ? b[l+1,m]-b[l,m] : b[l,m]-b[l-1,m]
	end
end

function Ac_mul_B!{T}(y::AbstractArray{T,2}, L::FiniteDiff{T,2,1}, b::AbstractArray{T,2})
	for l = 1:size(b,1), m = 1:size(b,2)
		y[l,m] =
		l == 1 ? -(b[l,m] + b[l+1,m]) :
		l == 2 ?   b[l,m] + b[l-1,m] - b[l+1,m] :
		l == size(b,1) ? b[l,m] : b[l,m]-b[l+1,m]
	end
end

function A_mul_B!{T}(y::AbstractArray{T,2},L::FiniteDiff{T,2,2},b::AbstractArray{T,2})
	for l = 1:size(b,1), m = 1:size(b,2)
		y[l,m] = m == 1 ? b[l,m+1]-b[l,m] : b[l,m]-b[l,m-1]
	end
end

function Ac_mul_B!{T}(y::AbstractArray{T,2},L::FiniteDiff{T,2,2},b::AbstractArray{T,2})
	for l = 1:size(b,1), m = 1:size(b,2)
		y[l,m] =
		m == 1 ? -(b[l,m] + b[l,m+1]) :
		m == 2 ?   b[l,m] + b[l,m-1] - b[l,m+1] :
		m == size(b,2) ? b[l,m] : b[l,m]-b[l,m+1]
	end
end

function A_mul_B!{T}(y::AbstractArray{T,3},L::FiniteDiff{T,3,1},b::AbstractArray{T,3})
	for l = 1:size(b,1), m = 1:size(b,2), n = 1:size(b,3)
		y[l,m,n] = l == 1 ? b[l+1,m,n]-b[l,m,n] : b[l,m,n]-b[l-1,m,n]
	end
end

function Ac_mul_B!{T}(y::AbstractArray{T,3},L::FiniteDiff{T,3,1},b::AbstractArray{T,3})
	for l = 1:size(b,1), m = 1:size(b,2), n = 1:size(b,3)
		y[l,m,n] =
		l == 1 ? -(b[l,m,n] + b[l+1,m,n]) :
		l == 2 ?   b[l,m,n] + b[l-1,m,n] - b[l+1,m,n] :
		l == size(b,1) ? b[l,m,n] : b[l,m,n]-b[l+1,m,n]
	end
end

function A_mul_B!{T}(y::AbstractArray{T,3},L::FiniteDiff{T,3,2},b::AbstractArray{T,3})
	for l = 1:size(b,1), m = 1:size(b,2), n = 1:size(b,3)
		y[l,m,n] = m == 1 ? b[l,m+1,n]-b[l,m,n] : b[l,m,n]-b[l,m-1,n]
	end
end

function Ac_mul_B!{T}(y::AbstractArray{T,3},L::FiniteDiff{T,3,2},b::AbstractArray{T,3})
	for l = 1:size(b,1), m = 1:size(b,2), n = 1:size(b,3)
		y[l,m,n] =
		m == 1 ? -(b[l,m,n] + b[l,m+1,n]) :
		m == 2 ?   b[l,m,n] + b[l,m-1,n] - b[l,m+1,n] :
		m == size(b,2) ? b[l,m,n] : b[l,m,n]-b[l,m+1,n]
	end
end

function A_mul_B!{T}(y::AbstractArray{T,3},L::FiniteDiff{T,3,3},b::AbstractArray{T,3})
	for l = 1:size(b,1), m = 1:size(b,2), n = 1:size(b,3)
		y[l,m,n] = n == 1 ? b[l,m,n+1]-b[l,m,n] : b[l,m,n]-b[l,m,n-1]
	end
end

function Ac_mul_B!{T}(y::AbstractArray{T,3},L::FiniteDiff{T,3,3},b::AbstractArray{T,3})
	for l = 1:size(b,1), m = 1:size(b,2), n = 1:size(b,3)
		y[l,m,n] =
		n == 1 ? -(b[l,m,n] + b[l,m,n+1]) :
		n == 2 ?   b[l,m,n] + b[l,m,n-1] - b[l,m,n+1] :
		n == size(b,3) ? b[l,m,n] : b[l,m,n]-b[l,m,n+1]
	end
end

# Properties

domainType{T, N}(L::FiniteDiff{T, N}) = T
codomainType{T, N}(L::FiniteDiff{T, N}) = T

size(L::FiniteDiff) = (L.dim_in, L.dim_in)

fun_name{T,N}(L::FiniteDiff{T,N,1})  = "δx"
fun_name{T,N}(L::FiniteDiff{T,N,2})  = "δy"
fun_name{T,N}(L::FiniteDiff{T,N,3})  = "δz"




