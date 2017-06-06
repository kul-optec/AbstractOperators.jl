export Variation

immutable Variation{T,N} <: LinearOperator
	dim_in::NTuple{N,Int}
end

# Constructors
#default constructor
function Variation{N}(domainType::Type, dim_in::NTuple{N,Int}) 
	N > 3 && error("Variation is currently implemented only for Arrays of ndims of 2 or 3")
	N == 1 && error("use FiniteDiff instead!")
	Variation{domainType,N}(dim_in)
end

Variation(domainType::Type, dim_in::Vararg{Int}) = Variation(domainType, dim_in)
Variation{N}(dim_in::NTuple{N,Int}) = Variation(Float64, dim_in)
Variation(dim_in::Vararg{Int}) = Variation(dim_in)
Variation(x::AbstractArray)  = Variation(eltype(x), size(x))

# Mappings

function A_mul_B!{T}(y::AbstractArray{T,2}, A::Variation{T,2}, b::AbstractArray{T,2})
	cnt = 0
	for m = 1:size(b,2), l = 1:size(b,1)
		cnt += 1
		y[cnt,1] = l == 1 ? b[l+1,m]-b[l,m] : b[l,m]-b[l-1,m]
		y[cnt,2] = m == 1 ? b[l,m+1]-b[l,m] : b[l,m]-b[l,m-1]
	end
end

function Ac_mul_B!{T}(y::AbstractArray{T,2}, A::Variation{T,2}, b::AbstractArray{T,2})

	cnt = 0
	Nx, Ny = size(y,1), size(y,2)
	for m = 1:Ny, l = 1:Nx
		cnt += 1
		y[l,m] = ((
	     l == 1  ? -(b[cnt,1] + b[cnt+1,1]) :
	     l == 2  ?   b[cnt,1] + b[cnt-1,1] - b[cnt+1,1] :
	     l == Nx ?   b[cnt,1] : b[cnt,1]   - b[cnt+1,1])
	    +(
       	     m == 1  ? -(b[cnt,2] + b[cnt+Nx,2]) :
             m == 2  ?   b[cnt,2] + b[cnt-Nx,2] - b[cnt+Nx,2] :
             m == Ny ?   b[cnt,2] : b[cnt,2]    - b[cnt+Nx,2] ))
	end
end

function A_mul_B!{T}(y::AbstractArray{T,2},A::Variation{T,3},b::AbstractArray{T,3})
	cnt = 0
	for n = 1:size(b,3), m = 1:size(b,2), l = 1:size(b,1)
		cnt += 1
		y[cnt,1] = l == 1 ? b[l+1,m,n]-b[l,m,n] : b[l,m,n]-b[l-1,m,n]
		y[cnt,2] = m == 1 ? b[l,m+1,n]-b[l,m,n] : b[l,m,n]-b[l,m-1,n]
		y[cnt,3] = n == 1 ? b[l,m,n+1]-b[l,m,n] : b[l,m,n]-b[l,m,n-1]
	end
end

function Ac_mul_B!{T}(y::AbstractArray{T,3},A::Variation{T,3},b::AbstractArray{T,2})
	cnt = 0
	Nx, Ny, Nz = size(y,1), size(y,2), size(y,3)
	Nxy = Nx*Ny
	for n = 1:Nz, m = 1:Ny, l = 1:Nx
		cnt += 1
		y[l,m,n] = ((
	        l == 1  ? -(b[cnt,1] + b[cnt+1,1]) :
	        l == 2  ?   b[cnt,1] + b[cnt-1,1] - b[cnt+1,1] :
	        l == Nx ?   b[cnt,1] : b[cnt,1]   - b[cnt+1,1])
	      +(
	 	m == 1  ? -(b[cnt,2] + b[cnt+Nx,2]) :
	 	m == 2  ?   b[cnt,2] + b[cnt-Nx,2] - b[cnt+Nx,2] :
	 	m == Ny ?   b[cnt,2] : b[cnt,2]    - b[cnt+Nx,2] )
	      +(
	 	n == 1  ? -(b[cnt,3] + b[cnt+Nxy,3]) :
	 	n == 2  ?   b[cnt,3] + b[cnt-Nxy,3] - b[cnt+Nxy,3] :
	 	n == Nz ?   b[cnt,3] : b[cnt,3]     - b[cnt+Nxy,3] ))
	end

end

# Properties

domainType{T,N}(L::Variation{T,N}) = T
codomainType{T,N}(L::Variation{T,N}) = T

size{T,N}(L::Variation{T,N}) = ((prod(L.dim_in), N), L.dim_in)

fun_name(L::Variation)  = "Ʋ"
