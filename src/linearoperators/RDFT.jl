
export RDFT, IRDFT

"""
`RDFT([domainType=Float64::Type,] dim_in::Tuple [,dims=1])`

`RDFT(dim_in...)`

`RDFT(x::AbstractArray [,dims=1])`

Creates a `LinearOperator` which, when multiplied with a real array `x`, returns the DFT over the dimension `dims`, exploiting Hermitian symmetry. 

```julia
julia> RDFT(Float64,(10,10))
ℱ  ℝ^(10, 10) -> ℂ^(6, 10)

julia> RDFT((10,10,10),2)
ℱ  ℝ^(10, 10, 10) -> ℂ^(10, 6, 10)

```

"""

struct RDFT{T <:Number,
	       N,
	       T1<:Base.DFT.Plan,
	       T2<:Base.DFT.Plan,
	       T3<:AbstractArray{Complex{T},N}
	      } <: LinearOperator 
	dim_in::NTuple{N,Int}
	dim_out::NTuple{N,Int}
	A::T1
	At::T2
	b2::T3
	y2::T3
	Zp::ZeroPad{Complex{T},N}
end


# Constructors
#standard constructor

function RDFT(x::AbstractArray{T,N}, dims::Int=1) where {T<:Real,N}
	A = plan_rfft(x,dims) 
	b2 = zeros(complex(T),size(x))
	y2 = zeros(complex(T),size(x))
	At = plan_bfft(y2,dims)
	dim_in = size(x)
	dim_out = ()
	for i = 1:N
		dim_out = i == dims ? (dim_out...,div(dim_in[i],2)+1) : (dim_out...,dim_in[i])
	end
	Z = ZeroPad(Complex{T},dim_out,size(b2))
	RDFT{T,N,typeof(A),typeof(At),typeof(b2)}(dim_in,dim_out,A,At,b2,y2,Z)
end

RDFT(T::Type,dim_in::NTuple{N,Int},dims::Int=1) where {N} = RDFT(zeros(T,dim_in),dims)
RDFT(dim_in::NTuple{N,Int},dims::Int=1) where {N} = RDFT(zeros(dim_in),dims)
RDFT(dim_in::Vararg{Int}) = RDFT(dim_in)
RDFT(T::Type,dim_in::Vararg{Int}) = RDFT(T,dim_in)


# Mappings

function A_mul_B!(y::T3,
		  L::RDFT{T,N,T1,T2,T3},
		  b::T4) where {N,T,T1,T2,T3,T4 <: AbstractArray{T,N}}
	A_mul_B!(y,L.A,b)
end

function Ac_mul_B!(y::T4,
		   L::RDFT{T,N,T1,T2,T3},
		   b::T3) where {N,T,T1,T2,T3,T4 <: AbstractArray{T,N}}
	A_mul_B!(L.b2,L.Zp,b)
	A_mul_B!(L.y2,L.At,L.b2)
	y .= real.(L.y2)
	return y

end


# Properties

size(L::RDFT) = (L.dim_out,L.dim_in)

fun_name(A::RDFT) = "ℱ"

domainType(L::RDFT{T,N}) where {T,N} = T
codomainType(L::RDFT{T,N}) where {T,N} = Complex{T}

is_AAc_diagonal(    L::RDFT) = false #TODO but might be true?
is_invertible(      L::RDFT) = true
is_full_row_rank(   L::RDFT) = true
