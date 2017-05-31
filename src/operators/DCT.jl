export DCT, IDCT

abstract type CosineTransform{N,C,T1,T2} <: LinearOperator end

immutable DCT{N,
	      C<:RealOrComplex,
	      T1<:Base.DFT.Plan,
	      T2<:Base.DFT.Plan} <: CosineTransform{N,C,T1,T2}
	dim_in::NTuple{N,Int}
	A::T1
	At::T2
end

immutable IDCT{N,
	       C<:RealOrComplex,
	       T1<:Base.DFT.Plan,
	       T2<:Base.DFT.Plan} <: CosineTransform{N,C,T1,T2}
	dim_in::NTuple{N,Int}
	A::T1
	At::T2
end

# Constructors
#standard constructor
DCT{N}(T::Type,dim_in::NTuple{N,Int}) = DCT(zeros(T,dim_in))

DCT{N}(dim_in::NTuple{N,Int}) = DCT(zeros(dim_in))
DCT(dim_in::Vararg{Int64}) = DCT(dim_in)
DCT(T::Type, dim_in::Vararg{Int64}) = DCT(T,dim_in)

function DCT{N,C<:RealOrComplex}(x::AbstractArray{C,N})
	A,At = plan_dct(x), plan_idct(x)
	DCT{N,C,typeof(A),typeof(At)}(size(x),A,At)
end

#standard constructor
IDCT{N}(T::Type,dim_in::NTuple{N,Int}) = IDCT(zeros(T,dim_in))

IDCT{N}(dim_in::NTuple{N,Int}) = IDCT(zeros(dim_in))
IDCT(dim_in::Vararg{Int64}) = IDCT(dim_in)
IDCT(T::Type, dim_in::Vararg{Int64}) = IDCT(T,dim_in)

function IDCT{N,C<:RealOrComplex}(x::AbstractArray{C,N})
	A,At = plan_idct(x), plan_dct(x)
	IDCT{N,C,typeof(A),typeof(At)}(size(x),A,At)
end


# Mappings

function A_mul_B!{N,C,T1,T2}(y::AbstractArray{C,N},A::DCT{N,C,T1,T2},b::AbstractArray{C,N})
	A_mul_B!(y,A.A,b)
end

function Ac_mul_B!{N,C,T1,T2}(y::AbstractArray{C,N},A::DCT{N,C,T1,T2},b::AbstractArray{C,N})
	y .= A.At*b
end

function A_mul_B!{N,C,T1,T2}(y::AbstractArray{C,N},A::IDCT{N,C,T1,T2},b::AbstractArray{C,N})
	y .= A.A*b
end

function Ac_mul_B!{N,C,T1,T2}(y::AbstractArray{C,N},A::IDCT{N,C,T1,T2},b::AbstractArray{C,N})
	A_mul_B!(y,A.At,b)
end

# Properties

size(L::CosineTransform) = (L.dim_in,L.dim_in)

fun_name(A::DCT)  = "ℱc"
fun_name(A::IDCT) = "ℱc^(-1)"

domainType{N,C,T1,T2}(L::CosineTransform{N,C,T1,T2}) = C
codomainType{N,C,T1,T2}(L::CosineTransform{N,C,T1,T2}) = C

is_gram_diagonal(L::CosineTransform) = true
