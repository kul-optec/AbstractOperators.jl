export DCT, IDCT

abstract type CosineTransform{N,C,T1,T2} <: LinearOperator end

"""
`DCT([domainType=Float64::Type,] dim_in::Tuple)`

`DCT(dim_in...)`

`DCT(x::AbstractArray)`

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the `N`-dimensional Inverse Discrete Cosine Transform of `x`. 

```julia
julia> DCT(Complex{Float64},(10,10))
ℱc  ℂ^(10, 10) -> ℂ^(10, 10) 

julia> DCT(10,10)
ℱc  ℝ^(10, 10) -> ℂ^(10, 10) 

julia> A = DCT(ones(3))
ℱc  ℝ^3 -> ℝ^3

julia> A*ones(3)
3-element Array{Float64,1}:
 1.73205
 0.0
 0.0

```

"""

immutable DCT{N,
	      C<:RealOrComplex,
	      T1<:Base.DFT.Plan,
	      T2<:Base.DFT.Plan} <: CosineTransform{N,C,T1,T2}
	dim_in::NTuple{N,Int}
	A::T1
	At::T2
end

"""
`IDCT([domainType=Float64::Type,] dim_in::Tuple)`

`IDCT(dim_in...)`

`IDCT(x::AbstractArray)`

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the `N`-dimensional Discrete Cosine Transform of `x`. 

```julia
julia> IDCT(Complex{Float64},(10,10))
ℱc^(-1)  ℂ^(10, 10) -> ℂ^(10, 10) 

julia> IDCT(10,10)
ℱc^(-1)  ℝ^(10, 10) -> ℂ^(10, 10) 

julia> A = IDCT(ones(3))
ℱc^(-1)  ℝ^3 -> ℝ^3

julia> A*[1.;0.;0.]
3-element Array{Float64,1}:
 0.57735
 0.57735
 0.57735

```

"""

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

is_AcA_diagonal(L::CosineTransform)     = true
is_AAc_diagonal(L::CosineTransform)     = true
is_orthogonal(  L::CosineTransform)     = true
is_invertible(L::CosineTransform)       = true
is_full_row_rank(L::CosineTransform)    = true
is_full_column_rank(L::CosineTransform) = true

diag_AcA(L::CosineTransform) = 1.
diag_AAc(L::CosineTransform) = 1.
