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
struct DCT{N,
	   C<:RealOrComplex,
	   T1<:AbstractFFTs.Plan,
	   T2<:AbstractFFTs.Plan} <: CosineTransform{N,C,T1,T2}
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
ℱc⁻¹  ℂ^(10, 10) -> ℂ^(10, 10) 

julia> IDCT(10,10)
ℱc⁻¹  ℝ^(10, 10) -> ℂ^(10, 10) 

julia> A = IDCT(ones(3))
ℱc⁻¹  ℝ^3 -> ℝ^3

julia> A*[1.;0.;0.]
3-element Array{Float64,1}:
 0.57735
 0.57735
 0.57735

```

"""
struct IDCT{N,
	    C<:RealOrComplex,
	    T1<:AbstractFFTs.Plan,
	    T2<:AbstractFFTs.Plan} <: CosineTransform{N,C,T1,T2}
	dim_in::NTuple{N,Int}
	A::T1
	At::T2
end

# Constructors
#standard constructor
DCT(T::Type,dim_in::NTuple{N,Int}) where {N} = DCT(zeros(T,dim_in))

DCT(dim_in::NTuple{N,Int}) where {N} = DCT(zeros(dim_in))
DCT(dim_in::Vararg{Int64}) = DCT(dim_in)
DCT(T::Type, dim_in::Vararg{Int64}) = DCT(T,dim_in)

function DCT(x::AbstractArray{C,N}) where {N,C<:RealOrComplex}
	A,At = plan_dct(x), plan_idct(x)
	DCT{N,C,typeof(A),typeof(At)}(size(x),A,At)
end

#standard constructor
IDCT(T::Type,dim_in::NTuple{N,Int}) where {N} = IDCT(zeros(T,dim_in))

IDCT(dim_in::NTuple{N,Int}) where {N} = IDCT(zeros(dim_in))
IDCT(dim_in::Vararg{Int64}) = IDCT(dim_in)
IDCT(T::Type, dim_in::Vararg{Int64}) = IDCT(T,dim_in)

function IDCT(x::AbstractArray{C,N}) where {N,C<:RealOrComplex}
	A,At = plan_idct(x), plan_dct(x)
	IDCT{N,C,typeof(A),typeof(At)}(size(x),A,At)
end


# Mappings

function mul!(y::AbstractArray{C,N},A::DCT{N,C,T1,T2},b::AbstractArray{C,N}) where {N,C,T1,T2}
	mul!(y,A.A,b)
end

function mul!(y::AbstractArray{C,N},A::AdjointOperator{DCT{N,C,T1,T2}},b::AbstractArray{C,N}) where {N,C,T1,T2}
	y .= A.A.At*b
end

function mul!(y::AbstractArray{C,N},A::IDCT{N,C,T1,T2},b::AbstractArray{C,N}) where {N,C,T1,T2}
	y .= A.A*b
end

function mul!(y::AbstractArray{C,N},A::AdjointOperator{IDCT{N,C,T1,T2}},b::AbstractArray{C,N}) where {N,C,T1,T2}
	mul!(y,A.A.At,b)
end

# Properties

size(L::CosineTransform) = (L.dim_in,L.dim_in)

fun_name(A::DCT)  = "ℱc"
fun_name(A::IDCT) = "ℱc⁻¹"

domainType(L::CosineTransform{N,C,T1,T2}) where {N,C,T1,T2} = C
codomainType(L::CosineTransform{N,C,T1,T2}) where {N,C,T1,T2} = C

is_AcA_diagonal(L::CosineTransform)     = true
is_AAc_diagonal(L::CosineTransform)     = true
is_orthogonal(  L::CosineTransform)     = true
is_invertible(L::CosineTransform)       = true
is_full_row_rank(L::CosineTransform)    = true
is_full_column_rank(L::CosineTransform) = true

diag_AcA(L::CosineTransform) = 1.
diag_AAc(L::CosineTransform) = 1.
