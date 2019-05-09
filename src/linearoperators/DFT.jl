export DFT, IDFT

abstract type FourierTransform{N,C,D,E,T1,T2} <: LinearOperator end

"""
`DFT([domainType=Float64::Type,] dim_in::Tuple [,dims])`

`DFT(dim_in...)`

`DFT(x::AbstractArray, [,dims])`

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the `N`-dimensional Discrete Fourier Transform over dimensions `dims` of `x`.

```julia
julia> DFT(Complex{Float64},(10,10))
ℱ  ℂ^(10, 10) -> ℂ^(10, 10)

julia> DFT(10,10)
ℱ  ℝ^(10, 10) -> ℂ^(10, 10)

julia> A = DFT(ones(3))
ℱ  ℝ^3 -> ℂ^3

julia> A*ones(3)
3-element Array{Complex{Float64},1}:
 3.0+0.0im
 0.0+0.0im
 0.0+0.0im
```

"""
struct DFT{N,
	      C<:RealOrComplex,
	      D<:RealOrComplex,
		  E,
	      T1<:AbstractFFTs.Plan,
	      T2<:AbstractFFTs.Plan} <: FourierTransform{N,C,D,E,T1,T2}
	dim_in::NTuple{N,Int}
	A::T1
	At::T2
end


"""
`IDFT([domainType=Float64::Type,] dim_in::Tuple)`

`IDFT(dim_in...)`

`IDFT(x::AbstractArray)`

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractArray{N}`, returns the `N`-dimensional Inverse Discrete Fourier Transform over dimensions `dims` of `x`.

```julia
julia> IDFT(Complex{Float64},(10,10))
ℱ⁻¹  ℂ^(10, 10) -> ℂ^(10, 10)

julia> IDFT(10,10)
ℱ⁻¹ ℝ^(10, 10) -> ℂ^(10, 10)

julia> A = IDFT(ones(3))
ℱ⁻¹  ℝ^3 -> ℂ^3

julia> A*ones(3)
3-element Array{Complex{Float64},1}:
 1.0+0.0im
 0.0+0.0im
 0.0+0.0im
```

"""
struct IDFT{N,
	       C<:RealOrComplex,
	       D<:RealOrComplex,
		   E,
	       T1<:AbstractFFTs.Plan,
	       T2<:AbstractFFTs.Plan} <: FourierTransform{N,C,D,E,T1,T2}
	dim_in::NTuple{N,Int}
	A::T1
	At::T2
end

# Constructors
#standard constructor
DFT(dim_in::NTuple{N,Int},dims=1:N) where {N} = DFT(zeros(dim_in),dims)

function DFT(x::AbstractArray{D,N},dims=1:ndims(x)) where {N,D<:Real}
	y2 = zeros(complex(D),size(x))
	A = plan_fft(x,dims)
	At = plan_bfft(y2,dims)
	dim_in = size(x)
	DFT{N,Complex{D},D,dims,typeof(A),typeof(At)}(dim_in,A,At)
end

function DFT(x::AbstractArray{D,N},dims=1:ndims(x)) where {N,D<:Complex}
	y2 = zeros(D,size(x))
	A = plan_fft(x,dims)
	At = plan_bfft(y2,dims)
	dim_in = size(x)
	DFT{N,D,D,dims,typeof(A),typeof(At)}(dim_in,A,At)
end

DFT(T::Type,dim_in::NTuple{N,Int},dims=1:N) where {N} = DFT(zeros(T,dim_in),dims)
DFT(dim_in::Vararg{Int}) = DFT(dim_in)
DFT(T::Type,dim_in::Vararg{Int}) = DFT(T,dim_in)

function IDFT(x::AbstractArray{D,N},dims=1:ndims(x)) where {N,D<:Real}
	y2 = zeros(complex(D),size(x))
	A = plan_ifft(x,dims)
	At = plan_fft(y2,dims)
	dim_in = size(x)
	IDFT{N,Complex{D},D,dims,typeof(A),typeof(At)}(dim_in,A,At)
end

#standard constructor
IDFT(T::Type,dim_in::NTuple{N,Int},dims=1:N) where {N} = IDFT(zeros(T,dim_in),dims)

function IDFT(x::AbstractArray{D,N},dims=1:ndims(x)) where {N,D<:Complex}
	y2 = zeros(D,size(x))
	A = plan_ifft(x,dims)
	At = plan_fft(y2,dims)
	dim_in = size(x)
	IDFT{N,D,D,dims,typeof(A),typeof(At)}(dim_in,A,At)
end

IDFT(dim_in::NTuple{N,Int},dims=1:N) where {N} = IDFT(zeros(dim_in),dims)
IDFT(dim_in::Vararg{Int}) = IDFT(dim_in)
IDFT(T::Type,dim_in::Vararg{Int}) = IDFT(T,dim_in)

# Mappings

function mul!(y::AbstractArray{C,N},
              L::DFT{N,C,C,E,T1,T2},
              b::AbstractArray{C,N}) where {N,C<:Complex,E,T1,T2}
	mul!(y,L.A,b)
end

function mul!(y::AbstractArray{C,N},
              L::DFT{N,C,D,E,T1,T2},
              b::AbstractArray{D,N}) where {N,C<:Complex,D<:Real,E,T1,T2}
	mul!(y,L.A,complex(b))
end

function mul!(y::AbstractArray{C,N},
              L::AdjointOperator{DFT{N,C,C,E,T1,T2}},
              b::AbstractArray{C,N}) where {N,C<:Complex,E,T1,T2}
	mul!(y,L.A.At,b)
end

function mul!(y::AbstractArray{D,N},
              L::AdjointOperator{DFT{N,C,D,E,T1,T2}},
              b::AbstractArray{C,N}) where {N,C<:Complex,D<:Real,E,T1,T2}
	y2 = complex(y)
	mul!(y2,L.A.At,b)
	y .= real.(y2)
end

function mul!(y::AbstractArray{C,N},
              L::IDFT{N,C,C,E,T1,T2},
              b::AbstractArray{C,N}) where {N,C<:Complex,E,T1,T2}
	mul!(y,L.A,b)
end

function mul!(y::AbstractArray{C,N},
              L::IDFT{N,C,D,E,T1,T2},
              b::AbstractArray{D,N}) where {N,C<:Complex,D<:Real,E,T1,T2}
	mul!(y,L.A,complex(b))
end

function mul!(y::AbstractArray{C,N},
              L::AdjointOperator{IDFT{N,C,C,E,T1,T2}},
              b::AbstractArray{C,N}) where {N,C<:Complex,E,T1,T2}
	mul!(y,L.A.At,b)
	y ./= prod(size(b,e) for e in E)
end

function mul!(y::AbstractArray{D,N},
              L::AdjointOperator{IDFT{N,C,D,E,T1,T2}},
              b::AbstractArray{C,N}) where {N,C<:Complex,D<:Real,E,T1,T2}

	y2 = complex(y)
	mul!(y2,L.A.At,b)
	y .= (/).(real.(y2), prod(size(b,e) for e in E))
end

# Properties

size(L::FourierTransform) = (L.dim_in,L.dim_in)

fun_name(A::DFT) = "ℱ"
fun_name(A::IDFT) = "ℱ⁻¹"

domainType(L::FourierTransform{N,C,D,E,T1,T2}) where {N,C,D,E,T1,T2} = D
codomainType(L::FourierTransform{N,C,D,E,T1,T2}) where {N,C,D,E,T1,T2} = C

is_AcA_diagonal(L::FourierTransform)     = true
is_AAc_diagonal(L::FourierTransform)     = true
is_invertible(L::FourierTransform)       = true
is_full_row_rank(L::FourierTransform)    = true
is_full_column_rank(L::FourierTransform) = true

diag_AcA(L::DFT) = float(prod(size(L,1)))
diag_AAc(L::DFT) = float(prod(size(L,1)))

diag_AcA(L::IDFT) = 1/prod(size(L,1))
diag_AAc(L::IDFT) = 1/prod(size(L,1))
