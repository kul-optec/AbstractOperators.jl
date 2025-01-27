export IRDFT

"""
	IRDFT([domainType=Float64::Type,] dim_in::Tuple, d::Int, [,dims=1])
	IRDFT(x::AbstractArray, d::Int, [,dims=1])

Creates a `LinearOperator` which, when multiplied with a complex array `x`, returns the IDFT over the dimension `dims`, exploiting Hermitian symmetry. Like in the function `BASE.irfft`, `d` must satisfy `div(d,2)+1 == size(x,dims)`.

```julia
julia> A = IRDFT(Complex{Float64},(10,),19)
ℱ⁻¹  ℂ^10 -> ℝ^19

julia> A = IRDFT((5,10,8),19,2)
ℱ⁻¹  ℂ^(5, 10, 8) -> ℝ^(5, 19, 8)
	
```
"""
struct IRDFT{T<:Number,N,D,T1<:AbstractFFTs.Plan,T2<:AbstractFFTs.Plan,T3<:NTuple{N,Any}} <:
	   LinearOperator
	dim_in::NTuple{N,Int}
	dim_out::NTuple{N,Int}
	A::T1
	At::T2
	idx::T3
end

# Constructors
#standard constructor

function IRDFT(x::AbstractArray{Complex{T},N}, d::Int, dims::Int=1) where {T<:Number,N}
	A = plan_irfft(x, d, dims)
	dim_in = size(x)
	dim_out = ()
	idx = ()
	for i in 1:N
		dim_out = i == dims ? (dim_out..., d) : (dim_out..., dim_in[i])
		idx = i == dims ? (idx..., 2:ceil(Int, d / 2)) : (idx..., Colon())
	end
	At = plan_rfft(similar(x, T, dim_out), dims)
	return IRDFT{T,N,dims,typeof(A),typeof(At),typeof(idx)}(dim_in, dim_out, A, At, idx)
end

function IRDFT(T::Type, dim_in::NTuple{N,Int}, d::Int, dims::Int=1) where {N}
	return IRDFT(zeros(T, dim_in), d, dims)
end
function IRDFT(dim_in::NTuple{N,Int}, d::Int, dims::Int=1) where {N}
	return IRDFT(zeros(Complex{Float64}, dim_in), d, dims)
end

# Mappings

function mul!(
	y::C1, L::IRDFT{T,N,D,T1,T2,T3}, b::C2
) where {N,T,D,T1,T2,T3,C1<:AbstractArray{T,N},C2<:AbstractArray{Complex{T},N}}
	return mul!(y, L.A, b)
end

function mul!(
	y::C2, L::AdjointOperator{IRDFT{T,N,D,T1,T2,T3}}, b::C1
) where {N,T,D,T1,T2,T3,C1<:AbstractArray{T,N},C2<:AbstractArray{Complex{T},N}}
	A = L.A
	mul!(y, A.At, b)
	y ./= size(b, D)
	@views y[A.idx...] .*= 2
	return y
end

# Properties

size(L::IRDFT) = (L.dim_out, L.dim_in)

fun_name(A::IRDFT) = "ℱ⁻¹"

domainType(::IRDFT{T}) where {T} = Complex{T}
codomainType(::IRDFT{T}) where {T} = T

is_AAc_diagonal(L::IRDFT) = false #TODO but might be true?
is_invertible(L::IRDFT) = true
is_full_row_rank(L::IRDFT) = true
