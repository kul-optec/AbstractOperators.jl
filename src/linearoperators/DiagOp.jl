export DiagOp

"""
	DiagOp(domainType::Type, dim_in::Tuple, d::AbstractArray)
	DiagOp(d::AbstractArray)

Creates a `LinearOperator` which, when multiplied with an array `x`, returns the elementwise product `d.*x`.

```jldoctest
julia> D = DiagOp(Float64, (2, 2,), [1. 2.; 3. 4.])
╲  ℝ^(2, 2) -> ℝ^(2, 2)

julia> D*ones(2,2)
2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0
	
```
"""
struct DiagOp{N,D,C,T<:Union{AbstractArray{C,N},Number}} <: LinearOperator
	dim_in::NTuple{N,Int}
	d::T
end

# Constructors

###standard constructor Operator{N}(DomainType::Type, DomainDim::NTuple{N,Int})
function DiagOp(DomainType::Type, DomainDim::NTuple{N,Int}, d::T) where {N,T<:AbstractArray}
	size(d) != DomainDim && error("dimension of d must coincide with DomainDim")
	C = eltype(d) <: Complex ? complex(DomainType) : DomainType
	return DiagOp{N,DomainType,C,T}(DomainDim, d)
end

###standard constructor with Scalar
function DiagOp(DomainType::Type, DomainDim::NTuple{N,Int}, d::T) where {N,T<:Number}
	C = eltype(d) <: Complex ? Complex{DomainType} : DomainType
	return DiagOp{N,DomainType,C,T}(DomainDim, d)
end

# other constructors
DiagOp(d::A) where {A<:AbstractArray} = DiagOp(eltype(d), size(d), d)
DiagOp(DomainDim::NTuple{N,Int}, d::A) where {N,A<:Number} = DiagOp(Float64, DomainDim, d)

# Mappings

function mul!(
	y::AbstractArray{C,N}, L::DiagOp{N,D,C,T}, b::AbstractArray{D,N}
) where {N,D,C,T}
	return y .= L.d .* b
end

function mul!(
	y::AbstractArray{D,N}, L::AdjointOperator{DiagOp{N,D,C,T}}, b::AbstractArray{C,N}
) where {N,D,C,T}
	return y .= conj.(L.A.d) .* b
end

function mul!(
	y::AbstractArray{D,N}, L::AdjointOperator{DiagOp{N,D,C,T}}, b::AbstractArray{C,N}
) where {N,D<:Real,C<:Complex,T}
	return y .= real.(conj.(L.A.d) .* b)
end

# Transformations (we'll see about this)
# inv(L::DiagOp) = DiagOp(L.domainType, L.dim_in, (L.d).^(-1))

# Properties

diag(L::DiagOp) = L.d
diag_AAc(L::DiagOp) = L.d .* conj.(L.d)
diag_AcA(L::DiagOp) = conj.(L.d) .* L.d

domainType(::DiagOp{N,D,C}) where {N,D,C} = D
codomainType(::DiagOp{N,D,C}) where {N,D,C} = C

size(L::DiagOp) = (L.dim_in, L.dim_in)

fun_name(L::DiagOp) = "╲"

is_diagonal(L::DiagOp) = true

# TODO: probably the following allows for too-close-to-singular matrices
is_invertible(L::DiagOp) = all(L.d .!= 0.0)
is_full_row_rank(L::DiagOp) = is_invertible(L)
is_full_column_rank(L::DiagOp) = is_invertible(L)
