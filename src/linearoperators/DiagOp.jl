export DiagOp

"""
`DiagOp(domainType::Type, dim_in::Tuple, d::AbstractArray)`

`DiagOp(d::AbstractArray)`

Creates a `LinearOperator` which, when multiplied with an array `x`, returns the elementwise product `d.*x`.

```julia
julia> D = DiagOp(Float64, (2, 2,), [1. 2.; 3. 4.])
╲  ℝ^(2, 2) -> ℝ^(2, 2)

julia> D*ones(2,2)
2×2 Array{Float64,2}:
 1.0  2.0
 3.0  4.0

```

"""

immutable DiagOp{T,N,D <: AbstractArray{T,N}} <: LinearOperator
	d::D
end


# Constructors

##TODO decide what to do when domainType is given, with conversion one loses pointer to data...
###standard constructor Operator{N}(DomainType::Type, DomainDim::NTuple{N,Int})
function DiagOp{N, D <: AbstractArray}(DomainType::Type, DomainDim::NTuple{N,Int}, d::D)  
	size(d) != DomainDim && error("dimension of d must coincide with DomainDim")
	DiagOp{DomainType, N, D}(d)
end
###

DiagOp{A <: AbstractArray}(d::A) = DiagOp(eltype(d),size(d),d)

# Mappings

function A_mul_B!{T,N,D}(y::AbstractArray{T,N}, L::DiagOp{T,N,D}, b::AbstractArray{T,N})
	y .= (*).(L.d, b)
end

function Ac_mul_B!{T,N,D}(y::AbstractArray{T,N}, L::DiagOp{T,N,D}, b::AbstractArray{T,N})
	y .= (*).(conj.(L.d), b)
end

# Transformations (we'll see about this)
# inv(L::DiagOp) = DiagOp(L.domainType, L.dim_in, (L.d).^(-1))

# Properties

diag(L::DiagOp) = L.d
diag_AAc(L::DiagOp) = L.d.*conj.(L.d)
diag_AcA(L::DiagOp) = conj.(L.d).*L.d

domainType{T,N,D}(L::DiagOp{T,N,D}) = T
codomainType{T,N,D}(L::DiagOp{T,N,D}) = T

size(L::DiagOp) = (size(L.d), size(L.d))

fun_name(L::DiagOp) = "╲"

is_diagonal(L::DiagOp) = true

# TODO: probably the following allows for too-close-to-singular matrices
is_invertible(L::DiagOp) = all(L.d .!= 0.)
is_full_row_rank(L::DiagOp) = is_invertible(L)
is_full_column_rank(L::DiagOp) = is_invertible(L)
