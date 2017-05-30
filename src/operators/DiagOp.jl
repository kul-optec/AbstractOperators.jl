export DiagOp

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
DiagOp{A <: AbstractArray}(T::Type, d::A) = DiagOp(T,size(d),d)

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

domainType{T,N,D}(L::DiagOp{T,N,D}) = T
codomainType{T,N,D}(L::DiagOp{T,N,D}) = T

size(L::DiagOp) = (size(L.d), size(L.d))

fun_name(L::DiagOp) = "Diagonal Operator"

is_diagonal(L::DiagOp) = true

# TODO: probably the following allows for too-close-to-singular matrices
is_invertible(L::DiagOp) = all(L.d .!= 0.)
is_full_row_rank(L::DiagOp) = is_invertible(L)
is_full_column_rank(L::DiagOp) = is_invertible(L)
