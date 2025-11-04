export DiagOp

"""
	DiagOp(domain_type::Type, dim_in::Tuple, d::AbstractArray)
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
struct DiagOp{N,D,C<:Number,dS,cS,T<:Union{AbstractArray{<:Number,N},Number},B} <: LinearOperator
	dim_in::NTuple{N,Int}
	d::T
end

# Constructors

###standard constructor Operator{N}(D::Type, domain_dim::NTuple{N,Int})
function DiagOp(D::Type, domain_dim::NTuple{N,Int}, d::T; threaded::Bool = true) where {N,T<:AbstractArray}
	size(d) != domain_dim && error("dimension of d must coincide with domain_dim")
	C = promote_type(eltype(d), D)
	threaded = threaded && Threads.nthreads() > 1 && length(d) * sizeof(D) > 2^16
	B = threaded ? FastBroadcast.True() : FastBroadcast.False()
	dS = typeof(d isa SubArray ? parent(d) : d).name.wrapper{D}
	cS = typeof(d isa SubArray ? parent(d) : d).name.wrapper{C}
	return DiagOp{N,D,C,dS,cS,T,B}(domain_dim, d)
end

###standard constructor with Scalar
function DiagOp(D::Type, domain_dim::NTuple{N,Int}, d::T; threaded::Bool = true) where {N,T<:Number}
	C = promote_type(eltype(d), D)
	threaded = threaded && Threads.nthreads() > 1 && length(d) > 2^16
	B = threaded ? FastBroadcast.True() : FastBroadcast.False()
	return DiagOp{N,D,C,Array{D},Array{C},T,B}(domain_dim, d)
end

# other constructors
function DiagOp(d::AbstractArray{T,N}; threaded::Bool = true) where {N,T<:Number}
	C = eltype(d)
	B = (threaded && length(d) > 2^16) ? FastBroadcast.True() : FastBroadcast.False()
	S = typeof(d isa SubArray ? parent(d) : d).name.wrapper{T}
	return DiagOp{N,eltype(d),C,S,S,typeof(d),B}(size(d), d)
end
DiagOp(domain_dim::NTuple{N,Int}, d::A; threaded::Bool = true) where {N,A<:Number} = DiagOp(Float64, domain_dim, d; threaded)

# scale of DiagOp
function Scale(coeff::T, L::DiagOp{N,D,C,dS,cS,T2,B}) where {T<:Number,N,D,C,dS,cS,T2,B}
	if coeff == 1
		return L
	end
	new_d = coeff * diag(L)
	new_C = promote_type(eltype(new_d), D)
	return DiagOp{N,D,new_C,dS,cS,typeof(new_d),B}(L.dim_in, new_d)
end

# Mappings

function mul!(
	y::AbstractArray{C,N}, L::DiagOp{N,D,C,dS,cS,T,B}, b::AbstractArray{D,N}
) where {N,D,C,dS,cS,T,B}
	return @.. thread=B y = L.d * b
end

function mul!(
	y::AbstractArray{D,N}, L::AdjointOperator{DiagOp{N,D,C,dS,cS,T,B}}, b::AbstractArray{C,N}
) where {N,D,C,dS,cS,T,B}
	return @.. thread=B y = conj(L.A.d) * b
end

function mul!(
	y::AbstractArray{D,N}, L::AdjointOperator{DiagOp{N,D,C,dS,cS,T,B}}, b::AbstractArray{C,N}
) where {N,D<:Real,C<:Complex,dS,cS,T,B}
	return @.. thread=B y = real(conj(L.A.d) * b)
end

# Transformations (we'll see about this)
# inv(L::DiagOp) = DiagOp(L.domain_type, L.dim_in, (L.d).^(-1))

# Properties

domain_storage_type(::DiagOp{N,D,C,dS,cS}) where {N,D,C,dS,cS} = dS
codomain_storage_type(::DiagOp{N,D,C,dS,cS}) where {N,D,C,dS,cS} = cS

diag(L::DiagOp) = L.d
diag_AAc(L::DiagOp{N,D,C,dS,cS,T,B}) where {N,D,C,dS,cS,T,B} = @.. thread=B L.d * conj(L.d)
diag_AcA(L::DiagOp{N,D,C,dS,cS,T,B}) where {N,D,C,dS,cS,T,B} = @.. thread=B conj(L.d) * L.d

domain_type(::DiagOp{N,D,C}) where {N,D,C} = D
codomain_type(::DiagOp{N,D,C}) where {N,D,C} = C
is_thread_safe(::DiagOp) = true

size(L::DiagOp) = (L.dim_in, L.dim_in)

fun_name(L::DiagOp) = "╲"

is_diagonal(L::DiagOp) = true

# TODO: probably the following allows for too-close-to-singular matrices
is_invertible(L::DiagOp) = 0 ∉ L.d
is_full_row_rank(L::DiagOp) = is_invertible(L)
is_full_column_rank(L::DiagOp) = is_invertible(L)

has_optimized_normalop(L::DiagOp) = true
has_optimized_normalop(L::AdjointOperator{<:DiagOp}) = true
function get_normal_op(L::DiagOp{N,D,C,dS,cS,T,B}) where {N,D,C,dS,cS,T,B}
	new_d = @.. thread=B L.d * conj(L.d)
	return DiagOp{N,D,D,dS,dS,typeof(new_d),B}(L.dim_in, new_d)
end

has_fast_opnorm(::DiagOp) = true
LinearAlgebra.opnorm(L::DiagOp) = maximum(abs, L.d)
