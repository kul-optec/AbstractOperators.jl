export BroadCast

"""
	BroadCast(A::AbstractOperator, dim_out...)

BroadCast the codomain dimensions of an `AbstractOperator`.

```jldoctest
julia> A = Eye(2)
I  ℝ^2 -> ℝ^2

julia> B = BroadCast(A,(2,3))
.I  ℝ^2 -> ℝ^(2, 3)

julia> B*[1.;2.]
2×3 Matrix{Float64}:
 1.0  1.0  1.0
 2.0  2.0  2.0
	
```
"""
struct BroadCast{
	N,
	L<:AbstractOperator,
	T<:AbstractArray,
	D<:AbstractArray,
	M,
	C<:NTuple{M,Colon},
	I<:CartesianIndices,
} <: AbstractOperator
	A::L
	dim_out::NTuple{N,Int}
	bufC::T
	bufD::D
	cols::C
	idxs::I

	function BroadCast(
		A::L, dim_out::NTuple{N,Int}, bufC::T, bufD::D
	) where {N,L<:AbstractOperator,T<:AbstractArray,D<:AbstractArray}
		Base.Broadcast.check_broadcast_shape(dim_out, size(A, 1))
		if size(A, 1) != (1,)
			M = length(size(A, 1))
			cols = ([Colon() for i in 1:M]...,)
			idxs = CartesianIndices((dim_out[(M + 1):end]...,))
			new{N,L,T,D,M,typeof(cols),typeof(idxs)}(A, dim_out, bufC, bufD, cols, idxs)
		else #singleton case
			M = 0
			idxs = CartesianIndices((1,))
			new{N,L,T,D,M,NTuple{0,Colon},typeof(idxs)}(A, dim_out, bufC, bufD, (), idxs)
		end
	end
end

# Constructors

function BroadCast(A::L, dim_out::NTuple{N,Int}) where {N,L<:AbstractOperator}
	return BroadCast(A, dim_out, allocate_in_codomain(A), allocate_in_domain(A))
end

# Mappings

function mul!(y::CC, R::BroadCast{N,L,T,D,M}, b::DD) where {N,L,T,D,M,CC,DD}
	mul!(R.bufC, R.A, b)
	return y .= R.bufC
end

function mul!(
	y::CC, A::AdjointOperator{BroadCast{N,L,T,D,M,C,I}}, b::DD
) where {N,L,T,D,M,C,I,CC,DD}
	R = A.A
	fill!(y, 0.0)
	for i in R.idxs
		@views mul!(R.bufD, R.A', b[R.cols..., i.I...])
		y .+= R.bufD
	end
	return y
end

#singleton
function mul!(
	y::CC, A::AdjointOperator{BroadCast{N,L,T,D,0,C,I}}, b::DD
) where {N,L,T,D,C,I,CC,DD}
	R = A.A
	fill!(y, 0.0)
	bii = allocate_in_codomain(R.A)
	for bi in b
		bii[1] = bi
		mul!(R.bufD, R.A', bii)
		y .+= R.bufD
	end
	return y
end

#TODO make this more general
#length(dim_out) == size(A,1) e.g. a .= b; size(a) = (m,n) size(b) = (1,n) matrix out, column in
function mul!(
	y::CC, A::AdjointOperator{BroadCast{2,L,T,D,2,C,I}}, b::DD
) where {L,T,D,C,I,CC,DD}
	R = A.A
	fill!(y, 0.0)
	for i in 1:size(b, 1)
		@views mul!(R.bufD, R.A, b[i, :]')
		y .+= R.bufD
	end
	return y
end

#singleton Eye
function mul!(
	y::CC, ::AdjointOperator{BroadCast{N,L,T,D,0,C,I}}, b::DD
) where {N,L<:Eye,T,D,C,I,CC,DD}
	return sum!(y, b)
end

# Properties

size(R::BroadCast) = (R.dim_out, size(R.A, 2))

domainType(R::BroadCast) = domainType(R.A)
codomainType(R::BroadCast) = codomainType(R.A)

is_linear(R::BroadCast) = is_linear(R.A)
is_null(R::BroadCast) = is_null(R.A)

fun_name(R::BroadCast) = "." * fun_name(R.A)
function remove_displacement(B::BroadCast)
	return BroadCast(remove_displacement(B.A), B.dim_out, B.bufC, B.bufD)
end
