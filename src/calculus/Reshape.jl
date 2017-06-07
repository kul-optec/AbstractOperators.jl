immutable Reshape{N,L<:LinearOperator} <: LinearOperator
	A::L
	dim_out::NTuple{N,Int}
end

# Constructors

Reshape{N,L<:LinearOperator}(A::L, dim_out::Vararg{Int,N}) =
Reshape{N,L}(A, dim_out)

# Mappings

function A_mul_B!{N,L,C,D}(y::C, R::Reshape{N,L}, b::D)
	y_res = reshape(y,size(R.A,1))
	b_res = reshape(b,size(R.A,2))
	A_mul_B!(y_res, R.A, b_res)
end

function Ac_mul_B!{N,L,C,D}(y::D, R::Reshape{N,L}, b::C)
	y_res = reshape(y,size(R.A,2))
	b_res = reshape(b,size(R.A,1))
	Ac_mul_B!(y_res, R.A, b_res)
end

# Properties

size(R::Reshape) = (R.dim_out, size(R.A,2))

  domainType(  R::Reshape) =   domainType(R.A)
codomainType(  R::Reshape) = codomainType(R.A)

is_diagonal(    R::Reshape) = is_diagonal(R.A)
is_gram_diagonal(R::Reshape) = is_gram_diagonal(R.A)
is_invertible(  R::Reshape) = is_invertible(R.A)

fun_name(R::Reshape) = "¶"*fun_name(R.A)
