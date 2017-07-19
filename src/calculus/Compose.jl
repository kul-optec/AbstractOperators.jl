export Compose

immutable Compose{N, M, L<:NTuple{N,Any}, T<:NTuple{M,Any}} <: AbstractOperator
	A::L
	mid::T       # memory in the middle of the operators
end

# Constructors

function Compose(L1::AbstractOperator, L2::AbstractOperator)
	if size(L1,2) != size(L2,1)
		throw(DimensionMismatch("cannot compose operators"))
	end
	if domainType(L1) != codomainType(L2)
		throw(DomainError())
	end
	Compose( L1, L2, Array{domainType(L1)}(size(L2,1)) )
end

Compose{N,M}(A::NTuple{N,Any},mid::NTuple{M,Any}) =
Compose{N,M,typeof(A),typeof(mid)}(A,mid)

Compose(L1::AbstractOperator,L2::AbstractOperator,mid::AbstractArray) =
Compose( (L2,L1), (mid,))

Compose(L1::Compose,       L2::AbstractOperator,mid::AbstractArray) =
Compose( (L2,L1.A...), (mid,L1.mid...))

Compose(L1::AbstractOperator,L2::Compose,       mid::AbstractArray) =
Compose((L2.A...,L1), (L2.mid...,mid))

Compose(L1::Compose,       L2::Compose,       mid::AbstractArray) =
Compose((L2.A...,L1.A...), (L2.mid...,mid,L1.mid...))

#special cases
Compose(L1::Scale,L2::AbstractOperator) = Scale(L1.coeff,L1.A*L2)
Compose(L1::AbstractOperator,L2::Scale) = Scale(L2.coeff,L1*L2.A)
Compose(L1::Scale,L2::Scale) = Scale(*(promote(L1.coeff,L2.coeff)...),L1.A*L2.A)

Compose(L1::AbstractOperator, L2::Eye) = L1
Compose(L1::Eye, L2::AbstractOperator) = L2
Compose(L1::Eye, L2::Eye) = L1

# Mappings

@generated function A_mul_B!{N,M,T1,T2,C,D}(y::C, L::Compose{N,M,T1,T2},b::D)
	ex = :(A_mul_B!(L.mid[1],L.A[1],b))
	for i = 2:M
		ex = quote
			$ex
			A_mul_B!(L.mid[$i],L.A[$i], L.mid[$i-1])
		end
	end
	ex = quote
		$ex
		A_mul_B!(y,L.A[N], L.mid[M])
		return y
	end
end

@generated function Ac_mul_B!{N,M,T1,T2,C,D}(y::D, L::Compose{N,M,T1,T2},b::C)
	ex = :(Ac_mul_B!(L.mid[M],L.A[N],b))
	for i = M:-1:2
		ex = quote
			$ex
			Ac_mul_B!(L.mid[$i-1],L.A[$i], L.mid[$i])
		end
	end
	ex = quote
		$ex
		Ac_mul_B!(y,L.A[1], L.mid[1])
		return y
	end
end
# jacobian 
function jacobian(L::Compose, x::AbstractArray)  
	Compose(jacobian.(L.A,(x,L.mid...)),L.mid)
end

# Properties

size(L::Compose) = ( size(L.A[end],1), size(L.A[1],2) )

fun_name(L::Compose) = length(L.A) == 2 ? fun_name(L.A[2])*"*"*fun_name(L.A[1]) : "Π"

domainType(L::Compose)   = domainType(L.A[1])
codomainType(L::Compose) = codomainType(L.A[end])

is_linear(L::Compose) = all(is_linear.(L.A))
is_diagonal(L::Compose) = all(is_diagonal.(L.A))
is_invertible(L::Compose) = all(is_invertible.(L.A))
