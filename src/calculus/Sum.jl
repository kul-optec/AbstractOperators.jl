export Sum

immutable Sum{M, N, K,
	      C <: Union{NTuple{M,AbstractArray}, AbstractArray},
	      D <: Union{NTuple{N,AbstractArray}, AbstractArray},
	      L<:NTuple{K,AbstractOperator}} <: AbstractOperator
	A::L
	midC::C
	midD::D
end

# Constructors

function Sum{C, D, K, L <: NTuple{K,AbstractOperator}}(A::L, midC::C, midD::D, M::Int, N::Int)
	if any([size(a) != size(A[1]) for a in A])
		throw(DimensionMismatch("cannot sum operator of different sizes"))
	end
	if any([codomainType(A[1]) != codomainType(a) for a in A]) ||
	   any([codomainType(A[1]) != codomainType(a) for a in A])
		throw(DomainError())
	end
	Sum{M, N, K, C, D, L}(A, midC, midD)
end

Sum(A::AbstractOperator) = A

function Sum(A::Vararg{AbstractOperator})
	s = size(A[1],1)
	t = codomainType(A[1])
	midC,M  = create_mid(t,s)

	s = size(A[1],2)
	t = domainType(A[1])
	midD,N  = create_mid(t,s)

	return Sum(A, midC, midD, M, N)
end

# special cases
Sum{M,N,K,C,D}(L1::AbstractOperator, L2::Sum{M,N,K,C,D}           ) =
Sum((L1,L2.A...),L2.midC,L2.midD, M, N)

# Mappings

@generated function A_mul_B!{M,N,K,C,D}(y::C, S::Sum{M,N,K,C,D}, b::D)
	ex = :(A_mul_B!(y,S.A[1],b))
	for i = 2:K
		ex = quote
			$ex
			A_mul_B!(S.midC,S.A[$i],b)
		end
		if C <: AbstractArray
			ex = :($ex; y .+= S.midC)
		else
			for ii = 1:M
				ex = :($ex; y[$ii] .+= S.midC[$ii])
			end
		end
	end
	ex = quote
		$ex
		return y
	end
end

@generated function Ac_mul_B!{M,N,K,C,D}(y::D, S::Sum{M,N,K,C,D}, b::C)
	ex = :(Ac_mul_B!(y,S.A[1],b))
	for i = 2:K
		ex = quote
			$ex
			Ac_mul_B!(S.midD,S.A[$i],b)
		end
		if D <: AbstractArray
			ex = :($ex; y .+= S.midD)
		else
			for ii = 1:N
				ex = :($ex; y[$ii] .+= S.midD[$ii])
			end
		end
	end
	ex = quote
		$ex
		return y
	end
end

# Properties

size(L::Sum) = size(L.A[1])

  domainType{M,N,K,C,D<:AbstractArray,L}(S::Sum{M, N, K, C, D, L}) =    domainType(S.A[1])
  domainType{M,N,K,C,D<:Tuple        ,L}(S::Sum{M, N, K, C, D, L}) =   domainType.(S.A[1])
codomainType{M,N,K,C<:AbstractArray,D,L}(S::Sum{M, N, K, C, D, L}) =  codomainType(S.A[1])
codomainType{M,N,K,C<:Tuple        ,D,L}(S::Sum{M, N, K, C, D, L}) = codomainType.(S.A[1])

fun_domain(S::Sum)   = fun_domain(S.A[1])
fun_codomain(S::Sum) = fun_codomain(S.A[1])

fun_name(S::Sum) =
length(S.A) == 2 ? fun_name(S.A[1])"+"fun_name(S.A[2]) : "Σ"


is_linear(L::Sum)        = all(is_linear.(L.A))            
is_null(L::Sum)          = all(is_null.(L.A))            
is_diagonal(L::Sum)      = all(is_diagonal.(L.A))        
is_full_row_rank(L::Sum) = any(is_full_row_rank.(L.A))
is_full_column_rank(L::Sum) = any(is_full_column_rank.(L.A))

diag(L::Sum) = sum(diag.(L.A))


