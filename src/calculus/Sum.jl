export Sum

struct Sum{M, N, K,
	   C <: Union{NTuple{M,AbstractArray}, AbstractArray},
	   D <: Union{NTuple{N,AbstractArray}, AbstractArray},
	   L<:NTuple{K,AbstractOperator}} <: AbstractOperator
	A::L
	bufC::C
	bufD::D
end

# Constructors

function Sum(A::L, bufC::C, bufD::D, M::Int, N::Int) where {C, D, K, L <: NTuple{K,AbstractOperator}}
	if any([size(a) != size(A[1]) for a in A])
		throw(DimensionMismatch("cannot sum operator of different sizes"))
	end
	if any([codomainType(A[1]) != codomainType(a) for a in A]) ||
	   any([codomainType(A[1]) != codomainType(a) for a in A])
		throw(DomainError())
	end
	Sum{M, N, K, C, D, L}(A, bufC, bufD)
end

Sum(A::AbstractOperator) = A

function Sum(A::Vararg{AbstractOperator})
	s = size(A[1],1)
	t = codomainType(A[1])
	(bufC, M) = eltype(s) <: Int ? (zeros(t,s), 1) : (zeros.(t,s), length(s))

	s = size(A[1],2)
	t = domainType(A[1])
	(bufD, N) = eltype(s) <: Int ? (zeros(t,s), 1) : (zeros.(t,s), length(s))

	return Sum(A, bufC, bufD, M, N)
end

# special cases
Sum(L1::AbstractOperator, L2::Sum{M,N,K,C,D}) where {M,N,K,C,D} =
Sum((L1,L2.A...),L2.bufC,L2.bufD, M, N)

# Mappings

@generated function A_mul_B!(y::C, S::Sum{M,N,K,C,D}, b::D) where {M,N,K,C,D}
	ex = :(A_mul_B!(y,S.A[1],b))
	for i = 2:K
		ex = quote
			$ex
			A_mul_B!(S.bufC,S.A[$i],b)
		end
		if C <: AbstractArray
			ex = :($ex; y .+= S.bufC)
		else
			for ii = 1:M
				ex = :($ex; y[$ii] .+= S.bufC[$ii])
			end
		end
	end
	ex = quote
		$ex
		return y
	end
end

@generated function Ac_mul_B!(y::D, S::Sum{M,N,K,C,D}, b::C) where {M,N,K,C,D}
	ex = :(Ac_mul_B!(y,S.A[1],b))
	for i = 2:K
		ex = quote
			$ex
			Ac_mul_B!(S.bufD,S.A[$i],b)
		end
		if D <: AbstractArray
			ex = :($ex; y .+= S.bufD)
		else
			for ii = 1:N
				ex = :($ex; y[$ii] .+= S.bufD[$ii])
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

  domainType(S::Sum{M, N, K, C, D, L}) where {M,N,K,C,D<:AbstractArray,L} =    domainType(S.A[1])
  domainType(S::Sum{M, N, K, C, D, L}) where {M,N,K,C,D<:Tuple        ,L} =   domainType.(S.A[1])
codomainType(S::Sum{M, N, K, C, D, L}) where {M,N,K,C<:AbstractArray,D,L} =  codomainType(S.A[1])
codomainType(S::Sum{M, N, K, C, D, L}) where {M,N,K,C<:Tuple        ,D,L} = codomainType.(S.A[1])

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


# utils
import Base: permute

function permute(S::Sum{M,N}, p::AbstractVector{Int}) where {M,N}
    AA = ([permute(A,p) for A in S.A]...) 
    return Sum(AA,S.bufC,S.bufD[p],M,N)
end

