export HCAT

immutable HCAT{M, N,
	       C <: Union{NTuple{M,AbstractArray}, AbstractArray},
	       D <: NTuple{N,AbstractArray},
	       L <: NTuple{N,LinearOperator}} <: LinearOperator
	A::L
	mid::C
end

# Constructors

function HCAT{N, C<:Union{Tuple,AbstractArray}, L <:NTuple{N,LinearOperator}}(A::L, mid::C, M::Int)
	if any([size(A[1],1) != size(a,1) for a in A])
		throw(DimensionMismatch("operators must have the same codomain dimension!"))
	end
	if any([codomainType(A[1]) != codomainType(a) for a in A])
		throw(error("operators must all share the same codomainType!"))
	end
	domType = domainType.(A)
	D = Tuple{[Array{domType[i],ndims(A[i],2)} for i in eachindex(domType)]...}
	HCAT{M,N,C,D,L}(A, mid)
end

HCAT(A::LinearOperator) = A

function HCAT(A::Vararg{LinearOperator})
	if any((<:).(typeof.(A), HCAT ))
		op = ()
		for a in A
			if typeof(a) <: HCAT
				op = (op...,a.A...)
			else
				op = (op...,a)
			end
		end
		h = A[findfirst((<:).(typeof.(A), HCAT ))]
		return HCAT(op, h.mid, get_M(h))
	else
		s = size(A[1],1)
		t = codomainType(A[1])
		mid,M  = create_mid(t,s)
		return HCAT(A, mid, M)
	end
end

get_M{M}(H::HCAT{M}) = M
create_mid{N}(t::NTuple{N,DataType},s::NTuple{N,NTuple}) = zeros.(t,s), N
create_mid{N}(t::Type,s::NTuple{N,Int}) = zeros(t,s), 1

# Mappings

@generated function A_mul_B!{M,N,C,D,L}(y::C, S::HCAT{M,N,C,D,L}, b::D)
	ex = :(A_mul_B!(y, S.A[1], b[1]))
	for i = 2:N
		ex = quote
			$ex
			A_mul_B!(S.mid, S.A[$i], b[$i])
		end

		if C <: AbstractArray
			ex = :($ex; y .+= S.mid)
		else
			for ii = 1:M
				ex = :($ex; y[$ii] .+= S.mid[$ii])
			end
		end
	end
	ex = quote
		$ex
		return y
	end
end

@generated function Ac_mul_B!{M,N,C,D,L}(y::D, H::HCAT{M,N,C,D,L}, b::C)
	ex = :()
	for i = 1:N
		ex = :($ex; Ac_mul_B!(y[$i],H.A[$i],b))
	end
	ex = quote
		$ex
		return y
	end
end

# Properties

size(L::HCAT) = size(L.A[1],1), size.(L.A, 2)

fun_name(L::HCAT) = length(L.A) == 2 ? "["fun_name(L.A[1])*","*fun_name(L.A[2])*"]" : "HCAT"

domainType(L::HCAT) = domainType.(L.A)
codomainType(L::HCAT) = codomainType.(L.A[1])

is_AAc_diagonal(L::HCAT) = all(is_AAc_diagonal.(L.A))
is_full_row_rank(L::HCAT) = any(is_full_row_rank.(L.A))

diag_AAc(L::HCAT) = sum(diag_AAc.(L.A))
