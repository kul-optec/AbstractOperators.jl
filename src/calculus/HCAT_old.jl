export FuseHCAT

immutable FuseHCAT{M, N,
	       C <: Union{NTuple{M,AbstractArray}, AbstractArray},
	       L <: NTuple{N,AbstractOperator}} <: AbstractOperator
	A::L
	mid::C
end

# Constructors

function FuseHCAT{N, C<:Union{Tuple,AbstractArray}, L <:NTuple{N,AbstractOperator}}(A::L, mid::C, M::Int)
	if any([size(A[1],1) != size(a,1) for a in A])
		throw(DimensionMismatch("operators must have the same codomain dimension!"))
	end
	if any([codomainType(A[1]) != codomainType(a) for a in A])
		throw(error("operators must all share the same codomainType!"))
	end
	FuseHCAT{M,N,C,L}(A, mid)
end

FuseHCAT(A::AbstractOperator) = A

function FuseHCAT(A::Vararg{AbstractOperator})
	if any((<:).(typeof.(A), FuseHCAT ))
		op = ()
		for a in A
			if typeof(a) <: FuseHCAT
				op = (op...,a.A...)
			else
				op = (op...,a)
			end
		end
		h = A[findfirst((<:).(typeof.(A), FuseHCAT ))]
		return FuseHCAT(op, h.mid, get_M(h))
	else
		s = size(A[1],1)
		t = codomainType(A[1])
		mid,M  = create_mid(t,s)
		return FuseHCAT(A, mid, M)
	end
end


# Mappings

@generated function A_mul_B!{M,N,C,D,L}(y::C, S::FuseHCAT{M,N,C,L}, b::D)
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

@generated function Ac_mul_B!{M,N,C,D,L}(y::D, H::FuseHCAT{M,N,C,L}, b::C)
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

size(L::FuseHCAT) = size(L.A[1],1), size.(L.A, 2)

fun_name(L::FuseHCAT) = length(L.A) == 2 ? "["fun_name(L.A[1])*","*fun_name(L.A[2])*"]" : "FuseHCAT"

domainType(L::FuseHCAT) = domainType.(L.A)
codomainType(L::FuseHCAT) = codomainType.(L.A[1])

is_linear(L::FuseHCAT) = all(is_linear.(L.A))
is_AAc_diagonal(L::FuseHCAT) = all(is_AAc_diagonal.(L.A))
is_full_row_rank(L::FuseHCAT) = any(is_full_row_rank.(L.A))

diag_AAc(L::FuseHCAT) = sum(diag_AAc.(L.A))
