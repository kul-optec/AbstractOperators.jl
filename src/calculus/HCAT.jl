
export HCAT

immutable HCAT{M, N, K,
	       L <: NTuple{N,AbstractOperator},
	       S <: NTuple{K,Tuple},
	       I <: NTuple{K,Int},
	       P <: NTuple{N,Any},
	       D <: NTuple{K,Type},
	       C <: Union{NTuple{M,AbstractArray}, AbstractArray},
	       } <: AbstractOperator
	A::L
	size_in::S
	input_idx::I
	partitions::P
	domain::D
	mid::C
end

# Constructors

function HCAT{N, K, 
	      L <: NTuple{N,AbstractOperator},
	      S <: NTuple{K,Tuple},
	      I <: NTuple{K,Int},
	      P <: NTuple{N,Any},
	      D <: NTuple{K,Type},
	      C
	      }(A::L, size_in::S, input_idx::I, partitions::P, domain::D, mid::C, M::Int)
	if any([size(A[1],1) != size(a,1) for a in A])
		throw(DimensionMismatch("operators must have the same codomain dimension!"))
	end
	if any([codomainType(A[1]) != codomainType(a) for a in A])
		throw(error("operators must all share the same codomainType!"))
	end
	HCAT{M,N,K,L,S,I,P,D,C}(A, size_in, input_idx, partitions, domain, mid)
end

function HCAT(A::Vararg{AbstractOperator})

	K = 0
	input_idx  = ()
	partitions = ()
	domain     = ()
	size_in = []
	for i in eachindex(ndoms.(A,2))
		if ndoms(A[i],2) == 1
			K += 1
			input_idx  = (input_idx... ,K)
			partitions = (partitions...,K)
			domain     = (domain...    ,domainType(A[i]))
			push!(size_in,size(A[i],2))
		else
			partitions = (partitions...,(collect(K+1:K+ndoms(A[i],2))...))
			for ii = 1:ndoms(A[i],2)
				K += 1
				input_idx  = (input_idx... ,K)
				domain     = (domain...    ,domainType(A[i])[ii])
				push!(size_in,size(A[i],2)[ii])
			end
		end
	end
	size_in = (size_in...)

	s = size(A[1],1)
	t = codomainType(A[1])
	mid, M  = create_mid(t,s)

	return HCAT(A, size_in, input_idx, partitions, domain, mid, M)
end

HCAT(A::AbstractOperator) = A

get_M{M}(H::HCAT{M}) = M
create_mid{N}(t::NTuple{N,DataType},s::NTuple{N,NTuple}) = zeros.(t,s), N
create_mid{N}(t::Type,s::NTuple{N,Int}) = zeros(t,s), 1

# Mappings

@generated function A_mul_B!{M,N,K,L,S,I,P,D,C,DD}(y::C, H::HCAT{M,N,K,L,S,I,P,D,C}, b::DD)

	ex = :()

	if fieldtype(P,1) <: Int
		idx = :(H.input_idx[H.partitions[1]])  
		ex = :($ex; A_mul_B!(y,H.A[1],b[$idx]))
	else
	
		bb = ""
		for ii in eachindex(fieldnames(fieldtype(P,1)))
			bb = bb*"b[H.input_idx[H.partitions[1][$ii]]],"
		end
		bb = parse(bb)
		ex = :($ex; A_mul_B!(y,H.A[1],$bb))
	end

	for i = 2:N
	
		if fieldtype(P,i) <: Int
			idx = :(H.input_idx[H.partitions[$i]])  
			ex = :($ex; A_mul_B!(H.mid,H.A[$i],b[$idx]))
		else
		
			bb = ""
			for ii in eachindex(fieldnames(fieldtype(P,i)))
				bb = bb*"b[H.input_idx[H.partitions[$i][$ii]]],"
			end
			bb = parse(bb)
			ex = :($ex; A_mul_B!(H.mid,H.A[$i],$bb))
		end
		
		if C <: AbstractArray
			ex = :($ex; y .+= H.mid)
		else
			for ii = 1:M
				ex = :($ex; y[$ii] .+= H.mid[$ii])
			end
		end
	end
	ex = :($ex; return y)
	return ex

end

@generated function Ac_mul_B!{M,N,K,L,S,I,P,D,C,DD}(y::DD, H::HCAT{M,N,K,L,S,I,P,D,C}, b::C)

	ex = :()

	for i = 1:N
		if fieldtype(P,i) <: Int
			idx = :(H.input_idx[H.partitions[$i]])  
			ex = :($ex; Ac_mul_B!(y[$idx],H.A[$i],b))
		else
		
			yy = ""
			for ii in eachindex(fieldnames(fieldtype(P,i)))
				yy = yy*"y[H.input_idx[H.partitions[$i][$ii]]],"
			end
			yy = parse(yy)
			ex = :($ex; Ac_mul_B!($yy,H.A[$i],b))
		end
	end
	ex = :($ex; return y)
	return ex

end

# Properties

size(L::HCAT) = size(L.A[1],1), L.size_in[[L.input_idx...]]

fun_name(L::HCAT) = length(L.A) == 2 ? "["fun_name(L.A[1])*","*fun_name(L.A[2])*"]" : "HCAT"

domainType(L::HCAT) = L.domain[[L.input_idx...]]
codomainType(L::HCAT) = codomainType.(L.A[1])

is_linear(L::HCAT) = all(is_linear.(L.A))
is_AAc_diagonal(L::HCAT) = all(is_AAc_diagonal.(L.A))
is_full_row_rank(L::HCAT) = any(is_full_row_rank.(L.A))

diag_AAc(L::HCAT) = sum(diag_AAc.(L.A))
