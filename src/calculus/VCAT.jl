

export VCAT

immutable VCAT{M, N, 
	       L <: NTuple{N,AbstractOperator},
	       P <: NTuple{N,Union{Int,Tuple}},
	       D <: Union{NTuple{M,AbstractArray}, AbstractArray},
	       } <: AbstractOperator
	A::L
	idxs::P
	mid::D
end

# Constructors

function VCAT{N,  
	      L <: NTuple{N,AbstractOperator},
	      P <: NTuple{N,Union{Int,Tuple}},
	      D
	      }(A::L, idxs::P, mid::D, M::Int)
	if any([size(A[1],2) != size(a,2) for a in A])
		throw(DimensionMismatch("operators must have the same codomain dimension!"))
	end
	if any([domainType(A[1]) != domainType(a) for a in A])
		throw(error("operators must all share the same domainType!"))
	end
	VCAT{M,N,L,P,D}(A, idxs, mid)
end

function VCAT(A::Vararg{AbstractOperator})

	if any((<:).(typeof.(A),VCAT)) #fuse VCATS
		AA = ()
		for a in A
			if typeof(a) <: VCAT
				AA = (AA...,a.A...)
			else
				AA = (AA...,a)
			end
		end
		mid = A[findfirst( (<:).(typeof.(A),VCAT) ) ].mid
		M = get_M( A[findfirst( (<:).(typeof.(A),VCAT) ) ]) 
	else
		AA = A
		s = size(AA[1],2)
		t = domainType(AA[1])
		mid, M  = create_mid(t,s)
	end

	K = 0
	idxs = []
	for i in eachindex(ndoms.(AA,1))
		if ndoms(AA[i],1) == 1
			K += 1
			push!(idxs,K)
		else
			idxs = push!(idxs,(collect(K+1:K+ndoms(AA[i],1))...) )
			for ii = 1:ndoms(AA[i],1)
				K += 1
			end
		end
	end

	return VCAT(AA, (idxs...), mid, M)
end

VCAT(A::AbstractOperator) = A

get_M{M}(H::VCAT{M}) = M

# Mappings

@generated function A_mul_B!{M,N,L,P,D,DD}(y::DD, H::VCAT{M,N,L,P,D}, b::D)

	ex = :()

	for i = 1:N

		if fieldtype(P,i) <: Int 
			yy = :(y[H.idxs[$i]])
		else
			yy = ""
			for ii in eachindex(fieldnames(fieldtype(P,i)))
				yy *= "y[H.idxs[$i][$ii]],"
			end
			yy = parse(yy)
		end
		
		ex = :($ex; A_mul_B!($yy,H.A[$i],b))

	end
	ex = :($ex; return y)
	return ex

end

@generated function Ac_mul_B!{M,N,L,P,D,DD}(y::D, H::VCAT{M,N,L,P,D}, b::DD)

	ex = :()

	if fieldtype(P,1) <: Int 
		bb = :(b[H.idxs[1]])
	else
		bb = ""
		for ii in eachindex(fieldnames(fieldtype(P,1)))
			bb *= "b[H.idxs[1][$ii]],"
		end
		bb = parse(bb)
	end
	ex = :($ex; Ac_mul_B!(y,H.A[1],$bb))

	for i = 2:N

		if fieldtype(P,i) <: Int 
			bb = :(b[H.idxs[$i]])
		else
			bb = ""
			for ii in eachindex(fieldnames(fieldtype(P,i)))
				bb *= "b[H.idxs[$i][$ii]],"
			end
			bb = parse(bb)
		end

		ex = :($ex; Ac_mul_B!(H.mid,H.A[$i],$bb))
		
		if D <: AbstractArray
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

# Properties

function size(H::VCAT) 
	size_out = []
	for s in size.(H.A,1)
		eltype(s) <: Int ? push!(size_out,s) : push!(size_out,s...) 
	end
	p = vcat([[idx... ] for idx in H.idxs]...)
	ipermute!(size_out,p)

	(size_out...), size(H.A[1],2)
end

fun_name(L::VCAT) = length(L.A) == 2 ? "["fun_name(L.A[1])*","*fun_name(L.A[2])*"]" : "VCAT"

domainType(L::VCAT) = domainType.(L.A[1])
function codomainType(H::VCAT) 
	codomain = vcat([typeof(d)<:Tuple ? [d...] : d  for d in codomainType.(H.A)]...)
	p = vcat([[idx... ] for idx in H.idxs]...)
	ipermute!(codomain,p)
	return (codomain...)
end

is_linear(L::VCAT) = all(is_linear.(L.A))
is_AcA_diagonal(L::VCAT) = all(is_AcA_diagonal.(L.A))
is_full_column_rank(L::VCAT) = any(is_full_column_rank.(L.A))

diag_AcA(L::VCAT) = sum(diag_AcA.(L.A))

# utils
import Base: permute

function permute{M,N,L,P,D}(H::VCAT{M,N,L,P,D}, p::AbstractVector{Int})


	unfolded = vcat([[idx... ] for idx in H.idxs]...) 
	ipermute!(unfolded,p)

	new_part = ()
	cnt = 0
	for z in length.(H.idxs)
		new_part = (new_part..., z == 1 ? unfolded[cnt+1] : (unfolded[cnt+1:z+cnt]...))
		cnt += z
	end

	VCAT{M,N,L,P,D}(H.A,new_part,H.mid)
end
