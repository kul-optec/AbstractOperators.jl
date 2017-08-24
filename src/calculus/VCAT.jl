
export VCAT

immutable VCAT{M, # number of domains  
	       N, # number of AbstractOperator 
	       L <: NTuple{N,AbstractOperator},
	       P <: NTuple{N,Union{Int,Tuple}},
	       C <: Union{NTuple{M,AbstractArray}, AbstractArray},
	       } <: AbstractOperator
	A::L     # tuple of AbstractOperators
	idxs::P  # indices 
	         # H = VCAT(Eye(n),VCAT(Eye(n),Eye(n))) has H.idxs = (1,2,3) 
		 # `AbstractOperators` are flatten
	         # H = VCAT(Eye(n),Compose(MatrixOp(randn(n,n)),VCAT(Eye(n),Eye(n)))) 
		 # has H.idxs = (1,(2,3))
		 # `AbstractOperators` are stack
	buf::C   # buffer memory
end

# Constructors

function VCAT{N,  
	      L <: NTuple{N,AbstractOperator},
	      P <: NTuple{N,Union{Int,Tuple}},
	      C
	      }(A::L, idxs::P, buf::C, M::Int)
	if any([size(A[1],2) != size(a,2) for a in A])
		throw(DimensionMismatch("operators must have the same domain dimension!"))
	end
	if any([domainType(A[1]) != domainType(a) for a in A])
		throw(error("operators must all share the same domainType!"))
	end
	VCAT{M,N,L,P,C}(A, idxs, buf)
end

function VCAT(A::Vararg{AbstractOperator})

	if any((<:).(typeof.(A),VCAT)) #there are VCATs in A
		AA = ()
		for a in A
			if typeof(a) <: VCAT # flatten 
				AA = (AA...,a.A...)
			else                 # stack
				AA = (AA...,a)
			end
		end
		# use buffer from VCAT in A
		buf = A[findfirst( (<:).(typeof.(A),VCAT) ) ].buf
	else 
		AA = A
		s = size(AA[1],2)
		t = domainType(AA[1])
		# generate buffer
		buf = eltype(s) <: Int ? zeros(t,s) : zeros.(t,s)
	end

	return VCAT(AA, buf)
end

function VCAT{N,C}(AA::NTuple{N,AbstractOperator}, buf::C) 
	if N == 1
		return AA[1]
	else
		# get number of domains
		M = C <: AbstractArray ? 1 : length(buf)
		# build H.idxs
		K = 0
		idxs = []
		for i in eachindex(ndoms.(AA,1))
			if ndoms(AA[i],1) == 1 # flatten operator
				K += 1
				push!(idxs,K)
			else                   # stacked operator 
				idxs = push!(idxs,(collect(K+1:K+ndoms(AA[i],1))...) )
				for ii = 1:ndoms(AA[i],1)
					K += 1
				end
			end
		end
		return VCAT(AA, (idxs...), buf, M)
	end
end

VCAT(A::AbstractOperator) = A

# Mappings

@generated function Ac_mul_B!{M,N,L,P,C,DD}(y::C, H::VCAT{M,N,L,P,C}, b::DD)

	ex = :()

	if fieldtype(P,1) <: Int 
		# flatten operator  
		# build Ac_mul_B!(y, H.A[1], b[H.idxs[1]])  
		bb = :(b[H.idxs[1]])
	else
		# staked operator 
		# build Ac_mul_B!(y, H.A[1],( b[H.idxs[1][1]], b[H.idxs[1][2]] ...  ))
		bb = ""
		for ii in eachindex(fieldnames(fieldtype(P,1)))
			bb *= "b[H.idxs[1][$ii]],"
		end
		bb = parse(bb)
	end
	ex = :($ex; Ac_mul_B!(y,H.A[1],$bb)) # write on y

	for i = 2:N

		if fieldtype(P,i) <: Int 
		# flatten operator  
		# build Ac_mul_B!(H.buf, H.A[i], b[H.idxs[i]])  
			bb = :(b[H.idxs[$i]])
		else
		# staked operator 
		# build Ac_mul_B!(H.buf, H.A[i],( b[H.idxs[i][1]], b[H.idxs[i][2]] ...  ))
			bb = ""
			for ii in eachindex(fieldnames(fieldtype(P,i)))
				bb *= "b[H.idxs[$i][$ii]],"
			end
			bb = parse(bb)
		end

		ex = :($ex; Ac_mul_B!(H.buf,H.A[$i],$bb)) # write on H.buf
		
		# sum H.buf with y
		if C <: AbstractArray
			ex = :($ex; y .+= H.buf)
		else
			for ii = 1:M
				ex = :($ex; y[$ii] .+= H.buf[$ii])
			end
		end

	end
	ex = :($ex; return y)
	return ex

end

@generated function A_mul_B!{M,N,L,P,C,DD}(y::DD, H::VCAT{M,N,L,P,C}, b::C)

	ex = :()

	for i = 1:N

		if fieldtype(P,i) <: Int 
		# flatten operator  
		# build A_mul_B!(y[H.idxs[i]], H.A[i], b)  
			yy = :(y[H.idxs[$i]])
		else
		# staked operator 
		# build A_mul_B!(( y[H.idxs[i][1]], y[H.idxs[i][2]] ...  ), H.A[i], b)
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

# same as Ac_mul_B but skips `Zeros`
@generated function Ac_mul_B_skipZeros!{M,N,L,P,C,DD}(y::C, H::VCAT{M,N,L,P,C}, b::DD)

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
		if !(fieldtype(L,i) <: Zeros)

			if fieldtype(P,i) <: Int 
				bb = :(b[H.idxs[$i]])
			else
				bb = ""
				for ii in eachindex(fieldnames(fieldtype(P,i)))
					bb *= "b[H.idxs[$i][$ii]],"
				end
				bb = parse(bb)
			end

			ex = :($ex; Ac_mul_B!(H.buf,H.A[$i],$bb))
			
			if C <: AbstractArray
				ex = :($ex; y .+= H.buf)
			else
				for ii = 1:M
					ex = :($ex; y[$ii] .+= H.buf[$ii])
				end
			end
		end

	end
	ex = :($ex; return y)
	return ex

end

# same as A_mul_B but skips `Zeros`
@generated function A_mul_B_skipZeros!{M,N,L,P,C,DD}(y::DD, H::VCAT{M,N,L,P,C}, b::C)

	ex = :()

	for i = 1:N

		if !(fieldtype(L,i) <: Zeros)
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

function codomainType(H::VCAT) 
	codomain = vcat([typeof(d)<:Tuple ? [d...] : d  for d in codomainType.(H.A)]...)
	p = vcat([[idx... ] for idx in H.idxs]...)
	ipermute!(codomain,p)
	return (codomain...)
end
domainType(L::VCAT) = domainType.(L.A[1])

is_linear(L::VCAT) = all(is_linear.(L.A))
is_AcA_diagonal(L::VCAT) = all(is_AcA_diagonal.(L.A))
is_full_column_rank(L::VCAT) = any(is_full_column_rank.(L.A))

diag_AcA(L::VCAT) = sum(diag_AcA.(L.A))

# utils
import Base: permute

function permute{M,N,L,P,C}(H::VCAT{M,N,L,P,C}, p::AbstractVector{Int})


	unfolded = vcat([[idx... ] for idx in H.idxs]...) 
	ipermute!(unfolded,p)

	new_part = ()
	cnt = 0
	for z in length.(H.idxs)
		new_part = (new_part..., z == 1 ? unfolded[cnt+1] : (unfolded[cnt+1:z+cnt]...))
		cnt += z
	end

	VCAT{M,N,L,P,C}(H.A,new_part,H.buf)
end
