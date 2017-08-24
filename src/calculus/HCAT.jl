
export HCAT

immutable HCAT{M, # number of codomains  
	       N, # number of AbstractOperator 
	       L <: NTuple{N,AbstractOperator},
	       P <: NTuple{N,Union{Int,Tuple}},
	       C <: Union{NTuple{M,AbstractArray}, AbstractArray},
	       } <: AbstractOperator
	A::L     # tuple of AbstractOperators
	idxs::P  # indices 
	         # H = HCAT(Eye(n),HCAT(Eye(n),Eye(n))) has H.idxs = (1,2,3) 
		 # `AbstractOperators` are flatten
	         # H = HCAT(Eye(n),Compose(MatrixOp(randn(n,n)),HCAT(Eye(n),Eye(n)))) 
		 # has H.idxs = (1,(2,3))
		 # `AbstractOperators` are stack
	buf::C   # buffer memory
end

# Constructors

function HCAT{N,  
	      L <: NTuple{N,AbstractOperator},
	      P <: NTuple{N,Union{Int,Tuple}},
	      C
	      }(A::L, idxs::P, buf::C, M::Int)
	if any([size(A[1],1) != size(a,1) for a in A])
		throw(DimensionMismatch("operators must have the same codomain dimension!"))
	end
	if any([codomainType(A[1]) != codomainType(a) for a in A])
		throw(error("operators must all share the same codomainType!"))
	end
	HCAT{M,N,L,P,C}(A, idxs, buf)
end

function HCAT(A::Vararg{AbstractOperator})

	if any((<:).(typeof.(A),HCAT)) #there are HCATs in A
		AA = ()
		for a in A
			if typeof(a) <: HCAT # flatten 
				AA = (AA...,a.A...)
			else                 # stack
				AA = (AA...,a)
			end
		end
		# use buffer from HCAT in A
		buf = A[findfirst( (<:).(typeof.(A),HCAT) ) ].buf
	else 
		AA = A
		s = size(AA[1],1)
		t = codomainType(AA[1])
		# generate buffer
		buf = eltype(s) <: Int ? zeros(t,s) : zeros.(t,s)
	end

	return HCAT(AA, buf)
end

function HCAT{N,C}(AA::NTuple{N,AbstractOperator}, buf::C) 
	if N == 1
		return AA[1]
	else
		# get number of codomains
		M = C <: AbstractArray ? 1 : length(buf)
		# build H.idxs
		K = 0
		idxs = []
		for i in eachindex(ndoms.(AA,2))
			if ndoms(AA[i],2) == 1 # flatten operator
				K += 1
				push!(idxs,K)
			else                   # stacked operator 
				idxs = push!(idxs,(collect(K+1:K+ndoms(AA[i],2))...) )
				for ii = 1:ndoms(AA[i],2)
					K += 1
				end
			end
		end
		return HCAT(AA, (idxs...), buf, M)
	end
end

HCAT(A::AbstractOperator) = A

# Mappings

@generated function A_mul_B!{M,N,L,P,C,DD}(y::C, H::HCAT{M,N,L,P,C}, b::DD)

	ex = :()

	if fieldtype(P,1) <: Int 
		# flatten operator  
		# build A_mul_B!(y, H.A[1], b[H.idxs[1]])  
		bb = :(b[H.idxs[1]])
	else
		# staked operator 
		# build A_mul_B!(y, H.A[1],( b[H.idxs[1][1]], b[H.idxs[1][2]] ...  ))
		bb = ""
		for ii in eachindex(fieldnames(fieldtype(P,1)))
			bb *= "b[H.idxs[1][$ii]],"
		end
		bb = parse(bb)
	end
	ex = :($ex; A_mul_B!(y,H.A[1],$bb)) # write on y

	for i = 2:N

		if fieldtype(P,i) <: Int 
		# flatten operator  
		# build A_mul_B!(H.buf, H.A[i], b[H.idxs[i]])  
			bb = :(b[H.idxs[$i]])
		else
		# staked operator 
		# build A_mul_B!(H.buf, H.A[i],( b[H.idxs[i][1]], b[H.idxs[1][2]] ...  ))
			bb = ""
			for ii in eachindex(fieldnames(fieldtype(P,i)))
				bb *= "b[H.idxs[$i][$ii]],"
			end
			bb = parse(bb)
		end

		ex = :($ex; A_mul_B!(H.buf,H.A[$i],$bb)) # write on H.buf
		
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

@generated function Ac_mul_B!{M,N,L,P,C,DD}(y::DD, H::HCAT{M,N,L,P,C}, b::C)

	ex = :()

	for i = 1:N

		if fieldtype(P,i) <: Int 
		# flatten operator  
		# build Ac_mul_B!(y[H.idxs[i]], H.A[i], b)  
			yy = :(y[H.idxs[$i]])
		else
		# staked operator 
		# build Ac_mul_B!(( y[H.idxs[i][1]], y[H.idxs[1][2]] ...  ), H.A[i], b)
			yy = ""
			for ii in eachindex(fieldnames(fieldtype(P,i)))
				yy *= "y[H.idxs[$i][$ii]],"
			end
			yy = parse(yy)
		end
		
		ex = :($ex; Ac_mul_B!($yy,H.A[$i],b))

	end
	ex = :($ex; return y)
	return ex

end

# same as A_mul_B but skips `Zeros`
@generated function A_mul_B_skipZeros!{M,N,L,P,C,DD}(y::C, H::HCAT{M,N,L,P,C}, b::DD)

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
	ex = :($ex; A_mul_B!(y,H.A[1],$bb))

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

			ex = :($ex; A_mul_B!(H.buf,H.A[$i],$bb))
			
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

# same as Ac_mul_B but skips `Zeros`
@generated function Ac_mul_B_skipZeros!{M,N,L,P,C,DD}(y::DD, H::HCAT{M,N,L,P,C}, b::C)

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
			
			ex = :($ex; Ac_mul_B!($yy,H.A[$i],b))
		end

	end
	ex = :($ex; return y)
	return ex

end

# Properties

function size(H::HCAT) 
	size_in = []
	for s in size.(H.A,2)
		eltype(s) <: Int ? push!(size_in,s) : push!(size_in,s...) 
	end
	p = vcat([[idx... ] for idx in H.idxs]...)
	ipermute!(size_in,p)

	size(H.A[1],1), (size_in...)
end

fun_name(L::HCAT) = length(L.A) == 2 ? "["fun_name(L.A[1])*","*fun_name(L.A[2])*"]" : "HCAT"

function domainType(H::HCAT) 
	domain = vcat([typeof(d)<:Tuple ? [d...] : d  for d in domainType.(H.A)]...)
	p = vcat([[idx... ] for idx in H.idxs]...)
	ipermute!(domain,p)
	return (domain...)
end
codomainType(L::HCAT) = codomainType.(L.A[1])

is_linear(L::HCAT) = all(is_linear.(L.A))
is_AAc_diagonal(L::HCAT) = all(is_AAc_diagonal.(L.A))
is_full_row_rank(L::HCAT) = any(is_full_row_rank.(L.A))

diag_AAc(L::HCAT) = sum(diag_AAc.(L.A))

# utils
import Base: permute

function permute{M,N,L,P,C}(H::HCAT{M,N,L,P,C}, p::AbstractVector{Int})


	unfolded = vcat([[idx... ] for idx in H.idxs]...) 
	ipermute!(unfolded,p)

	new_part = ()
	cnt = 0
	for z in length.(H.idxs)
		new_part = (new_part..., z == 1 ? unfolded[cnt+1] : (unfolded[cnt+1:z+cnt]...))
		cnt += z
	end

	HCAT{M,N,L,P,C}(H.A,new_part,H.buf)
end
