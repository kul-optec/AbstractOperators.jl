export DCAT

immutable DCAT{N,
	       L  <: NTuple{N,AbstractOperator},
	       P1 <: NTuple{N,Union{Int,Tuple}},
	       P2 <: NTuple{N,Union{Int,Tuple}}
	       } <: AbstractOperator
	A::L
	idxD::P1
	idxC::P2
	DCAT(A::L, idxD::P1, idxC::P2) where {L, P1, P2} = new{length(A), L, P1, P2}(A,idxD,idxC)
end

# Constructors
DCAT(A::AbstractOperator) = A

function DCAT(A::Vararg{AbstractOperator})

	# build H.idxs
	idxD,idxC = (),()
	for d in (1,2) 
		idxs = []
		K = 0
		for i in eachindex(ndoms.(A,d))
			if ndoms(A[i],d) == 1 # flatten operator
				K += 1
				push!(idxs,K)
			else                   # stacked operator 
				idxs = push!(idxs,(collect(K+1:K+ndoms(A[i],d))...) )
				for ii = 1:ndoms(A[i],d)
					K += 1
				end
			end
		end
		d == 1 ? (idxC = (idxs...)) : (idxD = (idxs...))
	end

	return DCAT(A,idxD,idxC)

end

# Mappings
@generated function A_mul_B!{N,L,P1,P2}(y, H::DCAT{N,L,P1,P2}, b)  

	ex = :()

	for i = 1:N

		if fieldtype(P2,i) <: Int 
		# flatten operator  
		# build A_mul_B!(y[H.idxC[i]], H.A[i], b)  
			yy = :(y[H.idxC[$i]])
		else
		# staked operator 
		# build A_mul_B!(( y[H.idxC[i][1]], y[H.idxC[i][2]] ...  ), H.A[i], b)
			yy = ""
			for ii in eachindex(fieldnames(fieldtype(P2,i)))
				yy *= "y[H.idxC[$i][$ii]],"
			end
			yy = parse(yy)
		end

		if fieldtype(P1,i) <: Int 
		# flatten operator  
		# build Ac_mul_B!(H.buf, H.A[i], b[H.idxD[i]])  
			bb = :(b[H.idxD[$i]])
		else
		# staked operator 
		# build Ac_mul_B!(H.buf, H.A[i],( b[H.idxD[i][1]], b[H.idxD[i][2]] ...  ))
			bb = ""
			for ii in eachindex(fieldnames(fieldtype(P1,i)))
				bb *= "b[H.idxD[$i][$ii]],"
			end
			bb = parse(bb)
		end
		
		ex = :($ex; A_mul_B!($yy,H.A[$i],$bb))

	end
	ex = :($ex; return y)
	return ex

end

@generated function Ac_mul_B!{N,L,P1,P2}(y, H::DCAT{N,L,P1,P2}, b)  

	ex = :()

	for i = 1:N

		if fieldtype(P1,i) <: Int 
		# flatten operator  
		# build Ac_mul_B!(y[H.idxD[i]], H.A[i], b)  
			yy = :(y[H.idxD[$i]])
		else
		# staked operator 
		# build Ac_mul_B!(( y[H.idxD[i][1]], y[H.idxD[i][2]] ...  ), H.A[i], b)
			yy = ""
			for ii in eachindex(fieldnames(fieldtype(P1,i)))
				yy *= "y[H.idxD[$i][$ii]],"
			end
			yy = parse(yy)
		end

		if fieldtype(P2,i) <: Int 
		# flatten operator  
		# build Ac_mul_B!(H.buf, H.A[i], b[H.idxC[i]])  
			bb = :(b[H.idxC[$i]])
		else
		# staked operator 
		# build Ac_mul_B!(H.buf, H.A[i],( b[H.idxC[i][1]], b[H.idxC[i][2]] ...  ))
			bb = ""
			for ii in eachindex(fieldnames(fieldtype(P2,i)))
				bb *= "b[H.idxC[$i][$ii]],"
			end
			bb = parse(bb)
		end
		
		ex = :($ex; Ac_mul_B!($yy,H.A[$i],$bb))

	end
	ex = :($ex; return y)
	return ex

end

# Properties
size(H::DCAT) = size(H,1),size(H,2) 

function size(H::DCAT, i::Int) 

	sz = []
	for s in size.(H.A,i)
		eltype(s) <: Int ? push!(sz,s) : push!(sz,s...) 
	end
	p = vcat([[idx... ] for idx in (i == 1? H.idxC : H.idxD) ]...)
	ipermute!(sz,p)

	(sz...)
end

fun_name(L::DCAT) = length(L.A) == 2 ? "["fun_name(L.A[1])*",0;0,"*fun_name(L.A[2])*"]" :
"DCAT"

function domainType(H::DCAT) 
	domain = vcat([typeof(d)<:Tuple ? [d...] : d  for d in domainType.(H.A)]...)
	p = vcat([[idx... ] for idx in H.idxD]...)
	ipermute!(domain,p)
	return (domain...)
end
function codomainType(H::DCAT) 
	codomain = vcat([typeof(d)<:Tuple ? [d...] : d  for d in codomainType.(H.A)]...)
	p = vcat([[idx... ] for idx in H.idxC]...)
	ipermute!(codomain,p)
	return (codomain...)
end

is_linear(L::DCAT) = all(is_linear.(L.A))
is_diagonal(L::DCAT) = all(is_diagonal.(L.A))
is_AcA_diagonal(L::DCAT) = all(is_AcA_diagonal.(L.A))
is_Ac_diagonal(L::DCAT) = all(is_Ac_diagonal.(L.A))
is_orthogonal(L::DCAT) = all(is_orthogonal.(L.A))
is_invertible(L::DCAT) = all(is_invertible.(L.A))
is_full_row_rank(L::DCAT) = all(is_full_row_rank.(L.A))
is_full_column_rank(L::DCAT) = all(is_full_column_rank.(L.A))

# utils
import Base: permute

function permute{N,L,P1,P2}(H::DCAT{N,L,P1,P2}, p::AbstractVector{Int})


	unfolded = vcat([[idx... ] for idx in H.idxD]...) 
	ipermute!(unfolded,p)

	new_part = ()
	cnt = 0
	for z in length.(H.idxD)
		new_part = (new_part..., z == 1 ? unfolded[cnt+1] : (unfolded[cnt+1:z+cnt]...))
		cnt += z
	end

	DCAT(H.A,new_part,H.idxC)
end
