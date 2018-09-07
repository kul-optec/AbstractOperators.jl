export DCAT

"""
`DCAT(A::AbstractOperator...)`

Shorthand constructor: 

`blkdiag(α::Number,A::AbstractOperator)` 

Block-diagonally concatenate `AbstractOperator`s.

```julia
julia> D = DCAT(HCAT(Eye(2),Eye(2)),DFT(3))
[[I,I],0;0,ℱ]  ℝ^2  ℝ^2  ℝ^4 -> ℝ^2  ℂ^3

julia> blkdiag(Eye(10),Eye(10),FiniteDiff((4,4)))
DCAT  ℝ^10  ℝ^10  ℝ^(4, 4) -> ℝ^10  ℝ^10  ℝ^(3, 4)
```

To evaluate `DCAT` operators multiply them with a `Tuple` of `AbstractArray` of the correct domain size and type. The output will consist as well of a `Tuple` with the codomain type and size of the `DCAT`.

```julia
julia> D*(ones(2),ones(2),ones(3))
([2.0, 2.0], Complex{Float64}[3.0+0.0im, 0.0+0.0im, 0.0+0.0im])

```

"""
struct DCAT{N,
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
				idxs = push!(idxs,(collect(K+1:K+ndoms(A[i],d))...,) )
				for ii = 1:ndoms(A[i],d)
					K += 1
				end
			end
		end
		d == 1 ? (idxC = (idxs...,)) : (idxD = (idxs...,))
	end

	return DCAT(A,idxD,idxC)

end

# Mappings
@generated function mul!(y, H::DCAT{N,L,P1,P2}, b) where {N,L,P1,P2} 

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
			yy = Meta.parse(yy)
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
			bb = Meta.parse(bb)
		end
		
		ex = :($ex; mul!($yy,H.A[$i],$bb))

	end
	ex = :($ex; return y)
	return ex

end

@generated function mul!(y, A::AdjointOperator{DCAT{N,L,P1,P2}}, b) where {N,L,P1,P2} 

	ex = :(H = A.A)

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
			yy = Meta.parse(yy)
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
			bb = Meta.parse(bb)
		end
		
		ex = :($ex; mul!($yy,H.A[$i]',$bb))

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
	p = vcat([[idx... ] for idx in (i == 1 ? H.idxC : H.idxD) ]...)
	invpermute!(sz,p)

	(sz...,)
end

fun_name(L::DCAT) = length(L.A) == 2 ? "["*fun_name(L.A[1])*",0;0,"*fun_name(L.A[2])*"]" :
"DCAT"

function domainType(H::DCAT) 
	domain = vcat([typeof(d)<:Tuple ? [d...] : d  for d in domainType.(H.A)]...)
	p = vcat([[idx... ] for idx in H.idxD]...)
	invpermute!(domain,p)
	return (domain...,)
end
function codomainType(H::DCAT) 
	codomain = vcat([typeof(d)<:Tuple ? [d...] : d  for d in codomainType.(H.A)]...)
	p = vcat([[idx... ] for idx in H.idxC]...)
	invpermute!(codomain,p)
	return (codomain...,)
end

is_eye(L::DCAT) = all(is_eye.(L.A))
is_linear(L::DCAT) = all(is_linear.(L.A))
is_diagonal(L::DCAT) = all(is_diagonal.(L.A))
is_AcA_diagonal(L::DCAT) = all(is_AcA_diagonal.(L.A))
is_AAc_diagonal(L::DCAT) = all(is_AAc_diagonal.(L.A))
is_orthogonal(L::DCAT) = all(is_orthogonal.(L.A))
is_invertible(L::DCAT) = all(is_invertible.(L.A))
is_full_row_rank(L::DCAT) = all(is_full_row_rank.(L.A))
is_full_column_rank(L::DCAT) = all(is_full_column_rank.(L.A))

# utils
function permute(H::DCAT{N,L,P1,P2}, p::AbstractVector{Int}) where {N,L,P1,P2}


	unfolded = vcat([[idx... ] for idx in H.idxD]...) 
	invpermute!(unfolded,p)

	new_part = ()
	cnt = 0
	for z in length.(H.idxD)
		new_part = (new_part..., z == 1 ? unfolded[cnt+1] : (unfolded[cnt+1:z+cnt]...,))
		cnt += z
	end

	DCAT(H.A,new_part,H.idxC)
end

remove_displacement(D::DCAT) = DCAT(remove_displacement.(D.A), D.idxD, D.idxC)

# special cases
# Eye constructor
Eye(x::A) where {N, A <: NTuple{N,AbstractArray}} = DCAT(Eye.(x)...)
diag(L::DCAT{N,NTuple{N,E}}) where {N, E <: Eye} = 1.
diag_AAc(L::DCAT{N,NTuple{N,E}}) where {N, E <: Eye} = 1.
diag_AcA(L::DCAT{N,NTuple{N,E}}) where {N, E <: Eye} = 1.
