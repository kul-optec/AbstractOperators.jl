export DCAT

"""
	DCAT(A::AbstractOperator...)

Block-diagonally concatenate `AbstractOperator`s.

```jldoctest
julia> D = DCAT(HCAT(Eye(2),Eye(2)),DFT(3))
[[I,I],0;0,ℱ]  ℝ^2  ℝ^2  ℝ^3 -> ℝ^2  ℂ^3

julia> DCAT(Eye(10),Eye(10),FiniteDiff((4,4)))
DCAT  ℝ^10  ℝ^10  ℝ^(4, 4) -> ℝ^10  ℝ^10  ℝ^(3, 4)

julia> #To evaluate `DCAT` operators multiply them with a `Tuple` of `AbstractArray` of the correct domain size and type. The output will consist as well of a `Tuple` with the codomain type and size of the `DCAT`.

julia> using RecursiveArrayTools

julia> D*ArrayPartition(ones(2),ones(2),ones(3))
([2.0, 2.0], ComplexF64[3.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im])
	
```
"""
struct DCAT{
	N,
	L<:NTuple{N,AbstractOperator},
	P1<:NTuple{N,Union{Int,Tuple}},
	P2<:NTuple{N,Union{Int,Tuple}},
} <: AbstractOperator
	A::L
	idxD::P1
	idxC::P2
	DCAT(A::L, idxD::P1, idxC::P2) where {L,P1,P2} = new{length(A),L,P1,P2}(A, idxD, idxC)
end

# Constructors
DCAT(A::AbstractOperator) = A

function DCAT(A::Vararg{AbstractOperator})

	# build H.idxs
	idxD, idxC = (), ()
	for d in (1, 2)
		idxs = []
		K = 0
		for i in eachindex(ndoms.(A, d))
			if ndoms(A[i], d) == 1 # flatten operator
				K += 1
				push!(idxs, K)
			else                   # stacked operator
				idxs = push!(idxs, (collect((K + 1):(K + ndoms(A[i], d)))...,))
				for ii in 1:ndoms(A[i], d)
					K += 1
				end
			end
		end
		d == 1 ? (idxC = (idxs...,)) : (idxD = (idxs...,))
	end

	return DCAT(A, idxD, idxC)
end

# Mappings
@generated function mul!(
	yy::ArrayPartition, H::DCAT{N,L,P1,P2}, bb::ArrayPartition
) where {N,L,P1,P2}

	# extract stuff
	ex = :(y = yy.x; b = bb.x)

	for i in 1:N
		if fieldtype(P2, i) <: Int
			# flatten operator
			# build mul!(y[H.idxC[i]], H.A[i], b)
			yy = :(y[H.idxC[$i]])
		else
			# stacked operator
			# build mul!(( y[H.idxC[i][1]], y[H.idxC[i][2]] ...  ), H.A[i], b)
			yy = [:(y[H.idxC[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P2, i)))]
			yy = :(ArrayPartition($(yy...)))
		end

		if fieldtype(P1, i) <: Int
			# flatten operator
			# build mul!(H.buf, H.A[i], b[H.idxD[i]])
			bb = :(b[H.idxD[$i]])
		else
			# stacked operator
			# build mul!(H.buf, H.A[i],( b[H.idxD[i][1]], b[H.idxD[i][2]] ...  ))
			bb = [:(b[H.idxD[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P1, i)))]
			bb = :(ArrayPartition($(bb...)))
		end

		ex = :($ex; mul!($yy, H.A[$i], $bb))
	end
	ex = :($ex; return y)
	return ex
end

@generated function mul!(
	yy::ArrayPartition, A::AdjointOperator{DCAT{N,L,P1,P2}}, bb::ArrayPartition
) where {N,L,P1,P2}

	# extract stuff
	ex = :(H = A.A; y = yy.x; b = bb.x)

	for i in 1:N
		if fieldtype(P1, i) <: Int
			# flatten operator
			# build mul!(y[H.idxD[i]], H.A[i]', b)
			yy = :(y[H.idxD[$i]])
		else
			# stacked operator
			# build mul!(( y[H.idxD[i][1]], y[H.idxD[i][2]] ...  ), H.A[i]', b)
			yy = [:(y[H.idxD[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P1, i)))]
			yy = :(ArrayPartition($(yy...)))
		end

		if fieldtype(P2, i) <: Int
			# flatten operator
			# build mul!(H.buf, H.A[i]', b[H.idxC[i]])
			bb = :(b[H.idxC[$i]])
		else
			# stacked operator
			# build mul!(H.buf, H.A[i]',( b[H.idxC[i][1]], b[H.idxC[i][2]] ...  ))
			bb = [:(b[H.idxC[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P2, i)))]
			bb = :(ArrayPartition($(bb...)))
		end

		ex = :($ex; mul!($yy, H.A[$i]', $bb))
	end
	ex = :($ex; return y)
	return ex
end

function get_normal_op(H::DCAT)
	# build normal operator
	normal = []
	for i in eachindex(H.A)
		push!(normal, get_normal_op(H.A[i]))
	end

	# build normal DCAT
	normal = DCAT(normal...)

	# build normal DCAT with the same idxs as H
	p = vcat([[idx...] for idx in H.idxC]...)
	invpermute!(normal.idxD, p)

	return normal
end

# Properties
size(H::DCAT) = size(H, 1), size(H, 2)

function size(H::DCAT, i::Int)
	sz = []
	for s in size.(H.A, i)
		eltype(s) <: Int ? push!(sz, s) : push!(sz, s...)
	end
	p = vcat([[idx...] for idx in (i == 1 ? H.idxC : H.idxD)]...)
	invpermute!(sz, p)

	return (sz...,)
end

function fun_name(L::DCAT)
	return if length(L.A) == 2
		"[" * fun_name(L.A[1]) * ",0;0," * fun_name(L.A[2]) * "]"
	else
		"DCAT"
	end
end

function domainType(H::DCAT)
	domain = vcat([typeof(d) <: Tuple ? [d...] : d for d in domainType.(H.A)]...)
	p = vcat([[idx...] for idx in H.idxD]...)
	invpermute!(domain, p)
	return (domain...,)
end
function codomainType(H::DCAT)
	codomain = vcat([typeof(d) <: Tuple ? [d...] : d for d in codomainType.(H.A)]...)
	p = vcat([[idx...] for idx in H.idxC]...)
	invpermute!(codomain, p)
	return (codomain...,)
end
is_thread_safe(::DCAT) = all(is_thread_safe.(H.A))

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
	unfolded = vcat([[idx...] for idx in H.idxD]...)
	invpermute!(unfolded, p)

	new_part = ()
	cnt = 0
	for z in length.(H.idxD)
		new_part = (
			new_part..., z == 1 ? unfolded[cnt + 1] : (unfolded[(cnt + 1):(z + cnt)]...,)
		)
		cnt += z
	end

	return DCAT(H.A, new_part, H.idxC)
end

remove_displacement(D::DCAT) = DCAT(remove_displacement.(D.A), D.idxD, D.idxC)

# special cases
# Eye constructor
Eye(x::ArrayPartition) = DCAT(Eye.(x.x)...)
diag(L::DCAT{N,Tuple{E,Vararg{E,M}}}) where {N,M,E<:Eye} = 1.0
diag_AAc(L::DCAT{N,Tuple{E,Vararg{E,M}}}) where {N,M,E<:Eye} = 1.0
diag_AcA(L::DCAT{N,Tuple{E,Vararg{E,M}}}) where {N,M,E<:Eye} = 1.0

LinearAlgebra.opnorm(L::DCAT) = maximum(opnorm.(L.A))
