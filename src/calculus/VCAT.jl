export VCAT

"""
	VCAT(A::AbstractOperator...)

Shorthand constructors:

	[A1; A2 ...]
	vcat(A...)

Vertically concatenate `AbstractOperator`s. Notice that all the operators must share the same domain dimensions and type, e.g. `size(A1,2) == size(A2,2)` and `domainType(A1) == domainType(A2)`.

```julia
julia> VCAT(DFT(4,4),Variation((4,4)))
[ℱ;Ʋ]  ℝ^(4, 4) -> ℂ^(4, 4)  ℝ^(16, 2)

julia> V = [Eye(3); DiagOp(2*ones(3))]
[I;╲]  ℝ^3 -> ℝ^3  ℝ^3

julia> vcat(V,FiniteDiff((3,)))
VCAT  ℝ^3 -> ℝ^3  ℝ^3  ℝ^2
```

When multiplying a `VCAT` with an array of the proper size, the result will be a `Tuple` containing arrays with the `VCAT`'s codomain type and size.

```julia
julia> V*ones(3)
([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
	
```
"""
struct VCAT{
	N, # number of AbstractOperator
	L<:NTuple{N,AbstractOperator},
	P<:Tuple,
	C<:AbstractArray,
} <: AbstractOperator
	A::L     # tuple of AbstractOperators
	idxs::P  # indices
	# H = VCAT(Eye(n),VCAT(Eye(n),Eye(n))) has H.idxs = (1,2,3)
	# `AbstractOperators` are flatten
	# H = VCAT(Eye(n),Compose(MatrixOp(randn(n,n)),VCAT(Eye(n),Eye(n))))
	# has H.idxs = (1,(2,3))
	# `AbstractOperators` are stack
	buf::C   # buffer memory
	function VCAT(
		A::L, idxs::P, buf::C
	) where {N,L<:NTuple{N,AbstractOperator},P<:Tuple,C<:AbstractArray}
		if any([size(A[1], 2) != size(a, 2) for a in A])
			throw(DimensionMismatch("operators must have the same domain dimension!"))
		end
		if any([domainType(A[1]) != domainType(a) for a in A])
			throw(error("operators must all share the same domainType!"))
		end
		return new{N,L,P,C}(A, idxs, buf)
	end
end

function VCAT(A::Vararg{AbstractOperator})
	if any((<:).(typeof.(A), VCAT)) #there are VCATs in A
		AA = ()
		for a in A
			if typeof(a) <: VCAT # flatten
				AA = (AA..., a.A...)
			else                 # stack
				AA = (AA..., a)
			end
		end
		# use buffer from VCAT in A
		buf = A[findfirst((<:).(typeof.(A), VCAT))].buf
	else
		AA = A
		# generate buffer
		buf = allocate_in_domain(AA[1])
	end

	return VCAT(AA, buf)
end

function VCAT(AA::NTuple{N,AbstractOperator}, buf::C) where {N,C}
	if N == 1
		return AA[1]
	else
		# build H.idxs
		K = 0
		idxs = []
		for i in eachindex(ndoms.(AA, 1))
			if ndoms(AA[i], 1) == 1 # flatten operator
				K += 1
				push!(idxs, K)
			else                   # stacked operator
				idxs = push!(idxs, (collect((K + 1):(K + ndoms(AA[i], 2)))...,))
				for ii in 1:ndoms(AA[i], 1)
					K += 1
				end
			end
		end
		return VCAT(AA, (idxs...,), buf)
	end
end

VCAT(A::AbstractOperator) = A

# Mappings
@generated function mul!(
	y::C, A::AdjointOperator{VCAT{N,L,P,C}}, b::DD
) where {N,L,P,C,DD<:ArrayPartition}
	ex = :(H = A.A)

	if fieldtype(P, 1) <: Int
		# flatten operator
		# build mul!(y, H.A[1]', b.x[H.idxs[1]])
		bb = :(b.x[H.idxs[1]])
	else
		# stacked operator
		# build mul!(y, H.A[1]',ArrayPartition( b.x[H.idxs[1][1]], b.x[H.idxs[1][2]] ...  ))
		bb = [:(b.x[H.idxs[1][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, 1)))]
		bb = :(ArrayPartition($(bb...)))
	end
	ex = :($ex; mul!(y, H.A[1]', $bb)) # write on y

	for i in 2:N
		if fieldtype(P, i) <: Int
			# flatten operator
			# build mul!(H.buf, H.A[i]', b.x[H.idxs[i]])
			bb = :(b.x[H.idxs[$i]])
		else
			# stacked operator
			# build mul!(H.buf, H.A[i]',( b.x[H.idxs[i][1]], b.x[H.idxs[i][2]] ...  ))
			bb = [:(b.x[H.idxs[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, i)))]
			bb = :(ArrayPartition($(bb...)))
		end
		ex = :($ex; mul!(H.buf, H.A[$i]', $bb)) # write on H.buf
		# sum H.buf with y
		ex = :($ex; y .+= H.buf)
	end
	ex = :($ex; return y)
	return ex
end

@generated function mul!(y::DD, H::VCAT{N,L,P,C}, b::C) where {N,L,P,C,DD<:ArrayPartition}
	ex = :()
	for i in 1:N
		if fieldtype(P, i) <: Int
			# flatten operator
			# build mul!(y.x[H.idxs[i]], H.A[i], b)
			yy = :(y.x[H.idxs[$i]])
		else
			# stacked operator
			# build mul!(ArrayPartition( y[.xH.idxs[i][1]], y.x[H.idxs[i][2]] ...  ), H.A[i], b)
			yy = [:(y.x[H.idxs[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, i)))]
			yy = :(ArrayPartition($(yy...)))
		end
		ex = :($ex; mul!($yy, H.A[$i], b))
	end
	ex = :($ex; return y)
	return ex
end

## same as mul! but skips `Zeros`
@generated function mul_skipZeros!(
	y::C, A::AdjointOperator{VCAT{N,L,P,C}}, b::DD
) where {N,L,P,C,DD<:ArrayPartition}
	ex = :(H = A.A)

	if fieldtype(P, 1) <: Int
		# flatten operator
		# build mul!(y, H.A[1]', b.x[H.idxs[1]])
		bb = :(b.x[H.idxs[1]])
	else
		# stacked operator
		# build mul!(y, H.A[1]',ArrayPartition( b.x[H.idxs[1][1]], b.x[H.idxs[1][2]] ...  ))
		bb = [:(b.x[H.idxs[1][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, 1)))]
		bb = :(ArrayPartition($(bb...)))
	end
	ex = :($ex; mul!(y, H.A[1]', $bb)) # write on y

	for i in 2:N
		if !(fieldtype(L, i) <: Zeros)
			if fieldtype(P, i) <: Int
				# flatten operator
				# build mul!(H.buf, H.A[i]', b.x[H.idxs[i]])
				bb = :(b.x[H.idxs[$i]])
			else
				# stacked operator
				# build mul!(H.buf, H.A[i]',( b.x[H.idxs[i][1]], b.x[H.idxs[i][2]] ...  ))
				bb = [
					:(b.x[H.idxs[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, i)))
				]
				bb = :(ArrayPartition($(bb...)))
			end
			ex = :($ex; mul!(H.buf, H.A[$i]', $bb)) # write on H.buf
			# sum H.buf with y
			ex = :($ex; y .+= H.buf)
		end
	end
	ex = :($ex; return y)
	return ex
end

@generated function mul_skipZeros!(
	y::DD, H::VCAT{N,L,P,C}, b::C
) where {N,L,P,C,DD<:ArrayPartition}
	ex = :()
	for i in 1:N
		if !(fieldtype(L, i) <: Zeros)
			if fieldtype(P, i) <: Int
				# flatten operator
				# build mul!(y.x[H.idxs[i]], H.A[i], b)
				yy = :(y.x[H.idxs[$i]])
			else
				# stacked operator
				# build mul!(ArrayPartition( y[.xH.idxs[i][1]], y.x[H.idxs[i][2]] ...  ), H.A[i], b)
				yy = [
					:(y.x[H.idxs[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P, i)))
				]
				yy = :(ArrayPartition($(yy...)))
			end
			ex = :($ex; mul!($yy, H.A[$i], b))
		end
	end
	ex = :($ex; return y)
	return ex
end

# Properties

function size(H::VCAT)
	size_out = []
	for s in size.(H.A, 1)
		eltype(s) <: Int ? push!(size_out, s) : push!(size_out, s...)
	end
	p = vcat([[idx...] for idx in H.idxs]...)
	invpermute!(size_out, p)

	return (size_out...,), size(H.A[1], 2)
end

function fun_name(L::VCAT)
	return length(L.A) == 2 ? "[" * fun_name(L.A[1]) * ";" * fun_name(L.A[2]) * "]" : "VCAT"
end

function codomainType(H::VCAT)
	codomain = vcat([typeof(d) <: Tuple ? [d...] : d for d in codomainType.(H.A)]...)
	p = vcat([[idx...] for idx in H.idxs]...)
	invpermute!(codomain, p)
	return (codomain...,)
end
domainType(L::VCAT) = domainType.(Ref(L.A[1]))

is_linear(L::VCAT) = all(is_linear.(L.A))
is_AcA_diagonal(L::VCAT) = all(is_AcA_diagonal.(L.A))
is_full_column_rank(L::VCAT) = any(is_full_column_rank.(L.A))

diag_AcA(L::VCAT) = (+).(diag_AcA.(L.A)...,)

# utils
function permute(H::VCAT{N,L,P,C}, p::AbstractVector{Int}) where {N,L,P,C}
	unfolded = vcat([[idx...] for idx in H.idxs]...)
	invpermute!(unfolded, p)

	new_part = ()
	cnt = 0
	for z in length.(H.idxs)
		new_part = (
			new_part..., z == 1 ? unfolded[cnt + 1] : (unfolded[(cnt + 1):(z + cnt)]...,)
		)
		cnt += z
	end

	return VCAT{M,N,L,P,C}(H.A, new_part, H.buf)
end

remove_displacement(V::VCAT) = VCAT(remove_displacement.(V.A), V.idxs, V.buf)
