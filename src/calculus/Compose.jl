export Compose

"""
	Compose(A::AbstractOperator,B::AbstractOperator)

Shorthand constructor:

	A*B

Compose different `AbstractOperator`s. Notice that the domain and codomain of the operators `A` and `B` must match, i.e. `size(A,2) == size(B,1)` and `domainType(A) == codomainType(B)`.

```jldoctest
julia> Compose(DFT(16,2),Variation((4,4)))
ℱ*Ʋ  ℝ^(4, 4) -> ℂ^(16, 2)

julia> MatrixOp(randn(20,10))*DCT(10)
▒*ℱc  ℝ^10 -> ℝ^20
	
```
"""
struct Compose{N,M,L<:NTuple{N,Any},T<:NTuple{M,Any}} <: AbstractOperator
	A::L
	buf::T       # memory in the bufdle of the operators
end

# Constructors

function Compose(L1::AbstractOperator, L2::AbstractOperator)
	if size(L1, 2) != size(L2, 1)
		throw(
			DimensionMismatch(
				"cannot compose operators with different domain and codomain sizes"
			),
		)
	end
	if domainType(L1) != codomainType(L2)
		throw(
			DomainError(
				(domainType(L1), codomainType(L2)),
				"cannot compose operators with different domain and codomain types",
			),
		)
	end
	if domain_storage_type(L1) != codomain_storage_type(L2)
		throw(
			DomainError(
				(domain_storage_type(L1), codomain_storage_type(L2)),
				"cannot compose operators with different input and output storage types",
			),
		)
	end
	return Compose(L1, L2, allocate_in_codomain(L2))
end

function Compose(L1::AbstractOperator, L2::AbstractOperator, buf::AbstractArray)
	return Compose((L2, L1), (buf,))
end

function Compose(L1::Compose, L2::AbstractOperator, buf::AbstractArray)
	return Compose((L2, L1.A...), (buf, L1.buf...))
end

function Compose(L1::AbstractOperator, L2::Compose, buf::AbstractArray)
	return Compose((L2.A..., L1), (L2.buf..., buf))
end

function Compose(L1::Compose, L2::Compose, buf::AbstractArray)
	return Compose((L2.A..., L1.A...), (L2.buf..., buf, L1.buf...))
end

#special cases
Scale(coeff, L::Compose) = Compose((L.A[1:(end - 1)]..., Scale(coeff, L.A[end])), L.buf)
Compose(L1::Scale, ::Eye) = L1
Compose(::Eye, L2::Scale) = L2

Compose(L1::AbstractOperator, ::Eye) = L1
Compose(::Eye, L2::AbstractOperator) = L2
Compose(L1::Eye, ::Eye) = L1

# Mappings

@generated function mul!(y::C, L::Compose{N,M,T1,T2}, b::D) where {N,M,T1,T2,C,D}
	ex = :(mul!(L.buf[1], L.A[1], b))
	for i in 2:M
		ex = quote
			$ex
			mul!(L.buf[$i], L.A[$i], L.buf[$i - 1])
		end
	end
	ex = quote
		$ex
		mul!(y, L.A[N], L.buf[M])
		return y
	end
end

@generated function mul!(
	y::D, L::AdjointOperator{Compose{N,M,T1,T2}}, b::C
) where {N,M,T1,T2,C,D}
	ex = :(mul!(L.A.buf[M], L.A.A[N]', b))
	for i in M:-1:2
		ex = quote
			$ex
			mul!(L.A.buf[$i - 1], L.A.A[$i]', L.A.buf[$i])
		end
	end
	ex = quote
		$ex
		mul!(y, L.A.A[1]', L.A.buf[1])
		return y
	end
end

# Properties

size(L::Compose) = (size(L.A[end], 1), size(L.A[1], 2))

fun_name(L::Compose) = length(L.A) == 2 ? fun_name(L.A[2]) * "*" * fun_name(L.A[1]) : "Π"

domainType(L::Compose) = domainType(L.A[1])
codomainType(L::Compose) = codomainType(L.A[end])

is_linear(L::Compose) = all(is_linear.(L.A))
function is_diagonal(L::Compose)
	return is_sliced(L) ? (length(L.A) == 2 && is_diagonal(L.A[2])) : all(is_diagonal.(L.A))
end
is_invertible(L::Compose) = all(is_invertible.(L.A))

is_sliced(L::Compose) = typeof(L.A[1]) <: GetIndex
is_AAc_diagonal(L::Compose) = is_sliced(L) && length(L.A) == 2 && is_AAc_diagonal(L.A[2])

diag(L::Compose) = is_sliced(L) ? diag(L.A[2]) : prod(diag.(L.A))
function diag_AAc(L::Compose)
	return if is_AAc_diagonal(L)
		diag_AAc(L.A[2])
	else
		error("is_AAc_diagonal( $(typeof(L) ) ) == false")
	end
end

# utils
function permute(C::Compose, p::AbstractVector{Int})
	i = findfirst(x -> ndoms(x, 2) > 1, C.A)
	P = permute(C.A[i], p)
	AA = (C.A[1:(i - 1)]..., P, C.A[(i + 1):end]...)
	return Compose(AA, C.buf)
end

remove_displacement(C::Compose) = Compose(remove_displacement.(C.A), C.buf)
