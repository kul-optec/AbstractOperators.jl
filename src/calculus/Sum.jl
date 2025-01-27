export Sum

"""
	Sum(A::AbstractOperator...)

Shorthand constructor:

	+(A::AbstractOperator...)

Sum of operators.

```julia
julia> Sum(DiagOp(rand(10)), DCT(10))
ℱ+ℱc  ℝ^10 -> ℝ^10

julia> MatrixOp(rand(ComplexF64,5,5)) + DFT(ComplexF64,5)
▒+ℱ  ℂ^5 -> ℂ^5
	
```
"""
struct Sum{K,C<:AbstractArray,D<:AbstractArray,L<:NTuple{K,AbstractOperator}} <:
	   AbstractOperator
	A::L
	bufC::C
	bufD::D
	function Sum(A::L, bufC::C, bufD::D) where {C,D,K,L<:NTuple{K,AbstractOperator}}
		if any([size(a) != size(A[1]) for a in A])
			throw(DimensionMismatch("cannot sum operator of different sizes"))
		end
		if any([codomainType(A[1]) != codomainType(a) for a in A]) || any([domainType(A[1]) != domainType(a) for a in A])
			throw(DomainError(A, "cannot sum operator with different codomains"))
		end
		return new{K,C,D,L}(A, bufC, bufD)
	end
end

Sum(A::AbstractOperator) = A

function Sum(A::Vararg{AbstractOperator})
	bufC = allocate_in_codomain(A[1])
	bufD = allocate_in_domain(A[1])
	return Sum(A, bufC, bufD)
end

# special cases
function Sum(L1::AbstractOperator, L2::Sum{K,C,D}) where {K,C,D}
	return Sum((L1, L2.A...), L2.bufC, L2.bufD)
end

# Mappings

@generated function mul!(y::C, S::Sum{K,C,D}, b::D) where {K,C,D}
	ex = :(mul!(y, S.A[1], b))
	for i in 2:K
		ex = quote
			$ex
			mul!(S.bufC, S.A[$i], b)
		end
		ex = :($ex; y .+= S.bufC)
	end
	ex = quote
		$ex
		return y
	end
end

@generated function mul!(y::D, A::AdjointOperator{Sum{K,C,D,L}}, b::C) where {K,C,D,L}
	ex = :(S = A.A; mul!(y, S.A[1]', b))
	for i in 2:K
		ex = quote
			$ex
			mul!(S.bufD, S.A[$i]', b)
		end
		ex = :($ex; y .+= S.bufD)
	end
	ex = quote
		$ex
		return y
	end
end

# Properties

size(L::Sum) = size(L.A[1])

domainType(S::Sum{K,C,D,L}) where {K,C,D<:AbstractArray,L} = domainType(S.A[1])
domainType(S::Sum{K,C,D,L}) where {K,C,D<:Tuple,L} = domainType.(Ref(S.A[1]))
codomainType(S::Sum{K,C,D,L}) where {K,C<:AbstractArray,D,L} = codomainType(S.A[1])
codomainType(S::Sum{K,C,D,L}) where {K,C<:Tuple,D,L} = codomainType.(Ref(S.A[1]))

fun_domain(S::Sum) = fun_domain(S.A[1])
fun_codomain(S::Sum) = fun_codomain(S.A[1])

fun_name(S::Sum) = length(S.A) == 2 ? fun_name(S.A[1]) * "+" * fun_name(S.A[2]) : "Σ"

is_linear(L::Sum) = all(is_linear.(L.A))
is_null(L::Sum) = all(is_null.(L.A))
is_diagonal(L::Sum) = all(is_diagonal.(L.A))
is_full_row_rank(L::Sum) = any(is_full_row_rank.(L.A))
is_full_column_rank(L::Sum) = any(is_full_column_rank.(L.A))

diag(L::Sum) = (+).(diag.(L.A)...,)

# utils
function permute(S::Sum, p::AbstractVector{Int})
	AA = ([permute(A, p) for A in S.A]...,)
	return Sum(AA, S.bufC, ArrayPartition(S.bufD.x[p]...))
end

remove_displacement(S::Sum) = Sum(remove_displacement.(S.A), S.bufC, S.bufD)
