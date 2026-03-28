export Sum

"""
	Sum(A::AbstractOperator...)

Shorthand constructor:

	+(A::AbstractOperator...)

Sum of operators.

```jldoctest
julia> Sum(DiagOp(rand(10)), Eye(10))
╲+I  ℝ^10 -> ℝ^10

julia> MatrixOp(randn(5,5)) + Eye(5)
▒+I  ℝ^5 -> ℝ^5
	
```
"""
struct Sum{K, C <: AbstractArray, D <: AbstractArray, L <: NTuple{K, AbstractOperator}} <:
    AbstractOperator
    A::L
    bufC::C
    bufD::D
    # Bare inner constructor — validation and flattening handled by the @generated outer constructor
    Sum{K, C, D, L}(A::L, bufC::C, bufD::D) where {K, C, D, L} = new{K, C, D, L}(A, bufC, bufD)
end

Sum(A::AbstractOperator) = A

function Sum(A::Vararg{AbstractOperator})
    bufC = allocate_in_codomain(A[1])
    bufD = allocate_in_domain(A[1])
    return Sum(A, bufC, bufD)
end

# special cases
function Sum(L1::AbstractOperator, L2::Sum{K, C, D}) where {K, C, D}
    return Sum((L1, L2.A...), L2.bufC, L2.bufD)
end

# @generated outer constructor: validates, flattens nested Sums, and filters Zeros — all
# with output type fully determined at compile time (zero runtime dispatch).
@generated function Sum(A::L, bufC::C, bufD::D) where {C, D, K, L <: NTuple{K, AbstractOperator}}
    # Compute the flattened operator types and access expressions at compile time.
    flat_types = Type[]
    extraction_exprs = Expr[]

    for i in 1:K
        Ti = fieldtype(L, i)
        if Ti <: Sum
            # Flatten this nested Sum one level: pull its inner operators directly.
            L_inner = fieldtype(Ti, :A)     # the NTuple type stored in field A of nested Sum
            for j in 1:fieldcount(L_inner)
                Tij = fieldtype(L_inner, j)
                if !(Tij <: Zeros)
                    push!(flat_types, Tij)
                    push!(extraction_exprs, :(A[$i].A[$j]))
                end
            end
        elseif !(Ti <: Zeros)
            push!(flat_types, Ti)
            push!(extraction_exprs, :(A[$i]))
        end
    end

    # Unroll validation at compile time — @generated functions cannot contain closures
    # or generator expressions, so we build the checks as explicit comparisons.
    size_checks = [:(size(A[$i]) != size(A[1]) && throw(DimensionMismatch("cannot sum operator of different sizes"))) for i in 2:K]
    ctype_checks = [:(codomain_type(A[$i]) != codomain_type(A[1]) && throw(DomainError(A, "cannot sum operator with different codomains"))) for i in 2:K]
    dtype_checks = [:(domain_type(A[$i]) != domain_type(A[1]) && throw(DomainError(A, "cannot sum operator with different codomains"))) for i in 2:K]
    validation = Expr(:block, size_checks..., ctype_checks..., dtype_checks...)

    n_flat = length(flat_types)

    if n_flat == 0
        # Degenerate: all elements are Zeros — return the first operator unchanged.
        return quote
            $validation
            return A[1]
        end
    end

    # Fast path: no Sum or Zeros in the original tuple — use types as-is.
    if n_flat == K && !any(fieldtype(L, i) <: Union{Sum, Zeros} for i in 1:K)
        return quote
            $validation
            return Sum{$K, $C, $D, $L}(A, bufC, bufD)
        end
    end

    # After flattening/filtering a single operator remains — return it directly.
    if n_flat == 1
        ex = extraction_exprs[1]
        return quote
            $validation
            return $ex
        end
    end

    # Multiple operators after flattening: build the new flat tuple with fully concrete types.
    L_new = Tuple{flat_types...}
    K_new = n_flat
    ops_expr = Expr(:tuple, extraction_exprs...)

    return quote
        $validation
        ops = $ops_expr
        return Sum{$K_new, $C, $D, $L_new}(ops, bufC, bufD)
    end
end

# Mappings

@generated function mul!(y::AbstractArray, S::Sum{K}, b::AbstractArray) where {K}
    ex = :(mul!(y, S.A[1], b))
    for i in 2:K
        ex = quote
            $ex
            mul!(S.bufC, S.A[$i], b)
        end
        ex = :($ex; y .+= S.bufC)
    end
    return ex = quote
        check(y, S, b)
        $ex
        return y
    end
end

@generated function mul!(y::AbstractArray, A::AdjointOperator{<:Sum{K}}, b::AbstractArray) where {K}
    ex = :(S = A.A; mul!(y, S.A[1]', b))
    for i in 2:K
        ex = quote
            $ex
            mul!(S.bufD, S.A[$i]', b)
        end
        ex = :($ex; y .+= S.bufD)
    end
    return ex = quote
        check(y, A, b)
        $ex
        return y
    end
end

# Properties

Base.:(==)(S1::Sum{K, C, D, L}, S2::Sum{K, C, D, L}) where {K, C, D, L} = all(S1.A .== S2.A)
size(L::Sum) = size(L.A[1])

domain_type(S::Sum{K, C, D, L}) where {K, C, D <: AbstractArray, L} = domain_type(S.A[1])
domain_type(S::Sum{K, C, D, L}) where {K, C, D <: Tuple, L} = domain_type.(Ref(S.A[1]))
codomain_type(S::Sum{K, C, D, L}) where {K, C <: AbstractArray, D, L} = codomain_type(S.A[1])
codomain_type(S::Sum{K, C, D, L}) where {K, C <: Tuple, D, L} = codomain_type.(Ref(S.A[1]))
domain_storage_type(S::Sum{K, C, D, L}) where {K, C <: AbstractArray, D, L} = domain_storage_type(S.A[1])
domain_storage_type(S::Sum{K, C, D, L}) where {K, C <: Tuple, D, L} = domain_storage_type.(Ref(S.A[1]))
codomain_storage_type(S::Sum{K, C, D, L}) where {K, C <: AbstractArray, D, L} = codomain_storage_type(S.A[1])
codomain_storage_type(S::Sum{K, C, D, L}) where {K, C <: Tuple, D, L} = codomain_storage_type.(Ref(S.A[1]))

fun_domain(S::Sum) = fun_domain(S.A[1])
fun_codomain(S::Sum) = fun_codomain(S.A[1])

# A Sum of multi-domain operators (e.g. Sum of HCATs) has the same domain arity
# as its first constituent.  This extends the HCAT machinery so that HCAT can
# correctly assign indices when a Sum appears as one of its sub-operators.
_ndoms_from_type(::Type{<:Sum{K, C, D, L}}, dim::Int) where {K, C, D, L} =
    _ndoms_from_type(fieldtype(L, 1), dim)

fun_name(S::Sum) = length(S.A) == 2 ? fun_name(S.A[1]) * "+" * fun_name(S.A[2]) : "Σ"

is_linear(L::Sum) = all(is_linear.(L.A))
is_null(L::Sum) = all(is_null.(L.A))
is_diagonal(L::Sum) = all(is_diagonal.(L.A))
is_full_row_rank(L::Sum) = any(is_full_row_rank.(L.A))
is_full_column_rank(L::Sum) = any(is_full_column_rank.(L.A))

diag(L::Sum) = (+).(diag.(L.A)...)

# utils
function permute(S::Sum, p::AbstractVector{Int})
    AA = ([permute(A, p) for A in S.A]...,)
    return Sum(AA, S.bufC, ArrayPartition(S.bufD.x[p]...))
end

remove_displacement(S::Sum) = Sum(remove_displacement.(S.A), S.bufC, S.bufD)
