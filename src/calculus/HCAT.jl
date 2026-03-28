export HCAT

"""
	HCAT(A::AbstractOperator...)

Shorthand constructors:

	[A1 A2 ...]
	hcat(A...)

Horizontally concatenate `AbstractOperator`s. Notice that all the operators must share the same codomain dimensions and type, e.g. `size(A1,1) == size(A2,1)` and `codomain_type(A1) == codomain_type(A2)`.

```jldoctest
julia> HCAT(Eye(10),FiniteDiff((20,))[1:10])
[I,↓*δx]  ℝ^10  ℝ^20 -> ℝ^10

julia> H = [Eye(10) DiagOp(2*ones(10))]
[I,╲]  ℝ^10  ℝ^10 -> ℝ^10

julia> hcat(H,FiniteDiff((11,)))
HCAT  ℝ^10  ℝ^10  ℝ^11 -> ℝ^10

julia> # To evaluate `HCAT` operators multiply them with a `Tuple` of `AbstractArray` of the correct dimensions and type.

julia> using RecursiveArrayTools

julia> H*ArrayPartition(ones(10),ones(10))
10-element Vector{Float64}:
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
 3.0
	
```
"""
struct HCAT{
        N, # number of AbstractOperator
        L <: NTuple{N, AbstractOperator},
        P <: Tuple,
        C <: AbstractArray,
    } <: AbstractOperator
    A::L     # tuple of AbstractOperators
    idxs::P  # indices
    # H = HCAT(Eye(n),HCAT(Eye(n),Eye(n))) has H.idxs = (1,2,3)
    # `AbstractOperators` are flatten
    # H = HCAT(Eye(n),Compose(MatrixOp(randn(n,n)),HCAT(Eye(n),Eye(n))))
    # has H.idxs = (1,(2,3))
    # `AbstractOperators` are stack
    buf::C   # buffer memory
    function HCAT(A::L, idxs::P, buf::C) where {N, L <: NTuple{N, AbstractOperator}, P <: Tuple, C}
        if any([size(A[1], 1) != size(a, 1) for a in A])
            throw(DimensionMismatch("operators must have the same codomain dimension!"))
        end
        if any([codomain_type(A[1]) != codomain_type(a) for a in A])
            throw(error("operators must all share the same codomain_type!"))
        end
        return new{N, L, P, C}(A, idxs, buf)
    end
end

function HCAT(A::Vararg{AbstractOperator})
    if any((<:).(typeof.(A), HCAT)) #there are HCATs in A
        AA = ()
        for a in A
            if typeof(a) <: HCAT # flatten
                AA = (AA..., a.A...)
            else                 # stack
                AA = (AA..., a)
            end
        end
        # use buffer from HCAT in A
        buf = A[findfirst((<:).(typeof.(A), HCAT))].buf
    else
        AA = A
        # generate buffer
        buf = allocate_in_codomain(AA[1])
    end

    return HCAT(AA, buf)
end

# Count actual domain slots from an HCAT's P (idxs) type.
# Each entry in P is either an Int (1 slot) or a NTuple{n,Int} (n slots).
_count_hcat_ndoms(::Type{<:Tuple{}}) = 0
@generated function _count_hcat_ndoms(::Type{P}) where {P <: Tuple}
    K = 0
    for i in 1:fieldcount(P)
        Pi = fieldtype(P, i)
        K += Pi <: Integer ? 1 : fieldcount(Pi)
    end
    return :($K)
end

# compile-time domain ndoms for HCAT's sub-operators:
# use the index-tuple type P (not N which only counts sub-operators) so that
# sub-operators with multi-component domains are accounted for correctly.
_ndoms_from_type(::Type{<:HCAT{N, L, P}}, dim::Int) where {N, L, P} =
    dim == 2 ? _count_hcat_ndoms(P) : 1

@generated function HCAT(AA::NTuple{N, AbstractOperator}, buf::C) where {N, C}
    N == 1 && return :(AA[1])
    # Build idxs at compile time using operator element types
    K = 0
    idx_exprs = []
    for i in 1:N
        nd = _ndoms_from_type(fieldtype(AA, i), 2)
        if nd == 1
            K += 1
            push!(idx_exprs, K)
        else
            K0 = K
            push!(idx_exprs, ntuple(j -> K0 + j, nd))
            K += nd
        end
    end
    idxs_literal = Expr(:tuple, idx_exprs...)
    return :(HCAT(AA, $idxs_literal, buf))
end

HCAT(A::AbstractOperator) = A

# Mappings
function mul!(y::AbstractArray, H::HCAT, b::ArrayPartition)
    check(y, H, b)
    return mul!(y, H, b.x)
end

function mul!(y::AbstractArray, H::HCAT, b::Tuple)
    if _hcat_has_natural_idxs(H)
        return _mul_hcat_natural!(y, H, b)
    end
    return _mul_hcat_indexed!(y, H, b)
end

function mul!(y::ArrayPartition, A::AdjointOperator{<:HCAT}, b::AbstractArray)
    check(y, A, b)
    mul!(y.x, A, b)
    return y
end

@generated function mul!(y::Tuple, A::AdjointOperator{<:HCAT{N, L, P}}, b::AbstractArray) where {N, L, P}
    K = 0
    function output_natural_expr(i)
        Pi = fieldtype(P, i)
        if Pi <: Integer
            K += 1
            return :(y[$K])
        else
            n = fieldcount(Pi)
            parts = [:(y[$(K + j)]) for j in 1:n]
            K += n
            return :(ArrayPartition($(parts...)))
        end
    end

    ex = :(H = A.A)
    function output_expr(i)
        Pi = fieldtype(P, i)
        if Pi <: Integer
            return :(y[H.idxs[$i]])
        else
            n = fieldcount(Pi)
            parts = [:(y[H.idxs[$i][$j]]) for j in 1:n]
            return :(ArrayPartition($(parts...)))
        end
    end

    ex_natural = ex
    for i in 1:N
        ex_natural = :($ex_natural; mul!($(output_natural_expr(i)), H.A[$i]', b))
    end
    ex_natural = :($ex_natural; return y)

    ex_indexed = ex
    for i in 1:N
        ex_indexed = :($ex_indexed; mul!($(output_expr(i)), H.A[$i]', b))
    end
    ex_indexed = :($ex_indexed; return y)

    return :(_hcat_has_natural_idxs(A.A) ? ($ex_natural) : ($ex_indexed))
end

_hcat_has_natural_idxs(H::HCAT{N, L, P}) where {N, L, P} = H.idxs == _hcat_natural_idxs(P)

@generated function _hcat_natural_idxs(::Type{P}) where {P <: Tuple}
    K = 0
    idx_exprs = []
    for i in 1:fieldcount(P)
        Pi = fieldtype(P, i)
        if Pi <: Integer
            K += 1
            push!(idx_exprs, K)
        else
            n = fieldcount(Pi)
            push!(idx_exprs, Expr(:tuple, (K + j for j in 1:n)...))
            K += n
        end
    end
    return Expr(:tuple, idx_exprs...)
end

@generated function _mul_hcat_natural!(y, H::HCAT{N, L, P}, b::Tuple) where {N, L, P}
    K = 0
    function input_expr(i)
        Pi = fieldtype(P, i)
        if Pi <: Integer
            K += 1
            return :(b[$K])
        else
            n = fieldcount(Pi)
            parts = [:(b[$(K + j)]) for j in 1:n]
            K += n
            return :(ArrayPartition($(parts...)))
        end
    end

    ex = :(mul!(y, H.A[1], $(input_expr(1))))
    for i in 2:N
        ex = :($ex; mul!(H.buf, H.A[$i], $(input_expr(i))))
        ex = :($ex; y .+= H.buf)
    end
    ex = :($ex; return y)
    return ex
end

@generated function _mul_hcat_indexed!(y, H::HCAT{N, L, P}, b::Tuple) where {N, L, P}
    function input_expr(i)
        Pi = fieldtype(P, i)
        if Pi <: Integer
            return :(b[H.idxs[$i]])
        else
            n = fieldcount(Pi)
            parts = [:(b[H.idxs[$i][$j]]) for j in 1:n]
            return :(ArrayPartition($(parts...)))
        end
    end

    ex = :(mul!(y, H.A[1], $(input_expr(1))))

    for i in 2:N
        ex = :($ex; mul!(H.buf, H.A[$i], $(input_expr(i))))
        ex = :($ex; y .+= H.buf)
    end
    ex = :($ex; return y)
    return ex
end

# Properties
Base.:(==)(H1::HCAT{N, L1, P1}, H2::HCAT{N, L2, P2}) where {N, L1, L2, P1, P2} = H1.A == H2.A && H1.idxs == H2.idxs

@generated function size(H::HCAT{N, L, P}) where {N, L, P}
    exprs = []
    for i in 1:N
        Pi = fieldtype(P, i)
        if Pi <: Integer
            push!(exprs, :(size(H.A[$i], 2)))
        else
            for ii in eachindex(fieldnames(Pi))
                push!(exprs, :(size(H.A[$i], 2)[$ii]))
            end
        end
    end
    natural_expr = Expr(:tuple, exprs...)
    return :(size(H.A[1], 1), _hcat_apply_invperm($natural_expr, H.idxs))
end

# Apply inverse permutation (from HCAT idxs) to a natural-order domain size/type tuple.
function _hcat_apply_invperm(natural::Tuple, idxs)
    p = vcat([[idx...] for idx in idxs]...)
    ip = invperm(p)
    return ntuple(j -> natural[ip[j]], Val(length(natural)))
end

function fun_name(L::HCAT)
    if length(L.A) == 2
        if L.idxs[1] == 2 || L.idxs[2] == 1
            return "[" * fun_name(L.A[2]) * "," * fun_name(L.A[1]) * "]"
        else
            return "[" * fun_name(L.A[1]) * "," * fun_name(L.A[2]) * "]"
        end
    else
        return "HCAT"
    end
end

@generated function domain_type(H::HCAT{N, L, P}) where {N, L, P}
    exprs = []
    for i in 1:N
        Pi = fieldtype(P, i)
        if Pi <: Integer
            push!(exprs, :(domain_type(H.A[$i])))
        else
            for ii in eachindex(fieldnames(Pi))
                push!(exprs, :(domain_type(H.A[$i])[$ii]))
            end
        end
    end
    natural_expr = Expr(:tuple, exprs...)
    return :(_hcat_apply_invperm($natural_expr, H.idxs))
end
codomain_type(L::HCAT) = codomain_type.(Ref(L.A[1]))
function domain_storage_type(H::HCAT)
    domain = vcat([d <: ArrayPartition ? [d.parameters[2].types...] : d for d in domain_storage_type.(H.A)]...)
    p = vcat([[idx...] for idx in H.idxs]...)
    invpermute!(domain, p)
    T = promote_type(domain_type(H)...)
    return ArrayPartition{T, Tuple{domain...}}
end
codomain_storage_type(L::HCAT) = codomain_storage_type.(Ref(L.A[1]))

is_linear(L::HCAT) = all(is_linear.(L.A))
is_AAc_diagonal(L::HCAT) = all(is_AAc_diagonal.(L.A))
is_full_row_rank(L::HCAT) = any(is_full_row_rank.(L.A))

is_sliced(L::HCAT) = any(is_sliced.(L.A))
function get_slicing_expr(L::HCAT)
    exprs = ()
    for i in eachindex(L.A)
        expr = get_slicing_expr(L[i])
        if expr isa Tuple && all(e -> e isa Tuple, expr)
            exprs = (exprs..., expr...)
        else
            exprs = (exprs..., expr)
        end
    end
    if length(exprs) == 1
        return exprs[1]
    else
        return exprs
    end
end
get_slicing_mask(L::HCAT) = get_slicing_mask.(L[i] for i in eachindex(L.A))
remove_slicing(L::HCAT) = HCAT(remove_slicing.(Tuple(A for A in L.A)), L.idxs, L.buf)

diag_AAc(L::HCAT) = (+).(diag_AAc.(L[i] for i in eachindex(L.A))...)

# utils
function permute(H::HCAT, p::AbstractVector{Int})
    unfolded = vcat([[idx...] for idx in H.idxs]...)
    invpermute!(unfolded, p)

    new_part = ()
    cnt = 0
    for z in length.(H.idxs)
        new_part = (
            new_part..., z == 1 ? unfolded[cnt + 1] : (unfolded[(cnt + 1):(z + cnt)]...,),
        )
        cnt += z
    end

    return HCAT(H.A, new_part, H.buf)
end

remove_displacement(H::HCAT) = HCAT(remove_displacement.(H.A), H.idxs, H.buf)
