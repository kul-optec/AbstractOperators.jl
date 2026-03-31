export VCAT

"""
	VCAT(A::AbstractOperator...)

Shorthand constructors:

	[A1; A2 ...]
	vcat(A...)

Vertically concatenate `AbstractOperator`s. Notice that all the operators must share the same domain dimensions and type, e.g. `size(A1,2) == size(A2,2)` and `domain_type(A1) == domain_type(A2)`.

```jldoctest
julia> VCAT(FiniteDiff((4,4)),Variation((4,4)))
[δx;Ʋ]  ℝ^(4, 4) -> ℝ^(3, 4)  ℝ^(16, 2)

julia> V = [Eye(3); DiagOp(2*ones(3))]
[I;╲]  ℝ^3 -> ℝ^3  ℝ^3

julia> vcat(V,FiniteDiff((3,)))
VCAT  ℝ^3 -> ℝ^3  ℝ^3  ℝ^2

julia> # When multiplying a `VCAT` with an array of the proper size, the result will be a `Tuple` containing arrays with the `VCAT`'s codomain type and size.

julia> V*ones(3)
([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
	
```
"""
struct VCAT{
        N, # number of AbstractOperator
        L <: NTuple{N, AbstractOperator},
        P <: Tuple,
        C <: AbstractArray,
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
        ) where {N, L <: NTuple{N, AbstractOperator}, P <: Tuple, C <: AbstractArray}
        if any([size(A[1], 2) != size(a, 2) for a in A])
            throw(DimensionMismatch("operators must have the same domain dimension!"))
        end
        if any([domain_type(A[1]) != domain_type(a) for a in A])
            throw(error("operators must all share the same domain_type!"))
        end
        return new{N, L, P, C}(A, idxs, buf)
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
        buf = A[findfirst((<:).(typeof.(A), VCAT))::Int].buf
    else
        AA = A
        # generate buffer
        buf = allocate_in_domain(AA[1])
    end

    return VCAT(AA, buf)
end

# compile-time codomain ndoms for VCAT's sub-operators
_ndoms_from_type(::Type{<:VCAT{N}}, dim::Int) where {N} = dim == 1 ? N : 1

@generated function VCAT(AA::NTuple{N, AbstractOperator}, buf) where {N}
    N == 1 && return :(AA[1])
    # Build idxs at compile time using operator element types
    K = 0
    idx_exprs = []
    for i in 1:N
        nd = _ndoms_from_type(fieldtype(AA, i), 1)
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
    return :(VCAT(AA, $idxs_literal, buf))
end

VCAT(A::AbstractOperator) = A

# Mappings

@generated function mul!(y::ArrayPartition, H::VCAT{N, L, P}, b::AbstractArray) where {N, L, P}
    ex = :(check(y, H, b))
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

@generated function mul!(
        y::AbstractArray, A::AdjointOperator{<:VCAT{N, L, P}}, b::ArrayPartition
    ) where {N, L, P}
    ex = :(check(y, A, b); H = A.A)

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

# Properties

Base.:(==)(H1::VCAT{N, L1, P1, C}, H2::VCAT{N, L2, P2, C}) where {N, L1, L2, P1, P2, C} = H1.A == H2.A && H1.idxs == H2.idxs

@generated function size(H::VCAT{N, L, P}) where {N, L, P}
    exprs = []
    for i in 1:N
        Pi = fieldtype(P, i)
        if Pi <: Integer
            push!(exprs, :(size(H.A[$i], 1)))
        else
            for ii in eachindex(fieldnames(Pi))
                push!(exprs, :(size(H.A[$i], 1)[$ii]))
            end
        end
    end
    natural_expr = Expr(:tuple, exprs...)
    return :((_vcat_apply_invperm($natural_expr, H.idxs), size(H.A[1], 2)))
end

# Apply inverse permutation (from VCAT idxs) to a natural-order codomain size/type tuple.
function _vcat_apply_invperm(natural::Tuple, idxs)
    p = vcat([[idx...] for idx in idxs]...)
    ip = invperm(p)
    return ntuple(j -> natural[ip[j]], Val(length(natural)))
end

function fun_name(L::VCAT)
    return length(L.A) == 2 ? "[" * fun_name(L.A[1]) * ";" * fun_name(L.A[2]) * "]" : "VCAT"
end

domain_type(L::VCAT) = domain_type.(Ref(L.A[1]))
@generated function codomain_type(H::VCAT{N, L, P}) where {N, L, P}
    exprs = []
    for i in 1:N
        Pi = fieldtype(P, i)
        if Pi <: Integer
            push!(exprs, :(codomain_type(H.A[$i])))
        else
            for ii in eachindex(fieldnames(Pi))
                push!(exprs, :(codomain_type(H.A[$i])[$ii]))
            end
        end
    end
    natural_expr = Expr(:tuple, exprs...)
    return :(_vcat_apply_invperm($natural_expr, H.idxs))
end
domain_storage_type(L::VCAT) = domain_storage_type.(Ref(L.A[1]))
function codomain_storage_type(H::VCAT)
    codomain = vcat([d <: ArrayPartition ? [d.parameters[2].types...] : d for d in codomain_storage_type.(H.A)]...)
    p = vcat([[idx...] for idx in H.idxs]...)
    invpermute!(codomain, p)
    T = promote_type(codomain_type(H)...)
    return ArrayPartition{T, Tuple{codomain...}}
end

is_linear(L::VCAT) = all(is_linear.(L.A))
is_AcA_diagonal(L::VCAT) = all(is_AcA_diagonal.(L.A))
is_AAc_diagonal(L::VCAT) = all(is_AAc_diagonal.(L.A))
is_full_column_rank(L::VCAT) = any(is_full_column_rank.(L.A))

is_sliced(L::VCAT) = any(is_sliced.(L.A))
function get_slicing_expr(L::VCAT)
    return get_slicing_expr.(Tuple(L.A[i] for i in eachindex(L.A)))
end
function remove_slicing(L::VCAT)
    new_ops = remove_slicing.(L[i] for i in eachindex(L.A))
    if !any(a -> a isa HCAT, new_ops) && all(i -> i isa Int, L.idxs)
        return DCAT(new_ops[collect(L.idxs)]...)
    elseif all(a -> a isa HCAT, L.A) && any(a -> any(is_null, a.A), L.A) && any(op -> size(op, 2) != size(new_ops[1], 2), new_ops)
        expected_hcat_domain_size = Vector{Any}(nothing, length(new_ops))
        for hcat_op in new_ops
            for i in eachindex(hcat_op.A)
                if !is_null(hcat_op[i]) || expected_hcat_domain_size[i] === nothing
                    expected_hcat_domain_size[i] = size(hcat_op[i], 2)
                end
            end
        end
        new_ops = collect(new_ops)
        for (i, hcat_op) in enumerate(new_ops)
            if any(is_null, hcat_op.A)
                ops = ()
                for j in eachindex(hcat_op.A)
                    op = if is_null(hcat_op[j])
                        Zeros(domain_type(hcat_op[j]), expected_hcat_domain_size[j], codomain_type(hcat_op[j]), size(hcat_op[j], 1))
                    else
                        hcat_op[j]
                    end
                    ops = (ops..., op)
                end
                new_ops[i] = hcat(ops...)
            end
        end
        return VCAT(tuple(new_ops...), L.idxs, L.buf)
    end
    error("remove_slicing is not implemented for this VCAT: $(typeof(L))")
end

diag_AcA(L::VCAT) = (+).(diag_AcA.(L.A)...)
diag_AAc(L::VCAT) = Tuple(diag_AAc.(L.A))

# utils
function permute(H::VCAT{N, L, P, C}, p::AbstractVector{Int}) where {N, L, P, C}
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

    return VCAT(H.A, new_part, H.buf)
end

remove_displacement(V::VCAT) = VCAT(remove_displacement.(V.A), V.idxs, V.buf)
