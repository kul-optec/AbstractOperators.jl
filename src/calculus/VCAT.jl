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
        CS <: AbstractArray,  # codomain storage type (fixed at construction)
        Th,                   # thread the forward block loop?
    } <: AbstractOperator
    A::L     # tuple of AbstractOperators
    idxs::P  # indices; always NTuple{N, Int} since inner VCATs are flattened at construction
    buf::C   # buffer memory
    function VCAT(
            A::L, idxs::P, buf::C; threaded::Bool = true
        ) where {N, L <: NTuple{N, AbstractOperator}, P <: Tuple, C <: AbstractArray}
        if any([size(A[1], 2) != size(a, 2) for a in A])
            throw(DimensionMismatch("operators must have the same domain dimension!"))
        end
        if any([domain_type(A[1]) != domain_type(a) for a in A])
            throw(error("operators must all share the same domain_type!"))
        end
        CS = _compute_vcat_cs(A, idxs)
        # Only the *forward* direction is block-parallel: it writes disjoint output blocks.
        # The adjoint accumulates every block into one shared `y`, so it stays serial.
        # See the note in DCAT: `A` must not be reassigned while a closure captures it,
        # or it gets boxed to `Any` and the whole constructor goes through runtime dispatch.
        th = _resolve_threaded(threaded) do
            default_block_threaded(VCAT, A)
        end
        blocks = th ? map(a -> adapt_operator(a; threaded = false), A) : A
        return new{N, typeof(blocks), P, C, CS, th}(blocks, idxs, buf)
    end
end

function _compute_vcat_cs(A, idxs)
    cs_list = [d <: ArrayPartition ? [d.parameters[2].types...] : d for d in codomain_array_type.(A)]
    codomain = vcat(cs_list...)
    p = vcat([[idx...] for idx in idxs]...)
    invpermute!(codomain, p)
    T = promote_type(map(_storage_eltype, codomain)...)
    return ArrayPartition{T, Tuple{codomain...}}
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

@generated function VCAT(AA::NTuple{N, AbstractOperator}, buf) where {N}
    if N isa Int
        N == 1 && return :(AA[1])
        # Build idxs at compile time: inner VCATs are always flattened, so all elements have nd=1
        idxs_literal = Expr(:tuple, (1:N)...)
        return :(VCAT(AA, $idxs_literal, buf))
    else
        # N is not statically known (e.g. built up in a loop); fall back to runtime length
        return :(VCAT(AA, ntuple(identity, length(AA)), buf))
    end
end

VCAT(A::AbstractOperator) = A

# Mappings

@generated function mul!(
        y::ArrayPartition, H::VCAT{N, L, P, C, CS, false}, b::AbstractArray
    ) where {N, L, P, C, CS}
    ex = :(check(y, H, b))
    for i in 1:N
        # P always has Int elements (inner VCATs are flattened at construction)
        ex = :($ex; mul!(y.x[H.idxs[$i]], H.A[$i], b))
    end
    ex = :($ex; return y)
    return ex
end

# Threaded forward. As in DCAT, the parallel loop cannot live in a `@generated` body (those
# must be pure, and the threading macros expand to closures), so only the single-block call
# is generated and selected by `Val(i)`.
@generated function _vcat_block_fwd!(y, H::VCAT{N, L, P}, b, ::Val{I}) where {N, L, P, I}
    return :(mul!(y.x[H.idxs[$I]], H.A[$I], b))
end

function mul!(
        y::ArrayPartition, H::VCAT{N, L, P, C, CS, true}, b::AbstractArray
    ) where {N, L, P, C, CS}
    check(y, H, b)
    @budgeted_threads for i in 1:N
        _vcat_block_fwd!(y, H, b, Val(i))
    end
    return y
end

@generated function mul!(
        y::AbstractArray, A::AdjointOperator{<:VCAT{N, L, P}}, b::ArrayPartition
    ) where {N, L, P}
    ex = :(check(y, A, b); H = A.A)

    # P always has Int elements (inner VCATs are flattened at construction)
    ex = :($ex; mul!(y, H.A[1]', b.x[H.idxs[1]])) # write on y

    for i in 2:N
        ex = :($ex; mul!(H.buf, H.A[$i]', b.x[H.idxs[$i]])) # write on H.buf
        # sum H.buf with y
        ex = :($ex; y .+= H.buf)
    end
    ex = :($ex; return y)
    return ex
end

# Properties

function Base.:(==)(H1::VCAT{N, L1, P1, C}, H2::VCAT{N, L2, P2, C}) where {N, L1, L2, P1, P2, C}
    return H1.A == H2.A && H1.idxs == H2.idxs
end

@generated function size(H::VCAT{N, L, P}) where {N, L, P}
    # P always has Int elements (inner VCATs are flattened at construction)
    exprs = [:(size(H.A[$i], 1)) for i in 1:N]
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
    # P always has Int elements (inner VCATs are flattened at construction)
    exprs = [:(codomain_type(H.A[$i])) for i in 1:N]
    natural_expr = Expr(:tuple, exprs...)
    return :(_vcat_apply_invperm($natural_expr, H.idxs))
end
domain_array_type(L::VCAT) = domain_array_type.(Ref(L.A[1]))
codomain_array_type(::VCAT{N, L, P, C, CS}) where {N, L, P, C, CS} = CS

is_linear(L::VCAT) = all(is_linear.(L.A))
is_AcA_diagonal(L::VCAT) = all(is_AcA_diagonal.(L.A))
is_AAc_diagonal(L::VCAT) = all(is_AAc_diagonal.(L.A))
is_full_column_rank(L::VCAT) = any(is_full_column_rank.(L.A))

is_sliced(L::VCAT) = any(is_sliced.(L.A))
function get_slicing_expr(L::VCAT)
    return get_slicing_expr.(Tuple(L.A[i] for i in eachindex(L.A)))
end
function remove_slicing(L::VCAT)
    new_ops = collect(map(remove_slicing, L.A))
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
        return VCAT(tuple(new_ops...), L.idxs, L.buf; threaded = is_block_threaded(L))
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

    return VCAT(H.A, new_part, H.buf; threaded = is_block_threaded(H))
end

function remove_displacement(V::VCAT)
    return VCAT(remove_displacement.(V.A), V.idxs, V.buf; threaded = is_block_threaded(V))
end

function _copy_operator_impl(
        op::VCAT{N, L, P, C, CS, Th}; storage_type = nothing, threaded = nothing
    ) where {N, L, P, C, CS, Th}
    new_threaded = threaded === nothing ? Th : threaded
    new_buf = _convert_buffer(op.buf, storage_type)
    # When the forward block loop threads, the constructor forces the blocks serial, so
    # only the storage request is forwarded down.
    child_threaded = new_threaded ? false : threaded
    new_ops = tuple([copy_operator(a; storage_type, threaded = child_threaded) for a in op.A]...)
    return VCAT(new_ops, op.idxs, new_buf; threaded = new_threaded)
end

# Whether the *block loop itself* threads, as distinct from `is_threaded`, which is also
# true when merely a child threads. Structural transforms need the former to round-trip.
# PROVENANCE: measured. VCAT-forward sweep, speedup by per-block size across 4/8/16
# blocks: 2^14 -> 0.31/0.39/0.85 (all lose), 2^15 -> 0.74/1.11/1.26 (mixed),
# 2^16 -> 1.32/1.74/1.60 (all win). 2^16 is the first size that wins at every block count.
block_threading_threshold(::Type{<:VCAT}) = THRESHOLD_BLOCK_PARALLEL

is_block_threaded(::VCAT{N, L, P, C, CS, Th}) where {N, L, P, C, CS, Th} = Th

_children(L::VCAT) = L.A
# Threaded if the forward block loop threads, or any block does.
is_threaded(L::VCAT{N, Ls, P, C, CS, Th}) where {N, Ls, P, C, CS, Th} =
    Th || _is_threaded_from_children(L)
supports_threading(::VCAT) = true
