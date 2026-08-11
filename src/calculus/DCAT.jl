export DCAT

"""
	DCAT(A::AbstractOperator...)

Block-diagonally concatenate `AbstractOperator`s.

```jldoctest
julia> D = DCAT(HCAT(Eye(2),Eye(2)),FiniteDiff((3,)))
[[I,I],0;0,δx]  ℝ^2  ℝ^2  ℝ^3 -> ℝ^2  ℝ^2

julia> DCAT(Eye(10),Eye(10),FiniteDiff((4,4)))
DCAT  ℝ^10  ℝ^10  ℝ^(4, 4) -> ℝ^10  ℝ^10  ℝ^(3, 4)

julia> #To evaluate `DCAT` operators multiply them with a `Tuple` of `AbstractArray` of the correct domain size and type. The output will consist as well of a `Tuple` with the codomain type and size of the `DCAT`.

julia> using RecursiveArrayTools

julia> D*ArrayPartition(ones(2),ones(2),ones(3))
([2.0, 2.0], [0.0, 0.0])
	
```
"""
struct DCAT{
        N,
        L <: NTuple{N, AbstractOperator},
        P1 <: NTuple{N, Union{Int, Tuple}},
        P2 <: NTuple{N, Union{Int, Tuple}},
        DS <: AbstractArray,  # domain storage type (fixed at construction)
        CS <: AbstractArray,  # codomain storage type (fixed at construction)
    } <: AbstractOperator
    A::L
    idxD::P1
    idxC::P2
    function DCAT(A::L, idxD::P1, idxC::P2) where {L, P1, P2}
        DS = _compute_dcat_ds(A, idxD)
        CS = _compute_dcat_cs(A, idxC)
        return new{length(A), L, P1, P2, DS, CS}(A, idxD, idxC)
    end
end

function _compute_dcat_ds(A, idxD)
    ds_list = [d <: ArrayPartition ? [d.parameters[2].types...] : d for d in domain_array_type.(A)]
    domain = vcat(ds_list...)
    p = vcat([[idx...] for idx in idxD]...)
    invpermute!(domain, p)
    T = promote_type(map(_storage_eltype, domain)...)
    return ArrayPartition{T, Tuple{domain...}}
end

function _compute_dcat_cs(A, idxC)
    cs_list = [d <: ArrayPartition ? [d.parameters[2].types...] : d for d in codomain_array_type.(A)]
    codomain = vcat(cs_list...)
    p = vcat([[idx...] for idx in idxC]...)
    invpermute!(codomain, p)
    T = promote_type(map(_storage_eltype, codomain)...)
    return ArrayPartition{T, Tuple{codomain...}}
end

# Flatten DCAT index structure (tuple of ints/tuples) into a flat tuple of ints.
# Used in _dcat_apply_invperm to avoid allocating intermediate Vectors.
@inline _dcat_flatten_idxs(::Tuple{}) = ()
@inline function _dcat_flatten_idxs(idxs::Tuple)
    head = first(idxs)
    tail = _dcat_flatten_idxs(Base.tail(idxs))
    return head isa Integer ? (Int(head), tail...) : (head..., tail...)
end

# Linear scan for value v in tuple t, returning its 1-based position.
@inline _dcat_find_in_tuple(::Tuple{}, ::Int, ::Int) = 0
@inline function _dcat_find_in_tuple(t::Tuple, v::Int, i::Int = 1)
    return first(t) == v ? i : _dcat_find_in_tuple(Base.tail(t), v, i + 1)
end

# Constructors
DCAT(A::AbstractOperator) = A

# compile-time ndoms for DCAT (both domain and codomain have N components)
_ndoms_from_type(::Type{<:DCAT{N}}, dim::Int) where {N} = N

function DCAT(A::Vararg{AbstractOperator})
    return _dcat_impl(A)
end

@generated function _dcat_impl(A::NTuple{N, AbstractOperator}) where {N}
    N == 1 && return :(A[1])
    # Build idxC (codomain, dim=1) and idxD (domain, dim=2) at compile time
    idx_exprs_C = []
    Kc = 0
    for i in 1:N
        ndc = _ndoms_from_type(fieldtype(A, i), 1)
        if ndc == 1
            Kc += 1
            push!(idx_exprs_C, Kc)
        else
            K0 = Kc
            push!(idx_exprs_C, ntuple(j -> K0 + j, ndc))
            Kc += ndc
        end
    end
    idx_exprs_D = []
    Kd = 0
    for i in 1:N
        ndd = _ndoms_from_type(fieldtype(A, i), 2)
        if ndd == 1
            Kd += 1
            push!(idx_exprs_D, Kd)
        else
            K0 = Kd
            push!(idx_exprs_D, ntuple(j -> K0 + j, ndd))
            Kd += ndd
        end
    end
    idxC_literal = Expr(:tuple, idx_exprs_C...)
    idxD_literal = Expr(:tuple, idx_exprs_D...)
    return :(DCAT(A, $idxD_literal, $idxC_literal))
end

# Mappings
@generated function mul!(
        yy::ArrayPartition, H::DCAT{N, L, P1, P2}, bb::ArrayPartition
    ) where {N, L, P1, P2}

    # extract stuff
    ex = :(check(yy, H, bb); y = yy.x; b = bb.x)

    for i in 1:N
        if fieldtype(P2, i) <: Int
            # flatten operator
            # build mul!(y[H.idxC[i]], H.A[i], b)
            yyi = :(y[H.idxC[$i]])
        else
            # stacked operator
            # build mul!(( y[H.idxC[i][1]], y[H.idxC[i][2]] ...  ), H.A[i], b)
            yyi = [:(y[H.idxC[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P2, i)))]
            yyi = :(ArrayPartition($(yyi...)))
        end

        if fieldtype(P1, i) <: Int
            # flatten operator
            # build mul!(H.buf, H.A[i], b[H.idxD[i]])
            bbi = :(b[H.idxD[$i]])
        else
            # stacked operator
            # build mul!(H.buf, H.A[i],( b[H.idxD[i][1]], b[H.idxD[i][2]] ...  ))
            bbi = [:(b[H.idxD[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P1, i)))]
            bbi = :(ArrayPartition($(bbi...)))
        end

        ex = :($ex; mul!($yyi, H.A[$i], $bbi))
    end
    ex = :($ex; return yy)
    return ex
end

@generated function mul!(
        yy::ArrayPartition, A::AdjointOperator{<:DCAT{N, L, P1, P2}}, bb::ArrayPartition
    ) where {N, L, P1, P2}

    # extract stuff
    ex = :(check(yy, A, bb); H = A.A; y = yy.x; b = bb.x)

    for i in 1:N
        if fieldtype(P1, i) <: Int
            # flatten operator
            # build mul!(y[H.idxD[i]], H.A[i]', b)
            yyi = :(y[H.idxD[$i]])
        else
            # stacked operator
            # build mul!(( y[H.idxD[i][1]], y[H.idxD[i][2]] ...  ), H.A[i]', b)
            yyi = [:(y[H.idxD[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P1, i)))]
            yyi = :(ArrayPartition($(yyi...)))
        end

        if fieldtype(P2, i) <: Int
            # flatten operator
            # build mul!(H.buf, H.A[i]', b[H.idxC[i]])
            bbi = :(b[H.idxC[$i]])
        else
            # stacked operator
            # build mul!(H.buf, H.A[i]',( b[H.idxC[i][1]], b[H.idxC[i][2]] ...  ))
            bbi = [:(b[H.idxC[$i][$ii]]) for ii in eachindex(fieldnames(fieldtype(P2, i)))]
            bbi = :(ArrayPartition($(bbi...)))
        end

        ex = :($ex; mul!($yyi, H.A[$i]', $bbi))
    end
    ex = :($ex; return yy)
    return ex
end

has_optimized_normalop(L::DCAT) = any(has_optimized_normalop.(L.A))
function get_normal_op(H::DCAT)
    idxs = tuple((1:length(H.A))...)
    return DCAT(tuple([get_normal_op(H.A[i]) for i in eachindex(H.A)]...), idxs, idxs)
end

# Apply inverse permutation encoded in idxs to natural, using only tuple
# operations so that no Vector is allocated on the hot path.
function _dcat_apply_invperm(natural::Tuple, idxs::Tuple)
    N = length(natural)
    p = _dcat_flatten_idxs(idxs)
    return ntuple(j -> natural[_dcat_find_in_tuple(p, j)], Val(N))
end

# Properties
Base.:(==)(H1::DCAT{N, L1, P1, P2}, H2::DCAT{N, L2, P1, P2}) where {N, L1, L2, P1, P2} = H1.A == H2.A && H1.idxD == H2.idxD && H1.idxC == H2.idxC

@generated function size(H::DCAT{N, L, P1, P2}) where {N, L, P1, P2}
    cod_exprs = []
    for i in 1:N
        Pi = fieldtype(P2, i)
        if Pi <: Integer
            push!(cod_exprs, :(size(H.A[$i], 1)))
        else
            for ii in eachindex(fieldnames(Pi))
                push!(cod_exprs, :(size(H.A[$i], 1)[$ii]))
            end
        end
    end
    dom_exprs = []
    for i in 1:N
        Pi = fieldtype(P1, i)
        if Pi <: Integer
            push!(dom_exprs, :(size(H.A[$i], 2)))
        else
            for ii in eachindex(fieldnames(Pi))
                push!(dom_exprs, :(size(H.A[$i], 2)[$ii]))
            end
        end
    end
    natural_cod = Expr(:tuple, cod_exprs...)
    natural_dom = Expr(:tuple, dom_exprs...)
    return :(
        _dcat_apply_invperm($natural_cod, H.idxC),
        _dcat_apply_invperm($natural_dom, H.idxD),
    )
end

function fun_name(L::DCAT)
    return if length(L.A) == 2
        "[" * fun_name(L.A[1]) * ",0;0," * fun_name(L.A[2]) * "]"
    else
        "DCAT"
    end
end

@generated function ndoms(H::DCAT{N, L}) where {N, L}
    nc = sum(_ndoms_from_type(fieldtype(L, i), 1) for i in 1:N)
    nd = sum(_ndoms_from_type(fieldtype(L, i), 2) for i in 1:N)
    return :(($(nc), $(nd)))
end

@generated function ndoms(H::DCAT{N, L}, dim::Int) where {N, L}
    nc = sum(_ndoms_from_type(fieldtype(L, i), 1) for i in 1:N)
    nd = sum(_ndoms_from_type(fieldtype(L, i), 2) for i in 1:N)
    return :(dim == 1 ? $(nc) : $(nd))
end

@generated function domain_type(H::DCAT{N, L, P1, P2}) where {N, L, P1, P2}
    exprs = []
    for i in 1:N
        Pi = fieldtype(P1, i)
        if Pi <: Integer
            push!(exprs, :(domain_type(H.A[$i])))
        else
            for ii in eachindex(fieldnames(Pi))
                push!(exprs, :(domain_type(H.A[$i])[$ii]))
            end
        end
    end
    natural_expr = Expr(:tuple, exprs...)
    return :(_dcat_apply_invperm($natural_expr, H.idxD))
end

function domain_array_type(::DCAT{N, L, P1, P2, DS, CS}) where {N, L, P1, P2, DS, CS}
    return DS
end

@generated function codomain_type(H::DCAT{N, L, P1, P2}) where {N, L, P1, P2}
    exprs = []
    for i in 1:N
        Pi = fieldtype(P2, i)
        if Pi <: Integer
            push!(exprs, :(codomain_type(H.A[$i])))
        else
            for ii in eachindex(fieldnames(Pi))
                push!(exprs, :(codomain_type(H.A[$i])[$ii]))
            end
        end
    end
    natural_expr = Expr(:tuple, exprs...)
    return :(_dcat_apply_invperm($natural_expr, H.idxC))
end

function codomain_array_type(::DCAT{N, L, P1, P2, DS, CS}) where {N, L, P1, P2, DS, CS}
    return CS
end
is_thread_safe(H::DCAT) = all(is_thread_safe.(H.A))

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
function permute(H::DCAT{N, L, P1, P2}, p::AbstractVector{Int}) where {N, L, P1, P2}
    unfolded = vcat([[idx...] for idx in H.idxD]...)
    invpermute!(unfolded, p)

    new_part = ()
    cnt = 0
    for z in length.(H.idxD)
        new_part = (
            new_part..., z == 1 ? unfolded[cnt + 1] : (unfolded[(cnt + 1):(z + cnt)]...,),
        )
        cnt += z
    end

    return DCAT(H.A, new_part, H.idxC)
end

remove_displacement(D::DCAT) = DCAT(remove_displacement.(D.A), D.idxD, D.idxC)

# special cases
# Eye constructor
Eye(x::ArrayPartition) = DCAT(Eye.(x.x)...)
diag(L::DCAT{N, Tuple{E, Vararg{E, M}}}) where {N, M, E <: Eye} = 1.0
diag_AAc(L::DCAT{N, Tuple{E, Vararg{E, M}}}) where {N, M, E <: Eye} = 1.0
diag_AcA(L::DCAT{N, Tuple{E, Vararg{E, M}}}) where {N, M, E <: Eye} = 1.0

has_fast_opnorm(L::DCAT) = all(has_fast_opnorm.(L.A))
LinearAlgebra.opnorm(L::DCAT) = maximum(opnorm.(L.A))
estimate_opnorm(L::DCAT) = maximum(estimate_opnorm.(L.A))
