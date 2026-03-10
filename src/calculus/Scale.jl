export Scale

"""
	Scale(α::Number,A::AbstractOperator)

Shorthand constructor:

	*(α::Number,A::AbstractOperator)

Scale an `AbstractOperator` by a factor of `α`.

```jldoctest
julia> A = FiniteDiff((10,2))
δx  ℝ^(10, 2) -> ℝ^(9, 2)

julia> S = Scale(10,A)
αδx  ℝ^(10, 2) -> ℝ^(9, 2)

julia> 10*A         #shorthand
αδx  ℝ^(10, 2) -> ℝ^(9, 2)
	
```
"""
struct Scale{Th, T <: Number, L <: AbstractOperator} <: AbstractOperator
    coeff::T
    coeff_conj::T
    A::L
    function Scale(coeff, coeff_conj, L; threaded = default_should_thread(L))
        cT = codomain_type(L)
        isCodomainReal = typeof(cT) <: Tuple ? all([t <: Real for t in cT]) : cT <: Real
        if isCodomainReal && typeof(coeff) <: Complex
            error(
                "Cannot Scale AbstractOperator with real codomain with complex scalar. Use `DiagOp` instead.",
            )
        end
        Th = threaded ? FastBroadcast.True() : FastBroadcast.False()
        return new{Th, typeof(coeff), typeof(L)}(coeff, coeff_conj, L)
    end
end

_ndoms_from_type(::Type{<:Scale{<:Any, <:Any, L}}, dim::Int) where {L} = _ndoms_from_type(L, dim)

# Constructors
function Scale(coeff, L; threaded = default_should_thread(L))
    if coeff == 1
        return L
    end
    coeff_conj = conj(coeff)
    coeff, coeff_conj = promote(coeff, coeff_conj)
    return Scale(coeff, coeff_conj, L; threaded)
end

get_output_length(L) = ndoms(L, 1) == 1 ? prod(size(L, 1)) : sum(prod.(size(L, 1)))
default_should_thread(L) = Threads.nthreads() > 1 && get_output_length(L) > 1.0e4

# Special Constructors
# scale of scale
function Scale(coeff::Number, L::Scale; threaded = default_should_thread(L))
    return Scale(*(promote(coeff, L.coeff)...), L.A; threaded)
end

# Mappings

function mul!(y::AbstractArray, L::Scale{Th}, x::AbstractArray) where {Th}
    check(y, L, x)
    mul!(y, L.A, x)
    return @.. thread = Th y *= L.coeff
end

function mul!(y::Tuple, L::Scale{Th}, x::AbstractArray) where {Th}
    check(y, L, x)
    mul!(y, L.A, x)
    for k in eachindex(y)
        @.. thread = Th y[k] *= L.coeff
    end
    return
end

function mul!(
        y::AbstractArray, S::AdjointOperator{<:Scale{Th}}, x::AbstractArray
    ) where {Th}
    check(y, S, x)
    L = S.A
    mul!(y, L.A', x)
    return @.. thread = Th y .*= L.coeff_conj
end

function mul!(y::Tuple, S::AdjointOperator{<:Scale{Th}}, x::AbstractArray) where {Th}
    check(y, S, x)
    L = S.A
    mul!(y, L.A', x)
    for k in eachindex(y)
        @.. thread = Th y[k] .*= L.coeff_conj
    end
    return
end

has_optimized_normalop(L::Scale) = is_linear(L.A) && has_optimized_normalop(L.A)
function get_normal_op(L::Scale)
    if is_linear(L.A)
        return Scale(L.coeff * L.coeff_conj, L.coeff * L.coeff_conj, get_normal_op(L.A))
    else
        return L' * L
    end
end

# Properties

function Base.:(==)(L1::Scale{<:Any, T, A}, L2::Scale{<:Any, T, A}) where {T, A}
    return L1.coeff == L2.coeff && L1.A == L2.A
end
size(L::Scale) = size(L.A)

domain_type(L::Scale) = domain_type(L.A)
codomain_type(L::Scale) = codomain_type(L.A)
domain_storage_type(L::Scale) = domain_storage_type(L.A)
codomain_storage_type(L::Scale) = codomain_storage_type(L.A)
is_thread_safe(L::Scale) = is_thread_safe(L.A)

is_linear(L::Scale) = is_linear(L.A)
is_sliced(L::Scale) = is_sliced(L.A)
get_slicing_expr(L::Scale) = get_slicing_expr(L.A)
get_slicing_mask(L::Scale) = get_slicing_mask(L.A)
remove_slicing(L::Scale) = L.coeff * remove_slicing(L.A)
is_null(L::Scale) = is_null(L.A)
is_diagonal(L::Scale) = is_diagonal(L.A)
is_invertible(L::Scale) = L.coeff == 0 ? false : is_invertible(L.A)
is_AcA_diagonal(L::Scale) = is_AcA_diagonal(L.A)
is_AAc_diagonal(L::Scale) = is_AAc_diagonal(L.A)
is_full_row_rank(L::Scale) = is_full_row_rank(L.A)
is_full_column_rank(L::Scale) = is_full_column_rank(L.A)

fun_name(L::Scale) = "α$(fun_name(L.A))"

diag(L::Scale) = L.coeff * diag(L.A)
diag_AcA(L::Scale) = (L.coeff)^2 * diag_AcA(L.A)
diag_AAc(L::Scale) = (L.coeff)^2 * diag_AAc(L.A)
remove_displacement(S::Scale) = Scale(S.coeff, S.coeff_conj, remove_displacement(S.A))

has_fast_opnorm(L::Scale) = has_fast_opnorm(L.A)
LinearAlgebra.opnorm(L::Scale) = abs(L.coeff) * LinearAlgebra.opnorm(L.A)
estimate_opnorm(L::Scale) = abs(L.coeff) * estimate_opnorm(L.A)

# utils

function permute(S::Scale, p::AbstractVector{Int})
    A = permute(S.A, p)
    return Scale(S.coeff, S.coeff_conj, A)
end
