#HadamardProd

export HadamardProd

"""
	HadamardProd(A::AbstractOperator,B::AbstractOperator)

Create an operator `P` such that:

	P*x == (Ax).*(Bx)

# Example

```jldoctest
julia> A,B = Sin(3), Cos(3);

julia> P = HadamardProd(A,B)
sin.*cos  ℝ^3 -> ℝ^3

julia> x = randn(3);

julia> P*x == (sin.(x).*cos.(x))
true
	
```
"""
struct HadamardProd{
        L1 <: AbstractOperator, L2 <: AbstractOperator, C <: AbstractArray, D <: AbstractArray,
    } <: NonLinearOperator
    A::L1
    B::L2
    bufA::C
    bufB::C
    bufD::D
    function HadamardProd(A::L1, B::L2, bufA::C, bufB::C, bufD::D) where {L1, L2, C, D}
        if size(A) != size(B)
            throw(DimensionMismatch("Cannot compose operators"))
        end
        return new{L1, L2, C, D}(A, B, bufA, bufB, bufD)
    end
end

struct HadamardProdJac{
        L1 <: AbstractOperator, L2 <: AbstractOperator, C <: AbstractArray, D <: AbstractArray,
    } <: LinearOperator
    A::L1
    B::L2
    bufA::C
    bufB::C
    bufD::D
end

# Constructors
function HadamardProd(A::AbstractOperator, B::AbstractOperator)
    bufA = allocate_in_codomain(A)
    bufB = allocate_in_codomain(B)
    bufD = allocate_in_domain(A)
    return HadamardProd(A, B, bufA, bufB, bufD)
end

# Jacobian
function Jacobian(P::HadamardProd{L1, L2, C, D}, x::AbstractArray) where {L1, L2, C, D}
    JA, JB = Jacobian(P.A, x), Jacobian(P.B, x)
    return HadamardProdJac{typeof(JA), typeof(JB), C, D}(JA, JB, P.bufA, P.bufB, P.bufD)
end

# Mappings
function mul!(y::AbstractArray, P::HadamardProd, b::AbstractArray)
    check(y, P, b)
    mul!(P.bufA, P.A, b)
    mul!(P.bufB, P.B, b)
    y .= P.bufA .* P.bufB
    return y
end

function mul!(y::AbstractArray, J::AdjointOperator{<:HadamardProdJac}, b::AbstractArray)
    #y .= J.A.B' * ( J.A.bufA .*b ) + J.A.A' * ( J.A.bufB .* b )
    check(y, J, b)
    J.A.bufA .*= b
    mul!(y, J.A.B', J.A.bufA)
    J.A.bufB .*= b
    mul!(J.A.bufD, J.A.A', J.A.bufB)
    y .+= J.A.bufD
    return y
end

# Properties
function Base.:(==)(P1::HadamardProd{L1, L2, C, D}, P2::HadamardProd{L1, L2, C, D}) where {L1, L2, C, D}
    return P1.A == P2.A && P1.B == P2.B
end
size(P::Union{HadamardProd, HadamardProdJac}) = (size(P.A, 1), size(P.A, 2))

fun_name(L::Union{HadamardProd, HadamardProdJac}) = fun_name(L.A) * ".*" * fun_name(L.B)

domain_type(L::Union{HadamardProd, HadamardProdJac}) = domain_type(L.A)
codomain_type(L::Union{HadamardProd, HadamardProdJac}) = codomain_type(L.A)
domain_array_type(L::Union{HadamardProd, HadamardProdJac}) = domain_array_type(L.A)
codomain_array_type(L::Union{HadamardProd, HadamardProdJac}) = codomain_array_type(L.A)

# utils
function permute(
        P::HadamardProd{L1, L2, C, D}, p::AbstractVector{Int}
    ) where {L1, L2, C, D <: ArrayPartition}
    return HadamardProd(
        permute(P.A, p), permute(P.B, p), P.bufA, P.bufB, ArrayPartition(P.bufD.x[p])
    )
end

function remove_displacement(P::HadamardProd)
    return HadamardProd(
        remove_displacement(P.A), remove_displacement(P.B), P.bufA, P.bufB, P.bufD
    )
end

function _copy_operator_impl(op::HadamardProd; storage_type = nothing, threaded = nothing)
    new_bufA = _convert_buffer(op.bufA, storage_type)
    new_bufB = _convert_buffer(op.bufB, storage_type)
    new_bufD = _convert_buffer(op.bufD, storage_type)
    new_A = copy_operator(op.A; storage_type, threaded)
    new_B = copy_operator(op.B; storage_type, threaded)
    return HadamardProd(new_A, new_B, new_bufA, new_bufB, new_bufD)
end

function _copy_operator_impl(op::HadamardProdJac; storage_type = nothing, threaded = nothing)
    # bufA/bufB hold the Jacobian linearization point (A*x, B*x) shared from the
    # originating HadamardProd — unlike bufD (pure scratch), their values must be
    # preserved, not just their shape/eltype.
    new_bufA = storage_type === nothing ? copy(op.bufA) :
        copyto!(similar(storage_type{eltype(op.bufA)}, size(op.bufA)), op.bufA)
    new_bufB = storage_type === nothing ? copy(op.bufB) :
        copyto!(similar(storage_type{eltype(op.bufB)}, size(op.bufB)), op.bufB)
    new_bufD = _convert_buffer(op.bufD, storage_type)
    new_A = copy_operator(op.A; storage_type, threaded)
    new_B = copy_operator(op.B; storage_type, threaded)
    return HadamardProdJac{typeof(new_A), typeof(new_B), typeof(new_bufA), typeof(new_bufD)}(
        new_A, new_B, new_bufA, new_bufB, new_bufD
    )
end

_children(L::HadamardProd) = (L.A, L.B)
is_threaded(L::HadamardProd) = _is_threaded_from_children(L)
_children(L::HadamardProdJac) = (L.A, L.B)
is_threaded(L::HadamardProdJac) = _is_threaded_from_children(L)
supports_threading(L::HadamardProd) = _supports_threading_from_children(L)
supports_threading(L::HadamardProdJac) = _supports_threading_from_children(L)
