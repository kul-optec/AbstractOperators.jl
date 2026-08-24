export LMatrixOp

"""
	LMatrixOp(domain_type=Float64::Type, dim_in::Tuple, b::Union{AbstractVector,AbstractMatrix}; threaded=true)
	LMatrixOp(b::AbstractVector, number_of_rows::Int; threaded=true)

Creates a `LinearOperator` which, when multiplied with a matrix `X::AbstractMatrix`, returns the product `X*b`.

- `threaded`: `false` forces `mul!` to run with BLAS restricted to a single thread; `true`
  (default) lets BLAS use its full available thread budget subject to the size policy and
  any outer `NestedThreading` restriction. Only the `X::AbstractMatrix` (BLAS `gemm!`) paths
  are affected -- the `Y::AbstractVector` adjoint path is a plain elementwise product, not a
  BLAS call.

```jldoctest
julia> op = LMatrixOp(Float64,(3,4),ones(4))
(⋅)b  ℝ^(3, 4) -> ℝ^3

julia> op = LMatrixOp(ones(4),3)
(⋅)b  ℝ^(3, 4) -> ℝ^3

julia> op*ones(3,4)
3-element Vector{Float64}:
 4.0
 4.0
 4.0

```
"""
struct LMatrixOp{T, A <: Union{AbstractVector, AbstractMatrix}, B <: AbstractMatrix, dS, cS} <:
    LinearOperator
    b::A
    bt::B
    n_row_in::Int
    threaded::Bool
end

##TODO decide what to do when domain_type is given, with conversion one loses pointer to data...
# Constructors
function LMatrixOp(
        domain_type::Type, DomainDim::Tuple{Int, Int}, b::A
        ; array_type::Type = _array_wrapper_type(A), threaded::Bool = true,
    ) where {A <: Union{AbstractVector, AbstractMatrix}}
    bt = b'
    dS = _normalize_array_type(array_type, domain_type)
    cS = _normalize_array_type(array_type, domain_type)
    th = _blas_threaded(threaded, domain_type, length(b), dS)
    return LMatrixOp{domain_type, A, typeof(bt), dS, cS}(b, bt, DomainDim[1], th)
end

function LMatrixOp(
        b::A, n_row_in::Int; threaded::Bool = true
    ) where {T, A <: Union{AbstractVector{T}, AbstractMatrix{T}}}
    return LMatrixOp(T, (n_row_in, size(b, 1)), b; array_type = _array_wrapper_type(A), threaded)
end

# Mappings
function mul!(y::AbstractArray, L::LMatrixOp, X::AbstractArray)
    check(y, L, X)
    return _with_blas_threading(L.threaded) do
        mul!(y, X, L.b)
    end
end

function mul!(y::AbstractArray, L::AdjointOperator{<:LMatrixOp}, Y::AbstractVector)
    check(y, L, Y)
    return y .= L.A.bt .* Y
end

function mul!(y::AbstractArray, L::AdjointOperator{<:LMatrixOp}, Y::AbstractMatrix)
    check(y, L, Y)
    return _with_blas_threading(L.A.threaded) do
        mul!(y, Y, L.A.b')
    end
end

# Properties
domain_type(::LMatrixOp{T}) where {T} = T
codomain_type(::LMatrixOp{T}) where {T} = T
domain_array_type(::LMatrixOp{T, A, B, dS}) where {T, A, B, dS} = dS
codomain_array_type(::LMatrixOp{T, A, B, dS, cS}) where {T, A, B, dS, cS} = cS
is_thread_safe(::LMatrixOp) = true
is_threaded(L::LMatrixOp) = L.threaded
supports_threading(::LMatrixOp) = true

fun_name(L::LMatrixOp) = "(⋅)b"

function size(L::LMatrixOp{T, A, B, dS, cS}) where {T, A <: AbstractVector, B <: Adjoint, dS, cS}
    return (L.n_row_in,), (L.n_row_in, length(L.b))
end
function size(L::LMatrixOp{T, A, B, dS, cS}) where {T, A <: AbstractMatrix, B <: AbstractMatrix, dS, cS}
    return (L.n_row_in, size(L.b, 2)), (L.n_row_in, size(L.b, 1))
end

#TODO

#is_full_row_rank(L::LMatrixOp) =
#is_full_column_rank(L::MatrixOp) =

function _copy_operator_impl(
        op::LMatrixOp{T, A, B, dS, cS}; storage_type = nothing, threaded = nothing
    ) where {T, A, B, dS, cS}
    new_threaded = threaded === nothing ? op.threaded : threaded
    new_b = storage_type === nothing ? op.b : similar(storage_type{eltype(op.b)}, size(op.b)) .= op.b
    return LMatrixOp(T, (op.n_row_in, size(op.b, 1)), new_b; threaded = new_threaded)
end
