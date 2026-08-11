export OperatorWrapper

"""
    OperatorWrapper{Op, DB, CB, DS, CS}

A wrapper that adapts any operator's storage type independently of the wrapped operator.
Useful for making CPU operators transparently usable with GPU arrays.

When `mul!(y, wrapper, x)` is called the wrapper:
1. Copies `x` to the preallocated domain buffer (same storage as the inner op)
2. Executes the inner operator on its native buffers
3. Copies the result to `y`

GPU `mul!` overrides are provided by the GpuExt extension (requires loading GPUArrays).

# Construction
    OperatorWrapper(op::AbstractOperator; array_type = Array)

`array_type` sets the *outer* storage type reported by the wrapper (used by `check`).
This is independent of the wrapped operator's storage type, which governs the internal
CPU buffers.

# Notes
- Thread safety: each wrapper has its own buffers; use `copy_operator` for parallel use.

# Example
```jldoctest
julia> using AbstractOperators

julia> op = FiniteDiff(Float64, (8,), 1)
δx  ℝ^8 -> ℝ^7

julia> w = OperatorWrapper(op)
CPU[δx]  ℝ^8 -> ℝ^7

julia> size(w) == size(op)
true
```
"""
struct OperatorWrapper{
        Op <: AbstractOperator,
        DB <: AbstractArray,  # inner domain buffer (always CPU-backed)
        CB <: AbstractArray,  # inner codomain buffer (always CPU-backed)
        DS <: AbstractArray,  # outer domain storage type (reported to check)
        CS <: AbstractArray,  # outer codomain storage type (reported to check)
    } <: LinearOperator
    op::Op
    dom_buf::DB
    cod_buf::CB
end

"""
    OperatorWrapper(op::AbstractOperator; array_type = Array)

Wrap `op`, preallocating internal CPU buffers from its domain/codomain.
`array_type` sets the outer storage type used in `check` (default: `Array`).
"""
function OperatorWrapper(op::AbstractOperator; array_type::Type = Array)
    dom_buf = allocate_in_domain(op)
    cod_buf = allocate_in_codomain(op)
    S = _array_wrapper_type(array_type)
    T_dom = domain_type(op)
    T_cod = codomain_type(op)
    DS = S{T_dom}
    CS = S{T_cod}
    return OperatorWrapper{typeof(op), typeof(dom_buf), typeof(cod_buf), DS, CS}(op, dom_buf, cod_buf)
end

# CPU mul! — copies through internal buffers
function mul!(y::AbstractArray, A::OperatorWrapper, x::AbstractArray)
    check(y, A, x)
    copyto!(A.dom_buf, x)
    mul!(A.cod_buf, A.op, A.dom_buf)
    copyto!(y, A.cod_buf)
    return y
end

function mul!(y::AbstractArray, Ac::AdjointOperator{<:OperatorWrapper}, x::AbstractArray)
    check(y, Ac, x)
    A = Ac.A
    copyto!(A.cod_buf, x)
    mul!(A.dom_buf, A.op', A.cod_buf)
    copyto!(y, A.dom_buf)
    return y
end

# Properties — size/types delegate to inner op

Base.size(A::OperatorWrapper) = size(A.op)
fun_name(A::OperatorWrapper) = "CPU[$(fun_name(A.op))]"

domain_type(A::OperatorWrapper) = domain_type(A.op)
codomain_type(A::OperatorWrapper) = codomain_type(A.op)

# Outer storage types are fixed at construction via DS/CS type parameters.
domain_array_type(::OperatorWrapper{Op, DB, CB, DS, CS}) where {Op, DB, CB, DS, CS} = DS
codomain_array_type(::OperatorWrapper{Op, DB, CB, DS, CS}) where {Op, DB, CB, DS, CS} = CS

# Forward all predicates to the wrapped operator.
import OperatorCore:
    is_linear, is_eye, is_null, is_diagonal,
    is_AcA_diagonal, is_AAc_diagonal, diag_AcA, diag_AAc,
    is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank,
    is_symmetric, is_positive_definite, is_positive_semidefinite

is_linear(A::OperatorWrapper) = is_linear(A.op)
is_eye(A::OperatorWrapper) = is_eye(A.op)
is_null(A::OperatorWrapper) = is_null(A.op)
is_diagonal(A::OperatorWrapper) = is_diagonal(A.op)
is_AcA_diagonal(A::OperatorWrapper) = is_AcA_diagonal(A.op)
is_AAc_diagonal(A::OperatorWrapper) = is_AAc_diagonal(A.op)
diag_AcA(A::OperatorWrapper) = diag_AcA(A.op)
diag_AAc(A::OperatorWrapper) = diag_AAc(A.op)
is_orthogonal(A::OperatorWrapper) = is_orthogonal(A.op)
is_full_row_rank(A::OperatorWrapper) = is_full_row_rank(A.op)
is_full_column_rank(A::OperatorWrapper) = is_full_column_rank(A.op)
is_symmetric(A::OperatorWrapper) = is_symmetric(A.op)
is_positive_definite(A::OperatorWrapper) = is_positive_definite(A.op)
is_positive_semidefinite(A::OperatorWrapper) = is_positive_semidefinite(A.op)

# OperatorWrapper has mutable buffers — never thread-safe regardless of inner op.
is_thread_safe(::OperatorWrapper) = false

displacement(A::OperatorWrapper) = displacement(A.op)
remove_displacement(A::OperatorWrapper) = OperatorWrapper(remove_displacement(A.op))

function _copy_operator_impl(
        A::OperatorWrapper{Op, DB, CB, DS, CS}; storage_type = nothing, threaded = nothing
    ) where {Op, DB, CB, DS, CS}
    new_op = copy_operator(A.op; storage_type = nothing, threaded)
    return OperatorWrapper{typeof(new_op), DB, CB, DS, CS}(
        new_op, similar(A.dom_buf), similar(A.cod_buf)
    )
end
