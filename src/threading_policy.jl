export is_threaded, adapt_operator, supports_threading

# ─── Thresholds ───────────────────────────────────────────────────────────────
#
# There is deliberately no single global constant: the size at which threading starts to
# pay depends on how much work each element costs. Operators with genuinely similar cost
# profiles share a constant, operators that differ get their own.
#
# Each constant carries a PROVENANCE line stating where its value came from. Values marked
# `measured` are transcriptions of a `benchmark/threading_sweep.jl` run whose machine and
# thread count are named; values marked `provisional` are order-of-magnitude starting
# points that the sweep has not yet confirmed. Never promote provisional -> measured
# without re-running the sweep and pasting its crossover.
#
# Measurement context for the `measured` values below:
#   AMD EPYC 7352 24-Core (96 logical), 8 Julia threads, OPENBLAS_NUM_THREADS=1, 2026-08-15
#   `OPENBLAS_NUM_THREADS=1 julia --project=benchmark -t 8 benchmark/threading_sweep.jl`
# Each threshold takes the *more conservative* of the Float64 and Float32 crossovers, so a
# threshold is never below the point where the narrower element type still pays.
#
# The sweep also settled the kernel choice per class, and it did not match the starting
# hypothesis everywhere -- see `THRESHOLD_MEMORY_BOUND`.

"""
Transcendental elementwise kernels (`Sin`, `Cos`, `Exp`, `Atan`, `Tanh`, `Sech`, `Sigmoid`,
`SoftPlus`, `Pow` with non-integer exponent). Each element costs tens of ns, so the thread
launch overhead is amortized early. Kernel: FastBroadcast `@..` with `thread = true`.

PROVENANCE: measured. `transcendental` sweep, `@..` crossover n = 1024 (Float64) and
n = 512 (Float32); taking the conservative Float64 value. At n = 2^20 threading is ~6x.
"""
const THRESHOLD_ELEMENTWISE_TRANSCENDENTAL = 2^10

"""
Cheap arithmetic elementwise kernels (`AffineAdd`, `HadamardProd`, `FiniteDiff`,
`Variation`, integer `Pow`). A few flops per element, so the crossover sits well above the
transcendental one but still below the pure-copy case. Kernel: FastBroadcast `@..` with
`thread = true`.

PROVENANCE: measured. `arithmetic` sweep, `@..` crossover n = 32768 for both Float64 and
Float32; the `finitediff` sweep independently lands on the same n = 32768, which is why
these two kernel shapes share one constant.
"""
const THRESHOLD_ELEMENTWISE_ARITHMETIC = 2^15

"""
Pure data movement (`Eye`, `ZeroPad`, `GetIndex`, `Zeros`, broadcasting). These are
memory-bandwidth-bound, so the win arrives late and is bandwidth- rather than core-limited.
Kernel: Polyester `@batch` -- **not** FastBroadcast.

PROVENANCE: measured. `memorybound` sweep. The result contradicted the starting hypothesis
that FastBroadcast wins or ties everywhere: for a plain `y = x` copy, `@..  thread = true`
is *time-identical to serial at every swept size* (FastBroadcast does not thread a pure
copy), first "winning" only at n = 2^22 where the difference is noise. `@batch` genuinely
threads it, crossing over at n = 32768 (Float64) and n = 262144 (Float32); this constant
takes the conservative Float32 value. `@budgeted_threads` never wins at Float64.
"""
const THRESHOLD_MEMORY_BOUND = 2^18

"""
Aggregate work across the blocks of a block-parallel calculus operator (`DCAT`). Unlike the
elementwise thresholds this counts *total elements over all blocks*, since the parallel
unit is a block, not an element. Kernel: `@budgeted_threads` (the loop body is a whole
child `mul!`, not element access).

PROVENANCE: measured. Block sweep over block count x block size, same machine and settings
as above. `@budgeted_threads` has a fixed ~25-40us setup cost here, so small blocks lose
badly (16 blocks x 2^12 = 0.56x, 8 x 2^12 = 0.16x) while large aggregates win big
(8 x 2^16 = 2.2x, 8 x 2^18 = 9.4x).
"""
const THRESHOLD_BLOCK_PARALLEL = 2^18

"""
Minimum number of blocks before block-level threading is considered, independent of total
size.

PROVENANCE: measured, and this constant exists *because* of the measurement: total work
alone is not a sufficient predictor. At 2 blocks x 2^18 elements the aggregate is 2^19 --
comfortably over `THRESHOLD_BLOCK_PARALLEL` -- yet the speedup is 1.02x, because two blocks
cap the achievable gain at 2x and the threading overhead eats it. The same 2^19 aggregate
split across 8 blocks gives 2.2x. Both conditions together admit every measured win and
exclude every measured loss.
"""
const MIN_BLOCKS_FOR_PARALLEL = 4

# ─── Storage classification ───────────────────────────────────────────────────

"""
	_is_cpu_storage(::Type{<:AbstractArray})

Whether Julia-level threading is meaningful for this storage type. GPU arrays are already
parallel at the kernel level, so wrapping their broadcasts in `@batch`/`@threads` adds
launch overhead and contention without adding parallelism.

Defaults to `false` for unknown array types — conservative, since the cost of wrongly
threading a device array is much higher than the cost of missing a thread-level win.
"""
_is_cpu_storage(::Type{<:AbstractArray}) = false
_is_cpu_storage(::Type{<:Array}) = true

# ─── Per-operator policy ──────────────────────────────────────────────────────

"""
	threading_threshold(::Type{<:AbstractOperator})

Number of elements at or above which threading is expected to pay for this operator type.
Defaults to `THRESHOLD_MEMORY_BOUND`, the most conservative choice.
"""
threading_threshold(::Type{<:AbstractOperator}) = THRESHOLD_MEMORY_BOUND

"""
	default_threaded(::Type{Op}, ::Type{T}, dims, ::Type{S}) -> Bool

Per-operator threading policy: whether an operator of type `Op` over element type `T`,
domain size `dims` and storage `S` should thread by default.

`dims` may be a plain dimension tuple or a tuple of tuples (multi-domain operators); both
are reduced to a total element count.

Operators specialize this when their policy is not purely size-driven; most only need to
specialize [`threading_threshold`](@ref).
"""
function default_threaded(
        ::Type{Op}, ::Type{T}, dims, ::Type{S}
    ) where {Op <: AbstractOperator, T, S <: AbstractArray}
    return _default_threaded(threading_threshold(Op), T, _total_elements(dims), S)
end

# Typed positional helper: keeps the decision statically resolvable for JET's `@test_opt`,
# which flags unparameterized `::Type` keywords as dynamic dispatch.
function _default_threaded(
        threshold::Int, ::Type{T}, n::Int, ::Type{S}
    ) where {T, S <: AbstractArray}
    return Threads.nthreads() > 1 && _is_cpu_storage(S) && n >= threshold
end

"""
	_elementwise_threaded(Op, threaded, T, dims, S) -> Bool

Resolve a constructor's `threaded` keyword for an elementwise operator: `nothing` means
"apply the per-operator policy", anything else is taken as an explicit override.

Returns a plain `Bool` so it can be spliced straight into a `Th` type parameter.
"""
function _elementwise_threaded(
        ::Type{Op}, threaded, ::Type{T}, dims, ::Type{S}
    ) where {Op <: AbstractOperator, T, S <: AbstractArray}
    threaded === nothing && return default_threaded(Op, T, dims, S)
    return threaded::Bool
end

"""
	default_block_threaded(blocks) -> Bool

Whether a block-parallel calculus operator over `blocks` should thread its block loop.

Requires **both** a large enough aggregate and enough blocks — see
[`MIN_BLOCKS_FOR_PARALLEL`](@ref) for why total work alone is not sufficient.
"""
function default_block_threaded(blocks)
    Threads.nthreads() > 1 || return false
    length(blocks) >= MIN_BLOCKS_FOR_PARALLEL || return false
    _is_cpu_storage(_array_wrapper_type(domain_array_type(first(blocks)))) || return false
    total = sum(b -> _total_elements(size(b, 2)), blocks; init = 0)
    return total >= THRESHOLD_BLOCK_PARALLEL
end

_total_elements(dims::Tuple{Vararg{Integer}}) = prod(dims; init = 1)
_total_elements(dims::Tuple) = sum(_total_elements, dims; init = 0)
_total_elements(n::Integer) = Int(n)

# ─── The `is_threaded` trait ──────────────────────────────────────────────────

"""
	is_threaded(L::AbstractOperator) -> Bool

Whether `L` executes its `mul!` (and adjoint `mul!`) using multiple Julia threads.

Defaults to `false`. Operators that can thread report the state they were constructed
with, so that `is_threaded(copy_operator(L; threaded = t)) == t`.
"""
is_threaded(::AbstractOperator) = false

"""
	_children(L::AbstractOperator) -> Tuple

The wrapped operators of a forwarding (calculus) operator, or `()` for a leaf.

Forwarders get `is_threaded` for free from this, which matters for more than tidiness:
`adapt_operator(op; threaded = false)` decides whether to copy by asking `is_threaded`, so
a forwarder that inherited the `false` default would report "constraint already satisfied"
and hand back an operator whose children are still threaded. That is a silent nesting bug,
not a missing optimization.
"""
_children(::AbstractOperator) = ()

# A forwarder is threaded if any child is: that is the property callers actually care
# about ("will running this spawn Julia threads?").
_is_threaded_from_children(L::AbstractOperator) = any(is_threaded, _children(L))

"""
	supports_threading(L::AbstractOperator) -> Bool

Whether `L` has a threaded execution path at all, i.e. whether `threaded` can change
anything about it.

This distinguishes the two ways `is_threaded(L) == false` can arise: "this operator can
thread and is currently not" versus "this operator has no threaded path". They need
different handling, because `threaded = true` is a **permission** ("thread where you can")
while `threaded = false` is a **demand** ("do not spawn threads", required for nesting
safety). Without the distinction, passing `threaded = true` down a `Compose` to a
memory-bound `Eye` would copy the `Eye` forever without ever satisfying the constraint.

Defaults to `false`; leaf operators that gained a `Th` parameter override it, forwarders
derive it from their children.
"""
supports_threading(::AbstractOperator) = false
_supports_threading_from_children(L::AbstractOperator) = any(supports_threading, _children(L))

# ─── FastBroadcast bridge ─────────────────────────────────────────────────────
#
# `DiagOp` and `Scale` keep FastBroadcast's singleton thread flag as their type parameter.
# These convert between that encoding and the plain `Bool` used everywhere else.

@inline _fbthread(::Val{true}) = FastBroadcast.True()
@inline _fbthread(::Val{false}) = FastBroadcast.False()
@inline _fbthread(b::Bool) = _fbthread(Val(b))

@inline _fbbool(::FastBroadcast.True) = true
@inline _fbbool(::FastBroadcast.False) = false
@inline _fbbool(::Type{FastBroadcast.True}) = true
@inline _fbbool(::Type{FastBroadcast.False}) = false

# ─── adapt_operator ───────────────────────────────────────────────────────────

"""
	adapt_operator(op; storage_type=nothing, threaded=nothing, require_thread_safe=false)

Return an operator equivalent to `op` that satisfies the requested constraints, **without
copying when `op` already satisfies them**.

This is the counterpart to [`copy_operator`](@ref), which always produces a new object:

- `adapt_operator` returns `op` itself (`===`) when every requested constraint already
  holds, and delegates to `copy_operator` otherwise.
- Use `adapt_operator` when you need *an* operator meeting a constraint (the common case
  when wrapping child operators); use `copy_operator` when you need a *distinct* object,
  e.g. one private copy per thread.

`require_thread_safe` demands `is_thread_safe(op)` before `op` may be returned as-is.

!!! warning
    `require_thread_safe` can only ever *withhold* the sharing fast path — it cannot
    manufacture thread safety. Thread safety is a property of an operator's type (does its
    `mul!` write into operator-owned buffers?), and copying an operator that keeps internal
    scratch buffers yields another operator that keeps internal scratch buffers. So when
    `is_thread_safe(op)` is `false`, this returns a **copy that is still not thread-safe**.

    Callers that need to run an operator concurrently must therefore branch on
    `is_thread_safe` themselves: share one instance when it is `true`, and allocate one
    private copy per thread when it is `false`. See `create_BatchOp` for that pattern.

```jldoctest
julia> op = MatrixOp(rand(4, 4));

julia> adapt_operator(op) === op   # already satisfies the (empty) constraints
true

julia> is_threaded(adapt_operator(FiniteDiff((4,)); threaded = false))
false
```
"""
function adapt_operator(
        op::AbstractOperator;
        storage_type = nothing,
        threaded = nothing,
        require_thread_safe::Bool = false,
    )
    if _satisfies_constraints(op, storage_type, threaded, require_thread_safe)
        return op
    end
    return copy_operator(op; storage_type, threaded)
end

function _satisfies_constraints(
        op::AbstractOperator, storage_type, threaded, require_thread_safe::Bool
    )
    require_thread_safe && !is_thread_safe(op) && return false
    # An operator with no threaded path satisfies any `threaded` request vacuously: there
    # is nothing a copy could change. See `supports_threading`.
    threaded !== nothing && supports_threading(op) && is_threaded(op) != threaded && return false
    storage_type !== nothing && !_storage_matches(op, storage_type) && return false
    return true
end

# An operator matches a requested storage type when both its domain and codomain already
# live in that array family. Compared on the *wrapper* (`Array`, `CuArray`, …) because the
# element type is fixed by the operator, not by the request.
function _storage_matches(op::AbstractOperator, storage_type::Type{<:AbstractArray})
    wrapper = _array_wrapper_type(storage_type)
    return _wrapper_of(domain_array_type(op)) === wrapper &&
        _wrapper_of(codomain_array_type(op)) === wrapper
end
_storage_matches(::AbstractOperator, ::Any) = false

_wrapper_of(::Type{A}) where {A <: AbstractArray} = _array_wrapper_type(A)
# Multi-domain operators report an ArrayPartition; its element storages are what matter,
# and they are not directly comparable to a single requested wrapper.
_wrapper_of(::Type{<:ArrayPartition}) = nothing
