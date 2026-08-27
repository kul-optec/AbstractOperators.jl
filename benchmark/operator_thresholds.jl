#!/usr/bin/env julia
#
# benchmark/operator_thresholds.jl
#
# Measures the threading crossover of each threading-capable operator **individually**, by
# timing its real `mul!` rather than a proxy kernel, so that `threading_threshold(::Type{Op})`
# can be a per-operator transcription instead of a shared cost-class guess.
#
# Why per-operator rather than per-cost-class: the class constants were a reasonable first
# approximation, but operators in the same class do not share a crossover once the real
# `mul!` is measured -- the surrounding indexing, buffer traffic and dispatch differ. The
# block-parallel case already demonstrated this concretely: VCAT-forward and HCAT-adjoint
# have identical inner work yet crossovers a full power of two apart, because HCAT's adjoint
# carries per-block ArrayPartition indexing.
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=benchmark -t 8 benchmark/operator_thresholds.jl
#   OPENBLAS_NUM_THREADS=1 julia --project=benchmark -t 8 benchmark/operator_thresholds.jl Sin Exp
#
# Output: a per-operator table on stdout and in `.temp/operator_thresholds.md`, ending in a
# block of ready-to-paste `threading_threshold` methods.
#
# Relation to `benchmark/threading_sweep.jl`: that script sweeps proxy kernels (plain
# broadcasts/loops shaped like the operators' inner work) to settle the cost-class defaults
# and kernel choice in `src/threading_policy.jl` (layer 1). This script instead times each
# operator's actual `mul!` end to end, to override those defaults with a per-operator
# `threading_threshold` (layer 2) where the class default is too coarse. See the header
# comment of `src/threading_policy.jl` for how the two layers combine.

using AbstractOperators
using BenchmarkTools
using LinearAlgebra
using Printf
using Random

BLAS.set_num_threads(1)

const SIZES = [2^k for k in 8:22]
const ELTYPES = [Float64, Float32]
const OUTDIR = joinpath(@__DIR__, "..", ".temp")

# Each entry builds (serial_op, threaded_op, x, y) for a given element type and size.
# `y` is preallocated so the measured call allocates nothing.
struct Case
    name::String
    build::Function      # (T, n) -> (serial, threaded, x, y)
    adjoint::Bool        # also measure the adjoint direction?
end

_elementwise(Op) = (T, n) -> begin
    x = randn(T, n)
    (Op(T, (n,); threaded = false), Op(T, (n,); threaded = true), x, similar(x))
end

# Builders are named functions rather than inline lambdas: a multi-line `->` inside a call
# argument list does not parse.
function _build_sigmoid(::Type{T}, n) where {T}
    x = randn(T, n)
    return (
        Sigmoid(T, (n,), one(T); threaded = false),
        Sigmoid(T, (n,), one(T); threaded = true), x, similar(x),
    )
end

function _build_pow(::Type{T}, n, p) where {T}
    x = abs.(randn(T, n)) .+ one(T)
    return (Pow(T, (n,), p; threaded = false), Pow(T, (n,), p; threaded = true), x, similar(x))
end

function _build_finitediff(::Type{T}, n) where {T}
    x = randn(T, n)
    return (
        FiniteDiff(T, (n,); threaded = false),
        FiniteDiff(T, (n,); threaded = true), x, zeros(T, n - 1),
    )
end

function _build_diagop(::Type{T}, n) where {T}
    d = randn(T, n)
    x = randn(T, n)
    return (DiagOp(d; threaded = false), DiagOp(d; threaded = true), x, similar(x))
end

# Variation needs >= 2 dims. The trailing dimension of 4 originally worked around a
# BoundsError in the adjoint for length-2 dimensions; that bug is fixed, but the shape is
# kept so the recorded `threading_threshold(::Type{<:Variation})` stays comparable to the
# run it was transcribed from.
function _build_variation(::Type{T}, n) where {T}
    m = max(4, n ÷ 4)
    x = randn(T, m, 4)
    return (
        Variation(T, (m, 4); threaded = false),
        Variation(T, (m, 4); threaded = true), x, zeros(T, m * 4, 2),
    )
end

# Scale wraps a deliberately *serial* DiagOp so the measurement isolates Scale's own
# broadcast rather than the child's threading.
function _build_scale(::Type{T}, n) where {T}
    d = randn(T, n)
    x = randn(T, n)
    return (
        Scale(T(2), DiagOp(d; threaded = false); threaded = false),
        Scale(T(2), DiagOp(d; threaded = false); threaded = true), x, similar(x),
    )
end

const CASES = Case[
    Case("Sin", _elementwise(Sin), false),
    Case("Cos", _elementwise(Cos), false),
    Case("Exp", _elementwise(Exp), false),
    Case("Atan", _elementwise(Atan), false),
    Case("Tanh", _elementwise(Tanh), false),
    Case("Sech", _elementwise(Sech), false),
    Case("SoftPlus", _elementwise(SoftPlus), false),
    Case("Sigmoid", _build_sigmoid, false),
    Case("Pow2", (T, n) -> _build_pow(T, n, 2), false),
    Case("PowHalf", (T, n) -> _build_pow(T, n, T(0.5)), false),
    Case("FiniteDiff", _build_finitediff, true),
    Case("DiagOp", _build_diagop, true),
    Case("Variation", _build_variation, true),
    Case("Scale", _build_scale, true),
]

"""
	crossover(rows) -> Union{Int,Nothing}

Smallest swept size at which the threaded operator beats the serial one **and keeps beating
it at every larger swept size**. Requiring the win to persist is what makes the result a
usable threshold rather than a lucky sample.
"""
function crossover(rows)
    sizes = sort(collect(keys(rows)))
    winning = nothing
    for n in reverse(sizes)
        if rows[n][2] < rows[n][1]
            winning = n
        else
            break
        end
    end
    return winning
end

function measure(case::Case, ::Type{T}, n::Int) where {T}
    Random.seed!(0)
    serial, threaded, x, y = case.build(T, n)
    mul!(y, serial, x)
    mul!(y, threaded, x)
    ts = 1.0e9 * @belapsed(mul!($y, $serial, $x), samples = 100, evals = 1)
    tt = 1.0e9 * @belapsed(mul!($y, $threaded, $x), samples = 100, evals = 1)
    if !case.adjoint
        return (ts, tt)
    end
    # Adjoint direction, measured on the same operators.
    xa = similar(x)
    ya = copy(y)
    as = serial'
    at = threaded'
    mul!(xa, as, ya)
    mul!(xa, at, ya)
    tsa = 1.0e9 * @belapsed(mul!($xa, $as, $ya), samples = 100, evals = 1)
    tta = 1.0e9 * @belapsed(mul!($xa, $at, $ya), samples = 100, evals = 1)
    # Report the *worse* of the two directions: a single threshold governs both, so it has
    # to be safe for whichever direction crosses over later.
    return (ts + tsa, tt + tta)
end

function sweep(case::Case; io::IO = stdout)
    println(io, "\n## $(case.name)\n")
    crossings = Int[]
    for T in ELTYPES
        rows = Dict{Int, Tuple{Float64, Float64}}()
        for n in SIZES
            rows[n] = measure(case, T, n)
        end
        println(io, "| n | serial (us) | threaded (us) | speedup |")
        println(io, "|---:|---:|---:|---:|")
        for n in sort(collect(keys(rows)))
            s, t = rows[n]
            println(io, @sprintf("| %d | %.2f | %.2f | %.2fx |", n, s / 1000, t / 1000, s / t))
        end
        c = crossover(rows)
        if c === nothing
            println(io, "\n- **$T: never wins** -> leave serial\n")
        else
            println(io, "\n- **$T crossover: n = $c** (2^$(round(Int, log2(c))))\n")
            push!(crossings, c)
        end
    end
    # Conservative across element types: a threshold must be safe for the narrower type too.
    return isempty(crossings) ? nothing : maximum(crossings)
end

function main(args)
    cases = isempty(args) ? CASES : filter(c -> c.name in args, CASES)
    isempty(cases) && error("no matching case; have $(join((c.name for c in CASES), ", "))")
    mkpath(OUTDIR)
    outfile = joinpath(OUTDIR, "operator_thresholds.md")
    results = Pair{String, Union{Int, Nothing}}[]
    open(outfile, "w") do io
        for sink in (stdout, io)
            println(sink, "# Per-operator threading thresholds")
            println(sink, "\n- Julia threads: $(Threads.nthreads())")
            println(sink, "- BLAS threads: $(BLAS.get_num_threads())")
            println(sink, "- CPU: $(Sys.cpu_info()[1].model) x $(length(Sys.cpu_info()))")
            println(sink, "- Date: $(strip(read(`date -I`, String)))")
        end
        for case in cases
            r = sweep(case; io = stdout)
            sweep(case; io = io)
            push!(results, case.name => r)
        end
        for sink in (stdout, io)
            println(sink, "\n## Suggested `threading_threshold` methods\n")
            println(sink, "```julia")
            for (name, r) in results
                if r === nothing
                    println(sink, "# $name: never wins -> leave unthreaded")
                else
                    println(sink, "threading_threshold(::Type{<:$name}) = 2^$(round(Int, log2(r)))")
                end
            end
            println(sink, "```")
        end
    end
    println("\nWrote $outfile")
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
