#!/usr/bin/env julia
#
# benchmark/threading_sweep.jl
#
# Measures where multithreading starts to pay for AbstractOperators' kernel shapes, so that
# every constant in `src/threading_policy.jl` is a transcription of a measurement rather
# than a guess.
#
# For each kernel it sweeps array size x element type x kernel variant, and reports the
# *crossover*: the smallest swept size at which a threaded variant beats the serial
# baseline, and stays ahead for every larger size. A kernel whose threaded variants never
# win has no crossover, and the honest outcome is to leave that operator unthreaded.
#
# The variants mirror the choices available inside the operators:
#   :serial    - plain broadcast / plain loop
#   :fastbcast - FastBroadcast `@..` with thread = true
#   :batch     - Polyester `@batch` (NestedThreading-guarded)
#   :threads   - NestedThreading `@budgeted_threads`
#
# Usage:
#   OPENBLAS_NUM_THREADS=1 julia --project=benchmark -t 8 benchmark/threading_sweep.jl
#   OPENBLAS_NUM_THREADS=1 julia --project=benchmark -t 8 benchmark/threading_sweep.jl transcendental
#
# Output: a markdown table per kernel on stdout, plus `.temp/threading_sweep.md`.
# BLAS is pinned to one thread so that Julia-level threading is what is being measured.
#
# Relation to `benchmark/operator_thresholds.jl`: this script settles the cost-class
# defaults and kernel choice (layer 1 in `src/threading_policy.jl`) from proxy kernels;
# `operator_thresholds.jl` instead times each operator's real `mul!` to derive a
# per-operator override (layer 2) where a class default is too coarse for that operator.

using BenchmarkTools
using LinearAlgebra
using Printf
using Random

using FastBroadcast: @..
using NestedThreading: @budgeted_threads
using Polyester: @batch

BLAS.set_num_threads(1)

const SIZES = [2^k for k in 8:22]
const ELTYPES = [Float64, Float32]
const OUTDIR = joinpath(@__DIR__, "..", ".temp")

# ─── Kernels ──────────────────────────────────────────────────────────────────
#
# Each kernel is (name, threshold-constant-it-informs, f!(variant, y, x)). They are written
# to match the shape of the real `mul!` bodies, not to be micro-optimal in isolation.

function transcendental!(::Val{:serial}, y, x)
    @. y = sin(x)
    return y
end
function transcendental!(::Val{:fastbcast}, y, x)
    @.. thread = true y = sin(x)
    return y
end
function transcendental!(::Val{:batch}, y, x)
    @batch for i in eachindex(y, x)
        @inbounds y[i] = sin(x[i])
    end
    return y
end
function transcendental!(::Val{:threads}, y, x)
    @budgeted_threads for i in eachindex(y, x)
        @inbounds y[i] = sin(x[i])
    end
    return y
end

function arithmetic!(::Val{:serial}, y, x)
    @. y = 2 * x + 1
    return y
end
function arithmetic!(::Val{:fastbcast}, y, x)
    @.. thread = true y = 2 * x + 1
    return y
end
function arithmetic!(::Val{:batch}, y, x)
    @batch for i in eachindex(y, x)
        @inbounds y[i] = 2 * x[i] + 1
    end
    return y
end
function arithmetic!(::Val{:threads}, y, x)
    @budgeted_threads for i in eachindex(y, x)
        @inbounds y[i] = 2 * x[i] + 1
    end
    return y
end

function memorybound!(::Val{:serial}, y, x)
    @. y = x
    return y
end
function memorybound!(::Val{:fastbcast}, y, x)
    @.. thread = true y = x
    return y
end
function memorybound!(::Val{:batch}, y, x)
    @batch for i in eachindex(y, x)
        @inbounds y[i] = x[i]
    end
    return y
end
function memorybound!(::Val{:threads}, y, x)
    @budgeted_threads for i in eachindex(y, x)
        @inbounds y[i] = x[i]
    end
    return y
end

# Forward finite difference: the FiniteDiff / Variation inner-loop shape.
function finitediff!(::Val{:serial}, y, x)
    @views @. y = x[2:end] - x[1:(end - 1)]
    return y
end
function finitediff!(::Val{:fastbcast}, y, x)
    @views @.. thread = true y = x[2:end] - x[1:(end - 1)]
    return y
end
function finitediff!(::Val{:batch}, y, x)
    @batch for i in eachindex(y)
        @inbounds y[i] = x[i + 1] - x[i]
    end
    return y
end
function finitediff!(::Val{:threads}, y, x)
    @budgeted_threads for i in eachindex(y)
        @inbounds y[i] = x[i + 1] - x[i]
    end
    return y
end

const KERNELS = Dict(
    "transcendental" => (fn = transcendental!, constant = "THRESHOLD_ELEMENTWISE_TRANSCENDENTAL", shrink_out = 0),
    "arithmetic" => (fn = arithmetic!, constant = "THRESHOLD_ELEMENTWISE_ARITHMETIC", shrink_out = 0),
    "memorybound" => (fn = memorybound!, constant = "THRESHOLD_MEMORY_BOUND", shrink_out = 0),
    "finitediff" => (fn = finitediff!, constant = "THRESHOLD_ELEMENTWISE_ARITHMETIC", shrink_out = 1),
)

const VARIANTS = [:serial, :fastbcast, :batch, :threads]

# ─── Sweep ────────────────────────────────────────────────────────────────────

"""
	measure(kernel, T, n) -> Dict{Symbol,Float64}

Median time in nanoseconds for each variant, at element type `T` and size `n`.
Deterministic input (`Random.seed!(0)`), allocation-free bodies (preallocated `y`).
"""
function measure(kernel, ::Type{T}, n::Int) where {T}
    Random.seed!(0)
    x = randn(T, n)
    y = zeros(T, n - kernel.shrink_out)
    out = Dict{Symbol, Float64}()
    for v in VARIANTS
        val = Val(v)
        f = kernel.fn
        f(val, y, x)  # warm up / compile
        b = @benchmark $f($val, $y, $x) samples = 200 evals = 1
        out[v] = median(b).time
    end
    return out
end

"""
	crossover(rows, variant) -> Union{Int,Nothing}

The smallest swept size at which `variant` beats `:serial` **and keeps beating it at every
larger swept size**. Requiring the win to persist is what makes this a usable threshold: a
single lucky size in the middle of the sweep is noise, not a crossover.
"""
function crossover(rows, variant)
    sizes = sort(collect(keys(rows)))
    winning = nothing
    for n in reverse(sizes)
        if rows[n][variant] < rows[n][:serial]
            winning = n
        else
            break   # walking down from the top, the first loss ends the winning suffix
        end
    end
    return winning
end

function sweep(name::String; io::IO = stdout)
    kernel = KERNELS[name]
    println(io, "\n## $name  (informs `$(kernel.constant)`)\n")
    for T in ELTYPES
        rows = Dict{Int, Dict{Symbol, Float64}}()
        for n in SIZES
            rows[n] = measure(kernel, T, n)
        end

        println(io, "### $T, $(Threads.nthreads()) threads\n")
        println(io, "| n | " * join(string.(VARIANTS), " | ") * " | best |")
        println(io, "|---:|" * repeat("---:|", length(VARIANTS) + 1))
        for n in sort(collect(keys(rows)))
            r = rows[n]
            best = argmin(v -> r[v], VARIANTS)
            cells = join((@sprintf("%.1f", r[v] / 1000) for v in VARIANTS), " | ")
            println(io, "| $n | $cells | $best |")
        end
        println(io, "\n(times in microseconds, median)\n")

        for v in VARIANTS
            v === :serial && continue
            c = crossover(rows, v)
            if c === nothing
                println(io, "- `$v`: **never wins** at any swept size -> leave serial")
            else
                println(io, "- `$v`: crossover at **n = $c** ($(round(Int, log2(c))) = log2)")
            end
        end
        println(io)
    end
    return
end

function main(args)
    names = isempty(args) ? sort(collect(keys(KERNELS))) : args
    mkpath(OUTDIR)
    outfile = joinpath(OUTDIR, "threading_sweep.md")
    open(outfile, "w") do io
        for sink in (stdout, io)
            println(sink, "# Threading sweep")
            println(sink, "\n- Julia threads: $(Threads.nthreads())")
            println(sink, "- BLAS threads: $(BLAS.get_num_threads())")
            println(sink, "- Host: $(gethostname())")
            println(sink, "- CPU: $(Sys.cpu_info()[1].model) x $(length(Sys.cpu_info()))")
            println(sink, "- Date: $(read(`date -I`, String) |> strip)")
        end
        for name in names
            haskey(KERNELS, name) || error("unknown kernel $name; have $(keys(KERNELS))")
            sweep(name; io = stdout)
            sweep(name; io = io)
        end
    end
    println("\nWrote $outfile")
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
