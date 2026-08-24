#!/usr/bin/env julia
# benchmark/compare.jl
#
# Repo-local benchmark comparison driver for CI.
# Runs benchmark/benchmarks.jl at two explicit source trees, serialises results
# with BenchmarkTools, then renders a markdown PR-comment body to body.md.
#
# Usage (called from .github/workflows/benchmark.yml):
#   julia --project=benchmark benchmark/compare.jl \
#       --base-dir  <path-to-base-checkout>  \
#       --head-dir  <path-to-head-checkout>  \
#       --output-dir <artifact-dir>           \
#       --pr        <PR-number>               \
#       --julia-version <julia-version-string>
#
# The script writes three files to --output-dir:
#   body.md          – markdown comment body ready for posting
#   pr_number.txt    – PR number (for the post-comment workflow)
#   julia_version.txt – Julia version string (for the post-comment workflow)

using ArgParse
using BenchmarkTools
using Pkg
using Printf
using Serialization
using Statistics

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function parse_args_local(args)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--base-dir"
        help = "Absolute path to base revision checkout"
        arg_type = String
        required = true
        "--head-dir"
        help = "Absolute path to head revision checkout"
        arg_type = String
        required = true
        "--output-dir"
        help = "Directory where body.md and metadata are written"
        arg_type = String
        default = "."
        "--pr"
        help = "Pull-request number"
        arg_type = String
        default = ""
        "--julia-version"
        help = "Julia version string for the comment header"
        arg_type = String
        default = string(VERSION)
    end
    return parse_args(args, s)
end

# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------

"""
Run benchmark/benchmarks.jl inside `repo_dir` in a clean subprocess and return
the serialised BenchmarkGroup results path.  The subprocess activates the
benchmark sub-project so all workspace/subproject resolution happens via the
standard local manifest.
"""
function run_suite(repo_dir::AbstractString, result_path::AbstractString)
    bench_project = joinpath(repo_dir, "benchmark")
    bench_script = joinpath(repo_dir, "benchmark", "benchmarks.jl")

    # The runner script activates the benchmark project, includes benchmarks.jl,
    # tunes and runs SUITE, then serialises results.
    runner = """
    using Pkg
    Pkg.activate($(repr(bench_project)); io = devnull)
    Pkg.instantiate(; io = devnull)
    include($(repr(bench_script)))
    results = BenchmarkTools.run(SUITE; verbose = true)
    using Serialization
    Serialization.serialize($(repr(result_path)), results)
    """

    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(bench_project) -e $(runner)`
    @info "Running benchmarks at $repo_dir"
    run(cmd)
    return result_path
end

# ---------------------------------------------------------------------------
# Comparison helpers  (logic mirrors AirspeedVelocity TableUtils / PR #140)
# ---------------------------------------------------------------------------

"""
Flatten a possibly nested BenchmarkGroup into a Dict{String,BenchmarkTools.Trial}
using "/" as the key separator.
"""
function flatten_group(group::BenchmarkGroup, prefix = "")
    out = Dict{String, BenchmarkTools.Trial}()
    for (k, v) in group
        key = isempty(prefix) ? string(k) : "$prefix/$k"
        if v isa BenchmarkGroup
            merge!(out, flatten_group(v, key))
        elseif v isa BenchmarkTools.Trial
            out[key] = v
        end
    end
    return out
end

# ---- time formatting -------------------------------------------------------

const TIME_UNITS = [(:ns, 1.0e0), (Symbol("μs"), 1.0e3), (:ms, 1.0e6), (:s, 1.0e9)]

function auto_time_unit(t_ns::Float64)
    for (unit, scale) in reverse(TIME_UNITS)
        t_ns / scale >= 1.0 && return (unit, scale)
    end
    return TIME_UNITS[1]
end

function format_time(t::BenchmarkTools.Trial)
    med = Statistics.median(t.times)   # nanoseconds
    lo = Statistics.quantile(t.times, 0.25)
    hi = Statistics.quantile(t.times, 0.75)
    unit, scale = auto_time_unit(med)
    v = med / scale
    verr = max(0.0, (hi - lo) / 2) / scale
    unit_str = string(unit)
    if isfinite(verr) && verr > 0
        return @sprintf("%.3g ± %.2g %s", v, verr, unit_str)
    else
        return @sprintf("%.3g %s", v, unit_str)
    end
end

# ---- memory formatting -----------------------------------------------------

function format_memory(t::BenchmarkTools.Trial)
    allocs = round(Int, Statistics.median(t.allocs))
    bytes = round(Int, Statistics.median(t.memory))
    if bytes == 0
        return "0 allocs (0 bytes)"
    elseif bytes < 1024
        return "$allocs allocs ($bytes bytes)"
    elseif bytes < 1024^2
        return @sprintf("%d allocs (%.2f KiB)", allocs, bytes / 1024)
    else
        return @sprintf("%d allocs (%.2f MiB)", allocs, bytes / 1024^2)
    end
end

# ---- ratio + emoji ---------------------------------------------------------

"""
Compute ratio base/head for the given mode (:time or :memory).
Returns `(ratio, ratio_err)` where `ratio_err` is finite only for time mode.
Semantics: ratio > 1 means head is faster (speedup), ratio < 1 means slowdown.
"""
function compute_ratio(base::BenchmarkTools.Trial, head::BenchmarkTools.Trial, mode::Symbol)
    if mode === :time
        base_med = Statistics.median(base.times)
        head_med = Statistics.median(head.times)
        ratio = base_med / head_med
        # interquartile-range based error propagation
        base_err = max(0.0, Statistics.quantile(base.times, 0.75) - Statistics.quantile(base.times, 0.25))
        head_err = max(0.0, Statistics.quantile(head.times, 0.75) - Statistics.quantile(head.times, 0.25))
        ratio_err = abs(ratio) * sqrt((base_err / base_med)^2 + (head_err / head_med)^2)
        return ratio, ratio_err
    else  # :memory
        base_mem = Statistics.median(base.memory)
        head_mem = Statistics.median(head.memory)
        if head_mem == 0
            return (base_mem == 0 ? 1.0 : Inf), NaN
        end
        return base_mem / head_mem, NaN
    end
end

"""
Emoji indicator for the ratio column (mirrors AirspeedVelocity PR #140).
ratio > 1  ⇒  head is faster than base
ratio < 1  ⇒  head is slower than base
"""
function ratio_emoji(ratio::Float64, ratio_err::Float64, mode::Symbol)
    if mode === :time && isfinite(ratio_err)
        ratio + ratio_err < 0.8  && return " 🐢"   # significant slowdown
        ratio - ratio_err > 1.2  && return " 🚀"   # significant speedup
    else
        ratio < 0.5              && return " 🐢"   # head uses >2× more memory
        ratio > 1.5              && return " 🚀"   # head uses >1.5× less memory
    end
    return ""
end

function format_ratio(ratio::Float64, ratio_err::Float64, mode::Symbol)
    emoji = ratio_emoji(ratio, ratio_err, mode)
    if !isfinite(ratio)
        return "N/A$emoji"
    end
    if isfinite(ratio_err) && ratio_err > 0
        return @sprintf("%.3g ± %.2g%s", ratio, ratio_err, emoji)
    else
        return @sprintf("%.3g%s", ratio, emoji)
    end
end

# ---- markdown table --------------------------------------------------------

function markdown_table(; header::AbstractVector, data::AbstractMatrix)
    @assert size(data, 2) == length(header)
    cw = [max(length(h), 4) for h in header]
    for row in eachrow(data)
        for (i, v) in enumerate(row)
            cw[i] = max(cw[i], length(string(v)))
        end
    end
    io = IOBuffer()
    print(io, "|")
    for (i, h) in enumerate(header)
        print(io, " $(rpad(h, cw[i])) |")
    end
    println(io)
    print(io, "|:$(repeat('-', cw[1]))|")
    for i in 2:length(header)
        print(io, ":$(repeat('-', cw[i] - 1)):|")
    end
    println(io)
    for row in eachrow(data)
        print(io, "|")
        for (i, v) in enumerate(row)
            s = string(v)
            print(io, " $(rpad(s, cw[i])) |")
        end
        println(io)
    end
    return String(take!(io))
end

"""
Build one markdown table (either :time or :memory) comparing `base_flat` and
`head_flat`.  Keys present in only one revision are still shown with "—" in
the missing column.
"""
function build_table(
        base_flat::Dict{String, BenchmarkTools.Trial},
        head_flat::Dict{String, BenchmarkTools.Trial},
        mode::Symbol,
        base_label::String,
        head_label::String,
    )
    all_keys = sort(collect(union(keys(base_flat), keys(head_flat))))
    header = ["Benchmark", base_label, head_label, "Ratio (base/head)"]

    rows = Vector{String}[]
    for k in all_keys
        has_base = haskey(base_flat, k)
        has_head = haskey(head_flat, k)
        b_str = has_base ? (mode === :time ? format_time(base_flat[k]) : format_memory(base_flat[k])) : "—"
        h_str = has_head ? (mode === :time ? format_time(head_flat[k]) : format_memory(head_flat[k])) : "—"
        ratio_str = if has_base && has_head
            ratio, ratio_err = compute_ratio(base_flat[k], head_flat[k], mode)
            format_ratio(ratio, ratio_err, mode)
        else
            "—"
        end
        push!(rows, [k, b_str, h_str, ratio_str])
    end

    data = isempty(rows) ? reshape(String[], 0, 4) : reduce(vcat, permutedims.(rows))
    return markdown_table(; header, data)
end

# ---------------------------------------------------------------------------
# Summary line
# ---------------------------------------------------------------------------

function build_summary(
        base_flat::Dict{String, BenchmarkTools.Trial},
        head_flat::Dict{String, BenchmarkTools.Trial},
    )
    common = intersect(keys(base_flat), keys(head_flat))

    time_speedups = 0
    time_regressions = 0
    mem_improvements = 0
    mem_regressions = 0

    for k in common
        r_t, re_t = compute_ratio(base_flat[k], head_flat[k], :time)
        if isfinite(r_t)
            r_t - re_t > 1.2 && (time_speedups += 1)
            r_t + re_t < 0.8 && (time_regressions += 1)
        end

        r_m, _ = compute_ratio(base_flat[k], head_flat[k], :memory)
        if isfinite(r_m)
            r_m > 1.5 && (mem_improvements += 1)
            r_m < 0.5 && (mem_regressions += 1)
        end
    end

    parts = String[]

    if time_speedups > 0
        n = time_speedups
        push!(parts, "🚀 $n $(n == 1 ? "benchmark" : "benchmarks") improved in time")
    end
    if time_regressions > 0
        n = time_regressions
        push!(parts, "🐢 $n $(n == 1 ? "time regression" : "time regressions") detected")
    end
    if mem_improvements > 0
        n = mem_improvements
        push!(parts, "🚀 $n $(n == 1 ? "benchmark" : "benchmarks") use less memory")
    end
    if mem_regressions > 0
        n = mem_regressions
        push!(parts, "🐢 $n $(n == 1 ? "memory regression" : "memory regressions") detected")
    end

    if isempty(parts)
        return "No significant performance or memory regressions detected."
    end
    return join(parts, " · ")
end

# ---------------------------------------------------------------------------
# Comment body assembly
# ---------------------------------------------------------------------------

function build_body(
        base_flat::Dict{String, BenchmarkTools.Trial},
        head_flat::Dict{String, BenchmarkTools.Trial},
        base_label::String,
        head_label::String,
        julia_version::String,
    )
    time_table = build_table(base_flat, head_flat, :time, base_label, head_label)
    memory_table = build_table(base_flat, head_flat, :memory, base_label, head_label)
    summary = build_summary(base_flat, head_flat)

    return """
    ## Benchmark Results (Julia v$(julia_version))

    $(summary)

    <details>
    <summary>Time benchmarks</summary>

    $(time_table)
    </details>

    <details>
    <summary>Memory benchmarks</summary>

    $(memory_table)
    </details>

    > **Ratio interpretation:** values > 1 mean the PR is faster; values < 1 mean slower.
    > 🚀 significant speedup · 🐢 significant slowdown
    """
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(argv = ARGS)
    opts = parse_args_local(argv)
    base_dir = opts["base-dir"]
    head_dir = opts["head-dir"]
    output_dir = opts["output-dir"]
    pr_number = opts["pr"]
    julia_version = opts["julia-version"]

    mkpath(output_dir)

    base_result_path = joinpath(output_dir, "results_base.jls")
    head_result_path = joinpath(output_dir, "results_head.jls")

    run_suite(base_dir, base_result_path)
    run_suite(head_dir, head_result_path)

    @info "Loading results …"
    base_results = Serialization.deserialize(base_result_path)
    head_results = Serialization.deserialize(head_result_path)

    base_flat = flatten_group(base_results)
    head_flat = flatten_group(head_results)

    # Use short SHA labels when directories carry them; otherwise use directory names.
    base_label = basename(rstrip(base_dir, '/'))
    head_label = basename(rstrip(head_dir, '/'))

    body = build_body(base_flat, head_flat, base_label, head_label, julia_version)

    write(joinpath(output_dir, "body.md"), body)
    write(joinpath(output_dir, "pr_number.txt"), pr_number)
    write(joinpath(output_dir, "julia_version.txt"), julia_version)

    @info "Benchmark comparison written to $(output_dir)/body.md"
    return
end

main()
