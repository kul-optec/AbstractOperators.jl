# Shared infrastructure and fixtures for the benchmark suites.
#
# `benchmarks.jl` is the entry point AirspeedVelocity loads: it includes this file, then every
# suite in `suites/`, then finalises the sampling parameters. Each suite can also be run on
# its own, which is the point of the split -- a change to one operator family does not require
# paying for the whole suite:
#
#     julia --project=benchmark benchmark/suites/linearoperators.jl
#     OPENBLAS_NUM_THREADS=1 julia --project=benchmark -t 8 benchmark/suites/calculus.jl
#
# What belongs here: the subpackage probes, the threading gate, the size constants, and the
# state builders used by more than one suite. A state builder used by exactly one suite lives
# in that suite's file.


using AbstractOperators
using BenchmarkTools
using LinearAlgebra
using Pkg
using Random
using RecursiveArrayTools: ArrayPartition
using Statistics: median

make_rng() = MersenneTwister(1234)

function _load_local_subpackage(pkg::Symbol, relpath::String)
    try
        @eval using $pkg
        return true
    catch err
        if !(err isa ArgumentError)
            rethrow(err)
        end
        local_path = joinpath(@__DIR__, "..", relpath)
        if !isdir(local_path)
            return false
        end
        try
            Pkg.develop(; path = local_path)
            @eval using $pkg
            return true
        catch
            return false
        end
    end
end

const HAS_DSP = _load_local_subpackage(:DSPOperators, "DSPOperators")
const HAS_FFTW = _load_local_subpackage(:FFTWOperators, "FFTWOperators")
const HAS_NFFT = _load_local_subpackage(:NFFTOperators, "NFFTOperators")
const HAS_WAVELET = _load_local_subpackage(:WaveletOperators, "WaveletOperators")

const SUITE = BenchmarkGroup()

"""
	BENCH_THREADED

Whether to register the multithreaded variants of the benchmarks.

Threaded timings are not comparable across CI runs. Hosted runners hand out 2-4 shared
vCPUs whose count and contention vary from job to job, so a threaded measurement tracks the
runner rather than the code, and AirspeedVelocity reports that noise as a regression or an
improvement. The multithreaded variants are therefore skipped whenever `CI` is set.

They are also skipped when Julia was started with a single thread, where every operator's
policy vetoes threading and a "threaded" entry would merely re-measure the serial path
under a misleading name.

Override with `ABSTRACTOPERATORS_BENCH_THREADED=true|false` -- set it to `true` on a
dedicated runner with a pinned core count, where threaded comparisons are meaningful.
"""
const BENCH_THREADED = let
    forced = get(ENV, "ABSTRACTOPERATORS_BENCH_THREADED", nothing)
    if forced !== nothing
        lowercase(strip(forced)) in ("1", "true", "yes")
    elseif haskey(ENV, "CI")
        false
    else
        Threads.nthreads() > 1
    end
end

"""
	check_threaded(op) -> op

Return `op`, first asserting that the threading policy actually granted it threading.

`threaded = true` is a *permission*, not a command: each operator's policy may still veto
it because the workload is below that operator's measured crossover, because there are
fewer blocks than `MIN_BLOCKS_FOR_PARALLEL`, or because the storage is not CPU-backed. A
benchmark that asks for threading and silently gets the serial path is worse than no
benchmark at all -- it reports a "threaded" number that never exercised a thread, and would
keep reporting it after a threading regression. Every threaded entry below routes its
operator through here so that mis-sizing fails loudly instead.
"""
function check_threaded(op)
    if !is_threaded(op)
        error(
            "benchmark asked for a threaded $(typeof(op).name.name) but the policy vetoed " *
                "it; the workload is probably below this operator's threshold " *
                "(see threading_threshold / block_threading_threshold) or has too few blocks"
        )
    end
    return op
end

"""
	check_block_threaded(op) -> op

`check_threaded` for HCAT/VCAT/DCAT, asserting that the *block loop* threads.

`is_threaded` is the wrong question for these: it reports `Th || children`, so it is already
true when the blocks happen to thread internally even though the block loop runs serially.
Only `is_block_threaded` distinguishes the two, and the block loop is what these benchmarks
exist to measure -- the blocks themselves are constructed serial so the number isolates it.
"""
function check_block_threaded(op)
    # Not exported: `is_block_threaded` is an internal trait, so it needs qualifying.
    if !AbstractOperators.is_block_threaded(op)
        error(
            "benchmark asked for a block-threaded $(typeof(op).name.name) but the policy " *
                "vetoed it; needs at least MIN_BLOCKS_FOR_PARALLEL blocks and a mean block " *
                "size of at least block_threading_threshold"
        )
    end
    return op
end

const BENCH_LINEAR_EYE_N = 1_048_576
const BENCH_LINEAR_DIAG_N = 524_288
const BENCH_LINEAR_MATRIX_SHAPE = (192, 192)
const BENCH_LINEAR_MATRIX_DOMAIN = 192
const BENCH_LINEAR_FD_N = 262_144
const BENCH_LINEAR_GETINDEX_DIM = (1536, 1024)
const BENCH_LINEAR_VARIATION_DIM = (512, 256)
const BENCH_LINEAR_ZEROPAD_DIM = (512, 256)
const BENCH_LINEAR_ZEROS_N = 2_000_000
const BENCH_LINEAR_LMATRIX_N = 1024
const BENCH_LINEAR_MYLIN_N = 524_288
const BENCH_LINEAR_LBFGS_N = 8192

const BENCH_NONLIN_POW_N = 131_072
const BENCH_NONLIN_EXP_N = 131_072
const BENCH_NONLIN_SIN_N = 32_768
const BENCH_NONLIN_COS_N = 32_768
const BENCH_NONLIN_ATAN_N = 32_768
const BENCH_NONLIN_TANH_N = 32_768
const BENCH_NONLIN_SECH_N = 32_768
const BENCH_NONLIN_SIGMOID_N = 65_536
const BENCH_NONLIN_SOFTMAX_N = 65_536
const BENCH_NONLIN_SOFTPLUS_N = 65_536

const BENCH_CALC_N = 32_768
const BENCH_CALC_2D = (256, 128)
const BENCH_CALC_SQ = 64

# Sizes used only by the threaded variants. The serial benchmarks are deliberately sized for
# a short run, which for several operators is *below* the measured crossover -- reusing those
# sizes would trip `check_threaded`. Each constant is set from the operator's own policy:
#
#   Scale            threading_threshold       2^22   (its broadcast is memory-bound)
#   VCAT / DCAT      block_threading_threshold 2^16   per block, and >= 4 blocks
#   HCAT             block_threading_threshold 2^17   per block, and >= 4 blocks
#
# `MIN_BLOCKS_FOR_PARALLEL` is 4, so the two-block *CAT states used by the serial benchmarks
# cannot be threaded at any size.
const BENCH_CALC_SCALE_THREADED_N = 4_194_304
const BENCH_CAT_THREADED_BLOCKS = 4
const BENCH_CAT_THREADED_N = 65_536
const BENCH_HCAT_THREADED_N = 131_072

const BENCH_DSP_FILT_N = 65_536
const BENCH_DSP_XCORR_N = 32_768
const BENCH_DSP_MIMO_SHAPE = (16_384, 2)
const BENCH_DFT_SHAPE = (128, 128)
const BENCH_WAVELET_N = 131_072
const BENCH_NFFT_IMAGE = (48, 48)
const BENCH_NFFT_NSAMP = 48
const BENCH_NFFT_NPROF = 24

# --- state builders shared by more than one suite ---------------------------------

function linear_state(op)
    rng = make_rng()
    x = randn(rng, domain_type(op), size(op, 2)...)
    y = zeros(codomain_type(op), size(op, 1)...)
    z = zeros(domain_type(op), size(op, 2)...)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function nonlinear_state(op; positive = false)
    rng = make_rng()
    x = if positive
        abs.(randn(rng, domain_type(op), size(op, 2)...))
    else
        randn(rng, domain_type(op), size(op, 2)...)
    end
    y = zeros(codomain_type(op), size(op, 1)...)
    return (op = op, x = x, y = y)
end

function jacobian_state(op; positive = false)
    rng = make_rng()
    x = if positive
        abs.(randn(rng, domain_type(op), size(op, 2)...))
    else
        randn(rng, domain_type(op), size(op, 2)...)
    end
    jac = Jacobian(op, x)
    b = randn(rng, codomain_type(jac), size(jac, 1)...)
    y = zeros(domain_type(jac), size(jac, 2)...)
    return (jac = jac, adj = jac', b = b, y = y)
end

function nfft_state(threaded = false)
    rng = make_rng()
    traj = rand(rng, 2, BENCH_NFFT_NSAMP, BENCH_NFFT_NPROF) .- 0.5
    dcf = ones(eltype(traj), BENCH_NFFT_NSAMP, BENCH_NFFT_NPROF)
    op = threaded ? check_threaded(NFFTOp(BENCH_NFFT_IMAGE, traj, dcf; threaded = true)) :
        NFFTOp(BENCH_NFFT_IMAGE, traj, dcf; threaded = false)
    x = randn(rng, ComplexF64, BENCH_NFFT_IMAGE...)
    y = zeros(ComplexF64, BENCH_NFFT_NSAMP, BENCH_NFFT_NPROF)
    z = zeros(ComplexF64, BENCH_NFFT_IMAGE...)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function normal_state(op)
    rng = make_rng()
    nop = AbstractOperators.get_normal_op(op)
    dT = domain_type(nop)
    cT = codomain_type(nop)
    dEl = dT <: AbstractArray ? eltype(dT) : dT
    cEl = cT <: AbstractArray ? eltype(cT) : cT
    x = randn(rng, dEl, size(nop, 2)...)
    y = zeros(cEl, size(nop, 1)...)
    return (op = nop, x = x, y = y)
end

nfft_normal_state() = normal_state(nfft_state().op)

# --- suite groups -----------------------------------------------------------------
#
# Created here rather than in the individual suite files so that every suite file has
# its group variable whether it is run standalone or included by `benchmarks.jl`, and
# so the SUITE key layout stays identical either way.

linear = SUITE["linearoperators"] = BenchmarkGroup()
calculus = SUITE["calculus"] = BenchmarkGroup()
nonlinear = SUITE["nonlinearoperators"] = BenchmarkGroup()
batching = SUITE["batching"] = BenchmarkGroup()
dsp = HAS_DSP ? (SUITE["dspoperators"] = BenchmarkGroup()) : nothing
fftw = HAS_FFTW ? (SUITE["fftwoperators"] = BenchmarkGroup()) : nothing
nfft = HAS_NFFT ? (SUITE["nfftoperators"] = BenchmarkGroup()) : nothing
wavelets = HAS_WAVELET ? (SUITE["waveletoperators"] = BenchmarkGroup()) : nothing
normal = SUITE["normaloperators"] = BenchmarkGroup()


"""
	BENCH_THREADED_SUFFIXES

Trailing name component that marks a leaf as exercising a threaded path. Any threaded entry
added later must end in one of these, or it will be sampled with the serial parameters.
"""
const BENCH_THREADED_SUFFIXES = ("threaded", "copying", "locking", "fixed_operator")

"""
	BENCH_THREADED_EVALS

Evaluations per sample for the threaded entries.

`setup` builds a fresh domain/codomain pair for every sample, so at `evals = 1` each timed
call pays first-touch page faults on several MB of newly allocated memory. That cost is
serial-ish in the kernel and swamps the parallel win: DiagOp's threaded forward measured a
median 4.15x its own minimum and came out *slower* than the serial path it beats by 8x once
the buffers are warm. Amortising over 50 evaluations reuses the buffers within a sample and
brings the spread down to 1.5x, with the minimum matching a standalone hot-loop measurement.

Applied only to the threaded entries: the serial ones keep `evals = 1` so their recorded
baselines stay comparable across this change.
"""
const BENCH_THREADED_EVALS = 50

"""
	finalize_suite!(suite)

Apply the sampling parameters. Called once by `benchmarks.jl`, and by
`run_suite_if_standalone` when a suite file is executed directly.
"""
function finalize_suite!(suite)
    # Cap run time so CI / ASV comparisons complete in reasonable time
    for (_, b) in BenchmarkTools.leaves(suite)
        b.params.seconds = 5
        b.params.samples = 10000
        b.params.evals = 1
    end
    for (k, b) in BenchmarkTools.leaves(suite)
        if any(endswith(last(k), sfx) for sfx in BENCH_THREADED_SUFFIXES)
            b.params.evals = BENCH_THREADED_EVALS
        end
    end
    return suite
end

"""
	run_suite_if_standalone(file, group)

Run and print just `SUITE[group]` when `file` was executed directly, and do nothing when it
was included by `benchmarks.jl`. Returns the trial group, or `nothing` when the suite is
unavailable because its subpackage could not be loaded.
"""
function run_suite_if_standalone(file, group)
    abspath(PROGRAM_FILE) == abspath(file) || return nothing
    if !haskey(SUITE, group)
        println("suite \"$group\" is unavailable (its subpackage did not load)")
        return nothing
    end
    finalize_suite!(SUITE)
    println("Running suite \"$group\"  (threads = $(Threads.nthreads()), BENCH_THREADED = $BENCH_THREADED)")
    results = BenchmarkTools.run(SUITE[group]; verbose = true)
    println()
    show(stdout, MIME"text/plain"(), median(results))
    println()
    return results
end

const BENCH_COMMON_LOADED = true
