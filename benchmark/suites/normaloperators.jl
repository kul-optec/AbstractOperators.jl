# Normal-operator (A'A) benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/normaloperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

normal["DiagOp"] = BenchmarkGroup()
normal["DiagOp"]["mul"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (rng = make_rng(); state = normal_state(DiagOp(randn(rng, BENCH_LINEAR_DIAG_N); threaded = false)))

if HAS_FFTW
    normal["DFT"] = BenchmarkGroup()
    normal["DFT"]["mul"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = normal_state(DFT(BENCH_DFT_SHAPE)))
end

if HAS_NFFT
    normal["NFFTOp"] = BenchmarkGroup()
    normal["NFFTOp"]["mul"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = nfft_normal_state())
end

run_suite_if_standalone(@__FILE__, "normaloperators")
