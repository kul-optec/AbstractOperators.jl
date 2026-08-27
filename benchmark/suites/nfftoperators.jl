# NFFTOperators benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/nfftoperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

if HAS_NFFT
    nfft["NFFTOp"] = BenchmarkGroup()
    # `threaded` left at its default: the per-thread-count run already covers serial and
    # threaded (see `BENCH_THREADED`), so no -single/-threaded split is needed here.
    nfft["NFFTOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = nfft_state())
    nfft["NFFTOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = nfft_state())
end

run_suite_if_standalone(@__FILE__, "nfftoperators")
