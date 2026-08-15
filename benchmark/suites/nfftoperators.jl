# NFFTOperators benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/nfftoperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

if HAS_NFFT
    nfft["NFFTOp"] = BenchmarkGroup()
    nfft["NFFTOp"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = nfft_state())
    nfft["NFFTOp"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = nfft_state())
    if BENCH_THREADED
        nfft["NFFTOp"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = nfft_state(true))
        nfft["NFFTOp"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = nfft_state(true))
    end
end

run_suite_if_standalone(@__FILE__, "nfftoperators")
