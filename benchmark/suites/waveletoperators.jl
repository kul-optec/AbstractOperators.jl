# WaveletOperators benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/waveletoperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

function wavelet_state()
    rng = make_rng()
    op = WaveletOp(wavelet(WT.db2), BENCH_WAVELET_N)
    x = randn(rng, BENCH_WAVELET_N)
    y = zeros(BENCH_WAVELET_N)
    z = zeros(BENCH_WAVELET_N)
    return (op = op, adj = op', x = x, y = y, z = z)
end

if HAS_WAVELET
    wavelets["WaveletOp"] = BenchmarkGroup()
    wavelets["WaveletOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = wavelet_state())
    wavelets["WaveletOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = wavelet_state())
end

run_suite_if_standalone(@__FILE__, "waveletoperators")
