# FFTWOperators benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/fftwoperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

function dft_state()
    rng = make_rng()
    op = DFT(BENCH_DFT_SHAPE)
    x = randn(rng, BENCH_DFT_SHAPE...)
    y = zeros(ComplexF64, BENCH_DFT_SHAPE...)
    z = zeros(Float64, BENCH_DFT_SHAPE...)
    return (op = op, adj = op', x = x, y = y, z = z)
end

if HAS_FFTW
    fftw["DFT"] = BenchmarkGroup()
    # `threaded` left at its default: the per-thread-count run already covers serial and
    # threaded (see `BENCH_THREADED`), so no -single/-threaded split is needed here.
    fftw["DFT"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = dft_state())
    fftw["DFT"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = dft_state())
end

run_suite_if_standalone(@__FILE__, "fftwoperators")
