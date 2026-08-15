# FFTWOperators benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/fftwoperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

function dft_state(threaded = false)
    rng = make_rng()
    # Explicit `threaded`: DFT defaults to `true` and BENCH_DFT_SHAPE (2^14 elements) clears
    # the 2^13 c2c threshold, so the default made this a threaded benchmark by accident.
    op = threaded ? check_threaded(DFT(BENCH_DFT_SHAPE; threaded = true)) : DFT(BENCH_DFT_SHAPE; threaded = false)
    x = randn(rng, BENCH_DFT_SHAPE...)
    y = zeros(ComplexF64, BENCH_DFT_SHAPE...)
    z = zeros(Float64, BENCH_DFT_SHAPE...)
    return (op = op, adj = op', x = x, y = y, z = z)
end

if HAS_FFTW
    fftw["DFT"] = BenchmarkGroup()
    fftw["DFT"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = dft_state())
    fftw["DFT"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = dft_state())
    if BENCH_THREADED
        fftw["DFT"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = dft_state(true))
        fftw["DFT"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = dft_state(true))
    end
end

run_suite_if_standalone(@__FILE__, "fftwoperators")
