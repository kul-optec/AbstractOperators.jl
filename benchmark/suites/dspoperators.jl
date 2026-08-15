# DSPOperators benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/dspoperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

function dsp_filt_state()
    rng = make_rng()
    op = Filt((BENCH_DSP_FILT_N, 1), randn(rng, 7))
    x = randn(rng, BENCH_DSP_FILT_N, 1)
    y = zeros(BENCH_DSP_FILT_N, 1)
    z = zeros(BENCH_DSP_FILT_N, 1)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function dsp_xcorr_state()
    rng = make_rng()
    h = randn(rng, 21)
    op = Xcorr(Float64, (BENCH_DSP_XCORR_N,), h)
    x = randn(rng, BENCH_DSP_XCORR_N)
    y = zeros(size(op, 1)...)
    z = zeros(BENCH_DSP_XCORR_N)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function dsp_mimofilt_state()
    rng = make_rng()
    taps = [randn(rng, 5) for _ in 1:4]
    op = MIMOFilt(BENCH_DSP_MIMO_SHAPE, taps)
    x = randn(rng, BENCH_DSP_MIMO_SHAPE...)
    y = zeros(size(op, 1)...)
    z = zeros(size(op, 2)...)
    return (op = op, adj = op', x = x, y = y, z = z)
end

if HAS_DSP
    dsp["Filt"] = BenchmarkGroup()
    dsp["Filt"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = dsp_filt_state())
    dsp["Filt"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = dsp_filt_state())

    dsp["Xcorr"] = BenchmarkGroup()
    dsp["Xcorr"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = dsp_xcorr_state())
    dsp["Xcorr"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = dsp_xcorr_state())

    dsp["MIMOFilt"] = BenchmarkGroup()
    dsp["MIMOFilt"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = dsp_mimofilt_state())
    dsp["MIMOFilt"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = dsp_mimofilt_state())
end

run_suite_if_standalone(@__FILE__, "dspoperators")
