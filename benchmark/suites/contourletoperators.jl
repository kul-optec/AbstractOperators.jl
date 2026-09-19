# ContourletOperators benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/contourletoperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

function _contourlet_params()
    return ContourletParams(J = BENCH_CONTOURLET_J, L_array = parabolic_levels(BENCH_CONTOURLET_J))
end

function contourlet_state()
    rng = make_rng()
    op = ContourletOp(_contourlet_params(), (BENCH_CONTOURLET_N, BENCH_CONTOURLET_N))
    x = randn(rng, BENCH_CONTOURLET_N, BENCH_CONTOURLET_N)
    y = op * x
    z = zeros(BENCH_CONTOURLET_N, BENCH_CONTOURLET_N)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function nsct_state()
    rng = make_rng()
    op = NSCTOp(_contourlet_params(), (BENCH_CONTOURLET_N, BENCH_CONTOURLET_N))
    x = randn(rng, BENCH_CONTOURLET_N, BENCH_CONTOURLET_N)
    y = op * x
    z = zeros(BENCH_CONTOURLET_N, BENCH_CONTOURLET_N)
    return (op = op, adj = op', x = x, y = y, z = z)
end

if HAS_CONTOURLET
    contourlets["ContourletOp"] = BenchmarkGroup()
    contourlets["ContourletOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = contourlet_state())
    contourlets["ContourletOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = contourlet_state())

    contourlets["NSCTOp"] = BenchmarkGroup()
    contourlets["NSCTOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = nsct_state())
    contourlets["NSCTOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = nsct_state())
end

run_suite_if_standalone(@__FILE__, "contourletoperators")
