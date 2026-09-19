# Non-linear operator benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/nonlinearoperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

for (name, builder, positive) in [
        ("Pow", () -> Pow(Float64, (BENCH_NONLIN_POW_N,), 2), false),
        ("Exp", () -> Exp(Float64, (BENCH_NONLIN_EXP_N,)), false),
        ("Sin", () -> Sin(Float64, (BENCH_NONLIN_SIN_N,)), false),
        ("Cos", () -> Cos(Float64, (BENCH_NONLIN_COS_N,)), false),
        ("Atan", () -> Atan(Float64, (BENCH_NONLIN_ATAN_N,)), false),
        ("Tanh", () -> Tanh(Float64, (BENCH_NONLIN_TANH_N,)), false),
        ("Sech", () -> Sech(Float64, (BENCH_NONLIN_SECH_N,)), false),
        ("Sigmoid", () -> Sigmoid(Float64, (BENCH_NONLIN_SIGMOID_N,), 1.5), false),
        ("SoftMax", () -> SoftMax(Float64, (BENCH_NONLIN_SOFTMAX_N,)), false),
        ("SoftPlus", () -> SoftPlus(Float64, (BENCH_NONLIN_SOFTPLUS_N,)), false),
    ]
    nonlinear[name] = BenchmarkGroup()
    # `threaded` left at its default: the per-thread-count run already covers serial and
    # threaded (see `BENCH_THREADED`), so no -single/-threaded split is needed here.
    nonlinear[name]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (
        state = nonlinear_state($builder(); positive = ($positive))
    )
    nonlinear[name]["jacobian-adjoint"] = @benchmarkable mul!(state.y, state.adj, state.b) setup = (
        state = jacobian_state($builder(); positive = ($positive))
    )
end

run_suite_if_standalone(@__FILE__, "nonlinearoperators")
