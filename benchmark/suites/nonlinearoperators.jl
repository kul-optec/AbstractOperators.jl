# Non-linear operator benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/nonlinearoperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

for (name, builder, positive, threadable) in [
        ("Pow", th -> Pow(Float64, (BENCH_NONLIN_N["Pow"],), 2; threaded = th), false, true),
        ("Exp", th -> Exp(Float64, (BENCH_NONLIN_N["Exp"],); threaded = th), false, true),
        ("Sin", th -> Sin(Float64, (BENCH_NONLIN_N["Sin"],); threaded = th), false, true),
        ("Cos", th -> Cos(Float64, (BENCH_NONLIN_N["Cos"],); threaded = th), false, true),
        ("Atan", th -> Atan(Float64, (BENCH_NONLIN_N["Atan"],); threaded = th), false, true),
        ("Tanh", th -> Tanh(Float64, (BENCH_NONLIN_N["Tanh"],); threaded = th), false, true),
        ("Sech", th -> Sech(Float64, (BENCH_NONLIN_N["Sech"],); threaded = th), false, true),
        ("Sigmoid", th -> Sigmoid(Float64, (BENCH_NONLIN_N["Sigmoid"],), 1.5; threaded = th), false, true),
        ("SoftMax", _ -> SoftMax(Float64, (BENCH_NONLIN_N["SoftMax"],)), false, false),
        ("SoftPlus", th -> SoftPlus(Float64, (BENCH_NONLIN_N["SoftPlus"],); threaded = th), false, true),
    ]
    nonlinear[name] = BenchmarkGroup()
    nonlinear[name]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (
        state = nonlinear_state($builder(false); positive = ($positive))
    )
    nonlinear[name]["jacobian-adjoint-single"] = @benchmarkable mul!(state.y, state.adj, state.b) setup = (
        state = jacobian_state($builder(false); positive = ($positive))
    )
    if BENCH_THREADED && threadable
        nonlinear[name]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (
            state = nonlinear_state(check_threaded($builder(true)); positive = ($positive))
        )
        nonlinear[name]["jacobian-adjoint-threaded"] = @benchmarkable mul!(state.y, state.adj, state.b) setup = (
            state = jacobian_state(check_threaded($builder(true)); positive = ($positive))
        )
    end
end

run_suite_if_standalone(@__FILE__, "nonlinearoperators")
