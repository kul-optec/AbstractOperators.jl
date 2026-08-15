# Batched operator benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/batching.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

function simple_batch_state(threaded)
    rng = make_rng()
    base = Compose(DiagOp(randn(rng, 255)), FiniteDiff(Float64, (256,), 1))
    op = BatchOp(base, (8, 8), (:_, :b, :b); threaded = threaded)
    x = randn(rng, 256, 8, 8)
    y = zeros(255, 8, 8)
    z = zeros(256, 8, 8)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function spreading_batch_state(threaded; strategy = nothing)
    rng = make_rng()
    ops = [DiagOp(randn(rng, 255)) * FiniteDiff(Float64, (256,), 1) for _ in 1:4]
    kwargs = threaded ? (; threaded = true, threading_strategy = strategy) : (; threaded = false)
    op = BatchOp(ops, 8, (:_, :s, :b); kwargs...)
    x = randn(rng, 256, 4, 8)
    y = zeros(255, 4, 8)
    z = zeros(256, 4, 8)
    return (op = op, adj = op', x = x, y = y, z = z)
end

batching["SimpleBatchOp"] = BenchmarkGroup()
batching["SimpleBatchOp"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = simple_batch_state(false))
batching["SimpleBatchOp"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = simple_batch_state(false))
if BENCH_THREADED
    batching["SimpleBatchOp"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = simple_batch_state(true))
    batching["SimpleBatchOp"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = simple_batch_state(true))
end

batching["SpreadingBatchOp"] = BenchmarkGroup()
batching["SpreadingBatchOp"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = spreading_batch_state(false))
batching["SpreadingBatchOp"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = spreading_batch_state(false))
if BENCH_THREADED
    for strategy in (ThreadingStrategy.COPYING, ThreadingStrategy.LOCKING, ThreadingStrategy.FIXED_OPERATOR)
        strategy_name = String(Symbol(strategy))
        batching["SpreadingBatchOp"]["forward-$(strategy_name)"] = @benchmarkable mul!(
            state.y, state.op, state.x
        ) setup = (state = spreading_batch_state(true; strategy = ($strategy)))
        batching["SpreadingBatchOp"]["adjoint-$(strategy_name)"] = @benchmarkable mul!(
            state.z, state.adj, state.y
        ) setup = (state = spreading_batch_state(true; strategy = ($strategy)))
    end
end

run_suite_if_standalone(@__FILE__, "batching")
