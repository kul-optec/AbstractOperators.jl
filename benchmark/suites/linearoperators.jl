# Linear operator benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/linearoperators.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

function mylinop_state()
    rng = make_rng()
    scale = randn(rng, BENCH_LINEAR_MYLIN_N)
    op = MyLinOp(
        Float64,
        (BENCH_LINEAR_MYLIN_N,),
        (BENCH_LINEAR_MYLIN_N,),
        (out, inp) -> (@. out = scale * inp),
        (out, inp) -> (@. out = scale * inp),
    )
    return linear_state(op)
end

function lbfgs_update_state()
    rng = make_rng()
    x = randn(rng, BENCH_LINEAR_LBFGS_N)
    x_prev = randn(rng, BENCH_LINEAR_LBFGS_N)
    grad = randn(rng, BENCH_LINEAR_LBFGS_N)
    grad_prev = randn(rng, BENCH_LINEAR_LBFGS_N)
    op = LBFGS(x, 5)
    return (op = op, x = x, x_prev = x_prev, grad = grad, grad_prev = grad_prev)
end

function lbfgs_mul_state()
    rng = make_rng()
    x_prev = randn(rng, BENCH_LINEAR_LBFGS_N)
    grad_prev = randn(rng, BENCH_LINEAR_LBFGS_N)
    op = LBFGS(x_prev, 5)
    x_curr = x_prev
    grad_curr = grad_prev
    for _ in 1:5
        x_next = randn(rng, BENCH_LINEAR_LBFGS_N)
        grad_next = randn(rng, BENCH_LINEAR_LBFGS_N)
        update!(op, x_next, x_curr, grad_next, grad_curr)
        x_curr = x_next
        grad_curr = grad_next
    end
    return (op = op, grad = grad_curr, d = zeros(BENCH_LINEAR_LBFGS_N))
end

linear["Eye"] = BenchmarkGroup()
linear["Eye"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Eye(Float64, (BENCH_LINEAR_EYE_N,))))

linear["DiagOp"] = BenchmarkGroup()
linear["DiagOp"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (rng = make_rng(); state = linear_state(DiagOp(randn(rng, BENCH_LINEAR_DIAG_N); threaded = false)))
linear["DiagOp"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (rng = make_rng(); state = linear_state(DiagOp(randn(rng, BENCH_LINEAR_DIAG_N); threaded = false)))
if BENCH_THREADED
    linear["DiagOp"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (rng = make_rng(); state = linear_state(check_threaded(DiagOp(randn(rng, BENCH_LINEAR_DIAG_N); threaded = true))))
    linear["DiagOp"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (rng = make_rng(); state = linear_state(check_threaded(DiagOp(randn(rng, BENCH_LINEAR_DIAG_N); threaded = true))))
end

linear["MatrixOp"] = BenchmarkGroup()
linear["MatrixOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (rng = make_rng(); state = linear_state(MatrixOp(randn(rng, BENCH_LINEAR_MATRIX_SHAPE...), BENCH_LINEAR_MATRIX_DOMAIN)))
linear["MatrixOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (rng = make_rng(); state = linear_state(MatrixOp(randn(rng, BENCH_LINEAR_MATRIX_SHAPE...), BENCH_LINEAR_MATRIX_DOMAIN)))

linear["FiniteDiff"] = BenchmarkGroup()
linear["FiniteDiff"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(FiniteDiff(Float64, (BENCH_LINEAR_FD_N,), 1; threaded = false)))
linear["FiniteDiff"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(FiniteDiff(Float64, (BENCH_LINEAR_FD_N,), 1; threaded = false)))
if BENCH_THREADED
    linear["FiniteDiff"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(check_threaded(FiniteDiff(Float64, (BENCH_LINEAR_FD_N,), 1; threaded = true))))
    linear["FiniteDiff"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(check_threaded(FiniteDiff(Float64, (BENCH_LINEAR_FD_N,), 1; threaded = true))))
end

linear["GetIndex"] = BenchmarkGroup()
linear["GetIndex"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(GetIndex(Float64, BENCH_LINEAR_GETINDEX_DIM, (25:1400, 10:800))))
linear["GetIndex"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(GetIndex(Float64, BENCH_LINEAR_GETINDEX_DIM, (25:1400, 10:800))))

linear["Variation"] = BenchmarkGroup()
linear["Variation"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Variation(Float64, BENCH_LINEAR_VARIATION_DIM; threaded = false)))
linear["Variation"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(Variation(Float64, BENCH_LINEAR_VARIATION_DIM; threaded = false)))
if BENCH_THREADED
    linear["Variation"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Variation(Float64, BENCH_LINEAR_VARIATION_DIM; threaded = true)))
    linear["Variation"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(Variation(Float64, BENCH_LINEAR_VARIATION_DIM; threaded = true)))
end

linear["ZeroPad"] = BenchmarkGroup()
linear["ZeroPad"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(ZeroPad(Float64, BENCH_LINEAR_ZEROPAD_DIM, (0, 256))))
linear["ZeroPad"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(ZeroPad(Float64, BENCH_LINEAR_ZEROPAD_DIM, (0, 256))))

linear["Zeros"] = BenchmarkGroup()
linear["Zeros"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Zeros(Float64, (BENCH_LINEAR_ZEROS_N,), Float64, (BENCH_LINEAR_ZEROS_N,))))

linear["LMatrixOp"] = BenchmarkGroup()
linear["LMatrixOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (rng = make_rng(); state = linear_state(LMatrixOp(randn(rng, BENCH_LINEAR_LMATRIX_N), BENCH_LINEAR_LMATRIX_N)))
linear["LMatrixOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (rng = make_rng(); state = linear_state(LMatrixOp(randn(rng, BENCH_LINEAR_LMATRIX_N), BENCH_LINEAR_LMATRIX_N)))

linear["MyLinOp"] = BenchmarkGroup()
linear["MyLinOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = mylinop_state())
linear["MyLinOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = mylinop_state())

linear["LBFGS"] = BenchmarkGroup()
linear["LBFGS"]["update"] = @benchmarkable update!(state.op, state.x, state.x_prev, state.grad, state.grad_prev) setup = (state = lbfgs_update_state())
linear["LBFGS"]["mul"] = @benchmarkable mul!(state.d, state.op, state.grad) setup = (state = lbfgs_mul_state())

run_suite_if_standalone(@__FILE__, "linearoperators")
