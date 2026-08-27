# Calculus / composition operator benchmarks.
#
# Standalone: julia --project=benchmark benchmark/suites/calculus.jl
isdefined(Main, :BENCH_COMMON_LOADED) || include(joinpath(@__DIR__, "..", "bench_common.jl"))

function hcat_state()
    rng = make_rng()
    op = HCAT(Eye(Float64, (BENCH_CALC_N,)), DiagOp(randn(rng, BENCH_CALC_N)))
    x = ArrayPartition(randn(rng, BENCH_CALC_N), randn(rng, BENCH_CALC_N))
    y = zeros(BENCH_CALC_N)
    z = ArrayPartition(zeros(BENCH_CALC_N), zeros(BENCH_CALC_N))
    return (op = op, adj = op', x = x, y = y, z = z)
end

# Threaded counterparts of the *CAT states. They differ from the serial ones in both block
# count and per-block size because block parallelism is gated on each independently; see the
# BENCH_CAT_THREADED_* constants.
function hcat_threaded_state()
    rng = make_rng()
    n = BENCH_HCAT_THREADED_N
    # Blocks are pinned serial so the measurement isolates the block loop, and because
    # HCAT/VCAT expose `threaded` only on their inner constructor -- the varargs form defaults
    # to `true` and lets the policy decide, which is what is being exercised here.
    blocks = ntuple(_ -> DiagOp(randn(rng, n); threaded = false), BENCH_CAT_THREADED_BLOCKS)
    op = check_block_threaded(HCAT(blocks...))
    x = ArrayPartition(ntuple(_ -> randn(rng, n), BENCH_CAT_THREADED_BLOCKS)...)
    y = zeros(n)
    z = ArrayPartition(ntuple(_ -> zeros(n), BENCH_CAT_THREADED_BLOCKS)...)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function vcat_threaded_state()
    rng = make_rng()
    n = BENCH_CAT_THREADED_N
    blocks = ntuple(_ -> DiagOp(randn(rng, n); threaded = false), BENCH_CAT_THREADED_BLOCKS)
    op = check_block_threaded(VCAT(blocks...))
    x = randn(rng, n)
    y = ArrayPartition(ntuple(_ -> zeros(n), BENCH_CAT_THREADED_BLOCKS)...)
    z = zeros(n)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function dcat_threaded_state()
    rng = make_rng()
    n = BENCH_CAT_THREADED_N
    blocks = ntuple(_ -> DiagOp(randn(rng, n); threaded = false), BENCH_CAT_THREADED_BLOCKS)
    op = check_block_threaded(DCAT(blocks...; threaded = true))
    x = ArrayPartition(ntuple(_ -> randn(rng, n), BENCH_CAT_THREADED_BLOCKS)...)
    y = ArrayPartition(ntuple(_ -> zeros(n), BENCH_CAT_THREADED_BLOCKS)...)
    z = ArrayPartition(ntuple(_ -> zeros(n), BENCH_CAT_THREADED_BLOCKS)...)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function vcat_state()
    rng = make_rng()
    op = VCAT(Eye(Float64, (BENCH_CALC_N,)), DiagOp(randn(rng, BENCH_CALC_N)))
    x = randn(rng, BENCH_CALC_N)
    y = ArrayPartition(zeros(BENCH_CALC_N), zeros(BENCH_CALC_N))
    z = zeros(BENCH_CALC_N)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function dcat_state()
    rng = make_rng()
    op = DCAT(Eye(Float64, (BENCH_CALC_N,)), DiagOp(randn(rng, BENCH_CALC_N)))
    x = ArrayPartition(randn(rng, BENCH_CALC_N), randn(rng, BENCH_CALC_N))
    y = ArrayPartition(zeros(BENCH_CALC_N), zeros(BENCH_CALC_N))
    z = ArrayPartition(zeros(BENCH_CALC_N), zeros(BENCH_CALC_N))
    return (op = op, adj = op', x = x, y = y, z = z)
end

function affineadd_state()
    rng = make_rng()
    A = Eye(Float64, (BENCH_CALC_N,))
    op = AffineAdd(A, randn(rng, BENCH_CALC_N))
    return linear_state(op)
end

hadamardprod_jacobian_state() = jacobian_state(HadamardProd(Sin((BENCH_CALC_N,)), Cos((BENCH_CALC_N,))))

function ax_mul_bxt_state()
    rng = make_rng()
    # Ax_mul_Bxt(A,B): computes A(x) * B(x)'. Requires same domain and A,B output same codomain.
    op = Ax_mul_Bxt(
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
    )
    return nonlinear_state(op)
end

function ax_mul_bxt_jacobian_state()
    rng = make_rng()
    op = Ax_mul_Bxt(
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
    )
    return jacobian_state(op)
end

function axt_mul_bx_state()
    rng = make_rng()
    # Axt_mul_Bx(A,B): computes A(x)' * B(x). Requires A and B share domain; rows(A)==rows(B).
    op = Axt_mul_Bx(
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
    )
    return nonlinear_state(op)
end

function axt_mul_bx_jacobian_state()
    rng = make_rng()
    op = Axt_mul_Bx(
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
    )
    return jacobian_state(op)
end

function ax_mul_bx_state()
    rng = make_rng()
    # Ax_mul_Bx(A,B): computes A(x)*B(x). Requires size(A,1)[2] == size(B,1)[1].
    # Requires square codomain shape where col(A) == row(B).
    op = Ax_mul_Bx(
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
    )
    return nonlinear_state(op)
end

function ax_mul_bx_jacobian_state()
    rng = make_rng()
    op = Ax_mul_Bx(
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
        MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ),
    )
    return jacobian_state(op)
end

calculus["Compose"] = BenchmarkGroup()
calculus["Compose"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (rng = make_rng(); state = linear_state(Compose(DiagOp(randn(rng, BENCH_CALC_N)), Eye(Float64, (BENCH_CALC_N,)))))
calculus["Compose"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (rng = make_rng(); state = linear_state(Compose(DiagOp(randn(rng, BENCH_CALC_N)), Eye(Float64, (BENCH_CALC_N,)))))

calculus["Reshape"] = BenchmarkGroup()
calculus["Reshape"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Reshape(Eye(Float64, (BENCH_CALC_N,)), BENCH_CALC_2D...)))
calculus["Reshape"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(Reshape(Eye(Float64, (BENCH_CALC_N,)), BENCH_CALC_2D...)))

calculus["Scale"] = BenchmarkGroup()
calculus["Scale"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Scale(2.0, Eye(Float64, (BENCH_CALC_N,)); threaded = false)))
calculus["Scale"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(Scale(2.0, Eye(Float64, (BENCH_CALC_N,)); threaded = false)))
if BENCH_THREADED
    # 128x the serial size: Scale's crossover is 2^22, so the shared BENCH_CALC_N would be
    # vetoed by the policy. The pair is therefore not comparable to the entries above -- it
    # is a threaded-vs-itself baseline for tracking regressions in the threaded path.
    calculus["Scale"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(check_threaded(Scale(2.0, Eye(Float64, (BENCH_CALC_SCALE_THREADED_N,)); threaded = true))))
    calculus["Scale"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(check_threaded(Scale(2.0, Eye(Float64, (BENCH_CALC_SCALE_THREADED_N,)); threaded = true))))
end

calculus["Sum"] = BenchmarkGroup()
calculus["Sum"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (rng = make_rng(); state = linear_state(Sum(Eye(Float64, (BENCH_CALC_N,)), DiagOp(randn(rng, BENCH_CALC_N)))))
calculus["Sum"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (rng = make_rng(); state = linear_state(Sum(Eye(Float64, (BENCH_CALC_N,)), DiagOp(randn(rng, BENCH_CALC_N)))))

calculus["HCAT"] = BenchmarkGroup()
calculus["HCAT"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = hcat_state())
calculus["HCAT"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = hcat_state())
if BENCH_THREADED
    calculus["HCAT"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = hcat_threaded_state())
    calculus["HCAT"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = hcat_threaded_state())
end

calculus["VCAT"] = BenchmarkGroup()
calculus["VCAT"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = vcat_state())
calculus["VCAT"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = vcat_state())
if BENCH_THREADED
    calculus["VCAT"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = vcat_threaded_state())
    calculus["VCAT"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = vcat_threaded_state())
end

calculus["DCAT"] = BenchmarkGroup()
calculus["DCAT"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = dcat_state())
calculus["DCAT"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = dcat_state())
if BENCH_THREADED
    calculus["DCAT"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = dcat_threaded_state())
    calculus["DCAT"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = dcat_threaded_state())
end

calculus["BroadCast"] = BenchmarkGroup()
# See the `DiagOp` comment in suites/linearoperators.jl: `threaded` left at its default,
# since the per-thread-count run already covers serial and threaded. `BENCH_CALC_N * 8` is
# exactly THRESHOLD_MEMORY_BOUND, so the policy declines at `-t 1` and grants from `-t 2` --
# which is precisely what the -single/-threaded pair used to spell out by hand.
calculus["BroadCast"]["identity"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(BroadCast(Eye(Float64, (BENCH_CALC_N,)), (BENCH_CALC_N, 8))))

# No threaded counterpart, deliberately: 256 * 8 = 2048 elements is three orders of
# magnitude below THRESHOLD_MEMORY_BOUND, so the policy vetoes threading at *every* thread
# count. The `operator-threaded-*` entries this replaces asked for `threaded = true`, were
# declined, and measured the serial path under a threaded name -- 261 ns against
# `operator-single-forward`'s 270 ns in the run that prompted this cleanup. They carried no
# `check_threaded` guard, which is the failure that guard exists to make loud.
#
# Kept small on purpose: this pair measures per-child dispatch overhead in a broadcast over
# a small wrapped operator, which is a different cost shape from `identity` above. Sizing it
# up to reach the threshold would measure the same memory-bound kernel twice instead.
calculus["BroadCast"]["operator-forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (rng = make_rng(); state = linear_state(BroadCast(DiagOp(randn(rng, 256)), (256, 8))))
calculus["BroadCast"]["operator-adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (rng = make_rng(); state = linear_state(BroadCast(DiagOp(randn(rng, 256)), (256, 8))))

calculus["AffineAdd"] = BenchmarkGroup()
calculus["AffineAdd"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = affineadd_state())
calculus["AffineAdd"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = affineadd_state())

calculus["Jacobian"] = BenchmarkGroup()
calculus["Jacobian"]["sigmoid-adjoint"] = @benchmarkable mul!(state.y, state.adj, state.b) setup = (state = jacobian_state(Sigmoid(Float64, (BENCH_CALC_N,), 1.5)))

calculus["HadamardProd"] = BenchmarkGroup()
calculus["HadamardProd"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = nonlinear_state(HadamardProd(Sin((BENCH_CALC_N,)), Cos((BENCH_CALC_N,)))))
calculus["HadamardProd"]["jacobian-adjoint"] = @benchmarkable mul!(state.y, state.adj, state.b) setup = (state = hadamardprod_jacobian_state())

calculus["Ax_mul_Bxt"] = BenchmarkGroup()
calculus["Ax_mul_Bxt"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = ax_mul_bxt_state())
calculus["Ax_mul_Bxt"]["jacobian-adjoint"] = @benchmarkable mul!(state.y, state.adj, state.b) setup = (state = ax_mul_bxt_jacobian_state())

calculus["Axt_mul_Bx"] = BenchmarkGroup()
calculus["Axt_mul_Bx"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = axt_mul_bx_state())
calculus["Axt_mul_Bx"]["jacobian-adjoint"] = @benchmarkable mul!(state.y, state.adj, state.b) setup = (state = axt_mul_bx_jacobian_state())

calculus["Ax_mul_Bx"] = BenchmarkGroup()
calculus["Ax_mul_Bx"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = ax_mul_bx_state())
calculus["Ax_mul_Bx"]["jacobian-adjoint"] = @benchmarkable mul!(state.y, state.adj, state.b) setup = (state = ax_mul_bx_jacobian_state())

# `builder` takes the `threaded` permission so that each operator is registered twice: once
# pinned serial and, when enabled, once threaded. Passing it explicitly matters -- these
# constructors default to `threaded = true` and every size below clears its operator's
# threshold, so leaving it implicit made the plain entries measure the *threaded* path on a
# multicore machine and the serial path on a single-core runner, under one name.
#
# SoftMax is absent from the threaded list on purpose: its `mul!` is a reduction over a
# shared buffer and `is_threaded(::SoftMax)` is permanently false, so a threaded entry would

run_suite_if_standalone(@__FILE__, "calculus")
