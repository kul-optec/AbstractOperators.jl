using AbstractOperators
using BenchmarkTools
using LinearAlgebra
using Pkg
using Random
using RecursiveArrayTools: ArrayPartition

function _load_local_subpackage(pkg::Symbol, relpath::String)
    try
        @eval using $pkg
        return true
    catch err
        if !(err isa ArgumentError)
            rethrow(err)
        end
        local_path = joinpath(@__DIR__, "..", relpath)
        if !isdir(local_path)
            return false
        end
        try
            Pkg.develop(path = local_path)
            @eval using $pkg
            return true
        catch
            return false
        end
    end
end

const HAS_DSP = _load_local_subpackage(:DSPOperators, "DSPOperators")
const HAS_FFTW = _load_local_subpackage(:FFTWOperators, "FFTWOperators")
const HAS_NFFT = _load_local_subpackage(:NFFTOperators, "NFFTOperators")
const HAS_WAVELET = _load_local_subpackage(:WaveletOperators, "WaveletOperators")

const SUITE = BenchmarkGroup()
const BENCH_LINEAR_EYE_N = 1_048_576
const BENCH_LINEAR_DIAG_N = 524_288
const BENCH_LINEAR_MATRIX_SHAPE = (192, 192)
const BENCH_LINEAR_MATRIX_DOMAIN = 192
const BENCH_LINEAR_FD_N = 262_144
const BENCH_LINEAR_GETINDEX_DIM = (1536, 1024)
const BENCH_LINEAR_VARIATION_DIM = (512, 256)
const BENCH_LINEAR_ZEROPAD_DIM = (512, 256)
const BENCH_LINEAR_ZEROS_N = 2_000_000
const BENCH_LINEAR_LMATRIX_N = 1024
const BENCH_LINEAR_MYLIN_N = 524_288
const BENCH_LINEAR_LBFGS_N = 8192

const BENCH_NONLIN_N = Dict(
    "Pow" => 131_072,
    "Exp" => 131_072,
    "Sin" => 32_768,
    "Cos" => 32_768,
    "Atan" => 32_768,
    "Tanh" => 32_768,
    "Sech" => 32_768,
    "Sigmoid" => 65_536,
    "SoftMax" => 65_536,
    "SoftPlus" => 65_536,
)

const BENCH_CALC_N = 32_768
const BENCH_CALC_2D = (256, 128)
const BENCH_CALC_SQ = 64

const BENCH_DSP_FILT_N = 65_536
const BENCH_DSP_XCORR_N = 32_768
const BENCH_DSP_MIMO_SHAPE = (16_384, 2)
const BENCH_DFT_SHAPE = (128, 128)
const BENCH_WAVELET_N = 131_072
const BENCH_NFFT_IMAGE = (48, 48)
const BENCH_NFFT_NSAMP = 48
const BENCH_NFFT_NPROF = 24

make_rng() = MersenneTwister(1234)

function linear_state(op)
    rng = make_rng()
    x = randn(rng, domain_type(op), size(op, 2)...)
    y = zeros(codomain_type(op), size(op, 1)...)
    z = zeros(domain_type(op), size(op, 2)...)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function nonlinear_state(op; positive = false)
    rng = make_rng()
    x = positive ? abs.(randn(rng, domain_type(op), size(op, 2)...)) : randn(rng, domain_type(op), size(op, 2)...)
    y = zeros(codomain_type(op), size(op, 1)...)
    return (op = op, x = x, y = y)
end

function jacobian_state(op; positive = false)
    rng = make_rng()
    x = positive ? abs.(randn(rng, domain_type(op), size(op, 2)...)) : randn(rng, domain_type(op), size(op, 2)...)
    jac = Jacobian(op, x)
    b = randn(rng, codomain_type(jac), size(jac, 1)...)
    y = zeros(domain_type(jac), size(jac, 2)...)
    return (jac = jac, adj = jac', b = b, y = y)
end

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

function hcat_state()
    rng = make_rng()
    op = HCAT(Eye(Float64, (BENCH_CALC_N,)), DiagOp(randn(rng, BENCH_CALC_N)))
    x = ArrayPartition(randn(rng, BENCH_CALC_N), randn(rng, BENCH_CALC_N))
    y = zeros(BENCH_CALC_N)
    z = ArrayPartition(zeros(BENCH_CALC_N), zeros(BENCH_CALC_N))
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
    op = Ax_mul_Bxt(MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ), MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ))
    return nonlinear_state(op)
end

function ax_mul_bxt_jacobian_state()
    rng = make_rng()
    op = Ax_mul_Bxt(MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ), MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ))
    return jacobian_state(op)
end

function axt_mul_bx_state()
    rng = make_rng()
    # Axt_mul_Bx(A,B): computes A(x)' * B(x). Requires A and B share domain; rows(A)==rows(B).
    op = Axt_mul_Bx(MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ), MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ))
    return nonlinear_state(op)
end

function axt_mul_bx_jacobian_state()
    rng = make_rng()
    op = Axt_mul_Bx(MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ), MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ))
    return jacobian_state(op)
end

function ax_mul_bx_state()
    rng = make_rng()
    # Ax_mul_Bx(A,B): computes A(x)*B(x). Requires size(A,1)[2] == size(B,1)[1].
    # Requires square codomain shape where col(A) == row(B).
    op = Ax_mul_Bx(MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ), MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ))
    return nonlinear_state(op)
end

function ax_mul_bx_jacobian_state()
    rng = make_rng()
    op = Ax_mul_Bx(MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ), MatrixOp(randn(rng, BENCH_CALC_SQ, BENCH_CALC_SQ), BENCH_CALC_SQ))
    return jacobian_state(op)
end

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

function dft_state()
    rng = make_rng()
    op = DFT(BENCH_DFT_SHAPE)
    x = randn(rng, BENCH_DFT_SHAPE...)
    y = zeros(ComplexF64, BENCH_DFT_SHAPE...)
    z = zeros(Float64, BENCH_DFT_SHAPE...)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function wavelet_state()
    rng = make_rng()
    op = WaveletOp(wavelet(WT.db2), BENCH_WAVELET_N)
    x = randn(rng, BENCH_WAVELET_N)
    y = zeros(BENCH_WAVELET_N)
    z = zeros(BENCH_WAVELET_N)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function nfft_state()
    rng = make_rng()
    traj = rand(rng, 2, BENCH_NFFT_NSAMP, BENCH_NFFT_NPROF) .- 0.5
    dcf = ones(eltype(traj), BENCH_NFFT_NSAMP, BENCH_NFFT_NPROF)
    op = NFFTOp(BENCH_NFFT_IMAGE, traj, dcf; threaded = false)
    x = randn(rng, ComplexF64, BENCH_NFFT_IMAGE...)
    y = zeros(ComplexF64, BENCH_NFFT_NSAMP, BENCH_NFFT_NPROF)
    z = zeros(ComplexF64, BENCH_NFFT_IMAGE...)
    return (op = op, adj = op', x = x, y = y, z = z)
end

function normal_state(op)
    nop = AbstractOperators.get_normal_op(op)
    rng = make_rng()
    x = randn(rng, domain_type(nop), size(nop, 2)...)
    y = zeros(codomain_type(nop), size(nop, 1)...)
    return (op = nop, x = x, y = y)
end

linear = SUITE["linearoperators"] = BenchmarkGroup()
calculus = SUITE["calculus"] = BenchmarkGroup()
nonlinear = SUITE["nonlinearoperators"] = BenchmarkGroup()
batching = SUITE["batching"] = BenchmarkGroup()
dsp = HAS_DSP ? (SUITE["dspoperators"] = BenchmarkGroup()) : nothing
fftw = HAS_FFTW ? (SUITE["fftwoperators"] = BenchmarkGroup()) : nothing
nfft = HAS_NFFT ? (SUITE["nfftoperators"] = BenchmarkGroup()) : nothing
wavelets = HAS_WAVELET ? (SUITE["waveletoperators"] = BenchmarkGroup()) : nothing
normal = SUITE["normaloperators"] = BenchmarkGroup()

linear["Eye"] = BenchmarkGroup()
linear["Eye"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Eye(Float64, (BENCH_LINEAR_EYE_N,))))

linear["DiagOp"] = BenchmarkGroup()
linear["DiagOp"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(DiagOp(randn(make_rng(), BENCH_LINEAR_DIAG_N); threaded = false)))
linear["DiagOp"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(DiagOp(randn(make_rng(), BENCH_LINEAR_DIAG_N); threaded = false)))
if Threads.nthreads() > 1
    linear["DiagOp"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(DiagOp(randn(make_rng(), BENCH_LINEAR_DIAG_N); threaded = true)))
    linear["DiagOp"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(DiagOp(randn(make_rng(), BENCH_LINEAR_DIAG_N); threaded = true)))
end

linear["MatrixOp"] = BenchmarkGroup()
linear["MatrixOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(MatrixOp(randn(make_rng(), BENCH_LINEAR_MATRIX_SHAPE...), BENCH_LINEAR_MATRIX_DOMAIN)))
linear["MatrixOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(MatrixOp(randn(make_rng(), BENCH_LINEAR_MATRIX_SHAPE...), BENCH_LINEAR_MATRIX_DOMAIN)))

linear["FiniteDiff"] = BenchmarkGroup()
linear["FiniteDiff"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(FiniteDiff(Float64, (BENCH_LINEAR_FD_N,), 1)))
linear["FiniteDiff"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(FiniteDiff(Float64, (BENCH_LINEAR_FD_N,), 1)))

linear["GetIndex"] = BenchmarkGroup()
linear["GetIndex"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(GetIndex(Float64, BENCH_LINEAR_GETINDEX_DIM, (25:1400, 10:800))))
linear["GetIndex"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(GetIndex(Float64, BENCH_LINEAR_GETINDEX_DIM, (25:1400, 10:800))))

linear["Variation"] = BenchmarkGroup()
linear["Variation"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Variation(Float64, BENCH_LINEAR_VARIATION_DIM; threaded = false)))
linear["Variation"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(Variation(Float64, BENCH_LINEAR_VARIATION_DIM; threaded = false)))
if Threads.nthreads() > 2
    linear["Variation"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Variation(Float64, BENCH_LINEAR_VARIATION_DIM; threaded = true)))
    linear["Variation"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(Variation(Float64, BENCH_LINEAR_VARIATION_DIM; threaded = true)))
end

linear["ZeroPad"] = BenchmarkGroup()
linear["ZeroPad"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(ZeroPad(Float64, BENCH_LINEAR_ZEROPAD_DIM, (0, 256))))
linear["ZeroPad"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(ZeroPad(Float64, BENCH_LINEAR_ZEROPAD_DIM, (0, 256))))

linear["Zeros"] = BenchmarkGroup()
linear["Zeros"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Zeros(Float64, (BENCH_LINEAR_ZEROS_N,), Float64, (BENCH_LINEAR_ZEROS_N,))))

linear["LMatrixOp"] = BenchmarkGroup()
linear["LMatrixOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(LMatrixOp(randn(make_rng(), BENCH_LINEAR_LMATRIX_N), BENCH_LINEAR_LMATRIX_N)))
linear["LMatrixOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(LMatrixOp(randn(make_rng(), BENCH_LINEAR_LMATRIX_N), BENCH_LINEAR_LMATRIX_N)))

linear["MyLinOp"] = BenchmarkGroup()
linear["MyLinOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = mylinop_state())
linear["MyLinOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = mylinop_state())

linear["LBFGS"] = BenchmarkGroup()
linear["LBFGS"]["update"] = @benchmarkable update!(state.op, state.x, state.x_prev, state.grad, state.grad_prev) setup = (state = lbfgs_update_state())
linear["LBFGS"]["mul"] = @benchmarkable mul!(state.d, state.op, state.grad) setup = (state = lbfgs_mul_state())

calculus["Compose"] = BenchmarkGroup()
calculus["Compose"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Compose(DiagOp(randn(make_rng(), BENCH_CALC_N)), Eye(Float64, (BENCH_CALC_N,)))))
calculus["Compose"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(Compose(DiagOp(randn(make_rng(), BENCH_CALC_N)), Eye(Float64, (BENCH_CALC_N,)))))

calculus["Reshape"] = BenchmarkGroup()
calculus["Reshape"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Reshape(Eye(Float64, (BENCH_CALC_N,)), BENCH_CALC_2D...)))
calculus["Reshape"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(Reshape(Eye(Float64, (BENCH_CALC_N,)), BENCH_CALC_2D...)))

calculus["Scale"] = BenchmarkGroup()
calculus["Scale"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Scale(2.0, Eye(Float64, (BENCH_CALC_N,)))))
calculus["Scale"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(Scale(2.0, Eye(Float64, (BENCH_CALC_N,)))))

calculus["Sum"] = BenchmarkGroup()
calculus["Sum"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(Sum(Eye(Float64, (BENCH_CALC_N,)), DiagOp(randn(make_rng(), BENCH_CALC_N)))))
calculus["Sum"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(Sum(Eye(Float64, (BENCH_CALC_N,)), DiagOp(randn(make_rng(), BENCH_CALC_N)))))

calculus["HCAT"] = BenchmarkGroup()
calculus["HCAT"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = hcat_state())
calculus["HCAT"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = hcat_state())

calculus["VCAT"] = BenchmarkGroup()
calculus["VCAT"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = vcat_state())
calculus["VCAT"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = vcat_state())

calculus["DCAT"] = BenchmarkGroup()
calculus["DCAT"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = dcat_state())
calculus["DCAT"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = dcat_state())

calculus["BroadCast"] = BenchmarkGroup()
calculus["BroadCast"]["identity-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(BroadCast(Eye(Float64, (BENCH_CALC_N,)), (BENCH_CALC_N, 8); threaded = false)))
if Threads.nthreads() > 2
    calculus["BroadCast"]["identity-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(BroadCast(Eye(Float64, (BENCH_CALC_N,)), (BENCH_CALC_N, 8); threaded = true)))
end
calculus["BroadCast"]["operator-single-forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(BroadCast(DiagOp(randn(make_rng(), 256)), (256, 8); threaded = false)))
calculus["BroadCast"]["operator-single-adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(BroadCast(DiagOp(randn(make_rng(), 256)), (256, 8); threaded = false)))
if Threads.nthreads() > 2
    calculus["BroadCast"]["operator-threaded-forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = linear_state(BroadCast(DiagOp(randn(make_rng(), 256)), (256, 8); threaded = true)))
    calculus["BroadCast"]["operator-threaded-adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = linear_state(BroadCast(DiagOp(randn(make_rng(), 256)), (256, 8); threaded = true)))
end

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

for (name, builder, positive) in [
        ("Pow", () -> Pow(Float64, (BENCH_NONLIN_N["Pow"],), 2), false),
        ("Exp", () -> Exp(Float64, (BENCH_NONLIN_N["Exp"],)), false),
        ("Sin", () -> Sin(Float64, (BENCH_NONLIN_N["Sin"],)), false),
        ("Cos", () -> Cos(Float64, (BENCH_NONLIN_N["Cos"],)), false),
        ("Atan", () -> Atan(Float64, (BENCH_NONLIN_N["Atan"],)), false),
        ("Tanh", () -> Tanh(Float64, (BENCH_NONLIN_N["Tanh"],)), false),
        ("Sech", () -> Sech(Float64, (BENCH_NONLIN_N["Sech"],)), false),
        ("Sigmoid", () -> Sigmoid(Float64, (BENCH_NONLIN_N["Sigmoid"],), 1.5), false),
        ("SoftMax", () -> SoftMax(Float64, (BENCH_NONLIN_N["SoftMax"],)), false),
        ("SoftPlus", () -> SoftPlus(Float64, (BENCH_NONLIN_N["SoftPlus"],)), false),
    ]
    nonlinear[name] = BenchmarkGroup()
    nonlinear[name]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = nonlinear_state($builder(); positive = $positive))
    nonlinear[name]["jacobian-adjoint"] = @benchmarkable mul!(state.y, state.adj, state.b) setup = (state = jacobian_state($builder(); positive = $positive))
end

batching["SimpleBatchOp"] = BenchmarkGroup()
batching["SimpleBatchOp"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = simple_batch_state(false))
batching["SimpleBatchOp"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = simple_batch_state(false))
if Threads.nthreads() > 2
    batching["SimpleBatchOp"]["forward-threaded"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = simple_batch_state(true))
    batching["SimpleBatchOp"]["adjoint-threaded"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = simple_batch_state(true))
end

batching["SpreadingBatchOp"] = BenchmarkGroup()
batching["SpreadingBatchOp"]["forward-single"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = spreading_batch_state(false))
batching["SpreadingBatchOp"]["adjoint-single"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = spreading_batch_state(false))
if Threads.nthreads() > 2
    for strategy in (ThreadingStrategy.COPYING, ThreadingStrategy.LOCKING, ThreadingStrategy.FIXED_OPERATOR)
        strategy_name = String(Symbol(strategy))
        batching["SpreadingBatchOp"]["forward-$(strategy_name)"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = spreading_batch_state(true; strategy = $strategy))
        batching["SpreadingBatchOp"]["adjoint-$(strategy_name)"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = spreading_batch_state(true; strategy = $strategy))
    end
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

if HAS_FFTW
    fftw["DFT"] = BenchmarkGroup()
    fftw["DFT"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = dft_state())
    fftw["DFT"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = dft_state())
end

if HAS_NFFT
    nfft["NFFTOp"] = BenchmarkGroup()
    nfft["NFFTOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = nfft_state())
    nfft["NFFTOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = nfft_state())
end

if HAS_WAVELET
    wavelets["WaveletOp"] = BenchmarkGroup()
    wavelets["WaveletOp"]["forward"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = wavelet_state())
    wavelets["WaveletOp"]["adjoint"] = @benchmarkable mul!(state.z, state.adj, state.y) setup = (state = wavelet_state())
end

normal["DiagOp"] = BenchmarkGroup()
normal["DiagOp"]["mul"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = normal_state(DiagOp(randn(make_rng(), BENCH_LINEAR_DIAG_N); threaded = false)))

if HAS_FFTW
    normal["DFT"] = BenchmarkGroup()
    normal["DFT"]["mul"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = normal_state(DFT(BENCH_DFT_SHAPE)))
end

if HAS_NFFT
    normal["NFFTOp"] = BenchmarkGroup()
    normal["NFFTOp"]["mul"] = @benchmarkable mul!(state.y, state.op, state.x) setup = (state = normal_state(nfft_state().op))
end

# Cap run time so CI / ASV comparisons complete in reasonable time
for (_, b) in BenchmarkTools.leaves(SUITE)
    b.params.seconds = 5
    b.params.samples = 10000
    b.params.evals = 1
end
