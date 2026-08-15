@testitem "Threading contract: transcendental nonlinear operators" tags = [
    :nonlinearoperator, :Threading, :Sin, :Cos, :Exp, :Atan, :Tanh, :Sech, :SoftPlus,
] setup = [TestUtils, ThreadingContract] begin
    using AbstractOperators, Random
    Random.seed!(0)

    n = 1 << 13
    # Captured once and reused across both operators: two separate `randn(n)` calls would
    # compare different inputs and the equality would be meaningless.
    x = randn(n)
    r = randn(n)

    for Op in (Sin, Cos, Exp, Atan, Tanh, Sech, SoftPlus)
        test_threading_contract((; threaded) -> Op(Float64, (n,); threaded), x)

        # Jacobian adjoints thread too, and must match the serial result exactly.
        serial = Op(Float64, (n,); threaded = false)
        threaded = Op(Float64, (n,); threaded = true)
        @test Jacobian(serial, x)' * r == Jacobian(threaded, x)' * r
    end
end

@testitem "Threading contract: Pow and Sigmoid" tags = [
    :nonlinearoperator, :Threading, :Pow, :Sigmoid,
] setup = [TestUtils, ThreadingContract] begin
    using AbstractOperators, Random
    Random.seed!(0)

    n = 1 << 13
    x = abs.(randn(n)) .+ 0.5   # keep `x^p` real for fractional p
    r = randn(n)

    for p in (2, 0.5)
        test_threading_contract((; threaded) -> Pow(Float64, (n,), p; threaded), x)
        s = Pow(Float64, (n,), p; threaded = false)
        t = Pow(Float64, (n,), p; threaded = true)
        @test Jacobian(s, x)' * r == Jacobian(t, x)' * r
    end

    test_threading_contract((; threaded) -> Sigmoid(Float64, (n,), 2.0; threaded), x)
    # Sigmoid's threaded Jacobian adjoint is a single fused expression while the serial one
    # is a four-statement in-place sequence, so this checks a genuine rewrite, not just a
    # scheduling change. Floating-point association differs, hence `≈` rather than `==`.
    s = Sigmoid(Float64, (n,), 2.0; threaded = false)
    t = Sigmoid(Float64, (n,), 2.0; threaded = true)
    @test Jacobian(s, x)' * r ≈ Jacobian(t, x)' * r
end

@testitem "Threading contract: SoftMax is deliberately unthreaded" tags = [
    :nonlinearoperator, :Threading, :SoftMax,
] setup = [TestUtils] begin
    using AbstractOperators

    op = SoftMax(Float64, (64,))
    @test is_threaded(op) == false
    @test supports_threading(op) == false
    # Asking for threading must fail loudly rather than hand back a serial operator.
    @test_throws ArgumentError copy_operator(op; threaded = true)

    # A copy gets its own scratch buffer -- sharing it is what makes SoftMax unsafe to
    # share between threads in the first place.
    c = copy_operator(op)
    @test c.buf !== op.buf
end

@testitem "Threading contract: FiniteDiff" tags = [:linearoperator, :Threading, :FiniteDiff] setup = [
    TestUtils, ThreadingContract,
] begin
    using AbstractOperators, LinearAlgebra, Random
    Random.seed!(0)

    n = 1 << 16
    x = randn(n)
    y = randn(n - 1)
    test_threading_contract((; threaded) -> FiniteDiff(Float64, (n,); threaded), x; adjoint_input = y)

    # The `@views` rewrite that made threading worthwhile also removed the two temporaries
    # the old `b[idx_1] .- b[idx_2]` materialised on every call. Guard that directly: an
    # allocating inner loop is a defect, not a style preference.
    # `op` and `op'` are hoisted into their own function so the measured call sees concrete
    # types; taking the adjoint inside `@allocated` measures the wrapper construction in a
    # type-unstable loop body rather than the kernel.
    function check_allocation_free(threaded, x, y, n)
        op = FiniteDiff(Float64, (n,); threaded)
        adj = op'
        out = zeros(n - 1)
        adj_out = zeros(n)
        mul!(out, op, x)                      # compile
        mul!(adj_out, adj, y)                 # compile
        return (@allocated mul!(out, op, x)), (@allocated mul!(adj_out, adj, y))
    end

    for threaded in (false, true)
        fwd_alloc, adj_alloc = check_allocation_free(threaded, x, y, n)
        @test fwd_alloc == 0
        @test adj_alloc == 0
    end
end

@testitem "Threading contract: forwarders report and forward threading" tags = [
    :calculus, :Threading, :Compose, :Sum, :Scale,
] setup = [TestUtils] begin
    using AbstractOperators, Random
    Random.seed!(0)

    n = 1 << 16
    threaded_leaf = FiniteDiff(Float64, (n,); threaded = true)
    serial_leaf = FiniteDiff(Float64, (n,); threaded = false)

    # A forwarder is threaded when any child is. Without this, `adapt_operator(op;
    # threaded=false)` would report the constraint satisfied and leave the child threaded
    # -- a silent nesting bug, which is exactly what the batching tests caught.
    @test is_threaded(Compose(DiagOp(randn(n - 1)), threaded_leaf)) == true
    @test is_threaded(Compose(DiagOp(randn(n - 1)), serial_leaf)) == false
    @test supports_threading(Compose(DiagOp(randn(n - 1)), serial_leaf)) == true

    # ...and adapting the forwarder actually reaches the child.
    op = Compose(DiagOp(randn(n - 1)), threaded_leaf)
    adapted = adapt_operator(op; threaded = false)
    @test is_threaded(adapted) == false
    @test all(!is_threaded, adapted.A)
    @test is_threaded(op) == true   # original untouched

    # Numerically unchanged by the adaptation.
    x = randn(n)
    @test op * x == adapted * x
end

@testitem "Threading contract: leaf operators without a threaded path" tags = [
    :linearoperator, :Threading, :Eye, :MatrixOp, :GetIndex, :ZeroPad, :Zeros,
] setup = [TestUtils] begin
    using AbstractOperators, Random
    Random.seed!(0)

    ops = (
        Eye(Float64, (16,)),
        MatrixOp(randn(8, 16)),
        GetIndex(Float64, (16,), (1:8,)),
        ZeroPad(Float64, (16,), (4,)),
        Zeros(Float64, (16,), Float64, (8,)),
    )
    for op in ops
        @test supports_threading(op) == false
        @test is_threaded(op) == false

        # `threaded` is accepted so forwarders can pass it down uniformly, and is vacuously
        # satisfied: there is no threaded path a copy could switch to, so adapt must share
        # rather than copy forever.
        @test adapt_operator(op; threaded = true) === op
        @test adapt_operator(op; threaded = false) === op

        # storage_type, by contrast, is honoured -- that is why these have a
        # `_copy_operator_impl` at all rather than falling back to deepcopy.
        c = copy_operator(op; storage_type = Array{Float64})
        @test domain_array_type(c) <: Array
    end
end

@testitem "Threading contract: batch operators disable threading in wrapped operators" tags = [
    :batching, :Threading, :SimpleBatchOp, :SpreadingBatchOp,
] setup = [TestUtils] begin
    using AbstractOperators, Random
    Random.seed!(0)

    if Threads.nthreads() > 1
        n = 1 << 16
        inner = FiniteDiff(Float64, (n,); threaded = true)
        @test is_threaded(inner) == true

        # The batch loop is the parallel layer, so every wrapped instance -- including the
        # first, which the pre-fix code left threaded -- must be serial.
        bop = BatchOp(inner, (4,); threaded = true)
        @test is_threaded(bop) == true
        @test all(!is_threaded, bop.operator)

        # Same for the spreading family.
        ops = [FiniteDiff(Float64, (n,); threaded = true) for _ in 1:3]
        sbop = BatchOp(ops, 2; threaded = true)
        @test all(!is_threaded, AbstractOperators._spreading_operators(sbop))

        # And the result is unchanged by any of it.
        serial_batch = BatchOp(FiniteDiff(Float64, (n,); threaded = false), (4,); threaded = false)
        x = randn(n, 4)
        @test bop * x == serial_batch * x
    end
end

@testitem "Threading contract: DCAT block-parallel loop" tags = [:calculus, :Threading, :DCAT] setup = [
    TestUtils,
] begin
    using AbstractOperators, LinearAlgebra, Random
    using RecursiveArrayTools: ArrayPartition
    Random.seed!(0)

    nb, bs = 8, 1 << 16
    blocks = [FiniteDiff(Float64, (bs,); threaded = false) for _ in 1:nb]

    serial = DCAT(blocks...; threaded = false)
    threaded = DCAT(blocks...; threaded = true)
    @test is_threaded(serial) == false
    @test is_threaded(threaded) == true
    @test supports_threading(serial) == true

    # Inputs captured once and shared by both operators.
    x = ArrayPartition([randn(bs) for _ in 1:nb]...)
    r = ArrayPartition([randn(bs - 1) for _ in 1:nb]...)

    # Blocks are independent and write disjoint outputs, so parallelising them must not
    # perturb a single bit.
    @test serial * x == threaded * x
    @test serial' * r == threaded' * r

    # copy_operator round-trips the block-loop flag.
    @test is_threaded(copy_operator(serial; threaded = true)) == true
    @test is_threaded(copy_operator(threaded; threaded = false)) == false

    # Nesting safety: when the block loop threads, the blocks themselves must not.
    nested = DCAT([FiniteDiff(Float64, (bs,); threaded = true) for _ in 1:nb]...; threaded = true)
    @test all(!is_threaded, nested.A)
end

@testitem "Threading contract: DCAT default policy tracks the benchmark" tags = [
    :calculus, :Threading, :DCAT,
] begin
    using AbstractOperators

    # These three cases are the measured boundary, not arbitrary sizes. Block-level
    # threading needs BOTH a large aggregate and enough blocks: 2 blocks x 2^18 has a 2^19
    # aggregate yet measured only 1.03x, while 8 x 2^16 (also 2^19) measured 1.9x.
    fd(n) = FiniteDiff(Float64, (n,); threaded = false)

    if Threads.nthreads() > 1
        @test is_threaded(DCAT([fd(1 << 16) for _ in 1:8]...)) == true    # measured 1.9x
        @test is_threaded(DCAT([fd(1 << 18) for _ in 1:2]...)) == false   # measured 1.03x
        @test is_threaded(DCAT([fd(1 << 12) for _ in 1:8]...)) == false   # measured 0.26x
    else
        # Single-threaded session: never thread the block loop, whatever the size.
        @test is_threaded(DCAT([fd(1 << 16) for _ in 1:8]...)) == false
    end
end

@testitem "Threading contract: batching an operator that has no threaded path" tags = [
    :batching, :Threading, :fftw, :dsp,
] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, DSPOperators, LinearAlgebra, Random
    Random.seed!(0)

    # Regression guard. FFTW/DSP operators are not thread-safe, so a threaded batch has to
    # make one private copy per thread; and they have no threaded path of their own, so
    # `threaded = false` asks nothing of them. The `copy_operator` fallback must therefore
    # accept that request rather than refuse it -- refusing broke batching for every
    # subpackage operator that owns scratch buffers.
    for op in (RDFT(Float64, (16,)), DCT(Float64, (16,)), Conv(Float64, (16,), randn(4)))
        @test is_thread_safe(op) == false
        @test supports_threading(op) == false
        @test copy_operator(op; threaded = false) isa typeof(op)

        if Threads.nthreads() > 1
            bop = BatchOp(op, (8,); threaded = true)
            @test is_threaded(bop) == true
        end
    end

    # A request the fallback genuinely cannot satisfy is still refused loudly.
    @test_throws ArgumentError copy_operator(RDFT(Float64, (16,)); storage_type = Array)
end

@testitem "Threading contract: subpackage operators declare their threading" tags = [
    :Threading, :fftw, :dsp, :nfft, :wavelets,
] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, DSPOperators, NFFTOperators, WaveletOperators
    using LinearAlgebra, Random
    Random.seed!(0)

    # FFTW is a *counted* thread pool: `threaded` picks a plan-time thread count, so it is
    # fixed at construction and `is_threaded` reads it back rather than switching a loop.
    serial_dft = DFT(Float64, (16,); threaded = false)
    threaded_dft = DFT(Float64, (16,); threaded = true)
    @test supports_threading(serial_dft) == true
    @test is_threaded(serial_dft) == false
    @test is_threaded(threaded_dft) == (Threads.nthreads() > 1)

    x = randn(16)
    @test serial_dft * x ≈ threaded_dft * x

    # `num_threads`, FFTW's own spelling, still works and still wins.
    @test is_threaded(DFT(Float64, (16,); num_threads = 1)) == false

    # Switching the flag has to replan, and must preserve the *domain* element type --
    # for a real-input DFT the codomain is complex, so replanning from the codomain would
    # silently produce an operator with the wrong domain.
    replanned = copy_operator(serial_dft; threaded = true)
    @test domain_type(replanned) == domain_type(serial_dft)
    @test replanned * x ≈ serial_dft * x

    # IDFT must stay an adjoint-wrapped DFT: that is what makes the DFT/IDFT combination
    # rules fire.
    @test IDFT(8) isa AbstractOperators.AdjointOperator
    @test AbstractOperators.can_be_combined(DFT(ComplexF64, 8), IDFT(8))

    # DSP and Wavelet operators have no Julia-level threaded path; they say so rather than
    # inheriting the default silently.
    for op in (Conv(Float64, (16,), randn(4)), Xcorr(Float64, (16,), randn(4)))
        @test supports_threading(op) == false
        @test is_threaded(op) == false
    end
end

@testitem "Threading contract: NFFTOp reports its plan-time threading" tags = [
    :Threading, :nfft,
] setup = [TestUtils] begin
    using AbstractOperators, NFFTOperators, LinearAlgebra, Random
    Random.seed!(0)

    traj = rand(2, 32, 8) .- 0.5
    dcf = rand(32, 8)
    op = NFFTOp((16, 16), traj, dcf; threaded = false)

    @test supports_threading(op) == true
    @test is_threaded(op) == false
    @test is_thread_safe(op) == false   # ksp_buffer is operator-owned scratch

    # A copy shares the immutable plan and dcf but gets its own scratch buffer -- sharing
    # that buffer is precisely what makes the operator unsafe to run from two threads.
    c = copy_operator(op)
    @test c.plan === op.plan
    @test c.ksp_buffer !== op.ksp_buffer
    x = rand(ComplexF64, 16, 16)
    @test op * x ≈ c * x

    # The plan is built for a fixed thread count and backend, so requests that would need a
    # different plan are refused rather than silently ignored.
    @test_throws ArgumentError copy_operator(op; threaded = true)
    @test_throws ArgumentError copy_operator(op; storage_type = Array)
end
