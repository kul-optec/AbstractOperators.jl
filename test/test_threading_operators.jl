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

@testitem "Threading contract: SoftMax" tags = [
    :nonlinearoperator, :Threading, :SoftMax,
] setup = [TestUtils] begin
    using AbstractOperators, Random
    Random.seed!(0)

    # Large enough to clear SoftMax's measured threading threshold (2^12).
    n = 1 << 15
    x = randn(n)
    r = randn(n)

    serial = SoftMax(Float64, (n,); threaded = false)
    threaded = SoftMax(Float64, (n,); threaded = true)
    @test is_threaded(serial) == false
    @test is_threaded(threaded) == true
    @test supports_threading(serial) == true
    @test supports_threading(threaded) == true

    # Not `==`: `mul!` is two reductions (max, then sum) around an elementwise `exp`, and
    # Polyester's `@batch reduction=...` sums in a different order than the serial `sum`,
    # so floating-point association differs even though the arithmetic is the same.
    @test serial * x ≈ threaded * x
    @test Jacobian(serial, x)' * r ≈ Jacobian(threaded, x)' * r

    # copy_operator round-trips the flag in both directions.
    @test is_threaded(copy_operator(serial; threaded = true)) == true
    @test is_threaded(copy_operator(threaded; threaded = false)) == false

    # adapt_operator: share when satisfied, copy when not.
    @test adapt_operator(serial; threaded = false) === serial
    adapted = adapt_operator(serial; threaded = true)
    @test is_threaded(adapted) == true
    @test is_threaded(serial) == false   # original untouched

    # A copy gets its own scratch buffer -- sharing it is what makes SoftMax unsafe to
    # share between threads in the first place, independent of whether a single `mul!`
    # call itself threads.
    c = copy_operator(serial)
    @test c.buf !== serial.buf
    @test is_thread_safe(serial) == false
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

    # The inner loop must not allocate temporaries per call -- an allocating inner loop is
    # a defect, not a style preference.
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
    #
    # The DiagOp is pinned to `threaded = false` rather than left to the policy: this
    # asserts the *forwarding rule*, and should not start failing because DiagOp's measured
    # threshold moved.
    diag = DiagOp(randn(n - 1); threaded = false)
    @test is_threaded(Compose(diag, threaded_leaf)) == true
    @test is_threaded(Compose(diag, serial_leaf)) == false
    @test supports_threading(Compose(diag, serial_leaf)) == true

    # ...and adapting the forwarder actually reaches the child.
    op = Compose(diag, threaded_leaf)
    adapted = adapt_operator(op; threaded = false)
    @test is_threaded(adapted) == false
    @test all(!is_threaded, adapted.A)
    @test is_threaded(op) == true   # original untouched

    # Numerically unchanged by the adaptation.
    x = randn(n)
    @test op * x == adapted * x
end

@testitem "Threading contract: leaf operators without a threaded path" tags = [
    :linearoperator, :Threading, :Eye, :GetIndex, :ZeroPad, :Zeros,
] setup = [TestUtils] begin
    using AbstractOperators, Random
    Random.seed!(0)

    ops = (
        Eye(Float64, (16,)),
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

@testitem "Threading contract: MatrixOp/LMatrixOp real BLAS threading" tags = [
    :linearoperator, :Threading, :MatrixOp, :LMatrixOp,
] setup = [TestUtils] begin
    using AbstractOperators, Random
    Random.seed!(0)

    # Large enough that BLAS actually threads (see MatrixOp.jl's benchmark note).
    A = randn(2000, 3000)
    serial = MatrixOp(A; threaded = false)
    threaded = MatrixOp(A; threaded = true)
    @test supports_threading(serial) == true
    @test is_threaded(serial) == false
    @test is_threaded(threaded) == true

    x = randn(3000)
    b = randn(2000)
    # Forward (gemv) splits by output row, so each element is one thread's own dot
    # product -- bit-identical regardless of thread count.
    @test serial * x == threaded * x
    # Adjoint (gemv on A') is not: BLAS reassociates the transposed-access reduction
    # differently across thread counts, so only an approximate match holds.
    @test serial' * b ≈ threaded' * b

    # copy_operator round-trips the flag in both directions.
    @test is_threaded(copy_operator(serial; threaded = true)) == true
    @test is_threaded(copy_operator(threaded; threaded = false)) == false

    # adapt_operator: share when satisfied, copy when not.
    @test adapt_operator(serial; threaded = false) === serial
    adapted = adapt_operator(serial; threaded = true)
    @test is_threaded(adapted) == true
    @test is_threaded(serial) == false   # original untouched

    # LMatrixOp mirrors MatrixOp's BLAS-backed gemm path.
    bmat = randn(3000, 200)
    lserial = LMatrixOp(bmat, 2000; threaded = false)
    lthreaded = LMatrixOp(bmat, 2000; threaded = true)
    @test supports_threading(lserial) == true
    @test is_threaded(lserial) == false
    @test is_threaded(lthreaded) == true
    X = randn(2000, 3000)
    @test lserial * X == lthreaded * X
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

    # Regression guard, in two flavours. Both kinds of operator are not thread-safe, so a
    # threaded batch has to make one private copy per thread -- and that copy request must
    # succeed. Refusing it broke batching for every subpackage operator with scratch
    # buffers.
    #
    # Filt has no threaded path at all (IIR filtering is a sequential recursion, not an
    # FFT), so `threaded = false` is vacuous for it and the deepcopy fallback answers the
    # request.
    for op in (Filt(4096, randn(4)),)
        @test is_thread_safe(op) == false
        @test supports_threading(op) == false
        @test copy_operator(op; threaded = false) isa typeof(op)

        if Threads.nthreads() > 1
            @test is_threaded(BatchOp(op, (8,); threaded = true)) == true
        end
    end

    # RDFT/DCT/Conv/Xcorr *do* have a threaded path (FFTW threads their plans), so they
    # answer the same request through their own `_copy_operator_impl` instead, replanning
    # if needed. Sized above MIN_BATCH_WORK_FOR_PARALLEL: the batch gate now has a size
    # component, so a batch over a 16-element operator deliberately stays serial (see the
    # gate test below).
    h = randn(4)
    for op in (
            RDFT(Float64, (4096,); threaded = false), DCT(Float64, (4096,); threaded = false),
            Conv(Float64, (4096,), h; threaded = false), Xcorr(Float64, (4096,), h; threaded = false),
        )
        @test is_thread_safe(op) == false
        @test supports_threading(op) == true
        @test is_threaded(op) == false
        @test copy_operator(op; threaded = false) isa typeof(op)

        if Threads.nthreads() > 1
            @test is_threaded(BatchOp(op, (8,); threaded = true)) == true
        end
    end

    # The batch gate has a size component: a batch over a tiny operator stays serial rather
    # than paying `@budgeted_threads` setup to parallelise microseconds of work.
    if Threads.nthreads() > 1
        @test is_threaded(BatchOp(Filt(16, randn(4)), (8,); threaded = true)) == false
    end

    # RDFT has no persistent data beyond its plans, so a storage_type request is
    # satisfiable by replanning against a same-shaped prototype on the new backend.
    rdft_copy = copy_operator(RDFT(Float64, (16,)); storage_type = Array)
    @test domain_array_type(rdft_copy) <: Array
end

@testitem "Threading contract: subpackage operators declare their threading" tags = [
    :Threading, :fftw, :dsp, :nfft, :wavelets,
] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, DSPOperators, NFFTOperators, WaveletOperators
    using LinearAlgebra, Random
    Random.seed!(0)

    # FFTW is a *counted* thread pool: `threaded` picks a plan-time thread count, so it is
    # fixed at construction and `is_threaded` reads it back rather than switching a loop.
    # Above the measured c2c crossover (2^13), so `threaded = true` is granted; below it the
    # policy declines and `is_threaded` would correctly report false.
    serial_dft = DFT(Float64, (1 << 14,); threaded = false)
    threaded_dft = DFT(Float64, (1 << 14,); threaded = true)
    @test supports_threading(serial_dft) == true
    @test is_threaded(serial_dft) == false
    @test is_threaded(threaded_dft) == (Threads.nthreads() > 1)

    x = randn(1 << 14)
    @test serial_dft * x ≈ threaded_dft * x

    # `num_threads`, FFTW's own spelling, still works and still wins.
    @test is_threaded(DFT(Float64, (1 << 14,); num_threads = 1)) == false

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

    # `Conv`/`Xcorr` plan their own FFTW transforms (DSPOperators does not depend on
    # DSP.jl), so they follow the same counted-thread-pool contract as DFT above: below
    # the size policy's threshold `threaded = true` is declined, above it it is granted.
    for ctor in (
            (dim, h; kw...) -> Conv(Float64, dim, h; kw...),
            (dim, h; kw...) -> Xcorr(Float64, dim, h; kw...),
        )
        local h_small = randn(4)
        local small_serial = ctor((16,), h_small; threaded = false)
        local small_threaded = ctor((16,), h_small; threaded = true)
        @test supports_threading(small_serial) == true
        @test is_threaded(small_serial) == false
        @test is_threaded(small_threaded) == false   # below threshold: declined

        local h_big = randn(4)
        local big_serial = ctor((1 << 14,), h_big; threaded = false)
        local big_threaded = ctor((1 << 14,), h_big; threaded = true)
        @test is_threaded(big_threaded) == (Threads.nthreads() > 1)
        local xb = randn(1 << 14)
        @test big_serial * xb ≈ big_threaded * xb

        local replanned = copy_operator(big_serial; threaded = true)
        @test is_threaded(replanned) == (Threads.nthreads() > 1)
        @test replanned * xb ≈ big_serial * xb
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

    # The plan is built for a fixed thread count and backend, so a request for either means
    # replanning -- recovering the trajectory from the existing plan (`plan.k`) and dcf
    # rather than refusing, since both are still available.
    threaded_copy = copy_operator(op; threaded = true)
    @test is_threaded(threaded_copy) == (Threads.nthreads() > 1)
    @test threaded_copy * x ≈ op * x

    storage_copy = copy_operator(op; storage_type = Array)
    @test domain_array_type(storage_copy) <: Array
    @test storage_copy * x ≈ op * x
end

@testitem "Threading contract: FFTW r2r/r2c transforms thread their plans" tags = [
    :Threading, :fftw, :DCT, :IDCT, :RDFT, :IRDFT,
] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, Random
    Random.seed!(0)

    # FFTW threads these transform kinds -- measured at n = 2^22 with 8 threads:
    # r2c (RDFT) 3.08x, r2r forward (DCT) 2.25x, r2r inverse (IDCT) 1.91x. That is why they
    # carry a plan-time thread count rather than being declared unthreaded.
    # Above every FFTW crossover measured for these kinds (c2c 2^13, r2r 2^15, r2c 2^15),
    # so `threaded = true` is a permission the policy grants.
    n = 1 << 16
    x = randn(n)
    z = randn(ComplexF64, n ÷ 2 + 1)

    for (serial, threaded, inp) in (
            (DCT(Float64, (n,); threaded = false), DCT(Float64, (n,); threaded = true), x),
            (IDCT(Float64, (n,); threaded = false), IDCT(Float64, (n,); threaded = true), x),
            (RDFT(Float64, (n,); threaded = false), RDFT(Float64, (n,); threaded = true), x),
            (
                IRDFT(ComplexF64, (n ÷ 2 + 1,), n; threaded = false),
                IRDFT(ComplexF64, (n ÷ 2 + 1,), n; threaded = true), z,
            ),
        )
        @test supports_threading(serial) == true
        @test is_threaded(serial) == false
        @test is_threaded(threaded) == (Threads.nthreads() > 1)

        # Threading an FFT changes the schedule inside FFTW, not the transform, but FFTW is
        # free to reassociate, so `≈` rather than `==`.
        @test serial * inp ≈ threaded * inp

        # Round-tripping the flag replans and stays numerically equivalent.
        replanned = copy_operator(serial; threaded = true)
        @test is_threaded(replanned) == (Threads.nthreads() > 1)
        @test replanned * inp ≈ serial * inp
        @test domain_type(replanned) == domain_type(serial)

        # A plain copy preserves the operator's own (serial) thread count.
        @test is_threaded(copy_operator(serial)) == false
    end

    # Scratch buffers are never shared with a copy.
    d = DCT(Float64, (n,); threaded = false)
    @test copy_operator(d).buf !== d.buf
    r = RDFT(Float64, (n,); threaded = false)
    @test copy_operator(r).b2 !== r.b2
end

@testitem "Threading contract: VCAT-forward and HCAT-adjoint block loops" tags = [
    :calculus, :Threading, :VCAT, :HCAT,
] setup = [TestUtils] begin
    using AbstractOperators, LinearAlgebra, Random
    using RecursiveArrayTools: ArrayPartition
    const AO = AbstractOperators
    Random.seed!(0)

    nb, bs = 8, 1 << 16
    blocks = [FiniteDiff(Float64, (bs,); threaded = false) for _ in 1:nb]

    # VCAT threads its *forward* direction (disjoint output blocks); the adjoint
    # accumulates into one shared `y` and stays serial. HCAT is the mirror image.
    V = VCAT(blocks...)
    vs = AO._copy_operator_impl(V; threaded = false)
    vt = AO._copy_operator_impl(V; threaded = true)
    @test AO.is_block_threaded(vs) == false
    @test AO.is_block_threaded(vt) == true
    @test is_threaded(vs) == false
    @test is_threaded(vt) == true
    @test supports_threading(vs) == true

    x = randn(bs)
    r = ArrayPartition([randn(bs - 1) for _ in 1:nb]...)
    @test vs * x == vt * x
    @test vs' * r == vt' * r

    H = HCAT(blocks...)
    hs = AO._copy_operator_impl(H; threaded = false)
    ht = AO._copy_operator_impl(H; threaded = true)
    xh = ArrayPartition([randn(bs) for _ in 1:nb]...)
    rh = randn(bs - 1)
    @test hs * xh == ht * xh
    @test hs' * rh == ht' * rh
    # `bs` sits below HCAT's own (higher) threshold, so `ht` above is not actually
    # block-threaded -- confirmed by the block-threshold tests below. Exercise the threaded
    # adjoint kernel itself (`_hcat_block_adj!`) with blocks sized to cross it.
    big_bs = 1 << 17
    big_blocks = [FiniteDiff(Float64, (big_bs,); threaded = false) for _ in 1:4]
    H_big = AO._copy_operator_impl(HCAT(big_blocks...); threaded = true)
    @test AO.is_block_threaded(H_big) == true
    H_big_serial = AO._copy_operator_impl(H_big; threaded = false)
    xhb = ArrayPartition([randn(big_bs) for _ in 1:4]...)
    rhb = randn(big_bs - 1)
    @test H_big * xhb == H_big_serial * xhb
    @test H_big' * rhb == H_big_serial' * rhb
    # Also exercise the natural-vs-indexed branch of `_hcat_block_adj!` through a permuted
    # (non-natural) index HCAT.
    p = [2, 1, 4, 3]
    H_big_permuted = AO.permute(H_big, p)
    @test AO.is_block_threaded(H_big_permuted) == true
    expected = (H_big_serial' * rhb).x[p]
    @test collect(H_big_permuted' * rhb) == collect(ArrayPartition(expected...))

    # Nesting safety: a threaded block loop must not nest each block's own threading
    # inside it. VCAT stores two block copies -- `A` (natural, used by the always-serial
    # adjoint) and `A_par` (forced serial, used only by the threaded forward loop) -- so
    # it's `A_par`, not `A`, that must be all-serial; `A` legitimately stays threaded since
    # nothing nests under the adjoint direction.
    nested = AO._copy_operator_impl(
        VCAT([FiniteDiff(Float64, (bs,); threaded = true) for _ in 1:nb]...); threaded = true
    )
    @test all(is_threaded, nested.A)
    @test all(!is_threaded, nested.A_par)
end

@testitem "Threading contract: block thresholds are per-operator and measured" tags = [
    :calculus, :Threading, :VCAT, :HCAT, :DCAT,
] begin
    using AbstractOperators
    const AO = AbstractOperators

    # HCAT's adjoint carries per-block ArrayPartition indexing the others do not, so it
    # needs more work per block before threading pays. These are the measured boundaries,
    # not round numbers: at 4 blocks of 2^16, VCAT-forward measures 1.32x but HCAT-adjoint
    # measures 0.76x -- a regression. One shared constant could not express both.
    @test AO.block_threading_threshold(VCAT) < AO.block_threading_threshold(HCAT)
    @test AO.block_threading_threshold(DCAT) == AO.block_threading_threshold(VCAT)

    fd(n) = FiniteDiff(Float64, (n,); threaded = false)

    if Threads.nthreads() > 1
        # 4 blocks x 2^16: VCAT wins (1.32x), HCAT loses (0.76x).
        @test AO.is_block_threaded(VCAT([fd(1 << 16) for _ in 1:4]...)) == true
        @test AO.is_block_threaded(HCAT([fd(1 << 16) for _ in 1:4]...)) == false
        # 4 blocks x 2^17: both win (1.32x / 1.27x).
        @test AO.is_block_threaded(HCAT([fd(1 << 17) for _ in 1:4]...)) == true
        # 16 blocks x 2^14: both lose (0.85x / 0.57x) despite a 2^18 aggregate -- which is
        # why the threshold is on per-block work, not on the total.
        @test AO.is_block_threaded(VCAT([fd(1 << 14) for _ in 1:16]...)) == false
        @test AO.is_block_threaded(HCAT([fd(1 << 14) for _ in 1:16]...)) == false
        # Too few blocks, however large: DCAT measures 1.03x at 2 blocks of 2^18.
        @test AO.is_block_threaded(DCAT([fd(1 << 18) for _ in 1:2]...)) == false
    end
end
