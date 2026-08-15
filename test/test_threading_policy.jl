@testitem "threading policy: default_threaded thresholds" tags = [:misc, :Threading] setup = [TestUtils] begin
    using AbstractOperators
    using AbstractOperators:
        default_threaded,
        threading_threshold,
        THRESHOLD_ELEMENTWISE_TRANSCENDENTAL,
        THRESHOLD_ELEMENTWISE_ARITHMETIC,
        THRESHOLD_MEMORY_BOUND,
        _is_cpu_storage,
        _total_elements

    # Thresholds are ordered by cost-per-element: the more work an element costs, the
    # earlier threading pays. This ordering is the whole reason there is no single global
    # constant, so assert it directly.
    @test THRESHOLD_ELEMENTWISE_TRANSCENDENTAL < THRESHOLD_ELEMENTWISE_ARITHMETIC
    @test THRESHOLD_ELEMENTWISE_ARITHMETIC < THRESHOLD_MEMORY_BOUND

    # default_threaded is a pure size/storage decision: below threshold false, at or above
    # true (given more than one thread).
    Op = typeof(FiniteDiff((4,)))
    thr = threading_threshold(Op)
    if Threads.nthreads() > 1
        @test default_threaded(Op, Float64, (thr,), Array{Float64}) == true
        @test default_threaded(Op, Float64, (thr - 1,), Array{Float64}) == false
    else
        # Single-threaded session: never thread, regardless of size.
        @test default_threaded(Op, Float64, (thr * 4,), Array{Float64}) == false
    end

    # GPU storage never gets Julia-level threading: the kernel is already parallel.
    @test _is_cpu_storage(Array{Float64}) == true
    @test _is_cpu_storage(Vector{Float32}) == true

    # Multi-domain sizes reduce to a total element count.
    @test _total_elements((2, 3)) == 6
    @test _total_elements(((2, 3), (4,))) == 10
    @test _total_elements(()) == 1
end

@testitem "threading policy: FastBroadcast bridge round-trips" tags = [:misc, :Threading] setup = [TestUtils] begin
    using AbstractOperators
    using AbstractOperators: _fbthread, _fbbool
    import FastBroadcast

    # DiagOp/Scale store FastBroadcast's singleton flag; everything else uses Bool. The
    # bridge must round-trip both ways or the trait and the kernel disagree.
    @test _fbthread(true) === FastBroadcast.True()
    @test _fbthread(false) === FastBroadcast.False()
    @test _fbbool(_fbthread(true)) == true
    @test _fbbool(_fbthread(false)) == false
    @test _fbbool(FastBroadcast.True) == true
    @test _fbbool(FastBroadcast.False) == false
end

@testitem "adapt_operator: shares when constraints hold, copies otherwise" tags = [:misc, :Threading] setup = [TestUtils] begin
    using AbstractOperators, Random
    Random.seed!(0)

    op = FiniteDiff(Float64, (64,); threaded = false)
    @test is_threaded(op) == false

    # Constraint already satisfied -> the very same object, no allocation.
    @test adapt_operator(op) === op
    @test adapt_operator(op; threaded = false) === op
    @test adapt_operator(op; storage_type = Array{Float64}) === op

    # Constraint unmet -> a copy that actually satisfies it.
    threaded_copy = adapt_operator(op; threaded = true)
    @test threaded_copy !== op
    @test is_threaded(threaded_copy) == true
    # ...and the original is untouched.
    @test is_threaded(op) == false

    # Numerically identical: threading changes the schedule, not the arithmetic.
    x = randn(64)
    @test op * x == threaded_copy * x

    # require_thread_safe is an additional constraint, not a replacement.
    @test is_thread_safe(op) == true
    @test adapt_operator(op; require_thread_safe = true) === op
end
