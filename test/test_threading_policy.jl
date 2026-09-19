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

    # Sized above FiniteDiff's measured threshold so that `threaded = true` is a permission
    # the policy will actually grant -- below it, `true` is correctly declined and there
    # would be nothing for `adapt_operator` to change.
    n = 1 << 16
    op = FiniteDiff(Float64, (n,); threaded = false)
    @test is_threaded(op) == false

    # Constraint already satisfied -> the very same object, no allocation.
    @test adapt_operator(op) === op
    @test adapt_operator(op; threaded = false) === op
    @test adapt_operator(op; storage_type = Array{Float64}) === op

    # Constraint unmet -> a copy that actually satisfies it (the policy agrees at this size).
    threaded_copy = adapt_operator(op; threaded = true)
    @test is_threaded(threaded_copy) == (Threads.nthreads() > 1)
    # ...and the original is untouched.
    @test is_threaded(op) == false

    # Numerically identical: threading changes the schedule, not the arithmetic.
    x = randn(n)
    @test op * x == threaded_copy * x

    # require_thread_safe is an additional constraint, not a replacement.
    @test is_thread_safe(op) == true
    @test adapt_operator(op; require_thread_safe = true) === op
end

@testitem "threading policy: thresholds are per-operator, not per-cost-class" tags = [:misc, :Threading] setup = [TestUtils] begin
    using AbstractOperators
    using AbstractOperators: threading_threshold

    # `threading_threshold` dispatches on the operator *type*, so each operator carries its
    # own measured crossover. This test pins the property that motivated the design: the
    # measured values genuinely differ within a single cost class, so a shared constant
    # could not represent them.
    #
    # All of these are transcriptions from benchmark/operator_thresholds.jl.
    @test threading_threshold(SoftPlus) == 2^8      # earliest of any operator
    @test threading_threshold(Cos) == 2^9
    @test threading_threshold(Sin) == 2^10
    @test threading_threshold(Sigmoid) == 2^10

    # An 8x spread inside what the cost-class scheme called one class.
    @test threading_threshold(SoftPlus) < threading_threshold(Sin)

    # Pow splits on the exponent kind: `x^0.5` lowers to `exp(p*log(x))` and so crosses over
    # three powers of two earlier than integer `x^2`. A single method could not express this.
    @test threading_threshold(typeof(Pow(Float64, (4,), 2))) == 2^11
    @test threading_threshold(typeof(Pow(Float64, (4,), 0.5))) == 2^8

    # Memory-bound work crosses over far later than compute-bound work.
    @test threading_threshold(FiniteDiff) == 2^16
    @test threading_threshold(DiagOp) == 2^17
    @test threading_threshold(Scale) == 2^22
    @test threading_threshold(Sin) < threading_threshold(FiniteDiff) < threading_threshold(Scale)
end

@testitem "threading policy: consistent thresholds across constructors and operators" tags = [
    :misc, :Threading,
] setup = [TestUtils] begin
    using AbstractOperators
    using AbstractOperators: _should_thread, MIN_BATCH_WORK_FOR_PARALLEL
    using Random
    Random.seed!(0)

    # 1. Variation's two constructors must agree: both threshold on *elements*, so the
    #    same input size produces the same threading decision regardless of which
    #    constructor is used.
    @test is_threaded(Variation(Float64, (100, 100))) == is_threaded(Variation(zeros(100, 100)))
    @test is_threaded(Variation(Float64, (4, 4))) == is_threaded(Variation(zeros(4, 4)))

    # 2. `_should_thread(::AbstractOperator)` has a size component: a batch over a
    #    four-element operator must not thread, only a sufficiently large one should.
    @test _should_thread(Eye(4)) == false
    if Threads.nthreads() > 1
        @test _should_thread(Eye(MIN_BATCH_WORK_FOR_PARALLEL)) == true
    end

    # Scale's threading threshold sits at its measured 2^22 crossover, so a 1e5-element
    # operator (well below that) must stay serial.
    @test is_threaded(Scale(2.0, DiagOp(randn(100_000); threaded = false))) == false
end

@testitem "threading policy: _policy_storage reduces a multi-domain operator to a plain array type" tags = [
    :misc, :Threading,
] setup = [TestUtils] begin
    using AbstractOperators
    using AbstractOperators: _policy_storage, _storage_eltype_or_float, _should_thread
    using RecursiveArrayTools: ArrayPartition

    # A single-domain operator's storage is used as-is.
    @test _policy_storage(Array{Float64}) === Array{Float64}

    # A multi-domain operator reports an `ArrayPartition`, which `_should_thread`'s CPU-storage
    # check cannot use directly -- `_policy_storage` reduces it to a plain CPU array of the
    # partition's element type instead.
    @test _policy_storage(ArrayPartition{Float64, Tuple{Vector{Float64}, Vector{Float64}}}) === Array{Float64}
    @test _storage_eltype_or_float(ArrayPartition{Float32, Tuple{Vector{Float32}}}) == Float32
    # An unparameterized/unrelated type falls back to Float64.
    @test _storage_eltype_or_float(ArrayPartition) == Float64

    # Exercise the real call path: a multi-domain (HCAT) operator's `domain_array_type` is an
    # `ArrayPartition`, and `_should_thread` (used by batch operators) must still resolve to a
    # plain Bool rather than erroring on it.
    op = HCAT(MatrixOp(randn(4, 3)), MatrixOp(randn(4, 5)))
    @test domain_array_type(op) <: ArrayPartition
    @test _should_thread(op) isa Bool
end

@testitem "threading policy: threaded=true is a permission, threaded=false a veto" tags = [
    :misc, :Threading,
] setup = [TestUtils] begin
    using AbstractOperators, FFTWOperators, NFFTOperators, Random
    Random.seed!(0)

    # The package-wide rule, asserted across every family that takes the keyword. The
    # keyword is a Bool -- there is no third state:
    #
    #   false -> veto. Never threads. Nesting safety depends on this being absolute.
    #   true  -> permission (and the default). Threads only if the policy also agrees.
    #
    # `true` is checked below the operator's threshold, where the policy must decline it.
    tiny_builders = (
        (; threaded) -> Sin(Float64, (4,); threaded),
        (; threaded) -> FiniteDiff(Float64, (4,); threaded),
        (; threaded) -> Variation(Float64, (4, 4); threaded),
        (; threaded) -> DiagOp(randn(4); threaded),
        (; threaded) -> Scale(2.0, DiagOp(randn(4); threaded = false); threaded),
        (; threaded) -> DFT(Float64, (256,); threaded),
        (; threaded) -> DCT(Float64, (256,); threaded),
        (; threaded) -> RDFT(Float64, (256,); threaded),
        (; threaded) -> BatchOp(FiniteDiff(Float64, (4,); threaded = false), (8,); threaded),
    )
    for build in tiny_builders
        @test is_threaded(build(; threaded = true)) == false    # permission declined
        @test is_threaded(build(; threaded = false)) == false   # veto
    end

    # Above threshold, the permission is granted (given more than one thread).
    if Threads.nthreads() > 1
        big_builders = (
            (; threaded) -> Sin(Float64, (1 << 20,); threaded),
            (; threaded) -> FiniteDiff(Float64, (1 << 20,); threaded),
            (; threaded) -> DiagOp(randn(1 << 20); threaded),
            (; threaded) -> DFT(Float64, (1 << 16,); threaded),
            (; threaded) -> DCT(Float64, (1 << 16,); threaded),
            (; threaded) -> RDFT(Float64, (1 << 16,); threaded),
            (; threaded) -> BatchOp(FiniteDiff(Float64, (1 << 20,); threaded = false), (8,); threaded),
        )
        for build in big_builders
            @test is_threaded(build(; threaded = true)) == true
            # The veto still wins over any amount of work.
            @test is_threaded(build(; threaded = false)) == false
        end
    end

    # FFTW's own `num_threads` remains an explicit *command* rather than a permission: it is
    # the escape hatch for callers who know their workload better than the policy does.
    @test is_threaded(DFT(Float64, (256,); num_threads = 8)) == (Threads.nthreads() >= 1)
end

@testitem "threading policy: `threaded` is Bool-only in constructors" tags = [:misc, :Threading] setup = [TestUtils] begin
    using AbstractOperators

    # `nothing` is not part of the constructor API -- the keyword is `Bool` with `true` as
    # the default, so a stray `nothing` fails loudly rather than being reinterpreted.
    @test_throws TypeError Sin(Float64, (4,); threaded = nothing)
    @test_throws TypeError FiniteDiff(Float64, (4,); threaded = nothing)

    # `copy_operator`/`adapt_operator` are the exception, and deliberately so: there
    # `threaded` is a *constraint*, and `nothing` means "no constraint", exactly as it does
    # for `storage_type`. Without it a plain copy could not preserve an explicitly serial
    # operator -- it would re-derive threading from the policy instead.
    op = Sin(Float64, (1 << 20,); threaded = false)
    @test is_threaded(op) == false
    @test is_threaded(copy_operator(op)) == false
    @test adapt_operator(op) === op
end

@testitem "adapt_operator: require_thread_safe and storage_type mismatch trigger a copy" tags = [
    :misc, :Threading,
] setup = [TestUtils] begin
    using AbstractOperators, Random, JLArrays
    Random.seed!(0)

    # require_thread_safe = true on a non-thread-safe operator cannot be satisfied by
    # sharing, so it must fall through to `copy_operator`.
    op = LBFGS(zeros(4), 3)
    @test is_thread_safe(op) == false
    copied = adapt_operator(op; require_thread_safe = true)
    @test copied !== op

    # storage_type mismatch: `_storage_matches` compares wrapper families (Array vs
    # JLArray), so a request for a different family cannot be satisfied by sharing either.
    op2 = MatrixOp(randn(4, 4))
    @test domain_array_type(op2) <: Array
    adapted = adapt_operator(op2; storage_type = JLArray)
    @test adapted !== op2
    @test domain_array_type(adapted) <: JLArray

    # Multi-domain operator: `domain_array_type` is an `ArrayPartition`, whose storage is
    # never comparable to a single requested wrapper, so `_storage_matches` is always false
    # and a `storage_type` request always copies -- even when it names the family already
    # in use.
    op3 = HCAT(MatrixOp(randn(4, 3)), MatrixOp(randn(4, 5)))
    adapted3 = adapt_operator(op3; storage_type = Array{Float64})
    @test adapted3 !== op3
end
