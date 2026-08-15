@testmodule ThreadingContract begin
    using Test
    using AbstractOperators
    using LinearAlgebra
    using Random

    export test_threading_contract, test_copy_contract

    """
    	test_copy_contract(build; storage_type = Array{Float64})

    Assert the `copy_operator` / `adapt_operator` contract for the operator returned by the
    zero-argument `build`.

    Deliberately *not* asserted: `copy_operator(op) !== op`. Most operators here are
    immutable structs holding only dimensions, and Julia makes two immutable structs with
    identical fields `===` regardless of how they were constructed. Asserting object
    identity would therefore test the compiler's egality rules rather than this package's
    copy semantics — the same reason such checks were removed in 4839a08. What matters, and
    what is checked instead, is that mutable buffers are never shared.
    """
    function test_copy_contract(build; storage_type = Array{Float64})
        op = build()

        # A no-constraint copy reproduces behaviour.
        c = copy_operator(op)
        @test typeof(c) === typeof(op)

        # adapt_operator shares when nothing needs changing...
        @test adapt_operator(op) === op
        # ...and honours a storage request that already holds.
        @test adapt_operator(op; storage_type = storage_type) === op

        # Mutable working buffers must never be shared between an operator and its copy:
        # that is what makes a copy usable from another thread.
        for f in fieldnames(typeof(op))
            v = getfield(op, f)
            v isa AbstractArray || continue
            # Read-only operator data (a matrix, a diagonal) is intentionally shared; only
            # buffers named as scratch are required to be distinct.
            startswith(String(f), "buf") || continue
            @test getfield(c, f) !== v
        end
        return c
    end

    """
    	test_threading_contract(build, x; adjoint_input = nothing)

    Assert the threading contract for an operator constructor `build(; threaded)`:

    - the `threaded` keyword is accepted and round-trips through `is_threaded`
    - threaded and serial results are **numerically identical**, not merely approximate:
      threading changes the schedule, not the arithmetic, so any difference is a bug
    - `copy_operator(op; threaded = t)` produces an operator reporting `t`
    - `adapt_operator` copies when the threading constraint is unmet and shares otherwise

    `x` is captured into a variable by the caller and reused for both operators — comparing
    `op1 * randn(n)` against `op2 * randn(n)` would compare two different inputs.
    """
    function test_threading_contract(build, x; adjoint_input = nothing)
        serial = build(; threaded = false)
        threaded = build(; threaded = true)

        @test is_threaded(serial) == false
        @test is_threaded(threaded) == true
        @test supports_threading(serial) == true

        # Identical, not approximate.
        @test serial * x == threaded * x

        if adjoint_input !== nothing
            y = adjoint_input
            @test serial' * y == threaded' * y
        end

        # copy_operator round-trips the flag in both directions.
        @test is_threaded(copy_operator(serial; threaded = true)) == true
        @test is_threaded(copy_operator(threaded; threaded = false)) == false

        # adapt_operator: share when satisfied, copy when not.
        @test adapt_operator(serial; threaded = false) === serial
        adapted = adapt_operator(serial; threaded = true)
        @test is_threaded(adapted) == true
        @test is_threaded(serial) == false   # original untouched
        return nothing
    end
end
