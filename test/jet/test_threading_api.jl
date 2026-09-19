# Standalone: julia --project=test test/jet/test_threading_api.jl
#
# JET coverage for the threading API introduced alongside `threading_policy.jl`. Kept in
# all three modes (@test_opt, @test_call, and the package-level check in test_package.jl)
# because the policy functions take type arguments, which is exactly where unparameterized
# `::Type` keywords would silently introduce runtime dispatch.
@testitem "@test_opt threading policy" tags = [:jet, :Threading] begin
    using JET, AbstractOperators
    const AO = AbstractOperators
    n = 8
    x = randn(n)

    # `default_threaded` / `_default_threaded` route the storage type through typed
    # positional parameters rather than a `storage_type::Type` keyword. If that ever
    # regresses to an unparameterized keyword, this is what catches it.
    @test_opt target_modules = (AO,) AO.default_threaded(
        typeof(FiniteDiff((n,))), Float64, (n,), Array{Float64}
    )
    @test_opt target_modules = (AO,) AO.threading_threshold(typeof(FiniteDiff((n,))))
    @test_opt target_modules = (AO,) AO._total_elements((4, 5))

    # Constructors with the `threaded` keyword.
    @test_opt target_modules = (AO,) FiniteDiff(Float64, (n,); threaded = true)
    @test_opt target_modules = (AO,) FiniteDiff(Float64, (n,); threaded = false)
    @test_opt target_modules = (AO,) Sin(Float64, (n,); threaded = true)
    @test_opt target_modules = (AO,) Pow(Float64, (n,), 2; threaded = true)
    @test_opt target_modules = (AO,) Sigmoid(Float64, (n,), 1.0; threaded = true)

    # Traits.
    @test_opt target_modules = (AO,) is_threaded(FiniteDiff((n,)))
    @test_opt target_modules = (AO,) supports_threading(FiniteDiff((n,)))
    @test_opt target_modules = (AO,) is_threaded(Compose(Eye(n), DiagOp(randn(n))))
end

@testitem "@test_call threading policy" tags = [:jet, :Threading] begin
    using JET, AbstractOperators, LinearAlgebra
    const AO = AbstractOperators
    n = 1 << 13
    x = randn(n)
    y = zeros(n - 1)

    @test_call target_modules = (AO,) FiniteDiff(Float64, (n,); threaded = true)
    @test_call target_modules = (AO,) is_threaded(FiniteDiff((n,)))
    @test_call target_modules = (AO,) adapt_operator(FiniteDiff((n,)); threaded = false)
    @test_call target_modules = (AO,) copy_operator(FiniteDiff((n,)); threaded = true)

    # Both threaded and serial `mul!` paths, forward and adjoint.
    for threaded in (false, true)
        op = FiniteDiff(Float64, (n,); threaded)
        @test_call target_modules = (AO,) mul!(y, op, x)
        @test_call target_modules = (AO,) mul!(zeros(n), op', randn(n - 1))
    end

    xs = randn(n)
    for threaded in (false, true)
        s = Sin(Float64, (n,); threaded)
        @test_call target_modules = (AO,) mul!(similar(xs), s, xs)
    end
end
