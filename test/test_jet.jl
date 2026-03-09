using AbstractOperators
using JET
using Test

const AO = AbstractOperators

@testset "JET static analysis" begin
    JET.test_package(AbstractOperators; target_modules = (AbstractOperators,))

    @testset "@test_opt" begin
        n = 8
        d = randn(n)

        @test_opt target_modules = (AO,) DiagOp(d)
        @test_opt target_modules = (AO,) GetIndex((n,), 1:4)
        @test_opt target_modules = (AO,) Variation(n, 2)
    end

    @testset "@test_call" begin
        n = 8
        d = randn(n)

        @test_call target_modules = (AO,) DiagOp(d)
        @test_call target_modules = (AO,) GetIndex((n,), 1:4)
        @test_call target_modules = (AO,) Variation(n, 2)
        @test_call target_modules = (AO,) FiniteDiff((n,))
    end
end
