# JET static analysis for ContourletOperators mul! methods
@testitem "@test_opt ContourletOperators mul!" tags = [:jet, :contourlet] begin
    using JET, Contourlets, ContourletOperators, AbstractOperators, RecursiveArrayTools

    n = 16
    params = ContourletParams(J = 2, L_array = [1, 2])

    let x = randn(n, n), op = ContourletOp(Float64, params, (n, n))
        y = op * x
        @test_opt target_modules = (ContourletOperators,) mul!(y, op, x)
        @test_opt target_modules = (ContourletOperators,) mul!(x, AdjointOperator(op), y)
    end

    let x = randn(n, n), op = NSCTOp(Float64, params, (n, n))
        y = op * x
        @test_opt target_modules = (ContourletOperators,) mul!(y, op, x)
        @test_opt target_modules = (ContourletOperators,) mul!(x, AdjointOperator(op), y)
    end
end

# JET call analysis for ContourletOperators constructors
@testitem "@test_call ContourletOperators" tags = [:jet, :contourlet] begin
    using JET, Contourlets, ContourletOperators
    n = 16
    params = ContourletParams(J = 2, L_array = [1, 2])

    @test_call target_modules = (ContourletOperators,) ContourletOp(Float64, params, (n, n))
    @test_call target_modules = (ContourletOperators,) NSCTOp(Float64, params, (n, n))
    @test_call target_modules = (ContourletOperators,) ContourletOp(
        params, (n, n); array_type = Array{Float32}
    )
    @test_call target_modules = (ContourletOperators,) NSCTOp(
        params, (n, n); array_type = Array{Float32}
    )
end

# JET package-level analysis for ContourletOperators
@testitem "JET test_package ContourletOperators" tags = [:jet, :contourlet] begin
    using JET, ContourletOperators
    JET.test_package(ContourletOperators; target_modules = (ContourletOperators,))
end
