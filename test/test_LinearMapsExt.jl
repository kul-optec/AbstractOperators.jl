@testset "LinearMapsExt" begin
    using AbstractOperators
    using LinearMaps
    using LinearAlgebra

    # Real Diagonal
    d = rand(10, 10);
    D = DiagOp(d)
    LM = LinearMaps.LinearMap(D)
    x = rand(100);
    @test LM * x == vec(D * reshape(x, 10, 10))
    @test LM' * x == vec(D' * reshape(x, 10, 10))
    @test issymmetric(LM)
    @test ishermitian(LM)
    @test is_linear(LM)
    @test is_invertible(LM)
    @test is_diagonal(LM)
    @test !is_eye(LM)

    # Complex Diagonal
    d = rand(ComplexF64, 10, 10)
    D = DiagOp(d)
    LM = LinearMaps.LinearMap(D)
    x = rand(ComplexF64, 100)
    @test LM * x == vec(D * reshape(x, 10, 10))
    @test LM' * x == vec(D' * reshape(x, 10, 10))
    @test !issymmetric(LM)
    @test ishermitian(LM)
    @test is_linear(LM)
    @test is_invertible(LM)
    @test is_diagonal(LM)
    @test !is_eye(LM)
end
