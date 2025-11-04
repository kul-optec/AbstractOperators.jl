@testset "LinearMapsExt" begin
    using AbstractOperators
    using LinearMaps

    # Real Diagonal
    d = rand(10, 10);
    D = DiagOp(d)
    LM = LinearMaps.LinearMap(D)
    x = rand(100);
    @test LM * x == vec(D * reshape(x, 10, 10))
    @test LM' * x == vec(D' * reshape(x, 10, 10))

    # Complex Diagonal
    d = rand(ComplexF64, 10, 10)
    D = DiagOp(d)
    LM = LinearMaps.LinearMap(D)
    x = rand(ComplexF64, 100)
    @test LM * x == vec(D * reshape(x, 10, 10))
    @test LM' * x == vec(D' * reshape(x, 10, 10))
end
