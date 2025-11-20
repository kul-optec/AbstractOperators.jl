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
    @test !is_null(LM)
    @test !is_eye(LM)
    @test is_symmetric(LM)
    @test is_diagonal(LM)
    @test is_AcA_diagonal(LM)
    @test is_AAc_diagonal(LM)
    @test diag_AcA(LM) == d.^2
    @test diag_AAc(LM) == d.^2
    @test !is_orthogonal(LM)
    @test is_invertible(LM)
    @test is_full_row_rank(LM)
    @test is_full_column_rank(LM)
    @test is_positive_definite(LM) == all(d .> 0)
    @test is_positive_semidefinite(LM) == all(d .>= 0)

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
