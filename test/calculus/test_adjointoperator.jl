if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)

@testset "AdjointOperator" begin
    verb && println(" --- Testing AdjointOperator --- ")

    m, n = 5, 7
    A1 =
        rand(m) * rand(n)' * 3.0 +
        rand(m) * rand(n)' * 2.0 +
        rand(m) * rand(n)' * 1.0 +
        0.01 * rand(m, n) # make it approximately rank 3
    opA1 = MatrixOp(A1)
    opA1t = MatrixOp(A1')
    opT = AdjointOperator(opA1)
    x1 = randn(m)
    y1 = test_op(opT, x1, randn(n), verb)
    y2 = A1' * x1
    @test norm(y1 - y2) <= 1e-12

    @test is_null(opT) == is_null(opA1t)
    @test is_eye(opT) == is_eye(opA1t)
    @test is_diagonal(opT) == is_diagonal(opA1t)
    @test is_AcA_diagonal(opT) == is_AcA_diagonal(opA1t)
    @test is_AAc_diagonal(opT) == is_AAc_diagonal(opA1t)
    @test is_orthogonal(opT) == is_orthogonal(opA1t)
    @test is_invertible(opT) == is_invertible(opA1t)
    @test is_full_row_rank(opT) == is_full_row_rank(opA1t)
    @test is_full_column_rank(opT) == is_full_column_rank(opA1t)

    d = randn(3)
    op = AdjointOperator(DiagOp(d))
    @test is_diagonal(op) == true
    @test diag(op) == d

    op = AdjointOperator(ZeroPad((10,), 5))
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test diag_AAc(op) == 1

    # Adjoint of adjoint returns original
    op2 = AdjointOperator(opT)
    @test op2 == opA1

    # Show/summary output
    io = IOBuffer()
    show(io, opT)
    str = String(take!(io))
    @test occursin("AdjointOperator", str) || occursin("ᵃ", str)

    # Error on nonlinear operator
    struct DummyNonlinearOp <: AbstractOperator end
    Base.size(::DummyNonlinearOp) = (2, 2)
    AbstractOperators.is_linear(::DummyNonlinearOp) = false
    @test_throws ErrorException AdjointOperator(DummyNonlinearOp())

    # Adjoint of adjoint returns original
    op2 = AdjointOperator(opT)
    @test op2 == opA1

    # storage & thread-safety propagation
    _dst = domain_storage_type(opT)
    _cst = codomain_storage_type(opT)
    @test _dst !== nothing && _cst !== nothing
    @test is_thread_safe(opT) == is_thread_safe(opA1)

    # size
    @test size(opT) == ((n,), (m,))

    # opnorm / estimate consistency (underlying matrix op)
    @test AbstractOperators.has_fast_opnorm(opT) == AbstractOperators.has_fast_opnorm(opA1)
    opnorm_opT = opnorm(opT)
    @test opnorm_opT ≈ opnorm(opA1) rtol=5e-6
    @test estimate_opnorm(opT) ≈ estimate_opnorm(opA1) rtol=0.05
    @test opnorm_opT ≈ estimate_opnorm(opT) rtol=0.05

    # For a diagonal operator, test diag_AcA matches original's diag_AAc
    d2 = randn(5)
    D = DiagOp(d2)
    AD = AdjointOperator(D)
    @test diag_AcA(AD) == diag_AAc(D)
    # Also diag_AAc(AD) == diag_AcA(D)
    @test diag_AAc(AD) == diag_AcA(D)

    # fun_name pattern (indirect via show)
    io2 = IOBuffer()
    show(io2, opT)
    str2 = String(take!(io2))
    @test occursin("ᵃ", str2)

    @test opA1' == opA1'
end
