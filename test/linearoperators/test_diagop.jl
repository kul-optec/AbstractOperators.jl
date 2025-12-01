if !isdefined(Main, :verb)
    const verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)

@testset "DiagOp" begin
    verb && println(" --- Testing DiagOp --- ")

    n = 4
    d = randn(n)
    op = DiagOp(Float64, (n,), d)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = d .* x1

    @test all(norm.(y1 .- y2) .<= 1e-12)

    n = 4
    d = randn(n) + im * randn(n)
    op = DiagOp(Float64, (n,), d)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n) .+ im * randn(n), verb)
    y2 = d .* x1

    @test all(norm.(y1 .- y2) .<= 1e-12)

    n = 4
    d = pi
    op = DiagOp(Float64, (n,), d)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = d .* x1

    @test all(norm.(y1 .- y2) .<= 1e-12)

    n = 4
    d = im
    op = DiagOp(Float64, (n,), d)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n) + im * randn(n), verb)
    @test domain_type(op) == Float64
    @test codomain_type(op) == Complex{Float64}
    y2 = d .* x1

    @test all(norm.(y1 .- y2) .<= 1e-12)

    # other constructors
    d = randn(4)
    op = DiagOp(d)

    d = randn(4) .+ im
    op = DiagOp(d)

    n = 4
    d = pi
    op = DiagOp((n,), d)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == true
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(DiagOp(ones(10))) == true
    @test is_invertible(DiagOp([ones(5); 0])) == false
    @test is_full_row_rank(op) == true
    @test is_full_row_rank(DiagOp([ones(5); 0])) == false
    @test is_full_column_rank(op) == true
    @test is_full_column_rank(DiagOp([ones(5); 0])) == false

    @test diag(op) == d
    @test norm(op' * (op * x1) .- diag_AcA(op) .* x1) <= 1e-12
    @test norm(op * (op' * x1) .- diag_AAc(op) .* x1) <= 1e-12

    n = 4
    d = pi
    op = DiagOp((n,), d)
    x1 = randn(n)

    @test diag(op) == d
    @test norm(op' * (op * x1) .- diag_AcA(op) .* x1) <= 1e-12
    @test norm(op * (op' * x1) .- diag_AAc(op) .* x1) <= 1e-12

    # Scale: create scaled operator and verify diagonal scaling and properties
    op_scaled = Scale(3.0, op)
    @test diag(op_scaled) == 3.0 .* diag(op)
    @test size(op_scaled) == size(op)

    # get_normal_op: should produce diagonal with abs2 of original diag
    normal_op = AbstractOperators.get_normal_op(op)
    @test diag(normal_op) == abs2.(diag(op))
    @test is_diagonal(normal_op) == true

    # storage and type related helpers
    @test is_thread_safe(op) == true
    @test AbstractOperators.has_fast_opnorm(op) == true
    @test opnorm(op) == maximum(abs, diag(op))
    @test estimate_opnorm(op) == maximum(abs, diag(op))
    @test AbstractOperators.has_optimized_normalop(op) == true
    @test AbstractOperators.has_optimized_normalop(op') == true

    # invertibility false path (contains a zero)
    op_sing = DiagOp([1.0, 0.0, 2.0, 3.0])
    @test is_invertible(op_sing) == false
    @test is_full_row_rank(op_sing) == false
    @test is_full_column_rank(op_sing) == false

    # size returns (domain_dim, domain_dim)
    @test size(op) == ((n,), (n,))
end
