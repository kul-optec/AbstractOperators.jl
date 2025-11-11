if !isdefined(Main, :verb)
    const verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end

@testset "Eye" begin
    verb && println(" --- Testing Eye --- ")

    n = 4
    op = Eye(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)

    @test all(norm.(y1 .- x1) .<= 1e-12)

    # other constructors
    op = Eye(Float64, (n,))
    op = Eye((n,))
    op = Eye(n)
    op = Eye(x1)

    # properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == true
    @test is_diagonal(op) == true
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == true
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true

    @test diag(op) == 1
    @test diag_AcA(op) == 1
    @test diag_AAc(op) == 1

    # Different element types
    opC = Eye(ComplexF64, (n,))
    xC = randn(ComplexF64, n)
    @test all(opC * xC .== xC)
    opI = Eye(Int, (n,))
    xI = rand(1:10, n)
    @test all(opI * xI .== xI)

    # Multi-dimensional
    op3 = Eye(Float64, (2,3,4))
    x3 = randn(2,3,4)
    @test all(op3 * x3 .== x3)
    @test size(op3) == ((2,3,4), (2,3,4))

    # Storage type helpers
    @test domain_type(op) == Float64
    @test codomain_type(op) == Float64
    @test domain_storage_type(op) == Array{Float64}
    @test codomain_storage_type(op) == Array{Float64}
    @test is_thread_safe(op) == true

    # Adjoint, opnorm, get_normal_op
    @test AdjointOperator(op) === op
    @test LinearAlgebra.opnorm(op) == 1.0
    @test LinearAlgebra.opnorm(op) == estimate_opnorm(op)
    @test AbstractOperators.has_optimized_normalop(op) == true
    @test AbstractOperators.get_normal_op(op) === op

    # In-place mul!
    y = similar(x1)
    fill!(y, 0)
    mul!(y, op, x1)
    @test all(y .== x1)

    # Symmetry
    @test is_symmetric(op) == true
end
