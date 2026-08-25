# Note: with the default biorthogonal (CDF97/Q2345) filters, ContourletOp/NSCTOp are
# *not* self-adjoint. `'` is wired to the declared inverse transform (perfect-reconstruction
# left inverse), not the literal linear-algebra transpose, so these tests verify round-trip
# reconstruction and in-place/shape/type correctness directly instead of using
# `TestUtils.test_op` (which asserts the true adjoint dot-product invariant).

@testitem "ContourletOp" tags = [:contourlet, :ContourletOp] setup = [TestUtils] begin
    using Contourlets, ContourletOperators, LinearAlgebra, Random, AbstractOperators

    n = 32
    params = ContourletParams(J = 2, L_array = [1, 2])
    op = ContourletOp(Float64, params, (n, n))

    @test domain_type(op) == Float64
    @test domain_array_type(op) == Array{Float64}
    @test codomain_array_type(op) <: ArrayPartition
    @test is_invertible(op)

    x1 = randn(n, n)
    y = op * x1
    @test y isa ArrayPartition
    @test length(y.x) == length(op.band_sizes)
    @test size(y.x[1]) == op.band_sizes[1]

    # preallocated in-place forward matches allocating forward
    y2 = similar(y)
    mul!(y2, op, x1)
    assert_cpu_approx(y, y2; atol = 1.0e-12)

    y3 = ct_forward(x1, params)
    @test norm(y.x[1] .- y3.coarse) <= 1.0e-10
    flat_subbands3 = vcat(y3.subbands...)
    @test all(norm.(y.x[2:end] .- flat_subbands3) .<= 1.0e-10)

    # preallocated in-place adjoint (inverse) matches allocating adjoint
    x_rec = op' * y
    x_rec2 = similar(x_rec)
    mul!(x_rec2, AdjointOperator(op), y)
    assert_cpu_approx(x_rec, x_rec2; atol = 1.0e-12)

    # perfect reconstruction
    assert_cpu_approx(x_rec, x1)
end

@testitem "NSCTOp" tags = [:contourlet, :NSCTOp] setup = [TestUtils] begin
    using Contourlets, ContourletOperators, LinearAlgebra, Random, AbstractOperators

    n = 32
    params = ContourletParams(J = 2, L_array = [1, 2])
    op = NSCTOp(Float64, params, (n, n))

    @test domain_type(op) == Float64
    @test domain_array_type(op) == Array{Float64}
    @test codomain_array_type(op) <: ArrayPartition
    @test is_invertible(op)
    @test all(sz == (n, n) for sz in op.band_sizes)

    x1 = randn(n, n)
    y = op * x1
    @test y isa ArrayPartition
    @test length(y.x) == length(op.band_sizes)

    y2 = similar(y)
    mul!(y2, op, x1)
    assert_cpu_approx(y, y2; atol = 1.0e-12)

    y3 = nsct_forward(x1, params)
    @test norm(y.x[1] .- y3.coarse) <= 1.0e-10
    flat_subbands3 = vcat(y3.subbands...)
    @test all(norm.(y.x[2:end] .- flat_subbands3) .<= 1.0e-10)

    x_rec = op' * y
    x_rec2 = similar(x_rec)
    mul!(x_rec2, AdjointOperator(op), y)
    assert_cpu_approx(x_rec, x_rec2; atol = 1.0e-12)

    assert_cpu_approx(x_rec, x1)
end

@testitem "ContourletOp / NSCTOp threading policies" tags = [:contourlet, :ContourletOp, :NSCTOp] setup = [TestUtils] begin
    using Contourlets, ContourletOperators, LinearAlgebra, Random, AbstractOperators

    n = 32
    params = ContourletParams(J = 2, L_array = [1, 2])
    x1 = randn(n, n)

    for threaded in (true, false)
        C = ContourletOp(Float64, params, (n, n); threaded = threaded)
        N = NSCTOp(Float64, params, (n, n); threaded = threaded)

        assert_cpu_approx(C' * (C * x1), x1)
        assert_cpu_approx(N' * (N * x1), x1)
    end
end

@testitem "ContourletOp / NSCTOp check() errors" tags = [:contourlet, :ContourletOp, :NSCTOp] setup = [TestUtils] begin
    using Contourlets, ContourletOperators, AbstractOperators

    n = 32
    params = ContourletParams(J = 2, L_array = [1, 2])
    C = ContourletOp(Float64, params, (n, n))
    N = NSCTOp(Float64, params, (n, n))

    # Wrong-size domain input
    @test_throws Exception C * randn(n + 1, n)
    @test_throws Exception N * randn(n + 1, n)

    # Domain must not be an ArrayPartition
    bad_x = ArrayPartition(randn(n, n))
    @test_throws Exception C * bad_x
    @test_throws Exception N * bad_x

    # Adjoint input must be an ArrayPartition
    @test_throws Exception C' * randn(n, n)
    @test_throws Exception N' * randn(n, n)
end
