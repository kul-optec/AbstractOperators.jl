@testitem "DCAT: nested DCAT" tags = [:calculus, :DCAT] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    # DCAT does NOT flatten inner DCATs: when an inner DCAT{2} is an element,
    # _ndoms_from_type returns 2, creating tuple indices in idxD/idxC.
    # This exercises the `else` (tuple-index) branches in @generated mul!.
    inner = DCAT(Eye(2), Eye(3))
    outer = DCAT(inner, Eye(4))

    # The domain storage type expands the inner DCAT's ArrayPartition:
    # DS = ArrayPartition{Float64, Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}
    # so input is a flat 3-element ArrayPartition.
    x = ArrayPartition(randn(2), randn(3), randn(4))
    y_ref = ArrayPartition(randn(2), randn(3), randn(4))

    # test_op verifies forward mul!, adjoint mul!, and adjoint consistency
    y = test_op(outer, x, y_ref, verb)

    # Since all sub-operators are identity, output == input
    @test norm(collect(y) - collect(x)) <= 1.0e-12
    @test norm(collect(outer' * y) - collect(x)) <= 1.0e-12
end

@testitem "DCAT: basic mul" tags = [:calculus, :DCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing DCAT --- ")

    m1, n1, m2, n2, m3, n3 = 4, 7, 5, 2, 5, 5
    A1 = randn(m1, n1)
    A2 = randn(m2, n2)
    A3 = randn(m3, n3)
    opD = DCAT(MatrixOp(A1), MatrixOp(A2), MatrixOp(A3))
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = test_op(
        opD, ArrayPartition(x1, x2, x3), ArrayPartition(randn(m1), randn(m2), randn(m3)), verb
    )
    @test norm(y1 .- ArrayPartition(A1 * x1, A2 * x2, A3 * x3)) .<= 1.0e-12

    n1, n2 = 4, 7
    opEye = Eye(ArrayPartition(randn(n1), randn(n2)))
    @test is_eye(opEye) == true && is_orthogonal(opEye) == true
end

@testitem "DCAT: properties" tags = [:calculus, :DCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, n1, m2, n2, m3, n3 = 4, 7, 5, 2, 5, 5
    opD = DCAT(MatrixOp(randn(m1, n1)), MatrixOp(randn(m2, n2)), MatrixOp(randn(m3, n3)))
    @test is_linear(opD) == true
    @test is_null(opD) == false
    @test is_eye(opD) == false
    @test is_diagonal(opD) == false
    @test is_AcA_diagonal(opD) == false
    @test is_full_row_rank(opD) == false
    @test is_full_column_rank(opD) == false

    m1, n1, m2, n2 = 4, 7, 5, 2
    A1 = MatrixOp(randn(m1, n1))
    A2 = MatrixOp(randn(m2, n2))
    op = DCAT(A1, A2)
    @test domain_array_type(op) !== nothing
    @test codomain_array_type(op) !== nothing
    @test remove_displacement(op) == op
    opd = DCAT(AffineAdd(A1, randn(m1)), AffineAdd(A2, randn(m2)))
    opd_removed = remove_displacement(opd)
    @test remove_displacement(opd_removed) == opd_removed
end

@testitem "DCAT: displacement" tags = [:calculus, :DCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    m1, n1, m2, n2, m3, n3 = 4, 7, 5, 2, 5, 5
    A1 = randn(m1, n1)
    A2 = randn(m2, n2)
    A3 = randn(m3, n3)
    d1 = randn(m1)
    d2 = randn(m2)
    d3 = randn(m3)
    opD = DCAT(
        AffineAdd(MatrixOp(A1), d1), AffineAdd(MatrixOp(A2), d2), AffineAdd(MatrixOp(A3), d3)
    )
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = opD * ArrayPartition(x1, x2, x3)
    @test norm(y1 .- ArrayPartition(A1 * x1 + d1, A2 * x2 + d2, A3 * x3 + d3)) .<= 1.0e-12
    @test norm(displacement(opD) .- ArrayPartition(d1, d2, d3)) .<= 1.0e-12
    y_rd = remove_displacement(opD) * ArrayPartition(x1, x2, x3)
    @test norm(y_rd .- ArrayPartition(A1 * x1, A2 * x2, A3 * x3)) .<= 1.0e-12
end

@testitem "DCAT: nonlinear" tags = [:calculus, :DCAT] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n, m = 4, 3
    x = ArrayPartition(randn(n), randn(m))
    r = ArrayPartition(randn(n), randn(m))
    A = randn(n, n)
    B = Sigmoid(Float64, (m,), 2)
    op = DCAT(MatrixOp(A), B)
    y, grad = test_NLop(op, x, r, verb)
    @test norm(ArrayPartition(A * x.x[1], B * x.x[2]) - y) < 1.0e-8
end

@testitem "DCAT (GPU)" tags = [:gpu, :calculus, :DCAT] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n1, n2 = 3, 4
        opD = DCAT(DiagOp(gpu_ones(backend, Float64, n1)), DiagOp(to_gpu(backend, 2 .* ones(n2))))
        test_op(
            opD,
            ArrayPartition(gpu_randn(backend, n1), gpu_randn(backend, n2)),
            ArrayPartition(gpu_randn(backend, n1), gpu_randn(backend, n2)),
            false,
        )

        m1, n1, m2, n2 = 4, 7, 5, 2
        A1 = gpu_randn(backend, m1, n1)
        A2 = gpu_randn(backend, m2, n2)
        opD2 = DCAT(MatrixOp(A1), MatrixOp(A2))
        test_op(
            opD2,
            ArrayPartition(gpu_randn(backend, n1), gpu_randn(backend, n2)),
            ArrayPartition(gpu_randn(backend, m1), gpu_randn(backend, m2)),
            false,
        )
    end
end
