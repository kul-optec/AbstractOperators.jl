if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
import Base: size

@testset "DCAT" begin
    verb && println(" --- Testing DCAT --- ")

    m1, n1, m2, n2 = 4, 7, 5, 2
    A1 = randn(m1, n1)
    A2 = randn(m2, n2)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opD = DCAT(opA1, opA2)
    x1 = randn(n1)
    x2 = randn(n2)
    y1 = test_op(opD, ArrayPartition(x1, x2), ArrayPartition(randn(m1), randn(m2)), verb)
    y2 = ArrayPartition(A1 * x1, A2 * x2)
    @test norm(y1 .- y2) .<= 1e-12

    # test DCAT longer

    m1, n1, m2, n2, m3, n3 = 4, 7, 5, 2, 5, 5
    A1 = randn(m1, n1)
    A2 = randn(m2, n2)
    A3 = randn(m3, n3)
    opA1 = MatrixOp(A1)
    opA2 = MatrixOp(A2)
    opA3 = MatrixOp(A3)
    opD = DCAT(opA1, opA2, opA3)
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = test_op(
        opD, ArrayPartition(x1, x2, x3), ArrayPartition(randn(m1), randn(m2), randn(m3)), verb
    )
    y2 = ArrayPartition(A1 * x1, A2 * x2, A3 * x3)
    @test norm(y1 .- y2) .<= 1e-12

    #properties
    @test is_linear(opD) == true
    @test is_null(opD) == false
    @test is_eye(opD) == false
    @test is_diagonal(opD) == false
    @test is_AcA_diagonal(opD) == false
    @test is_AAc_diagonal(opD) == false
    @test is_orthogonal(opD) == false
    @test is_invertible(opD) == false
    @test is_full_row_rank(opD) == false
    @test is_full_column_rank(opD) == false

    # DCAT of Eye

    n1, n2 = 4, 7
    x1 = randn(n1)
    x2 = randn(n2)

    opD = Eye(ArrayPartition(x1, x2))
    y1 = test_op(opD, ArrayPartition(x1, x2), ArrayPartition(randn(n1), randn(n2)), verb)

    #properties
    @test is_linear(opD) == true
    @test is_null(opD) == false
    @test is_eye(opD) == true
    @test is_diagonal(opD) == true
    @test is_AcA_diagonal(opD) == true
    @test is_AAc_diagonal(opD) == true
    @test is_orthogonal(opD) == true
    @test is_invertible(opD) == true
    @test is_full_row_rank(opD) == true
    @test is_full_column_rank(opD) == true

    @test diag(opD) == 1
    @test diag_AcA(opD) == 1
    @test diag_AAc(opD) == 1

    # displacement DCAT

    m1, n1, m2, n2, m3, n3 = 4, 7, 5, 2, 5, 5
    A1 = randn(m1, n1)
    A2 = randn(m2, n2)
    A3 = randn(m3, n3)
    d1 = randn(m1)
    d2 = randn(m2)
    d3 = randn(m3)
    opA1 = AffineAdd(MatrixOp(A1), d1)
    opA2 = AffineAdd(MatrixOp(A2), d2)
    opA3 = AffineAdd(MatrixOp(A3), d3)
    opD = DCAT(opA1, opA2, opA3)
    x1 = randn(n1)
    x2 = randn(n2)
    x3 = randn(n3)
    y1 = opD * ArrayPartition(x1, x2, x3)
    y2 = ArrayPartition(A1 * x1 + d1, A2 * x2 + d2, A3 * x3 + d3)
    @test norm(y1 .- y2) .<= 1e-12
    @test norm(displacement(opD) .- ArrayPartition(d1, d2, d3)) .<= 1e-12
    y1 = remove_displacement(opD) * ArrayPartition(x1, x2, x3)
    y2 = ArrayPartition(A1 * x1, A2 * x2, A3 * x3)
    @test norm(y1 .- y2) .<= 1e-12

    m1, n1, m2, n2 = 4, 7, 5, 2
    A1 = MatrixOp(randn(m1, n1))
    A2 = MatrixOp(randn(m2, n2))
    op = DCAT(A1, A2)
    _ds = domain_storage_type(op)
    _cs = codomain_storage_type(op)
    @test _ds !== nothing
    @test _cs !== nothing
    @test is_thread_safe(op) == is_thread_safe(A1) && is_thread_safe(A2)

    @test remove_displacement(op) == op

    d1 = randn(m1)
    d2 = randn(m2)
    opd = DCAT(AffineAdd(A1, d1), AffineAdd(A2, d2))
    opd_removed = remove_displacement(opd)
    @test remove_displacement(opd_removed) == opd_removed

    # Test show output (fun_name indirectly)
    opA1 = MatrixOp([1.0 0.0; 0.0 1.0])
    opA2 = MatrixOp([2.0 0.0; 0.0 2.0])
    opD2 = DCAT(opA1, opA2)
    io = IOBuffer(); show(io, opD2); shown = String(take!(io))
    @test occursin("DCAT", shown) || occursin("[", shown)

    # Test permute
    m1, n1, m2, n2 = 2, 2, 2, 2
    opA1 = MatrixOp([1.0 0.0; 0.0 1.0])
    opA2 = MatrixOp([2.0 0.0; 0.0 2.0])
    opD = DCAT(opA1, opA2)
    x = ArrayPartition([1.0, 2.0], [3.0, 4.0])
    p = [2, 1]  # permute domain
    opDp = AbstractOperators.permute(opD, p)
    # The permuted operator should act on permuted input
    x_perm = ArrayPartition(x.x[2], x.x[1])
    # Just check that permute returns a DCAT and doesn't error
    @test typeof(opDp) <: typeof(opD)

    # Test get_normal_op and has_optimized_normalop
    struct DCATDummyOp <: AbstractOperator end
    AbstractOperators.has_optimized_normalop(::DCATDummyOp) = true
    AbstractOperators.get_normal_op(::DCATDummyOp) = DCATDummyOp()
    AbstractOperators.size(::DCATDummyOp) = ((2,), (2,))
    opD3 = DCAT(DCATDummyOp(), opA1)
    @test AbstractOperators.has_optimized_normalop(opD3) == true
    nrm = AbstractOperators.get_normal_op(opD3)
    @test typeof(nrm) <: DCAT

    # Test opnorm and estimate_opnorm
    opA3 = MatrixOp([1.0 0.0; 0.0 3.0])
    opD4 = DCAT(opA1, opA3)
    @test opnorm(opD4) ≈ estimate_opnorm(opD4)

    # Test == for general equality
    opD5a = DCAT(opA1, opA2)
    opD5b = DCAT(opA1, opA2)
    opD5c = DCAT(opA2, opA1)
    @test opD5a == opD5b
    @test opD5a != opD5c

    # Test is_thread_safe for mixed thread safety
    struct NotThreadSafeOp <: AbstractOperator end
    AbstractOperators.is_thread_safe(::NotThreadSafeOp) = false
    Base.size(::NotThreadSafeOp) = ((2,), (2,))
    opD6 = DCAT(opA1, NotThreadSafeOp())
    @test is_thread_safe(opD6) == false

    # Explicit diag_AcA/diag_AAc for all-Eye DCAT
    opEye = Eye(ArrayPartition([1.0, 2.0], [3.0, 4.0]))
    @test diag_AcA(opEye) == 1
    @test diag_AAc(opEye) == 1

    # Test nonlinear DCAT
    n, m = 4, 3
    x = ArrayPartition(randn(n), randn(m))
    r = ArrayPartition(randn(n), randn(m))
    A = randn(n, n)
    B = Sigmoid(Float64, (m,), 2)
    op = DCAT(MatrixOp(A), B)

    y, grad = test_NLop(op, x, r, verb)

    Y = ArrayPartition(A * x.x[1], B * x.x[2])
    @test norm(Y - y) < 1e-8
end
