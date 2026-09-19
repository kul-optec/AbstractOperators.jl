@testitem "Axt_mul_Bx: basic mul" tags = [:calculus, :Axt_mul_Bx] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing Axt_mul_Bx: basic mul --- ")

    n = 10
    A, B = Eye(n), Sin(n)
    P = Axt_mul_Bx(A, B)

    x = randn(n)
    r = randn(1)
    y, grad = test_NLop(P, x, r, verb)
    @test norm([(A * x)' * (B * x)] - y) < 1.0e-8

    n, m = 3, 4
    A, B = MatrixOp(randn(n, m)), MatrixOp(randn(n, m))
    P = Axt_mul_Bx(A, B)

    x = randn(m)
    r = randn(1)
    y, grad = test_NLop(P, x, r, verb)
    @test norm([(A * x)' * (B * x)] - y) < 1.0e-8

    n, m, l = 3, 7, 5
    A, B = MatrixOp(randn(n, m), l), MatrixOp(randn(n, m), l)
    P = Axt_mul_Bx(A, B)
    x = randn(m, l)
    r = randn(l, l)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((A * x)' * (B * x) - y) < 1.0e-8

    n, m = 3, 7
    A, B = Sin(n, m), Cos(n, m)
    P = Axt_mul_Bx(A, B)
    x = randn(n, m)
    r = randn(m, m)
    y, grad = test_NLop(P, x, r, verb)
    @test norm((A * x)' * (B * x) - y) < 1.0e-8
end

@testitem "Axt_mul_Bx: HCAT and permute" tags = [:calculus, :Axt_mul_Bx] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing Axt_mul_Bx: HCAT and permute --- ")

    # testing with HCAT
    m, n = 3, 5
    x = ArrayPartition(randn(m), randn(n))
    r = randn(1)
    b = randn(m)
    A = AffineAdd(Sin(Float64, (m,)), b)
    B = MatrixOp(randn(m, n))
    op1 = HCAT(A, B)
    C = Cos(Float64, (m,))
    D = MatrixOp(randn(m, n))
    op2 = HCAT(C, D)
    P = Axt_mul_Bx(op1, op2)
    y, grad = test_NLop(P, x, r, verb)
    @test norm([(op1 * x)' * (op2 * x)] - y) < 1.0e-8

    #test remove_displacement
    y2, grad = test_NLop(remove_displacement(P), x, r, verb)
    @test norm([(op1 * x - b)' * (op2 * x)] - y2) < 1.0e-8

    # test permute
    p = [2, 1]
    Pp = AbstractOperators.permute(P, p)
    xp = ArrayPartition(x.x[p])
    y2, grad = test_NLop(Pp, xp, r, verb)
    @test norm(y2 - y) < 1.0e-8

    @test_throws Exception Axt_mul_Bx(Eye(2, 2), Eye(2, 1))
    @test_throws Exception Axt_mul_Bx(Eye(2, 2, 2), Eye(2, 2, 2))
end

@testitem "Axt_mul_Bx: error paths and equality" tags = [:calculus, :Axt_mul_Bx] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    verb && println(" --- Testing Axt_mul_Bx: error paths and equality --- ")

    # ndims==2 branch with mismatched first codomain dimension
    struct AxtDummy2D <: AbstractOperator
        dim_out::Tuple{Int, Int}
        dim_in::Tuple{Int, Int}
    end
    Base.size(op::AxtDummy2D) = (op.dim_out, op.dim_in)
    AbstractOperators.domain_type(::AxtDummy2D) = Float64
    AbstractOperators.codomain_type(::AxtDummy2D) = Float64
    AbstractOperators.domain_array_type(::AxtDummy2D) = Array{Float64}
    AbstractOperators.codomain_array_type(::AxtDummy2D) = Array{Float64}

    struct AxtDummyMixed <: AbstractOperator
        dim_out::Tuple{Int}
        dim_in::Tuple{Int, Int}
    end
    Base.size(op::AxtDummyMixed) = (op.dim_out, op.dim_in)
    AbstractOperators.domain_type(::AxtDummyMixed) = Float64
    AbstractOperators.codomain_type(::AxtDummyMixed) = Float64
    AbstractOperators.domain_array_type(::AxtDummyMixed) = Array{Float64}
    AbstractOperators.codomain_array_type(::AxtDummyMixed) = Array{Float64}

    A2d = AxtDummy2D((2, 3), (4, 3))
    B2d_bad = AxtDummy2D((3, 3), (4, 3))
    @test_throws DimensionMismatch Axt_mul_Bx(A2d, B2d_bad)

    # NOTE: mixed codomain dimensionality currently dispatches to MethodError in the public constructor.

    # test equality
    n, m = 3, 4
    A, B = MatrixOp(randn(n, m)), MatrixOp(randn(n, m))
    # x matches the HCAT ArrayPartition context used in original test
    x = ArrayPartition(randn(3), randn(5))
    @test Axt_mul_Bx(A, B) == Axt_mul_Bx(A, B)
    @test Jacobian(Axt_mul_Bx(A, B), x) == Jacobian(Axt_mul_Bx(A, B), x)
end

@testitem "Axt_mul_Bx (GPU)" tags = [:gpu, :calculus, :Axt_mul_Bx] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, LinearAlgebra, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)

        n, m = 3, 4
        AL = gpu_randn(backend, n, m)
        BL = gpu_randn(backend, n, m)
        A, B = MatrixOp(AL), MatrixOp(BL)
        P = Axt_mul_Bx(A, B)
        x = gpu_randn(backend, m)
        y = gpu_zeros(backend, Float64, 1)
        mul!(y, P, x)
        Acpu, Bcpu, xcpu = Array(AL), Array(BL), Array(x)
        @test Array(y)[1] ≈ dot(Acpu * xcpu, Bcpu * xcpu)

        n, m, l = 3, 5, 4
        A2, B2 = MatrixOp(gpu_randn(backend, n, m), l), MatrixOp(gpu_randn(backend, n, m), l)
        P2 = Axt_mul_Bx(A2, B2)
        x2 = gpu_randn(backend, m, l)
        y2 = gpu_zeros(backend, Float64, l, l)
        mul!(y2, P2, x2)
        Ax = Array(A2.A) * Array(x2)
        Bx = Array(B2.A) * Array(x2)
        @test Array(y2) ≈ Ax' * Bx
    end
end

@testitem "Axt_mul_Bx 2D operator DimensionMismatch" tags = [:calculus, :Axt_mul_Bx] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    # 2D operators with same domain but different first codomain dimension (line 39)
    A = MatrixOp(randn(3, 5), 4)
    B = MatrixOp(randn(6, 5), 4)
    @test_throws DimensionMismatch Axt_mul_Bx(A, B)
end

@testitem "Axt_mul_Bx: size mismatch error (1D)" tags = [:calculus, :Axt_mul_Bx] setup = [TestUtils] begin
    using AbstractOperators
    # Two 1D operators with different sizes -> triggers size(A) != size(B) in inner constructor
    A = MatrixOp(randn(3, 4))   # size = ((3,), (4,))
    B = MatrixOp(randn(5, 4))   # size = ((5,), (4,)) -- different codomain
    @test_throws DimensionMismatch Axt_mul_Bx(A, B)
end

@testitem "Axt_mul_Bx: size mismatch error (1D)" tags = [:calculus, :Axt_mul_Bx] setup = [TestUtils] begin
    using AbstractOperators
    # Two 1D operators with different sizes -> triggers size(A) != size(B) in inner constructor
    A = MatrixOp(randn(3, 4))   # size = ((3,), (4,))
    B = MatrixOp(randn(5, 4))   # size = ((5,), (4,)) -- different codomain
    @test_throws DimensionMismatch Axt_mul_Bx(A, B)
end

@testitem "Axt_mul_Bx: threading traits and copy_operator" tags = [:calculus, :Axt_mul_Bx] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(1)

    n = 2000
    threaded_leaf = Sin(Float64, (n,); threaded = true)
    serial_leaf = Cos(Float64, (n,); threaded = false)
    @test is_threaded(Axt_mul_Bx(threaded_leaf, serial_leaf)) == true
    @test is_threaded(Axt_mul_Bx(serial_leaf, serial_leaf)) == false
    @test supports_threading(Axt_mul_Bx(serial_leaf, serial_leaf)) == true

    n2 = 10
    A, B = Eye(n2), Sin(n2)
    P = Axt_mul_Bx(A, B)
    P2 = copy_operator(P; threaded = true)
    @test P2 isa Axt_mul_Bx
    x = randn(n2)
    @test P * x ≈ P2 * x
end
