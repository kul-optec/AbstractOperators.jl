using SparseArrays

if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end

@testset "MatrixOp" begin
    verb && println(" --- Testing MatrixOp --- ")

    # real matrix, real input
    n, m = 5, 4
    A = randn(n, m)
    op = MatrixOp(Float64, (m,), A)
    x1 = randn(m)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = A * x1

    # real matrix, complex input
    n, m = 5, 4
    A = randn(n, m)
    op = MatrixOp(Complex{Float64}, (m,), A)
    x1 = randn(m) + im .* randn(m)
    y1 = test_op(op, x1, randn(n) + im * randn(n), verb)
    y2 = A * x1

    # complex matrix, complex input
    n, m = 5, 4
    A = randn(n, m) + im * randn(n, m)
    op = MatrixOp(Complex{Float64}, (m,), A)
    x1 = randn(m) + im .* randn(m)
    y1 = test_op(op, x1, randn(n) + im * randn(n), verb)
    y2 = A * x1

    # complex matrix, real input
    n, m = 5, 4
    A = randn(n, m) + im * randn(n, m)
    op = MatrixOp(Float64, (m,), A)
    x1 = randn(m)
    y1 = test_op(op, x1, randn(n) + im * randn(n), verb)
    y2 = A * x1

    @test all(norm.(y1 .- y2) .<= 1e-12)

    # complex matrix, real matrix input
    c = 3
    op = MatrixOp(Float64, (m, c), A)
    @test_throws ErrorException MatrixOp(Float64, (m, c, 3), A)
    @test_throws MethodError MatrixOp(Float64, (m, c), randn(n, m, 2))
    x1 = randn(m, c)
    y1 = test_op(op, x1, randn(n, c) .+ randn(n, c), verb)
    y2 = A * x1

    # other constructors
    op = MatrixOp(A)
    op = MatrixOp(Float64, A)
    op = MatrixOp(A, c)
    op = MatrixOp(Float64, A, c)

    op = convert(LinearOperator, A)
    op = convert(LinearOperator, A, c)
    op = convert(LinearOperator, Complex{Float64}, size(x1), A)

    #properties
    @test is_sliced(op) == false
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(MatrixOp(randn(Random.seed!(0), 3, 4))) == true
    @test is_full_column_rank(MatrixOp(randn(Random.seed!(0), 3, 4))) == false

    # Square invertible matrix
    B = randn(4,4)
    invop = MatrixOp(Float64, (4,), B)
    @test is_invertible(invop) == (det(B) != 0)

    # Orthogonal case (approx)
    Q,_ = qr(randn(5,5))
    Qmat = Matrix(Q)
    Qop = MatrixOp(Float64, (5,), Qmat)
    @test is_orthogonal(Qop) == true

    # Diagonal detection path
    D = diagm(0 => randn(5))
    Dop = MatrixOp(Float64, (5,), D)
    @test is_diagonal(Dop) == true
    @test is_AcA_diagonal(Dop) == true
    @test is_AAc_diagonal(Dop) == true

    # Scale: real codomain, complex coeff should error
    @test_throws ErrorException Scale(1.0 + 2im, invop)
    Sop = Scale(2.0, invop)
    xS = randn(4)
    @test Sop * xS ≈ 2.0 * (invop * xS)

    # Normal operator A'A
    Nop = AbstractOperators.get_normal_op(invop)
    @test size(Nop) == ((4,), (4,))
    xN = randn(4)
    @test Nop * xN ≈ invop' * (invop * xN)

    # Adjoint mapping with complex matrix and real input (special method)
    Cmat = randn(3,3) + im*randn(3,3)
    Cop = MatrixOp(Float64, (3,), Cmat)
    xr = randn(3)
    yr = Cop' * xr
    # Compare with manual real of complex multiplication
    @test yr ≈ real.(Cmat' * xr)

    # In-place mul!
    xvec = randn(5)
    yvec = zeros(5)
    mul!(yvec, Qop, xvec)
    @test yvec ≈ Qmat * xvec

    # Batched (matrix input) in-place multiplication
    Xmat = randn(5,3)
    Ymat = zeros(5,3)
    mul!(Ymat, Qop, Xmat)
    @test Ymat ≈ Qmat * Xmat

    # opnorm vs estimate_opnorm
    @test LinearAlgebra.opnorm(invop) ≈ estimate_opnorm(invop)
    @test LinearAlgebra.opnorm(invop) == LinearAlgebra.opnorm(B)

    # Size variations: single vs multi-column
    Tall = randn(8,3)
    TallOp = MatrixOp(Float64, (3,), Tall)
    TallOpMulti = MatrixOp(Float64, (3,2), Tall)
    @test size(TallOp) == ((8,), (3,))
    @test size(TallOpMulti) == ((8,2), (3,2))

    # Show output symbol
    io = IOBuffer(); show(io, TallOp); str = String(take!(io)); @test occursin("▒", str)
end
