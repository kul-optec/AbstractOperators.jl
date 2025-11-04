if !isdefined(Main, :verb)
    const verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
using SparseArrays

@testset "FiniteDiff" begin
    verb && println(" --- Testing FiniteDiff --- ")

    n = 10
    op = FiniteDiff(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n - 1), verb)
    y1 = op * collect(range(0; stop=1, length=n))
    @test all(norm.(y1 .- 1 / 9) .<= 1e-12)

    I1, J1, V1 = SparseArrays.spdiagm_internal(0 => ones(n - 1))
    I2, J2, V2 = SparseArrays.spdiagm_internal(1 => ones(n - 1))
    B = -sparse(I1, J1, V1, n - 1, n) + sparse(I2, J2, V2, n - 1, n)

    @test norm(B * x1 - op * x1) <= 1e-8

    n, m = 10, 5
    op = FiniteDiff(Float64, (n, m))
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(n - 1, m), verb)
    y1 = op * repeat(collect(range(0; stop=1, length=n)), 1, m)
    @test all(norm.(y1 .- 1 / 9) .<= 1e-12)

    n, m = 10, 5
    op = FiniteDiff(Float64, (n, m), 2)
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(n, m - 1), verb)
    y1 = op * repeat(collect(range(0; stop=1, length=n)), 1, m)
    @test all(norm.(y1) .<= 1e-12)

    n, m, l = 10, 5, 7
    op = FiniteDiff(Float64, (n, m, l))
    x1 = randn(n, m, l)
    y1 = test_op(op, x1, randn(n - 1, m, l), verb)
    y1 = op * reshape(repeat(collect(range(0; stop=1, length=n)), 1, m * l), n, m, l)
    @test all(norm.(y1 .- 1 / 9) .<= 1e-12)

    n, m, l = 10, 5, 7
    op = FiniteDiff(Float64, (n, m, l), 2)
    x1 = randn(n, m, l)
    y1 = test_op(op, x1, randn(n, m - 1, l), verb)
    y1 = op * reshape(repeat(collect(range(0; stop=1, length=n)), 1, m * l), n, m, l)
    @test all(norm.(y1) .<= 1e-12)

    n, m, l = 10, 5, 7
    op = FiniteDiff(Float64, (n, m, l), 3)
    x1 = randn(n, m, l)
    y1 = test_op(op, x1, randn(n, m, l - 1), verb)
    y1 = op * reshape(repeat(collect(range(0; stop=1, length=n)), 1, m * l), n, m, l)
    @test all(norm.(y1) .<= 1e-12)

    n, m, l, i = 5, 6, 2, 3
    op = FiniteDiff(Float64, (n, m, l, i), 1)
    x1 = randn(n, m, l, i)
    y1 = test_op(op, x1, randn(n - 1, m, l, i), verb)
    y1 = op * reshape(repeat(collect(range(0; stop=1, length=n)), 1, m * l * i), n, m, l, i)
    @test all(norm.(y1 .- 1 / (n - 1)) .<= 1e-12)

    n, m, l, i = 5, 6, 2, 3
    op = FiniteDiff(Float64, (n, m, l, i), 4)
    x1 = randn(n, m, l, i)
    y1 = test_op(op, x1, randn(n, m, l, i - 1), verb)
    y1 = op * reshape(repeat(collect(range(0; stop=1, length=n)), 1, m * l * i), n, m, l, i)
    @test norm(y1) <= 1e-12

    @test_throws ErrorException FiniteDiff(Float64, (n, m, l), 4)

    ## other constructors
    FiniteDiff((n, m))
    FiniteDiff(x1)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == false

    # Invalid direction > number of dims on construction (simple 2D example)
    @test_throws ErrorException FiniteDiff(Float64, (4,5), 3)

    # Minimal dimension case (size 2 along diff axis) gives single difference
    op_min = FiniteDiff(Float64, (2,))
    x_min = randn(2)
    @test (op_min * x_min)[1] ≈ x_min[2] - x_min[1]
    @test size(op_min) == ((1,), (2,))

    # Complex input - ensure same behavior and real/imag handled
    n = 6
    op_c = FiniteDiff(ComplexF64, (n,))
    x_c = randn(n) .+ im * randn(n)
    y_c = similar(x_c, n-1)
    y_c .= op_c * x_c
    @test all(y_c .≈ x_c[2:end] .- x_c[1:end-1])

    # Adjoint consistency: <Fx,g> == <x,F'*g>
    x = randn(n)
    F = FiniteDiff(Float64, (n,))
    g = randn(n-1)
    lhs = dot(F * x, g)
    tmp = zeros(n)
    mul!(tmp, F', g)
    rhs = dot(x, tmp)
    @test lhs ≈ rhs atol=1e-10

    # In-place mul! forward and adjoint
    y_store = zeros(n-1)
    mul!(y_store, F, x)
    @test all(y_store .== (F * x))
    x_store = zeros(n)
    mul!(x_store, F', y_store)
    @test dot(F * x, y_store) ≈ dot(x, x_store) atol=1e-10

    # show() output should contain the symbolic direction labels (instead of testing non-exported fun_name)
    io = IOBuffer(); show(io, F); strF = String(take!(io)); @test occursin("δx", strF)
    Fy = FiniteDiff(Float64, (3,4), 2)
    io = IOBuffer(); show(io, Fy); strFy = String(take!(io)); @test occursin("δy", strFy)

    # size correctness for higher-dim
    F3 = FiniteDiff(Float64, (3,4,5), 2)
    @test size(F3) == ((3,3,5), (3,4,5))
end
