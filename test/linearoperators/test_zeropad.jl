if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)

@testset "ZeroPad" begin
    verb && println(" --- Testing ZeroPad --- ")

    n = (3,)
    z = (5,)
    op = ZeroPad(Float64, n, z)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n .+ z), verb)
    @test all(norm.(y1 .- [x1; zeros(5)]) .<= 1e-12)
    @test size(op) == (n .+ z, n)
    @test domain_type(op) == Float64
    @test codomain_type(op) == Float64
    @test is_thread_safe(op) == true
    @test opnorm(op) == 1

    n = (3, 2)
    z = (5, 3)
    op = ZeroPad(Float64, n, z)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n .+ z), verb)
    y2 = zeros(n .+ z)
    y2[1:n[1], 1:n[2]] = x1
    @test all(norm.(y1 .- y2) .<= 1e-12)
    @test size(op) == (n .+ z, n)

    n = (3, 2, 2)
    z = (5, 3, 1)
    op = ZeroPad(Float64, n, z)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n .+ z), verb)
    y2 = zeros(n .+ z)
    y2[1:n[1], 1:n[2], 1:n[3]] = x1
    @test all(norm.(y1 .- y2) .<= 1e-12)
    @test size(op) == (n .+ z, n)

    # Normal operator should be identity on input space
    Nop = AbstractOperators.get_normal_op(op)
    xin = randn(n)
    @test Nop * xin ≈ xin

    # Adjoint: crop back
    ybig = zeros(n .+ z)
    ybig[1:n[1], 1:n[2], 1:n[3]] .= x1
    xcropped = zeros(n)
    mul!(xcropped, op', ybig)
    @test xcropped ≈ x1

    # In-place forward padding
    ybuf = similar(ybig)
    mul!(ybuf, op, x1)
    @test ybuf ≈ y2

    # Scaling (real ok, complex rejected for real domain)
    Sop = Scale(2.0, op)
    @test Sop * x1 ≈ 2.0 * (op * x1)
    @test_throws ErrorException Scale(1 + 2im, op)

    # other constructors
    ZeroPad(n, z...)
    ZeroPad(Float64, n, z...)
    ZeroPad(n, z...)
    ZeroPad(x1, z)
    ZeroPad(x1, z...)

    #errors
    @test_throws ErrorException ZeroPad(Float64, n, (1, 2))
    @test_throws ErrorException ZeroPad(Float64, n, (1, -2, 3))

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == false
    @test is_full_column_rank(op) == true

    @test diag_AcA(op) == 1

    # Show output symbol
    io = IOBuffer(); show(io, op); s = String(take!(io)); @test occursin("[I;0]", s)
end
