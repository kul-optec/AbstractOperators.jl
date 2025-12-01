if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)

@testset "MyLinOp" begin
    verb && println(" --- Testing MyLinOp --- ")

    n, m = 5, 4
    A = randn(n, m)
    op = MyLinOp(Float64, (m,), (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x))
    x1 = randn(m)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = A * x1
    @test y1 ≈ y2

    # size & types
    @test size(op) == ((n,), (m,))
    @test domain_type(op) == Float64
    @test codomain_type(op) == Float64

    # adjoint application
    z = randn(n)
    out_adj = similar(x1)
    mul!(out_adj, op', z)
    @test out_adj ≈ A' * z

    # In-place forward mul! on matrix input via broadcasting columns
    X = randn(m, 3)
    Y = zeros(n, 3)
    for j in 1:3
        mul!(view(Y, :, j), op, view(X, :, j))
    end
    @test Y ≈ A * X

    # Scaling operator (real scale ok, complex should fail for real op)
    Sop = Scale(3.0, op)
    @test Sop * x1 ≈ 3.0 * (op * x1)
    @test_throws ErrorException Scale(1 + 2im, op)

    # Error path: dimension mismatch input length
    @test_throws DimensionMismatch op * randn(m + 1)

    # Show output
    io = IOBuffer(); show(io, op); s = String(take!(io)); @test occursin("A", s)

    # other constructors
    op = MyLinOp(
        Float64, (m,), Float64, (n,), (y, x) -> mul!(y, A, x), (y, x) -> mul!(y, A', x)
    )
    @test size(op) == ((n,), (m,))
    @test op * x1 ≈ A * x1
end
