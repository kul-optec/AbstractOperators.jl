if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end

@testset "Zeros" begin
    verb && println(" --- Testing Zeros --- ")

    n = (3, 4)
    D = Float64
    m = (5, 2)
    C = Complex{Float64}
    op = Zeros(D, n, C, m)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(m) + im * randn(m), verb)
    @test y1 == zeros(eltype(y1), m)
    @test size(op) == (m, n)
    @test domain_type(op) == D
    @test codomain_type(op) == C
    @test is_thread_safe(op) == true
    @test LinearAlgebra.opnorm(op) == 0

    # Adjoint returns zeros of domain shape
    z = zeros(n)
    zbuf = similar(z)
    mul!(zbuf, op', y1)
    @test zbuf == z

    # In-place forward
    ybuf = similar(y1)
    mul!(ybuf, op, x1)
    @test ybuf == zeros(eltype(ybuf), m)

    # Normal operator should also be zero
    Nop = AbstractOperators.get_normal_op(op)
    @test Nop * x1 == zeros(eltype(x1), n)

    # Scaling real ok, complex scale allowed since codomain complex; verify
    Sop = Scale(2.5, op)
    @test Sop * x1 == zeros(eltype(y1), m)
    # Complex scaling also yields zeros (no error expected)
    Sopc = Scale(1 + 2im, op)
    @test Sopc * x1 == zeros(eltype(y1), m)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == true
    @test is_eye(op) == false
    @test is_diagonal(op) == true
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == false
    @test is_full_column_rank(op) == false

    @test diag_AcA(op) == 0
    @test diag_AAc(op) == 0

    # Show output symbol
    io = IOBuffer(); show(io, op); s = String(take!(io)); @test occursin("0", s)
end
