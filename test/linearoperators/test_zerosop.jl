@testitem "Zeros" tags = [:linearoperator, :Zeros] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)

    n = (3, 4)
    m = (5, 2)
    op = Zeros(Float64, n, Float64, m)
    y1 = test_op(op, randn(n), randn(m), verb)
    @test y1 == zeros(Float64, m)

    function test_zeros_mul(conv, verb)
        n = (3, 4)
        m = (5, 2)
        ST = Base.typename(typeof(conv(zeros(Float64, 1)))).wrapper
        op = Zeros(Float64, n, Float64, m; array_type = ST)
        y1 = test_op(op, conv(randn(n)), conv(randn(m)), verb)
        @test to_cpu(y1) == zeros(Float64, m)
    end

    test_zeros_mul(identity, verb)

    n = (3, 4)
    D = Float64
    m = (5, 2)
    C = Complex{Float64}
    op = Zeros(D, n, C, m)
    op_array_type = Zeros(D, n, C, m; array_type = Array{ComplexF32, 2})
    @test domain_array_type(op_array_type) == Array{Float64}
    @test codomain_array_type(op_array_type) == Array{ComplexF64}
    x1 = randn(n)
    y1 = test_op(op, x1, randn(m) + im * randn(m), verb)
    @test y1 == zeros(eltype(y1), m)
    @test size(op) == (m, n)
    @test domain_type(op) == D
    @test codomain_type(op) == C
    @test is_thread_safe(op) == true
    @test AbstractOperators.has_fast_opnorm(op) == true
    @test opnorm(op) == 0

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
    io = IOBuffer()
    show(io, op)
    s = String(take!(io))
    @test occursin("0", s)
end

@testitem "Zeros (GPU)" tags = [:gpu, :linearoperator, :Zeros] setup = [TestUtils] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)
        n = (3, 4)
        m = (5, 2)
        op = Zeros(Float64, n, Float64, m; array_type = gpu_wrapper(backend, Float64, 1))
        y = test_op(op, gpu_randn(backend, n...), gpu_randn(backend, m...), false)
        @test collect(y) == zeros(Float64, m)
    end
end

@testitem "Zeros: same-type constructor and Zeros(A) shortcut" tags = [:linearoperator, :Zeros] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    Random.seed!(1)

    # Zeros(T, dim_in, dim_out) — same domain/codomain type, no storage keyword
    n = (3,)
    m = (5,)
    op = Zeros(Float64, n, m)
    @test domain_type(op) == Float64
    @test codomain_type(op) == Float64
    x = randn(n...)
    y = zeros(m...)
    mul!(y, op, x)
    @test all(iszero, y)
    mul!(x, op', y)
    @test all(iszero, x)

    # Zeros(A::AbstractOperator) shortcut
    ref_op = MatrixOp(randn(4, 3))
    zop = Zeros(ref_op)
    @test domain_type(zop) == domain_type(ref_op)
    @test codomain_type(zop) == codomain_type(ref_op)
    @test size(zop) == size(ref_op)
end

@testitem "Zeros: multi-domain (HCAT) and multi-codomain (VCAT) constructors" tags = [
    :linearoperator, :Zeros,
] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(2)

    # Multi-domain → returns HCAT of Zeros
    z_hcat = Zeros((Float64, Float64), ((3,), (4,)), Float64, (5,))
    @test z_hcat isa HCAT
    x1 = randn(3)
    x2 = randn(4)
    y = z_hcat * ArrayPartition(x1, x2)
    @test all(iszero, y)

    # Multi-codomain → returns VCAT of Zeros
    z_vcat = Zeros(Float64, (3,), (Float64, Float64), ((5,), (6,)))
    @test z_vcat isa VCAT
    x = randn(3)
    yv = z_vcat * x
    @test all(iszero, yv)
end
