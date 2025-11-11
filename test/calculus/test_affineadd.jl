if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end

@testset "AffineAdd" begin
    verb && println(" --- Testing AffineAdd --- ")

    n, m = 5, 6
    A = randn(n, m)
    opA = MatrixOp(A)
    d = randn(n)
    T = AffineAdd(opA, d)

    verb && println(T)
    x1 = randn(m)
    y1 = T * x1
    @test norm(y1 - (A * x1 + d)) < 1e-9
    r = randn(n)
    @test norm(T' * r - (A' * r)) < 1e-9
    @test displacement(T) == d
    @test norm(remove_displacement(T) * x1 - A * x1) < 1e-9

    # with sign
    T = AffineAdd(opA, d, false)
    @test sign(T) == -1

    verb && println(T)
    x1 = randn(m)
    y1 = T * x1
    @test norm(y1 - (A * x1 - d)) < 1e-9
    r = randn(n)
    @test norm(T' * r - (A' * r)) < 1e-9
    @test displacement(T) == -d
    @test norm(remove_displacement(T) * x1 - A * x1) < 1e-9

    # with scalar
    T = AffineAdd(opA, pi)
    @test sign(T) == 1

    verb && println(T)
    x1 = randn(m)
    y1 = T * x1
    @test norm(y1 - (A * x1 .+ pi)) < 1e-9
    r = randn(n)
    @test norm(T' * r - (A' * r)) < 1e-9
    @test displacement(T) .- pi < 1e-9
    @test norm(remove_displacement(T) * x1 - A * x1) < 1e-9

    @test_throws DimensionMismatch AffineAdd(MatrixOp(randn(2, 5)), randn(5))
    opD = DiagOp(Float64, (4,), randn(ComplexF64, 4))
    @test_throws ErrorException AffineAdd(opD, randn(4))
    AffineAdd(opD, pi)
    @test_throws ErrorException AffineAdd(Eye(4), im * pi)

    # with scalar and vector 
    d = randn(n)
    T = AffineAdd(AffineAdd(opA, pi), d, false)

    verb && println(T)
    x1 = randn(m)
    y1 = T * x1
    @test norm(y1 - (A * x1 .+ pi .- d)) < 1e-9
    r = randn(n)
    @test norm(T' * r - (A' * r)) < 1e-9
    @test norm(displacement(T) .- (pi .- d)) < 1e-9

    T2 = remove_displacement(T)
    @test norm(T2 * x1 - (A * x1)) < 1e-9

    # permute AddAffine 
    n, m = 5, 6
    A = randn(n, m)
    d = randn(n)
    opH = HCAT(Eye(n), MatrixOp(A))
    x = ArrayPartition(randn(n), randn(m))
    opHT = AffineAdd(opH, d)

    @test norm(opHT * x - (x.x[1] + A * x.x[2] .+ d)) < 1e-12
    p = [2; 1]
    @test norm(
        AbstractOperators.permute(opHT, p) * ArrayPartition(x.x[p]...) -
        (x.x[1] + A * x.x[2] .+ d),
    ) < 1e-12

    n, m = 5, 6
    A = randn(n, m)
    opA = MatrixOp(A)
    d = randn(n)
    Tplus = AffineAdd(opA, d)
    Tminus = AffineAdd(opA, d, false)
    Tscalar = AffineAdd(opA, pi)
    io = IOBuffer(); show(io, Tplus); splus = String(take!(io)); @test occursin("+d", splus)
    io = IOBuffer(); show(io, Tminus); sminus = String(take!(io)); @test occursin("-d", sminus)
    io = IOBuffer(); show(io, Tscalar); sscal = String(take!(io)); @test occursin("+d", sscal) || occursin("-d", sscal)

    # Scaling of AffineAdd (array displacement)
    α = 2.0
    Tscaled = Scale(α, Tplus)
    x = randn(m)
    @test Tscaled * x ≈ α * (A * x + d)
    # scaling of negative sign variant
    Tscaled_neg = Scale(α, Tminus)
    @test Tscaled_neg * x ≈ α * (A * x - d)

    # Diagonal passthrough and diag/diag_AcA/diag_AAc
    dvec = randn(n)
    D = DiagOp(dvec)
    Tdiag = AffineAdd(D, zeros(n))
    @test is_diagonal(Tdiag) == true
    @test diag(Tdiag) == diag(D)

    # is_null / is_eye cases
    Zop = Zeros(Float64, (n,), Float64, (n,))
    Tzero = AffineAdd(Zop, zeros(n))
    @test is_null(Tzero) == true
    Eop = Eye(n)
    Teye = AffineAdd(Eop, zeros(n))
    @test is_eye(Teye) == true

    # remove_displacement idempotence (already covered for simple, add explicit check)
    rd1 = remove_displacement(Tplus)
    rd2 = remove_displacement(rd1)
    @test rd1 * x == rd2 * x

    # AffineAdd with NonLinearOperator
    n = 10
    d = randn(n)
    T = AffineAdd(Exp(n), d, false)

    r = randn(n)
    x = randn(size(T, 2))
    y, grad = test_NLop(T, x, r, verb)
    @test norm(y - (exp.(x) - d)) < 1e-8
end
