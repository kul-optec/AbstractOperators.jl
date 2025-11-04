@testset "DCT" begin
    n = 4
    op = DCT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = dct(x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    # other constructors
    op = DCT((n,))
    op = DCT(n, n)
    op = DCT(Complex{Float64}, n, n)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == true
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true

    m = 10
    op = DCT(n, m)
    x1 = randn(n, m)

    @test norm(op' * (op * x1) - x1) <= 1e-12
    @test diag_AAc(op) == 1.0
    @test diag_AcA(op) == 1.0
end

@testset "IDCT" begin
    n = 4
    op = IDCT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = idct(x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    # other constructors
    op = IDCT((n,))
    op = IDCT(n, n)
    op = IDCT(Complex{Float64}, n, n)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == true
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true

    m = 10
    op = IDCT(n, m)
    x1 = randn(n, m)

    @test norm(op' * (op * x1) - x1) <= 1e-12
    @test diag_AAc(op) == 1.0
    @test diag_AcA(op) == 1.0
end

@testset "DFT" begin
    n, m = 4, 7

    op = DFT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    op = DFT(Complex{Float64}, (n,))
    x1 = randn(n) + im * randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    op = DFT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    op = DFT(Complex{Float64}, (n,))
    x1 = randn(n) + im * randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    op = DFT(Float64, (n,), 1)
    x1 = randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1, 1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    op = DFT(Complex{Float64}, (n,), 1)
    x1 = randn(n) + im * randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1, 1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    op = DFT(Float64, (n, m))
    x1 = randn(n, m)
    y1 = test_op(op, x1, fft(randn(n, m)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    op = DFT(Complex{Float64}, (n, m))
    x1 = randn(n, m) + im * randn(n, m)
    y1 = test_op(op, x1, fft(randn(n, m)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    op = DFT(Float64, (m, n), 1)
    x1 = randn(m, n)
    y1 = test_op(op, x1, fft(randn(m, n)), verb)
    y2 = fft(x1, 1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    op = DFT(Complex{Float64}, (n, m), 2)
    x1 = randn(n, m) + im * randn(n, m)
    y1 = test_op(op, x1, fft(randn(n, m)), verb)
    y2 = fft(x1, 2)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    # other constructors
    op = DFT((n,))
    op = DFT(n, n)
    op = DFT(Complex{Float64}, n, n)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true

    op = DFT(n, m)
    x1 = randn(n, m)
    y1 = op * x1
    @test norm(op' * (op * x1) - diag_AcA(op) * x1) <= 1e-12
    @test norm(op * (op' * y1) - diag_AAc(op) * y1) <= 1e-12
end

@testset "IDFT" begin
    n, m = 5, 6

    @test_throws AssertionError IDFT(Float64, (n,))

    op = IDFT(Complex{Float64}, (n,))
    x1 = randn(ComplexF64, n)
    @test op * x1 ≈ ifft(x1)

    @test_throws AssertionError IDFT(Float64, (n,), 1)

    op = IDFT(Complex{Float64}, (n,), 1)
    x1 = randn(ComplexF64, n)
    @test op * x1 ≈ ifft(x1, 1)

    @test_throws AssertionError IDFT(Float64, (n, m))

    op = IDFT(Complex{Float64}, (n, m))
    x1 = randn(ComplexF64, n, m)
    @test op * x1 ≈ ifft(x1)

    @test_throws AssertionError IDFT(Float64, (m, n), 1)

    op = IDFT(Complex{Float64}, (n, m), 2)
    x1 = randn(ComplexF64, n, m)
    @test op * x1 ≈ ifft(x1, 2)

    n, m, l = 4, 19, 5
    op = IDFT(Complex{Float64}, (n, m, l), 2)
    x1 = fft(randn(n, m, l), 2)
    @test op * x1 ≈ ifft(x1, 2)

    n, m, l = 4, 18, 5
    op = IDFT(Complex{Float64}, (n, m, l), (1, 2))
    x1 = fft(randn(n, m, l), (1, 2))
    @test op * x1 ≈ ifft(x1, (1, 2))

    op = IDFT(Complex{Float64}, (n, m, l), (3, 2))
    x1 = fft(randn(n, m, l), (3, 2))
    @test op * x1 ≈ ifft(x1, (3, 2))

    # other constructors
    op = IDFT((n,))
    op = IDFT(n, n)
    op = IDFT(Complex{Float64}, n, n)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true

    n, m = 4, 10
    op = IDFT(n, m)
    x1 = randn(ComplexF64, n, m)
    y1 = op * x1
    @test norm(op' * (op * x1) - diag_AcA(op) * x1) <= 1e-12
    @test norm(op * (op' * y1) - diag_AAc(op) * y1) <= 1e-12
end

@testset "RDFT" begin
    n = 4
    op = RDFT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, rfft(x1), verb)
    y2 = rfft(x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    n, m, l = 4, 8, 5
    op = RDFT(Float64, (n, m, l), 2)
    x1 = randn(n, m, l)
    y1 = test_op(op, x1, rfft(x1, 2), verb)
    y2 = rfft(x1, 2)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    # other constructors
    op = RDFT((n,))
    op = RDFT(n, n)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == false
end

@testset "IRDFT" begin
    n = 12
    op = IRDFT(Complex{Float64}, (div(n, 2) + 1,), n)
    x1 = rfft(randn(n))
    y1 = test_op(op, x1, irfft(randn(div(n, 2) + 1), n), verb)
    y2 = irfft(x1, n)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    n = 11
    op = IRDFT(Complex{Float64}, (div(n, 2) + 1,), n)
    x1 = rfft(randn(n))
    y1 = test_op(op, x1, irfft(randn(div(n, 2) + 1), n), verb)
    y2 = irfft(x1, n)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    n, m, l = 4, 19, 5
    op = IRDFT(Complex{Float64}, (n, div(m, 2) + 1, l), m, 2)
    x1 = rfft(randn(n, m, l), 2)
    y1 = test_op(op, x1, irfft(x1, m, 2), verb)
    y2 = irfft(x1, m, 2)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    n, m, l = 4, 18, 5
    op = IRDFT(Complex{Float64}, (n, div(m, 2) + 1, l), m, 2)
    x1 = rfft(randn(n, m, l), 2)
    y1 = test_op(op, x1, irfft(x1, m, 2), verb)
    y2 = irfft(x1, m, 2)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    n, m, l = 5, 18, 5
    op = IRDFT(Complex{Float64}, (div(n, 2) + 1, m, l), n, 1)
    x1 = rfft(randn(n, m, l), 1)
    y1 = test_op(op, x1, irfft(x1, n, 1), verb)
    y2 = irfft(x1, n, 1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    ## other constructors
    op = IRDFT((10,), 19)

    #properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == false
end
