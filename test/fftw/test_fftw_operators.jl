@testitem "DCT" tags = [:fftw, :DCT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n = 4
    op = DCT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = dct(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

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

    @test norm(op' * (op * x1) - x1) <= 1.0e-12
    @test diag_AAc(op) == 1.0
    @test diag_AcA(op) == 1.0
end

@testitem "IDCT" tags = [:fftw, :IDCT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n = 4
    op = IDCT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = idct(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

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

    @test norm(op' * (op * x1) - x1) <= 1.0e-12
    @test diag_AAc(op) == 1.0
    @test diag_AcA(op) == 1.0
end

@testitem "DFT" tags = [:fftw, :DFT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n, m = 4, 7

    op = DFT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Complex{Float64}, (n,))
    x1 = randn(n) + im * randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Complex{Float64}, (n,))
    x1 = randn(n) + im * randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Float64, (n,), 1)
    x1 = randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1, 1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Complex{Float64}, (n,), 1)
    x1 = randn(n) + im * randn(n)
    y1 = test_op(op, x1, fft(randn(n)), verb)
    y2 = fft(x1, 1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Float64, (n, m))
    x1 = randn(n, m)
    y1 = test_op(op, x1, fft(randn(n, m)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Complex{Float64}, (n, m))
    x1 = randn(n, m) + im * randn(n, m)
    y1 = test_op(op, x1, fft(randn(n, m)), verb)
    y2 = fft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Float64, (m, n), 1)
    x1 = randn(m, n)
    y1 = test_op(op, x1, fft(randn(m, n)), verb)
    y2 = fft(x1, 1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    op = DFT(Complex{Float64}, (n, m), 2)
    x1 = randn(n, m) + im * randn(n, m)
    y1 = test_op(op, x1, fft(randn(n, m)), verb)
    y2 = fft(x1, 2)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

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
    @test norm(op' * (op * x1) - diag_AcA(op) * x1) <= 1.0e-12
    @test norm(op * (op' * y1) - diag_AAc(op) * y1) <= 1.0e-12
end

@testitem "DFT ORTHO normalization" tags = [:fftw, :DFT] setup = [TestUtils] begin
    using AbstractOperators, FFTW, LinearAlgebra, Random, FFTWOperators
    Random.seed!(0)
    n = 8
    op = DFT(Float64, (n,); normalization = FFTWOperators.ORTHO)
    x1 = randn(n)
    y1 = op * x1
    y2 = fft(x1) ./ sqrt(n)
    @test norm(y1 .- y2) <= 1.0e-12
    # Adjoint applies ORTHO scaling too
    z1 = op' * y1
    @test norm(z1 .- x1) <= 1.0e-12
end

@testitem "IDFT" tags = [:fftw, :IDFT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

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
    @test norm(op' * (op * x1) - diag_AcA(op) * x1) <= 1.0e-12
    @test norm(op * (op' * y1) - diag_AAc(op) * y1) <= 1.0e-12
end

@testitem "RDFT" tags = [:fftw, :RDFT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n = 4
    op = RDFT(Float64, (n,))
    x1 = randn(n)
    y1 = test_op(op, x1, rfft(x1), verb)
    y2 = rfft(x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n, m, l = 4, 8, 5
    op = RDFT(Float64, (n, m, l), 2)
    x1 = randn(n, m, l)
    y1 = test_op(op, x1, rfft(x1, 2), verb)
    y2 = rfft(x1, 2)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

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

@testitem "IRDFT" tags = [:fftw, :IRDFT] setup = [TestUtils] begin
    using AbstractOperators
    using FFTW, LinearAlgebra, Random, FFTWOperators
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n = 12
    op = IRDFT(Complex{Float64}, (div(n, 2) + 1,), n)
    x1 = rfft(randn(n))
    y1 = test_op(op, x1, irfft(randn(div(n, 2) + 1), n), verb)
    y2 = irfft(x1, n)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n = 11
    op = IRDFT(Complex{Float64}, (div(n, 2) + 1,), n)
    x1 = rfft(randn(n))
    y1 = test_op(op, x1, irfft(randn(div(n, 2) + 1), n), verb)
    y2 = irfft(x1, n)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n, m, l = 4, 19, 5
    op = IRDFT(Complex{Float64}, (n, div(m, 2) + 1, l), m, 2)
    x1 = rfft(randn(n, m, l), 2)
    y1 = test_op(op, x1, irfft(x1, m, 2), verb)
    y2 = irfft(x1, m, 2)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n, m, l = 4, 18, 5
    op = IRDFT(Complex{Float64}, (n, div(m, 2) + 1, l), m, 2)
    x1 = rfft(randn(n, m, l), 2)
    y1 = test_op(op, x1, irfft(x1, m, 2), verb)
    y2 = irfft(x1, m, 2)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    n, m, l = 5, 18, 5
    op = IRDFT(Complex{Float64}, (div(n, 2) + 1, m, l), n, 1)
    x1 = rfft(randn(n, m, l), 1)
    y1 = test_op(op, x1, irfft(x1, n, 1), verb)
    y2 = irfft(x1, n, 1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

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

@testitem "DCT/IDCT (GPU)" tags = [:gpu, :fftw, :DCT, :IDCT] setup = [TestUtils, GpuEnvSetup] begin
    using AbstractOperators, FFTW, FFTWOperators, GPUEnv, Random
    has_accelerated_dcts = Base.find_package("AcceleratedDCTs") !== nothing

    for backend in gpu_backends(; include_jlarrays = false, supports_fftw = true)
        Random.seed!(0)

        has_accelerated_dcts || continue
        import AcceleratedDCTs

        dims = (8, 10)
        x_cpu = randn(Float32, dims...)
        x_gpu = to_gpu(backend, x_cpu)

        dct_op = DCT(x_gpu)
        y_gpu = similar(x_gpu)
        mul!(y_gpu, dct_op, x_gpu)
        @test y_gpu isa typeof(x_gpu)
        @test collect(y_gpu) ≈ dct(x_cpu) rtol = 1.0e-4 atol = 1.0e-4

        idct_op = IDCT(x_gpu)
        x_rec_gpu = similar(x_gpu)
        mul!(x_rec_gpu, idct_op, y_gpu)
        @test collect(x_rec_gpu) ≈ x_cpu rtol = 1.0e-4 atol = 1.0e-4

        x_adj_gpu = similar(x_gpu)
        mul!(x_adj_gpu, dct_op', y_gpu)
        @test collect(x_adj_gpu) ≈ x_cpu rtol = 1.0e-4 atol = 1.0e-4
    end
end

@testitem "DFT/RDFT/IRDFT (GPU)" tags = [:gpu, :fftw, :DFT, :RDFT, :IRDFT] setup = [TestUtils, GpuEnvSetup] begin
    using FFTW, FFTWOperators, GPUEnv, LinearAlgebra, Random, AbstractOperators

    for backend in gpu_backends(; include_jlarrays = false, supports_fftw = true)
        Random.seed!(0)
        n, m = 8, 6

        x_cpu = randn(Float64, n, m)
        op = DFT(to_gpu(backend, x_cpu))
        @test op isa DFT
        x_gpu = to_gpu(backend, x_cpu)
        y_gpu = similar(x_gpu, Complex{Float64})
        mul!(y_gpu, op, x_gpu)
        @test Array(y_gpu) ≈ fft(x_cpu)
        r_gpu = similar(x_gpu, Float64)
        mul!(r_gpu, op', y_gpu)
        @test Array(r_gpu) ≈ real.(bfft(Array(y_gpu)))

        xc_cpu = randn(ComplexF64, n, m)
        opc = DFT(to_gpu(backend, xc_cpu))
        @test opc isa DFT
        xc_gpu = to_gpu(backend, xc_cpu)
        yc_gpu = similar(xc_gpu)
        mul!(yc_gpu, opc, xc_gpu)
        @test Array(yc_gpu) ≈ fft(xc_cpu)

        opr = RDFT(to_gpu(backend, x_cpu))
        yr_gpu = similar(x_gpu, Complex{Float64}, n ÷ 2 + 1, m)
        mul!(yr_gpu, opr, x_gpu)
        @test Array(yr_gpu) ≈ rfft(x_cpu, 1)

        opir = IRDFT(to_gpu(backend, Array(yr_gpu)), n)
        out_gpu = similar(x_gpu)
        mul!(out_gpu, opir, yr_gpu)
        @test Array(out_gpu) ≈ x_cpu rtol = 1.0e-12
    end
end
