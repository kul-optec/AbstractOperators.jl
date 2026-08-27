@testitem "Conv" tags = [:dsp, :Conv] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    using AbstractOperators: is_linear, is_null, is_eye, is_diagonal, is_AcA_diagonal, is_AAc_diagonal, is_orthogonal, is_invertible, is_full_row_rank, is_full_column_rank

    n, m = 5, 6
    h = randn(m)
    op = Conv(Float64, (n,), h)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n + m - 1), verb)
    y2 = conv(x1, h)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    z1 = op' * y1
    z2 = xcorr(y1, h)[size(op.h, 1)[1]:(end - length(op.h) + 1)]
    @test all(norm.(z1 .- z2) .<= 1.0e-12)

    # other constructors
    op = Conv(x1, h)

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
    @test is_full_column_rank(op) == true
end

@testitem "Conv complex domain" tags = [:dsp, :Conv] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    Random.seed!(0)
    n, m = 5, 6
    h = randn(ComplexF64, m)
    op = Conv(ComplexF64, (n,), h)
    x1 = randn(ComplexF64, n)
    y1 = test_op(op, x1, randn(ComplexF64, n + m - 1), verb)
    y2 = conv(x1, h)
    @test all(norm.(y1 .- y2) .<= 1.0e-10)
end

@testitem "Filt: a[1] != 1 normalization" tags = [:dsp, :Filt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random
    Random.seed!(0)
    n = 10
    b = [2.0; 0.0; 2.0; 0.0; 0.0]
    a = [2.0; 2.0; 2.0]    # a[1] != 1, triggers normalization
    op = Filt(Float64, (n,), b, a)
    # after normalization b/2 and a/2 => same IIR
    op_ref = Filt(Float64, (n,), b ./ 2, a ./ 2)
    x1 = randn(n)
    @test op * x1 ≈ op_ref * x1
end

@testitem "Filt: IIR and FIR mappings" tags = [:dsp, :Filt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random

    n, m = 15, 2
    b, a = [1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 1.0; 1.0]
    op = Filt(Float64, (n,), b, a)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = filt(b, a, x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)

    h = randn(10)
    op = Filt(Float64, (n, m), h)
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(n, m), verb)
    y2 = filt(h, [1.0], x1)

    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "Filt: constructors and properties" tags = [:dsp, :Filt] setup = [TestUtils] begin
    using DSPOperators
    using AbstractOperators:
        is_linear,
        is_null,
        is_eye,
        is_diagonal,
        is_AcA_diagonal,
        is_AAc_diagonal,
        is_orthogonal,
        is_invertible,
        is_full_row_rank,
        is_full_column_rank

    n, m = 15, 2
    b, a = [1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 1.0; 1.0]
    h = randn(10)
    x1 = randn(n, m)
    op = Filt(Float64, (n, m), h)

    # other constructors
    Filt(n, b, a)
    Filt((n, m), b, a)
    Filt(n, h)
    Filt((n,), h)
    Filt(x1, b, a)
    Filt(x1, b)

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
    @test is_full_column_rank(op) == true
end

@testitem "MIMOFilt: 2-input 1-output mapping" tags = [:dsp, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random

    m, n = 10, 2
    b = [[1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 0.0; 1.0; 0.0; 0.0]]
    a = [[1.0; 1.0; 1.0], [2.0; 2.0; 2.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    x1 = randn(m, n)
    y1 = test_op(op, x1, randn(m, 1), verb)
    y2 = filt(b[1], a[1], x1[:, 1]) + filt(b[2], a[2], x1[:, 2])

    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "MIMOFilt: 3-input 2-output mapping" tags = [:dsp, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random

    m, n = 10, 3  #time samples, number of inputs
    b = [
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
    ]
    a = [[1.0; 1.0; 1.0], [2.0; 2.0; 2.0], [3.0], [4.0], [5.0], [6.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    x1 = randn(m, n)
    y1 = test_op(op, x1, randn(m, 2), verb)
    col1 = filt(b[1], a[1], x1[:, 1]) + filt(b[2], a[2], x1[:, 2]) + filt(b[3], a[3], x1[:, 3])
    col2 = filt(b[4], a[4], x1[:, 1]) + filt(b[5], a[5], x1[:, 2]) + filt(b[6], a[6], x1[:, 3])
    y2 = [col1 col2]

    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "MIMOFilt: FIR mapping" tags = [:dsp, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators, DSP, LinearAlgebra, Random

    m, n = 10, 3
    b = [randn(10), randn(5), randn(10), randn(2), randn(10), randn(10)]
    a = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    x1 = randn(m, n)
    y1 = test_op(op, x1, randn(m, 2), verb)
    col1 = filt(b[1], a[1], x1[:, 1]) + filt(b[2], a[2], x1[:, 2]) + filt(b[3], a[3], x1[:, 3])
    col2 = filt(b[4], a[4], x1[:, 1]) + filt(b[5], a[5], x1[:, 2]) + filt(b[6], a[6], x1[:, 3])
    y2 = [col1 col2]

    @test all(norm.(y1 .- y2) .<= 1.0e-12)
end

@testitem "MIMOFilt: constructors and errors" tags = [:dsp, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators

    m, n = 10, 3
    b = [randn(10), randn(5), randn(10), randn(2), randn(10), randn(10)]
    a = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    x1 = randn(m, n)

    ## other constructors
    MIMOFilt((10, 3), b, a)
    MIMOFilt((10, 3), b)
    MIMOFilt(x1, b, a)
    MIMOFilt(x1, b)

    #errors
    @test_throws ErrorException MIMOFilt(Float64, (10, 3, 2), b, a)
    a2 = [[1.0f0], [1.0f0], [1.0f0], [1.0f0], [1.0f0], [1.0f0]]
    b2 = convert.(Array{Float32, 1}, b)
    @test_throws ErrorException MIMOFilt(Float64, (m, n), b2, a2)
    @test_throws ErrorException MIMOFilt(Float64, (m, n), b, a[1:(end - 1)])
    push!(a2, [1.0f0])
    push!(b2, randn(Float32, 10))
    @test_throws ErrorException MIMOFilt(Float32, (m, n), b2, a2)
    a[1][1] = 0.0
    @test_throws ErrorException MIMOFilt(Float64, (m, n), b, a)
end

@testitem "MIMOFilt: properties" tags = [:dsp, :MIMOFilt] setup = [TestUtils] begin
    using DSPOperators
    using AbstractOperators:
        is_linear,
        is_null,
        is_eye,
        is_diagonal,
        is_AcA_diagonal,
        is_AAc_diagonal,
        is_orthogonal,
        is_invertible,
        is_full_row_rank,
        is_full_column_rank

    m, n = 10, 3
    b = [randn(10), randn(5), randn(10), randn(2), randn(10), randn(10)]
    a = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    ##properties
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == false
    @test is_AAc_diagonal(op) == false
    @test is_orthogonal(op) == false
    @test is_invertible(op) == true
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == true
end

@testitem "Conv (GPU)" tags = [:gpu, :dsp, :Conv] setup = [TestUtils, GpuEnvSetup] begin
    using DSPOperators, DSP, GPUEnv, LinearAlgebra, Random
    for backend in gpu_backends(; include_jlarrays = false, supports_fftw = true)
        Random.seed!(0)
        n, m = 20, 6
        h_cpu = randn(m)
        x_cpu = randn(n)
        h = to_gpu(backend, h_cpu)
        op = Conv(Float64, (n,), h)
        x = to_gpu(backend, x_cpu)
        y = gpu_zeros(backend, Float64, n + m - 1)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        op_cpu = Conv(Float64, (n,), h_cpu)
        y_cpu = zeros(n + m - 1)
        mul!(y_cpu, op_cpu, x_cpu)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        b_cpu = randn(n + m - 1)
        b = to_gpu(backend, b_cpu)
        z = gpu_zeros(backend, Float64, n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', b_cpu)
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "Xcorr (GPU)" tags = [:gpu, :dsp, :Xcorr] setup = [TestUtils, GpuEnvSetup] begin
    using DSPOperators, DSP, GPUEnv, LinearAlgebra, Random
    for backend in gpu_backends(; include_jlarrays = false, supports_fftw = true)
        Random.seed!(0)
        n, m = 15, 5
        h_cpu = randn(m)
        x_cpu = randn(n)
        h = to_gpu(backend, h_cpu)
        outlen = 2 * max(n, m) - 1
        op = Xcorr(Float64, (n,), h)
        x = to_gpu(backend, x_cpu)
        y = gpu_zeros(backend, Float64, outlen)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        op_cpu = Xcorr(Float64, (n,), h_cpu)
        y_cpu = zeros(outlen)
        mul!(y_cpu, op_cpu, x_cpu)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        b_cpu = randn(outlen)
        b = to_gpu(backend, b_cpu)
        z = gpu_zeros(backend, Float64, n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', b_cpu)
        mul!(z, op', b)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "Filt (GPU, FIR)" tags = [:gpu, :dsp, :Filt] setup = [TestUtils, GpuEnvSetup] begin
    using DSPOperators, DSP, GPUEnv, LinearAlgebra, Random
    for backend in gpu_backends(; include_jlarrays = false, supports_fftw = true)
        Random.seed!(42)
        n = 20
        b = randn(5)
        x_cpu = randn(n)
        op_cpu = Filt(Float64, (n,), b)
        y_cpu = zeros(n)
        mul!(y_cpu, op_cpu, x_cpu)
        x = to_gpu(backend, x_cpu)
        op = Filt(x, b)
        y = gpu_zeros(backend, Float64, n)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        r_cpu = randn(n)
        z_cpu = zeros(n)
        mul!(z_cpu, op_cpu', r_cpu)
        r = to_gpu(backend, r_cpu)
        z = gpu_zeros(backend, Float64, n)
        mul!(z, op', r)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "MIMOFilt (GPU, FIR)" tags = [:gpu, :dsp, :MIMOFilt] setup = [TestUtils, GpuEnvSetup] begin
    using DSPOperators, DSP, GPUEnv, LinearAlgebra, Random
    for backend in gpu_backends(; include_jlarrays = false, supports_fftw = true)
        Random.seed!(7)
        m, n = 10, 3
        b = [randn(5), randn(3), randn(4), randn(5), randn(3), randn(4)]
        a = [[1.0] for _ in b]
        op_cpu = MIMOFilt(Float64, (m, n), b, a)
        x_cpu = randn(m, n)
        y_cpu = zeros(m, 2)
        mul!(y_cpu, op_cpu, x_cpu)
        x = to_gpu(backend, x_cpu)
        op = MIMOFilt(x, b, a)
        y = gpu_zeros(backend, Float64, m, 2)
        mul!(y, op, x)
        y_alloc = op * x
        @test y_alloc isa typeof(y)
        @test collect(y) ≈ y_cpu atol = 1.0e-10
        @test collect(y_alloc) ≈ y_cpu atol = 1.0e-10
        r_cpu = randn(m, 2)
        z_cpu = zeros(m, n)
        mul!(z_cpu, op_cpu', r_cpu)
        r = to_gpu(backend, r_cpu)
        z = gpu_zeros(backend, Float64, m, n)
        mul!(z, op', r)
        @test collect(z) ≈ z_cpu atol = 1.0e-10
    end
end

@testitem "Xcorr complex domain" tags = [:dsp, :Xcorr] setup = [TestUtils] begin
    using DSPOperators, LinearAlgebra, Random
    Random.seed!(0)
    n, m = 5, 6
    h = randn(ComplexF64, m)
    op = Xcorr(ComplexF64, (n,), h)
    x1 = randn(ComplexF64, n)
    y1 = op * x1
    @test length(y1) == 2 * max(n, m) - 1
    z1 = op' * y1
    @test length(z1) == n
    @test z1 ≈ op' * (op * x1)
end

@testitem "Conv/Xcorr: copy_operator honours storage_type and threaded" tags = [
    :dsp, :Conv, :Xcorr,
] setup = [TestUtils] begin
    using DSPOperators, AbstractOperators, LinearAlgebra, Random
    Random.seed!(3)

    n, m = 5, 6
    h = randn(m)
    x = randn(n)

    conv_op = Conv(Float64, (n,), h)
    y_conv = conv_op * x
    conv_copy = copy_operator(conv_op; storage_type = Array{Float64})
    @test conv_copy isa Conv
    @test conv_copy.h !== conv_op.h
    @test conv_copy.h == conv_op.h
    @test conv_copy * x ≈ y_conv

    # storage_type and threaded together
    conv_copy2 = copy_operator(conv_op; storage_type = Array{Float64}, threaded = true)
    @test conv_copy2 * x ≈ y_conv

    xcorr_op = Xcorr(Float64, (n,), h)
    y_xcorr = xcorr_op * x
    xcorr_copy = copy_operator(xcorr_op; storage_type = Array{Float64})
    @test xcorr_copy isa Xcorr
    @test xcorr_copy.h !== xcorr_op.h
    @test xcorr_copy.h == xcorr_op.h
    @test xcorr_copy * x ≈ y_xcorr

    xcorr_copy2 = copy_operator(xcorr_op; storage_type = Array{Float64}, threaded = true)
    @test xcorr_copy2 * x ≈ y_xcorr
end
