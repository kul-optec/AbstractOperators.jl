
@testset "Conv" begin
    n, m = 5, 6
    h = randn(m)
    op = Conv(Float64, (n,), h)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n + m - 1), verb)
    y2 = conv(x1, h)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    z1 = op' * y1;
    z2 = xcorr(y1, h)[size(op.h, 1)[1]:(end-length(op.h)+1)];
    @test all(norm.(z1 .- z2) .<= 1e-12)

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

@testset "Filt" begin
    n, m = 15, 2
    b, a = [1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 1.0; 1.0]
    op = Filt(Float64, (n,), b, a)
    x1 = randn(n)
    y1 = test_op(op, x1, randn(n), verb)
    y2 = filt(b, a, x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

    h = randn(10)
    op = Filt(Float64, (n, m), h)
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(n, m), verb)
    y2 = filt(h, [1.0], x1)

    @test all(norm.(y1 .- y2) .<= 1e-12)

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

@testset "MIMOFilt" begin
    m, n = 10, 2
    b = [[1.0; 0.0; 1.0; 0.0; 0.0], [1.0; 0.0; 1.0; 0.0; 0.0]]
    a = [[1.0; 1.0; 1.0], [2.0; 2.0; 2.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    x1 = randn(m, n)
    y1 = test_op(op, x1, randn(m, 1), verb)
    y2 = filt(b[1], a[1], x1[:, 1]) + filt(b[2], a[2], x1[:, 2])

    @test all(norm.(y1 .- y2) .<= 1e-12)

    m, n = 10, 3; #time samples, number of inputs
    b = [
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
        [1.0; 0.0; 1.0],
    ];
    a = [[1.0; 1.0; 1.0], [2.0; 2.0; 2.0], [3.0], [4.0], [5.0], [6.0]];
    op = MIMOFilt(Float64, (m, n), b, a)

    x1 = randn(m, n)
    y1 = test_op(op, x1, randn(m, 2), verb)
    col1 = filt(b[1], a[1], x1[:, 1]) + filt(b[2], a[2], x1[:, 2]) + filt(b[3], a[3], x1[:, 3])
    col2 = filt(b[4], a[4], x1[:, 1]) + filt(b[5], a[5], x1[:, 2]) + filt(b[6], a[6], x1[:, 3])
    y2 = [col1 col2]

    @test all(norm.(y1 .- y2) .<= 1e-12)

    m, n = 10, 3
    b = [randn(10), randn(5), randn(10), randn(2), randn(10), randn(10)]
    a = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
    op = MIMOFilt(Float64, (m, n), b, a)

    x1 = randn(m, n)
    y1 = test_op(op, x1, randn(m, 2), verb)
    col1 = filt(b[1], a[1], x1[:, 1]) + filt(b[2], a[2], x1[:, 2]) + filt(b[3], a[3], x1[:, 3])
    col2 = filt(b[4], a[4], x1[:, 1]) + filt(b[5], a[5], x1[:, 2]) + filt(b[6], a[6], x1[:, 3])
    y2 = [col1 col2]

    @test all(norm.(y1 .- y2) .<= 1e-12)

    ## other constructors
    MIMOFilt((10, 3), b, a)
    MIMOFilt((10, 3), b)
    MIMOFilt(x1, b, a)
    MIMOFilt(x1, b)

    #errors
    @test_throws ErrorException MIMOFilt(Float64, (10, 3, 2), b, a)
    a2 = [[1.0f0], [1.0f0], [1.0f0], [1.0f0], [1.0f0], [1.0f0]]
    b2 = convert.(Array{Float32,1}, b)
    @test_throws ErrorException MIMOFilt(Float64, (m, n), b2, a2)
    @test_throws ErrorException MIMOFilt(Float64, (m, n), b, a[1:(end-1)])
    push!(a2, [1.0f0])
    push!(b2, randn(Float32, 10))
    @test_throws ErrorException MIMOFilt(Float32, (m, n), b2, a2)
    a[1][1] = 0.0
    @test_throws ErrorException MIMOFilt(Float64, (m, n), b, a)

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
