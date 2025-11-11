@testset "FFTShift/IFFTShift Operators" begin
    # 1D even length
    n = 4
    x = collect(1.0:n)
    A = FFTShift((n,), (1,))
    B = IFFTShift((n,), (1,))

    @test A * x == [3.0, 4.0, 1.0, 2.0]
    @test B * (A * x) == x
    @test A' * (A * x) == x  # orthogonal permutation
    @test B' * (B * x) == x

    # 1D odd length
    n = 5
    x = collect(1.0:n)
    A = FFTShift((n,), (1,))
    B = IFFTShift((n,), (1,))

    @test A * x == [4.0, 5.0, 1.0, 2.0, 3.0]
    @test B * (A * x) == x

    # 2D, both dims
    n, m = 2, 3
    X = reshape(1.0:(n*m), n, m)
    A = FFTShift((n, m), (1, 2))
    B = IFFTShift((n, m), (1, 2))
    Y = A * X
    @test size(Y) == size(X)
    @test B * Y == X

    # Properties
    @test is_orthogonal(A)
    @test is_invertible(A)
    @test is_full_row_rank(A)
    @test is_full_column_rank(A)
    @test diag_AAc(A) == 1.0
    @test diag_AcA(A) == 1.0
end

@testset "SignAlternation Operator" begin
    # 1D
    n = 6
    x = ones(n)
    S = SignAlternation((n,), (1,))
    y = S * x
    @test y == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    @test S' * (S * x) == x
    @test AbstractOperators.is_symmetric(S)
    @test is_orthogonal(S)

    # 2D across both dims
    X = ones(2, 2)
    S2 = SignAlternation((2, 2), (1, 2))
    @test S2 * X == [1.0 -1.0; -1.0 1.0]

    # mul! variants
    y1 = similar(x)
    mul!(y1, S, x)
    @test y1 == y

    y2 = similar(X)
    mul!(y2, S2, X)
    @test y2 == S2 * X
end

@testset "alternate_sign helpers" begin
    # in-place
    v = collect(1.0:4.0)
    alternate_sign!(v, 1)
    @test v == [1.0, -2.0, 3.0, -4.0]

    # out-of-place with dest
    x = reshape(1.0:4.0, 2, 2)
    y = similar(x)
    alternate_sign!(y, x, 1, 2)
    @test y == [1.0 -3.0; -2.0 4.0]

    # functional copy
    u = alternate_sign(reshape(1.0:4.0, 2, 2), 1, 2)
    @test u == [1.0 -3.0; -2.0 4.0]

    # no dirs → identity
    z = collect(1.0:3.0)
    @test_throws ArgumentError alternate_sign!(z)

    y = similar(z)
    @test_throws ArgumentError alternate_sign!(y, z)
end

@testset "fftshift/ifftshift wrappers around operators" begin
    using FFTW
    # Even length
    n = 4
    A = DFT(n)
    x = randn(n)

    # Codomain shift: should equal shifting output of A*x
    T = fftshift_op(A; codomain_shifts=(1,))
    y1 = T * x
    y2 = FFTW.fftshift(A * x, (1,))
    @test y1 ≈ y2

    # Domain shift: should equal A applied to shifted input
    T2 = fftshift_op(A; domain_shifts=(1,))
    y1 = T2 * x
    y2 = A * FFTW.fftshift(x, (1,))
    @test y1 ≈ y2

    # ifftshift variants
    T3 = ifftshift_op(A; codomain_shifts=(1,))
    y1 = T3 * x
    y2 = FFTW.ifftshift(A * x, (1,))
    @test y1 ≈ y2

    T4 = ifftshift_op(A; domain_shifts=(1,))
    y1 = T4 * x
    y2 = A * FFTW.ifftshift(x, (1,))
    @test y1 ≈ y2

    # Odd length
    n = 5
    A = DFT(n)
    x = randn(n)

    T5 = fftshift_op(A; codomain_shifts=(1,))
    @test (T5 * x) ≈ FFTW.fftshift(A * x, (1,))

    T6 = ifftshift_op(A; domain_shifts=(1,))
    @test (T6 * x) ≈ (A * FFTW.ifftshift(x, (1,)))
end

@testset "Combination rules: FFTShift/IFFTShift with DFT/IDFT" begin
    using AbstractOperators: can_be_combined, combine

    # Even length → can be combined; replacement with SignAlternation
    n = 8
    x = randn(ComplexF64, n)
    dft = DFT(ComplexF64, n)
    sh = FFTShift(ComplexF64, (n,), (1,))
    ish = IFFTShift(ComplexF64, (n,), (1,))

    @test can_be_combined(dft, sh)
    @test can_be_combined(sh, dft)
    @test can_be_combined(dft', sh)
    @test can_be_combined(sh, dft')

    c1 = combine(dft, sh)
    @test c1 isa AbstractOperator
    @test (c1 * x) ≈ (SignAlternation(codomain_type(dft), size(dft, 1), (1,)) * (dft * x))

    c2 = combine(sh, dft)
    @test (c2 * x) ≈ (dft * (SignAlternation(domain_type(dft), size(dft, 2), (1,)) * x))

    # IDFT variants
    idft = IDFT(ComplexF64, n)
    @test can_be_combined(idft, sh)
    @test can_be_combined(sh, idft)

    c3 = combine(idft, sh)
    @test (c3 * x) ≈ (SignAlternation(codomain_type(idft), size(idft, 1), (1,)) * (idft * x))

    c4 = combine(sh, idft)
    @test (c4 * x) ≈ (idft * (SignAlternation(domain_type(idft), size(idft, 2), (1,)) * x))

    # Odd length → cannot be combined
    n = 7
    dft = DFT(ComplexF64, n)
    sh = FFTShift(ComplexF64, (n,), (1,))
    @test !can_be_combined(dft, sh)
    @test !can_be_combined(sh, dft)

    # Multi-dim dirs
    n, m = 6, 10
    dft2 = DFT(ComplexF64, n, m)
    sh2 = FFTShift(ComplexF64, (n, m), (1, 2))
    @test can_be_combined(dft2, sh2)  # both even

    c5 = combine(dft2, sh2)
    X = randn(ComplexF64, n, m)
    @test (c5 * X) ≈ ((dft2 * sh2) * X)
end
