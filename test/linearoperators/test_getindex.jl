@testitem "GetIndex: basic mul" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, k = 5, 3
    test_op(GetIndex(zeros(Float64, n), (1:k,)), randn(n), randn(k), verb)
    n, m = 5, 4
    k = 3
    test_op(GetIndex(zeros(Float64, n, m), (1:k, :)), randn(n, m), randn(k, m), verb)

    op = GetIndex(Float64, (n,), (1:k,); array_type = Array{ComplexF32, 2})
    @test domain_array_type(op) == Array{Float64}
    @test codomain_array_type(op) == Array{Float64}

    op2d = GetIndex(Float64, (n, m), (:, 2))
    x1 = randn(n, m)
    y1 = test_op(op2d, x1, randn(n), verb)
    @test all(norm.(y1 .- x1[:, 2]) .<= 1.0e-12)

    @test_throws BoundsError GetIndex(Float64, (n, m), (1:k, :, :))
    @test typeof(GetIndex(Float64, (n, m), (1:n, 1:m))) <: Eye
end

@testitem "GetIndex: properties" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, m, k = 5, 4, 3
    op = GetIndex(Float64, (n,), (1:k,))
    @test is_sliced(op) == true
    @test is_linear(op) == true
    @test is_null(op) == false
    @test is_eye(op) == false
    @test is_diagonal(op) == false
    @test is_AcA_diagonal(op) == true
    @test is_AAc_diagonal(op) == true
    @test is_orthogonal(op) == false
    @test is_invertible(op) == false
    @test is_full_row_rank(op) == true
    @test is_full_column_rank(op) == false
    @test diag_AAc(op) == 1
    @test AbstractOperators.has_fast_opnorm(op) == true
    @test opnorm(op) == estimate_opnorm(op)
end

@testitem "GetIndex: boolean mask and CartesianIndex" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, m = 5, 4
    mask = falses(n, m)
    mask[1:2, 2] .= true
    op_mask = GetIndex(Float64, (n, m), mask)
    xmask = randn(n, m)
    ymask = op_mask * xmask
    @test length(ymask) == sum(mask) && all(ymask .== xmask[mask])

    fullmask = trues(n, m)
    op_full = GetIndex(Float64, (n, m), fullmask)
    xfull = randn(n, m)
    yfull = op_full * xfull
    back = zeros(size(xfull))
    mul!(back, op_full', yfull)
    @test back ≈ xfull

    mask_vec = [isodd(i) for i in 1:n]
    op_mask_vec = GetIndex(Float64, (n,), mask_vec)
    xmask_vec = randn(n)
    @test op_mask_vec * xmask_vec == xmask_vec[mask_vec]

    op_mask_vec_tuple = GetIndex(Float64, (n,), (mask_vec,))
    @test op_mask_vec_tuple * xmask_vec == xmask_vec[mask_vec]

    cart = collect(CartesianIndices((n, m))[1:4])
    op_cart = GetIndex(Float64, (n, m), cart)
    xcart = randn(n, m)
    @test op_cart * xcart == xcart[cart]
end

@testitem "GetIndex: normal op and slicing helpers" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(0)

    n, m = 5, 4
    A = GetIndex(Float64, (n, m), (1:3, :))
    xA = randn(n, m)
    normal = AbstractOperators.get_normal_op(A)
    tmp = similar(xA)
    mul!(tmp, normal, xA)
    proj = zeros(size(xA))
    proj[1:3, :] .= xA[1:3, :]
    @test tmp == proj
    @test typeof(AbstractOperators.get_normal_op(A')) <: Eye

    @test AbstractOperators.get_slicing_expr(A) == (1:3, :)
    mask = falses(n, m)
    mask[1:2, 2] .= true
    op_mask = GetIndex(Float64, (n, m), mask)
    @test sum(AbstractOperators.get_slicing_mask(op_mask)) == sum(mask)

    base_eye = AbstractOperators.remove_slicing(A)
    @test typeof(base_eye) <: Eye
    @test size(base_eye) == (size(A, 1), size(A, 1))

    io = IOBuffer()
    show(io, A)
    @test occursin("↓", String(take!(io)))
end

@testitem "GetIndex (GPU)" tags = [:linearoperator, :GetIndex, :gpu] setup = [TestUtils] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(0)
        n, m, k = 5, 4, 3
        backend_type_wrapper = gpu_wrapper(backend, randn(1))

        test_op(GetIndex(gpu_zeros(backend, Float64, n), (1:k,)), gpu_randn(backend, n), gpu_randn(backend, k), false)
        test_op(
            GetIndex(gpu_zeros(backend, Float64, n, m), (1:k, :)),
            gpu_randn(backend, n, m),
            gpu_randn(backend, k, m),
            false,
        )
        test_op(
            GetIndex(gpu_zeros(backend, Float64, n, m), (:, 2)),
            gpu_randn(backend, n, m),
            gpu_randn(backend, n),
            false,
        )
        test_op(
            GetIndex(gpu_zeros(backend, Float64, n, m), (2:k, 2:m)),
            gpu_randn(backend, n, m),
            gpu_randn(backend, k - 1, m - 1),
            false,
        )
        test_op(
            GetIndex(gpu_zeros(backend, Float64, n, m), (1:k, 2)),
            gpu_randn(backend, n, m),
            gpu_randn(backend, k),
            false,
        )

        idx_vec = collect(1:2:n)
        op_vec = GetIndex(gpu_zeros(backend, Float64, n), idx_vec)
        @test Base.typename(typeof(op_vec.idx[1])).wrapper == backend_type_wrapper
        test_op(op_vec, gpu_randn(backend, n), gpu_randn(backend, length(idx_vec)), false)
        op_vec_kw = GetIndex(Float64, (n,), idx_vec; array_type = gpu_wrapper(backend, Float64, 1))
        @test Base.typename(typeof(op_vec_kw.idx[1])).wrapper == backend_type_wrapper

        mask = falses(n)
        mask[2:2:n] .= true
        op_mask = GetIndex(gpu_zeros(backend, Float64, n), mask)
        @test Base.typename(typeof(op_mask.idx[1])).wrapper == backend_type_wrapper
        @test length(op_mask.idx[1]) == sum(mask)
        test_op(op_mask, gpu_randn(backend, n), gpu_randn(backend, sum(mask)), false)
        op_mask_kw = GetIndex(Float64, (n,), mask; array_type = gpu_wrapper(backend, Float64, 1))
        @test Base.typename(typeof(op_mask_kw.idx[1])).wrapper == backend_type_wrapper
    end
end

@testitem "GetIndex scalar Int specialized path" tags = [:linearoperator, :GetIndex] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 6, 7
    op = GetIndex(Float64, (n, m), (2, 3))
    x = randn(n, m)
    y = op * x
    @test size(op) == ((1,), (n, m))
    @test length(y) == 1
    @test y[1] == x[2, 3]
end

@testitem "GetIndex boolean mask inside tuple branch" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m, l = 5, 4, 3
    mask_first = falses(n)
    mask_first[2:4] .= true
    op = GetIndex(Float64, (n, m, l), (mask_first, :, 2))
    x = randn(n, m, l)
    y = op * x
    @test length(y) == sum(mask_first) * m
    @test y == x[mask_first, :, 2]
end

@testitem "GetIndex AbstractArray of Int indices (multi-dim)" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    arr_idx = reshape([1, 2, 3, 4], 2, 2)
    op = GetIndex(Float64, (10,), (arr_idx,))
    x = randn(10)
    y = op * x
    @test size(y) == (2, 2)
    @test y == x[arr_idx]
end

@testitem "GetIndex unsupported index type error path" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    @test_throws ArgumentError GetIndex(Float64, (5, 4), (1:2, "bad"))
end

@testitem "NormalGetIndex non-tuple idx conversion" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n = 8
    N1 = AbstractOperators.NormalGetIndex(Float64, Array{Float64}, (n,), 3:3)
    @test size(N1) == ((n,), (n,))
    @test AbstractOperators.domain_type(N1) == Float64
    @test AbstractOperators.domain_array_type(N1) == Array{Float64}
    expected = zeros(n)
    expected[3] = 1
    @test AbstractOperators.diag(N1) == expected
end

@testitem "NormalGetIndex vector idx conversion" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    idx_vec = [2, 4, 5]
    N2 = AbstractOperators.NormalGetIndex(Float64, Array{Float64}, (9,), idx_vec)
    d = AbstractOperators.diag(N2)
    @test d[idx_vec] == ones(length(idx_vec))
    @test sum(d) == length(idx_vec)
end

@testitem "GetIndex get_idx accessor" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    @test AbstractOperators.get_idx(GetIndex(Float64, (7,), (2:5,))) == (2:5,)
end

@testitem "GetIndex array-first overloads and BitArray mask" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    x2 = randn(4, 3)
    op_eye_x = GetIndex(x2, (:, :))
    @test op_eye_x isa Eye
    @test op_eye_x * x2 == x2
    xv = randn(6)
    op_eye_vec = GetIndex(xv, collect(1:length(xv)))
    @test op_eye_vec isa Eye
    @test op_eye_vec * xv == xv
    n, m = 3, 5
    bmask = trues(n, m)
    op_bmask = GetIndex(Float64, (n, m), bmask)
    if op_bmask isa GetIndex
        @test AbstractOperators.get_slicing_mask(op_bmask) === bmask
    else
        @test size(op_bmask) == ((n * m,), (n, m))
    end

    bool_vec = [i % 2 == 0 for i in 1:length(xv)]
    op_bool_vec = GetIndex(xv, bool_vec)
    @test op_bool_vec * xv == xv[bool_vec]
end

@testitem "get_dim_out missing indices error" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    @test_throws ErrorException AbstractOperators.get_dim_out((2, 3))
end

@testitem "AdjointOperator(NormalGetIndex) identity" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    N = AbstractOperators.NormalGetIndex(Float64, Array{Float64}, (5,), 2:3)
    @test AbstractOperators.AdjointOperator(N) === N
    @test AbstractOperators.codomain_array_type(N) == Array{Float64}
end

@testitem "GetIndex vector of CartesianIndex via tuple" tags = [:linearoperator, :GetIndex] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(0)
    n, m = 6, 5
    cart = collect(CartesianIndices((n, m))[1:6])
    op = GetIndex(Float64, (n, m), (cart,))
    x = randn(n, m)
    y = op * x
    @test size(y) == (length(cart),)
    @test y == x[cart]
end
