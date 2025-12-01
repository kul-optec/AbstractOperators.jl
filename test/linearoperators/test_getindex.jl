if !isdefined(Main, :verb)
    const verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)
using LinearAlgebra

@testset "GetIndex" begin
    verb && println(" --- Testing GetIndex --- ")

    n, m = 5, 4
    k = 3
    op = GetIndex(Float64, (n,), (1:k,))
    x1 = randn(n)
    y1 = test_op(op, x1, randn(k), verb)

    @test all(norm.(y1 .- x1[1:k]) .<= 1e-12)

    n, m = 5, 4
    k = 3
    op = GetIndex(Float64, (n, m), (1:k, :))
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(k, m), verb)

    @test all(norm.(y1 .- x1[1:k, :]) .<= 1e-12)

    n, m = 5, 4
    op = GetIndex(Float64, (n, m), (:, 2))
    x1 = randn(n, m)
    y1 = test_op(op, x1, randn(n), verb)

    @test all(norm.(y1 .- x1[:, 2]) .<= 1e-12)

    n, m, l = 5, 4, 3
    op = GetIndex(Float64, (n, m, l), (1:3, 2, :))
    x1 = randn(n, m, l)
    y1 = test_op(op, x1, randn(3, 3), verb)

    @test all(norm.(y1 .- x1[1:3, 2, :]) .<= 1e-12)

    # other constructors
    GetIndex((n, m), (1:k, :))
    GetIndex(x1, (1:k, :, :))

    @test_throws BoundsError GetIndex(Float64, (n, m), (1:k, :, :))
    op = GetIndex(Float64, (n, m), (1:n, 1:m))
    @test typeof(op) <: Eye

    op = GetIndex(Float64, (n,), (1:k,))

    ##properties
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

    # Boolean mask partial selection
    mask = falses(n, m)
    mask[1:2, 2] .= true
    op_mask = GetIndex(Float64, (n, m), mask)
    xmask = randn(n, m)
    ymask = op_mask * xmask
    @test length(ymask) == sum(mask)
    @test all(ymask .== xmask[mask])

    # Boolean mask selecting all => should behave like Eye reshaped
    fullmask = trues(n, m)
    op_full = GetIndex(Float64, (n, m), fullmask)
    xfull = randn(n, m)
    yfull = op_full * xfull
    @test prod(size(xfull)) == length(yfull)
    # Applying adjoint should scatter back
    back = zeros(size(xfull))
    mul!(back, op_full', yfull)
    @test back ≈ xfull

    # Vector of CartesianIndex selection
    cart = collect(CartesianIndices((n, m))[1:4])
    op_cart = GetIndex(Float64, (n, m), cart)
    xcart = randn(n, m)
    ycart = op_cart * xcart
    @test ycart == xcart[cart]

    # Normal operator A'A (should be diagonal identity restricted)
    A = GetIndex(Float64, (n, m), (1:3, :))
    xA = randn(n, m)
    yA = A * xA
    normal = AbstractOperators.get_normal_op(A)
    # normal maps full shape -> full shape
    tmp = similar(xA)
    mul!(tmp, normal, xA)
    # For GetIndex selecting subset rows, A'A should be a projector with ones on selected entries
    proj = zeros(size(xA))
    proj[1:3, :] .= xA[1:3, :]
    @test tmp == proj

    # normal of adjoint is Eye on domain
    normal_adj = AbstractOperators.get_normal_op(A')
    @test typeof(normal_adj) <: Eye

    # Slicing helpers
    @test AbstractOperators.get_slicing_expr(A) == (1:3, :)
    mask_expr = AbstractOperators.get_slicing_mask(op_mask)
    @test sum(mask_expr) == sum(mask)

    # remove_slicing returns Eye of original domain
    base_eye = AbstractOperators.remove_slicing(A)
    @test typeof(base_eye) <: Eye
    @test size(base_eye) == (size(A, 1), size(A, 1))

    # opnorm vs estimate_opnorm (no direct has_fast_opnorm check)
    @test opnorm(A) == estimate_opnorm(A)

    # show output should contain arrow-like symbol for GetIndex
    io = IOBuffer(); show(io, A); strA = String(take!(io)); @test occursin("↓", strA)

    # Dimension mismatch errors in mul!
    bad_y = zeros(size(A, 1)..., 2)  # deliberately wrong extra dim
    @test_throws ArgumentError mul!(bad_y, A, xA)

    # Additional coverage-focused tests for uncovered GetIndex / NormalGetIndex branches

    @testset "GetIndex scalar Int... specialized path" begin
        n, m = 6, 7
        op = GetIndex(Float64, (n, m), (2, 3))  # both indices scalar Int → get_dim_out(::Dims, Int...) method
        x = randn(n, m)
        y = op * x
        @test size(op) == ((1,), (n, m))
        @test length(y) == 1
        @test y[1] == x[2, 3]
    end

    @testset "GetIndex boolean mask inside tuple branch" begin
        n, m, l = 5, 4, 3
        mask_first = falses(n); mask_first[2:4] .= true  # Boolean vector for first dimension
        op = GetIndex(Float64, (n, m, l), (mask_first, :, 2))  # mask used as one index among others
        x = randn(n, m, l)
        y = op * x
        @test length(y) == sum(mask_first) * m  # last dim fixed → product of kept rows and full second dim
        expected = x[mask_first, :, 2]
        @test y == expected
    end

    @testset "GetIndex AbstractArray of Int indices (multi-dim)" begin
        n = 10
        arr_idx = reshape([1,2,3,4], 2, 2)  # proper reshape with dimensions
        op = GetIndex(Float64, (n,), (arr_idx,))  # index array inside tuple triggers AbstractArray branch
        x = randn(n)
        y = op * x
        @test size(y) == (2, 2)
        @test y == x[arr_idx]
    end

    @testset "GetIndex unsupported index type error path" begin
        n, m = 5, 4
        @test_throws ArgumentError GetIndex(Float64, (n, m), (1:2, "bad"))
    end

    @testset "NormalGetIndex non-tuple idx conversion" begin
        n = 8
        N1 = AbstractOperators.NormalGetIndex(Float64, Array{Float64}, (n,), 3:3)
        @test size(N1) == ((n,), (n,))
        @test AbstractOperators.domain_type(N1) == Float64
        @test AbstractOperators.domain_storage_type(N1) == Array{Float64}
        # Use diag to avoid 0-dim view assignment in mul! for scalar idx
        d = AbstractOperators.diag(N1)
        expected = zeros(n); expected[3] = 1
        @test d == expected
    end

    @testset "NormalGetIndex vector idx conversion" begin
        n = 9
        idx_vec = [2,4,5]
        N2 = AbstractOperators.NormalGetIndex(Float64, Array{Float64}, (n,), idx_vec)
        @test size(N2) == ((n,), (n,))
        d = AbstractOperators.diag(N2)
        @test d[idx_vec] == ones(length(idx_vec))
        @test sum(d) == length(idx_vec)
    end

    @testset "GetIndex get_idx accessor" begin
        n = 7
        op = GetIndex(Float64, (n,), (2:5,))
        idx_back = AbstractOperators.get_idx(op)
        @test idx_back == (2:5,)
    end

    # Cover x::AbstractArray overloads and BitArray-specific slicing mask
    @testset "GetIndex array-first overloads and BitArray mask" begin
        # Tuple indices equal to full dims -> Eye path in x::Tuple overload
        x2 = randn(4, 3)
        op_eye_x = GetIndex(x2, (:, :))
        @test op_eye_x isa Eye
        @test op_eye_x * x2 == x2

        # Vector-of-Int indices equal to full length on 1D -> Eye path in x::Vector overload
        xv = randn(6)
        op_eye_vec = GetIndex(xv, collect(1:length(xv)))
        @test op_eye_vec isa Eye
        @test op_eye_vec * xv == xv

        # BitArray mask-specific get_slicing_mask specialization
        n, m = 3, 5
        bmask = trues(n, m)  # BitMatrix <: BitArray
        op_bmask = GetIndex(Float64, (n, m), bmask)
        # Only call get_slicing_mask if result is actually a GetIndex; otherwise it may be a Reshape(Eye)
        if op_bmask isa GetIndex
            m2 = AbstractOperators.get_slicing_mask(op_bmask)
            @test m2 === bmask
        else
            @test size(op_bmask) == ((n*m,), (n, m))  # reshaped Eye path
        end
    end

    # Small utility error path in get_dim_out(::Dims)
    @testset "get_dim_out missing indices error" begin
        @test_throws ErrorException AbstractOperators.get_dim_out((2, 3))
    end

    # Specialized adjoint for NormalGetIndex returns itself
    @testset "AdjointOperator(NormalGetIndex) identity" begin
        n = 5
        N = AbstractOperators.NormalGetIndex(Float64, Array{Float64}, (n,), 2:3)
        @test AbstractOperators.AdjointOperator(N) === N
        @test AbstractOperators.codomain_storage_type(N) == Array{Float64}
    end

    @testset "GetIndex vector of CartesianIndex via tuple (hits get_dim_out branch)" begin
        n, m = 6, 5
        cart = collect(CartesianIndices((n, m))[1:6])
        # Wrap as a tuple to force the GetIndex(domain_type, dim_in, idx::Tuple) constructor,
        # which calls get_dim_out and should take the AbstractVector{CartesianIndex} branch
        op = GetIndex(Float64, (n, m), (cart,))
        x = randn(n, m)
        y = op * x
        @test size(y) == (length(cart),)
        @test y == x[cart]
    end

end

