if !isdefined(Main, :verb)
    const verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
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
    @test size(base_eye) == ((n, m), (n, m))

    # opnorm vs estimate_opnorm (no direct has_fast_opnorm check)
    @test LinearAlgebra.opnorm(A) == estimate_opnorm(A)

    # show output should contain arrow-like symbol for GetIndex
    io = IOBuffer(); show(io, A); strA = String(take!(io)); @test occursin("↓", strA)

    # Dimension mismatch errors in mul!
    bad_y = zeros(size(A, 1)..., 2)  # deliberately wrong extra dim
    @test_throws MethodError mul!(bad_y, A, xA)
end
