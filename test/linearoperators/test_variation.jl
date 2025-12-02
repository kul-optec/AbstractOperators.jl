if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)

using SparseArrays

@testset "Variation" begin
    verb && println(" --- Testing Variation --- ")

    for threaded in (false, true)
        n, m = 10, 5
        verb && println("  - threaded = $threaded")
        op = Variation(Float64, (n, m); threaded)
        x1 = randn(n, m)
        y1 = test_op(op, x1, randn(m * n, 2), verb)
        # size & types
        @test size(op) == ((n * m, 2), (n, m))
        @test domain_type(op) == Float64
        @test codomain_type(op) == Float64
        @test is_thread_safe(op) == true

        # Forward difference on simple ramp in first dimension, constant in second
        y1 = op * repeat(collect(range(0; stop=1, length=n)), 1, m)
        @test all(norm.(y1[:, 1] .- 1 / (n - 1)) .<= 1e-12)
        @test all(norm.(y1[:, 2]) .<= 1e-12)
        # Constant input gives zero
        const_in = fill(3.14, n, m)
        @test op * const_in ≈ zeros(n * m, 2)

        Dx = spdiagm(0 => ones(n), -1 => -ones(n - 1))
        Dx[1, 1], Dx[1, 2] = -1, 1
        Dy = spdiagm(0 => ones(m), -1 => -ones(m - 1))
        Dy[1, 1], Dy[1, 2] = -1, 1

        Dxx = kron(sparse(I, m, m), Dx)
        Dyy = kron(Dy, sparse(I, n, n))
        TV = [Dxx; Dyy]

        x1 = randn(n, m)
        @test norm(op * x1 - reshape(TV * (x1[:]), n * m, 2)) < 1e-12

        n, m, l = 100, 50, 30
        verb && println("  - threaded = $threaded")
        op = Variation(Float64, (n, m, l); threaded)
        x1 = randn(n, m, l)
        y1 = test_op(op, x1, randn(m * n * l, 3), verb)
        @test size(op) == ((n * m * l, 3), (n, m, l))
        y1 = op * reshape(repeat(collect(range(0; stop=1, length=n)), 1, m * l), n, m, l)
        @test all(norm.(y1[:, 1] .- 1 / (n - 1)) .<= 1e-12)
        @test all(norm.(y1[:, 2]) .<= 1e-12)
        @test all(norm.(y1[:, 3]) .<= 1e-12)
        # Constant 3D input zero output
        const3 = fill(-2.0, n, m, l)
        @test op * const3 ≈ zeros(n * m * l, 3)

        ### other constructors
        Variation(Float64, n, m)
        Variation((n, m))
        Variation(n, m)
        Variation(x1)

        ##errors
        @test_throws ErrorException Variation(Float64, (n,))
        badX = randn(n, m + 1)
        @test_throws ArgumentError op * badX

        # Adjoint consistency: <Vx, Y> == <x, V'Y>
        x_test = randn(n, m)
        verb && println("  - threaded = $threaded")
        V = Variation(Float64, (n, m); threaded)
        Y = randn(n * m, 2)
        lhs = dot(V * x_test |> vec, vec(Y))  # vec(Vx) ⋅ vec(Y)
        z = zeros(n, m)
        mul!(z, V', Y)
        rhs = dot(vec(x_test), vec(z))
        @test abs(lhs - rhs) <= 1e-10 * (1 + abs(lhs))

        # In-place mul! forward and adjoint
        Yf = zeros(n * m, 2)
        mul!(Yf, V, x_test)
        @test Yf == V * x_test
        Zb = zeros(n, m)
        mul!(Zb, V', Yf)
        @test Zb == V' * Yf

        # Scaling
        S = Scale(2.0, V)
        @test S * x_test ≈ 2.0 * (V * x_test)
        @test_throws ErrorException Scale(1 + 2im, V)

        # Show output symbol
        io = IOBuffer(); show(io, V); s = String(take!(io)); @test occursin("Ʋ", s)

        ###properties
        @test is_linear(op) == true
        @test is_null(op) == false
        @test is_eye(op) == false
        @test is_diagonal(op) == false
        @test is_AcA_diagonal(op) == false
        @test is_AAc_diagonal(op) == false
        @test is_orthogonal(op) == false
        @test is_invertible(op) == false
        @test is_full_row_rank(op) == false
        @test is_full_column_rank(op) == false
    end
end
