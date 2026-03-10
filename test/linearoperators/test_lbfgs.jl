@testitem "L-BFGS" tags = [:linearoperator, :LBFGS] setup = [TestUtils] begin
    using AbstractOperators
    using AbstractOperators: LBFGS, update!, mul!, reset!
    verb && println(" --- Testing L-BFGS --- ")

    Q = [
        32.0 13.1 -4.9 -3.0 6.0 2.2 2.6 3.4 -1.9 -7.5
        13.1 18.3 -5.3 -9.5 3.0 2.1 3.9 3.0 -3.6 -4.4
        -4.9 -5.3 7.7 2.1 -0.4 -3.4 -0.8 -3.0 5.3 5.5
        -3.0 -9.5 2.1 20.1 1.1 0.8 -12.4 -2.5 5.5 2.1
        6.0 3.0 -0.4 1.1 3.8 0.6 0.5 0.9 -0.4 -2.0
        2.2 2.1 -3.4 0.8 0.6 7.8 2.9 -1.3 -4.3 -5.1
        2.6 3.9 -0.8 -12.4 0.5 2.9 14.5 1.7 -4.9 1.2
        3.4 3.0 -3.0 -2.5 0.9 -1.3 1.7 6.6 -0.8 2.7
        -1.9 -3.6 5.3 5.5 -0.4 -4.3 -4.9 -0.8 7.9 5.7
        -7.5 -4.4 5.5 2.1 -2.0 -5.1 1.2 2.7 5.7 16.1
    ]

    q = [
        2.9, 0.8, 1.3, -1.1, -0.5, -0.3, 1.0, -0.3, 0.7, -2.1,
    ]

    xs =
        [
        1.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
        0.09 1.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08
        0.08 0.09 1.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07
        0.07 0.08 0.09 1.0 0.01 0.02 0.03 0.04 0.05 0.06
        0.06 0.07 0.08 0.09 1.0 0.01 0.02 0.03 0.04 0.05
    ]'

    dirs_ref = [
        -3.476e+1 -6.861170733797231e-1 -1.621334774299757e-1 -2.008976150849174e-1 -2.317011191832649e-1
        -1.3677e+1 -1.661270665201917e+0 2.870743130038511e-1 2.237224648542354e-1 2.980080835636926e-2
        2.961e+0 2.217225828759783e-1 -5.485761164147891e-1 4.811889625788801e-2 -1.267017945785352e-1
        3.756e+0 5.615134140894827e-1 9.992734938824949e-2 -6.855884193567087e-1 4.328230970765587e-2
        -5.618000000000001e+0 -1.922426760799171e-1 -1.332550298134261e-2 -2.729265954345345e-2 -2.437461022925742e-1
        -1.571e+0 -8.961101045874649e-2 5.326252573648003e-2 3.651730112313705e-2 1.349716200511426e-2
        -4.121e+0 -3.044802963260585e-1 -6.2994080682891e-2 6.325330777317102e-2 -7.155992987801297e-4
        -3.709e+0 -1.996235459345302e-1 1.525398352758626e-2 2.871281112230844e-2 -3.513449694839536e-3
        4.01e-1 1.267604425710271e-1 -7.776943954825602e-2 -1.285590864125103e-1 -5.603489763638488e-2
        7.639999999999999e+0 3.360845247013288e-1 -2.3358849535076e-2 -3.204963735369062e-3 5.612114259243499e-2
    ]

    dirs = zeros(10, 5) # matrix of directions (to be filled in)

    mem = 3
    x = zeros(10)

    H = LBFGS(x, mem)
    # Basic properties after construction (no memory yet)
    @test size(H) == (size(x), size(x))
    @test domain_type(H) == eltype(x)
    @test codomain_type(H) == eltype(x)
    @test is_thread_safe(H) == false
    # Initial operator should act like identity (H.H = 1, empty memory)
    @test H * x == x # stochastic but value not stored; run separately below
    dir = zeros(10)
    verb && println(H)

    HH = LBFGS(ArrayPartition(x, x), mem)
    @test size(HH) == (size.(ArrayPartition(x, x).x), size.(ArrayPartition(x, x).x))
    dirdir = ArrayPartition(zeros(10), zeros(10))
    verb && println(HH)

    let x_old = [], grad_old = []
        for i in 1:5
            x = xs[:, i]
            grad = Q * x + q

            if i > 1
                xx = ArrayPartition(x, copy(x))
                xx_old = ArrayPartition(x_old, copy(x_old))
                gradgrad = ArrayPartition(grad, copy(grad))
                gradgrad_old = ArrayPartition(grad_old, copy(grad_old))
                if verb
                    @time update!(H, x, x_old, grad, grad_old)
                    @time update!(HH, xx, xx_old, gradgrad, gradgrad_old)
                else
                    update!(H, x, x_old, grad, grad_old)
                    update!(HH, xx, xx_old, gradgrad, gradgrad_old)
                end
            end

            dir_ref = dirs_ref[:, i]

            gradm = -grad
            if verb
                @time mul!(dir, H, gradm)
            else
                mul!(dir, H, gradm)
            end
            @test norm(dir - dir_ref, Inf) / (1 + norm(dir_ref, Inf)) <= 1.0e-15

            gradm2 = ArrayPartition(-grad, copy(-grad))
            if verb
                @time mul!(dirdir, HH, gradm2)
            else
                mul!(dirdir, HH, gradm2)
            end
            @test norm(dirdir.x[1] - dir_ref, Inf) / (1 + norm(dir_ref, Inf)) <= 1.0e-15
            @test norm(dirdir.x[2] - dir_ref, Inf) / (1 + norm(dir_ref, Inf)) <= 1.0e-15
            # Symmetry check: (H * g)' * h == g' * (H * h)
            g = randn(10)
            h = randn(10)
            Hg = similar(g)
            Hh = similar(h)
            mul!(Hg, H, g)
            mul!(Hh, H, h)
            @test abs(dot(Hg, h) - dot(g, Hh)) <= 1.0e-12 * (1 + norm(g) * norm(h))

            x_old = x
            grad_old = grad
        end
    end  # let x_old, grad_old

    # Memory limit: ensure no more than mem updates stored
    @test H.currmem <= mem
    @test 0 <= H.curridx <= mem

    # Zero curvature pair (ys <= 0) is skipped: craft y with negative curvature
    x_new = randn(10)
    grad_new = randn(10)
    # Force y = grad_new - grad_new (zero) -> ys == 0 and update should not change currmem
    prev_mem = H.currmem
    update!(H, x_new, x_new, grad_new, grad_new)
    @test H.currmem == prev_mem

    # Show output symbol
    io = IOBuffer(); show(io, H); s = String(take!(io)); @test occursin("LBFGS", s)

    # Testing reset

    @test ones(size(H, 1)) != H * ones(size(H, 1))
    @test ArrayPartition(ones.(size(HH, 1))) != HH * ArrayPartition(ones.(size(HH, 1))...)

    AbstractOperators.reset!(H)
    AbstractOperators.reset!(HH)

    @test ones(size(H, 1)) == H * ones(size(H, 1))
    @test ArrayPartition(ones.(size(HH, 1))) == HH * ArrayPartition(ones.(size(HH, 1))...)
end
