if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("../utils.jl")
end
Random.seed!(0)

@testset "BroadCast" begin
    verb && println(" --- Testing BroadCast --- ")

    function test_broadcast_op(verb, threaded)
        m, n = 8, 4
        dim_out = (m, 10)
        A1 = randn(m, n)
        opA1 = MatrixOp(A1)
        opR = BroadCast(opA1, dim_out; threaded)
        x1 = randn(n)
        y1 = test_op(opR, x1, randn(dim_out), verb)
        y2 = zeros(dim_out)
        y2 .= A1 * x1
        @test norm(y1 - y2) <= 1e-12

        m, n, l, k = 8, 4, 5, 7
        dim_out = (m, n, l, k)
        opA1 = Eye(m, n)
        opR = BroadCast(opA1, dim_out; threaded)
        x1 = randn(m, n)
        y1 = test_op(opR, x1, randn(dim_out), verb)
        y2 = zeros(dim_out)
        y2 .= x1
        @test norm(y1 - y2) <= 1e-12

        @test_throws Exception BroadCast(opA1, (m, m))

        m, l = 1, 5
        dim_out = (m, l)
        opA1 = Eye(m)
        opR = BroadCast(opA1, dim_out; threaded)
        x1 = randn(m)
        y1 = test_op(opR, x1, randn(dim_out), verb)
        y2 = zeros(dim_out)
        y2 .= x1
        @test norm(y1 - y2) <= 1e-12

        #colum in - matrix out
        m, l = 4, 5
        dim_out = (m, l)
        opA1 = Eye(1, l)
        opR = BroadCast(opA1, dim_out; threaded)
        x1 = randn(1, l)
        y1 = test_op(opR, x1, randn(dim_out), verb)
        y2 = zeros(dim_out)
        y2 .= x1
        @test norm(y1 - y2) <= 1e-12

        op = HCAT(Eye(m, l), opR)
        x1 = ArrayPartition(randn(m, l), randn(1, l))
        y1 = test_op(op, x1, randn(dim_out), verb)
        y2 = x1.x[1] .+ x1.x[2]
        @test norm(y1 - y2) <= 1e-12

        m, n, l = 2, 5, 8
        dim_out = (m, n, l)
        opA1 = Eye(m)
        opR = BroadCast(opA1, dim_out; threaded)
        x1 = randn(m)
        y1 = test_op(opR, x1, randn(dim_out), verb)
        y2 = zeros(dim_out)
        y2 .= x1
        @test norm(y1 - y2) <= 1e-12

        m, n, l = 1, 5, 8
        dim_out = (m, n, l)
        opA1 = Eye(m)
        opR = BroadCast(opA1, dim_out; threaded)
        x1 = randn(m)
        y1 = test_op(opR, x1, randn(dim_out), verb)
        y2 = zeros(dim_out)
        y2 .= x1
        @test norm(y1 - y2) <= 1e-12

        m, n, l = 1, 5, 8
        dim_out = (m, n, l)
        opA1 = Scale(2.4, Eye(m))
        opR = BroadCast(opA1, dim_out; threaded)
        x1 = randn(m)
        y1 = test_op(opR, x1, randn(dim_out), verb)
        y2 = zeros(dim_out)
        y2 .= 2.4 * x1
        @test norm(y1 - y2) <= 1e-12

        @test is_null(opR) == is_null(opA1)
        @test is_eye(opR) == false
        @test is_diagonal(opR) == false
        @test is_AcA_diagonal(opR) == false
        @test is_AAc_diagonal(opR) == false
        @test is_orthogonal(opR) == false
        @test is_invertible(opR) == false
        @test is_full_row_rank(opR) == false
        @test is_full_column_rank(opR) == false
    end

    @testset "non-threaded" test_broadcast_op(verb, false)
    @testset "threaded" test_broadcast_op(verb, true)

    # NoOperatorBroadCast path (Eye case returning NoOperatorBroadCast rather than OperatorBroadCast)
    # Use dim_out that differs to avoid early return of original operator
    m = 3
    E = Eye(m)
    SB = BroadCast(E, (m, 2))  # should be NoOperatorBroadCast
    @test SB isa AbstractOperators.NoOperatorBroadCast
    x = randn(m)
    y = SB * x
    @test y[:,1] == x && y[:,2] == x
    # fun_name for NoOperatorBroadCast (".I") appears in show
    io = IOBuffer()
    show(io, SB)
    s = String(take!(io))
    @test occursin(".I", s)
    # opnorm / estimate consistency
    @test opnorm(SB) == √2
    @test estimate_opnorm(SB) == √2
    # remove_displacement on NoOperatorBroadCast is identity
    @test remove_displacement(SB) === SB

    # remove_displacement on non-threaded OperatorBroadCast returns structurally equal broadcast
    A = MatrixOp(randn(m, m))
    OB = BroadCast(A, (m, 2); threaded=false)
    @test remove_displacement(OB) == OB
    # For a displaced inner operator
    d = randn(m)
    AD = AffineAdd(A, d)
    OBD = BroadCast(AD, (m, 2); threaded=false)
    rd = remove_displacement(OBD)
    @test rd * x == BroadCast(remove_displacement(AD), (m, 2); threaded=false) * x
    @test remove_displacement(rd) == rd  # idempotent

    # permute test (domain permutation) using HCAT to create partition domain
    A1 = HCAT(Eye(m), Eye(m))  # domain is ArrayPartition
    B1 = BroadCast(A1, (m, 2); threaded=false)
    xpart = ArrayPartition(randn(m), randn(m))
    y_base = B1 * xpart
    p = [2,1]
    B1p = AbstractOperators.permute(B1, p)
    xpart_p = ArrayPartition(xpart.x[p]...)
    y_perm = B1p * xpart_p
    @test y_perm == y_base  # same broadcasted sum after permutation inversion

    # Adjoint path exercise for OperatorBroadCast (non-threaded) to hit get_input_slice
    r = randn(size(B1,1))
    g = B1' * r
    @test length(g) == length(xpart.x[1]) + length(xpart.x[2])

    # If multi-threading available, test threaded variant basics (skip if only 1 thread)
    if Threads.nthreads() > 1
        AT = BroadCast(A, (m, 3); threaded=true)
        xt = randn(m)
        yt = AT * xt
        @test yt[:,2] == A * xt  # replicated columns
        # remove_displacement idempotence for threaded case
        @test remove_displacement(AT) == AT
        # displaced inner operator threaded
        ADT = BroadCast(AffineAdd(A, d), (m, 3); threaded=true)
        rdt = remove_displacement(ADT)
        @test rdt * xt == BroadCast(A, (m, 3); threaded=true) * xt
    end

    # test displacement

    m, n = 8, 4
    dim_out = (m, 10)
    A1 = randn(m, n)
    d1 = randn(m)
    opA1 = AffineAdd(MatrixOp(A1), d1)
    opR = BroadCast(opA1, dim_out)
    x1 = randn(n)
    y1 = opR * x1
    y2 = zeros(dim_out)
    y2 .= A1 * x1 + d1
    @test norm(y1 - y2) <= 1e-12
    x1 = randn(n)
    y1 = remove_displacement(opR) * x1
    y2 = zeros(dim_out)
    y2 .= A1 * x1
    @test norm(y1 - y2) <= 1e-12

    # Storage types / thread safety (non-thread-safe expected)
    _ds = domain_storage_type(opR)
    _cs = codomain_storage_type(opR)
    @test _ds !== nothing
    @test _cs !== nothing
    @test is_thread_safe(opR) == false

    # In-place multiplication
    m, n = 6, 3
    dim_out = (m, 5)
    A1 = randn(m, n)
    opA1 = MatrixOp(A1)
    opR = BroadCast(opA1, dim_out)
    x1 = randn(n)
    y = zeros(dim_out)
    mul!(y, opR, x1)
    yref = zeros(dim_out)
    yref .= A1 * x1
    @test norm(y - yref) <= 1e-12

    # Adjoint of BroadCast
    opR_adj = adjoint(opR)
    # If defined, test shape
    @test size(opR_adj, 1) == (n,)

    # Show/summary output
    io = IOBuffer(); show(io, opR); str = String(take!(io)); @test occursin(".", str)

    # Edge: BroadCast of Zeros
    opZ = Zeros(Float64, (3,), Float64, (2,))
    opRz = BroadCast(opZ, (2, 4))
    xz = randn(3)
    y = opRz * xz
    @test all(y .== 0)

    # Edge: BroadCast of Eye
    opE = Eye(2)
    opRe = BroadCast(opE, (2, 3))
    xe = randn(2)
    y = opRe * xe
    @test all(y[:,1] .== xe)

    # Edge: BroadCast of DiagOp
    d = randn(2)
    opD = DiagOp(d)
    opRd = BroadCast(opD, (2, 3))
    xd = randn(2)
    y = opRd * xd
    @test all(y[:,1] .== d .* xd)

    # Singleton and empty array broadcast
    opS = MatrixOp(randn(1,1))
    opRs = BroadCast(opS, (1,))
    xs = [1.0]
    ys = opRs * xs
    @test length(ys) == 1

    # Testing nonlinear BroadCast
    n, l = 4, 7
    x = randn(n)
    r = randn(n, l)
    opS = Sigmoid(Float64, (n,), 2)
    op = BroadCast(opS, (n, l))

    y, grad = test_NLop(op, x, r, verb)

    Y = (opS * x) .* ones(n, l)
    @test norm(Y - y) < 1e-8

    n, l = 1, 7
    x = randn(n)
    r = randn(n, l)
    opS = Sigmoid(Float64, (n,), 2)
    op = BroadCast(opS, (n, l))

    y, grad = test_NLop(op, x, r, verb)

    Y = (opS * x) .* ones(n, l)
    @test norm(Y - y) < 1e-8

    # Additional coverage tests
    @testset "BroadCast error cases and edge paths" begin
        m, n = 4, 3
        A = MatrixOp(randn(m, n))
        
        # Error: dim_out that doesn't match broadcast rules
        @test_throws DimensionMismatch BroadCast(A, (2,))
        
        # NoOperatorBroadCast equality
        E1 = Eye(3)
        E2 = Eye(3)
        B1 = BroadCast(E1, (3, 2); threaded=false)
        B2 = BroadCast(E2, (3, 2); threaded=false)
        B3 = BroadCast(E1, (3, 3); threaded=false)
        @test B1 == B2
        @test B1 != B3
        
        # opnorm for OperatorBroadCast
        A_op = MatrixOp(randn(3, 2))
        B_op = BroadCast(A_op, (3, 4); threaded=false)
        @test opnorm(B_op) ≈ opnorm(A_op)
        
        if Threads.nthreads() > 1
            B_op_t = BroadCast(A_op, (3, 4); threaded=true)
            @test opnorm(B_op_t) ≈ opnorm(A_op)
        end
    end

    @testset "Threaded NoOperatorBroadCast" begin
        if Threads.nthreads() > 1
            m = 1000
            E = Eye(m)
            B_threaded = BroadCast(E, (m, 100); threaded=true)
            @test B_threaded isa AbstractOperators.NoOperatorBroadCast
            
            x = randn(m)
            y = B_threaded * x
            
            for i in 1:100
                @test y[:, i] ≈ x
            end
            
            y_adj = randn(m, 100)
            x_back = B_threaded' * y_adj
            @test x_back ≈ dropdims(sum(y_adj, dims=2), dims=2)
        end
    end

    @testset "Non-compact threaded OperatorBroadCast" begin
        if Threads.nthreads() > 1
            m, n = 3, 2
            A = reshape(MatrixOp(randn(m, n)), 1, m)
            dim_out = (4, m, 5)
            B_noncompact = BroadCast(A, dim_out; threaded=true)
            
            x = randn(n)
            y = B_noncompact * x
            
            ref = A * x
            for i in 1:4, j in 1:5
                @test y[i, :, j] ≈ vec(ref)
            end
            
            y_test = randn(dim_out)
            x_back = B_noncompact' * y_test
            @test size(x_back) == (n,)
            @test x_back ≈ A' * dropdims(sum(y_test, dims=(1,3)), dims=3)
        end
    end
end
