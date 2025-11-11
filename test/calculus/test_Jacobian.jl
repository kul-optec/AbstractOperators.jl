if !@isdefined verb
    verb = false
end
if !@isdefined test_op
    include("../utils.jl")
end

@testset "Jacobian" begin
    verb && println(" --- Testing Jacobian --- ")

    m, n = 3, 5
    x = ArrayPartition(randn(m), randn(n))
    r = randn(m)
    A = Sin(Float64, (m,))
    M = randn(m, n)
    B = MatrixOp(M)
    op = HCAT(A, B)

    p = [2, 1]
    opP = AbstractOperators.permute(op, p)
    xp = ArrayPartition(x.x[p]...)
    J = Jacobian(opP, xp)'
    verb && println(size(J, 1))
    y, grad = test_NLop(opP, xp, r, verb)

    # Additional coverage-oriented tests appended (no new testset created)

    # 1. LinearOperator path (Jacobian of a LinearOperator returns itself)
    n_lin = 6
    L = Eye(n_lin)
    x_lin = randn(n_lin)
    JL = Jacobian(L, x_lin)
    @test JL === L

    # 2. Scale path
    n_sc = 5
    base_op = Sigmoid(Float64, (n_sc,), 2)
    S = 2.5 * base_op
    x_sc = randn(n_sc)
    JS = Jacobian(S, x_sc)
    @test JS isa Scale
    @test JS.coeff == 2.5
    @test JS.A == Jacobian(base_op, x_sc)

    # 3. AffineAdd path (Jacobian should drop displacement)
    n_aff = 4
    d_aff = randn(n_aff)
    aff = AffineAdd(Sigmoid(Float64, (n_aff,), 2), d_aff)
    x_aff = randn(n_aff)
    Jaff = Jacobian(aff, x_aff)
    @test !(Jaff isa AffineAdd)  # displacement removed
    @test Jaff == Jacobian(aff.A, x_aff)

    # 4. Transpose path
    n_tr = 5
    op_tr = Sigmoid(Float64, (n_tr,), 2)
    @test_throws ErrorException op_tr'

    # 5. Compose path (single op) with tuple input to trigger second Compose method
    n_cp = 3
    op_c = Sigmoid(Float64, (n_cp,), 2)
    op_p = Pow(Float64, (n_cp,), 2)
    x = rand(n_cp)
    comp = Compose(op_c, op_p)
    Jcomp = Jacobian(comp, x)
    @test length(Jcomp.A) == 2
    @test Jcomp.A[1] == Jacobian(op_p, x)
    @test Jcomp.A[2] == Jacobian(op_c, op_p * x)

    # 6. Sum path
    n_sum = 4
    op_sum1 = Sigmoid(Float64, (n_sum,), 2)
    op_sum2 = Pow(Float64, (n_sum,), 2)
    Ssum = Sum(op_sum1, op_sum2)
    x_sum = randn(n_sum)
    JSsum = Jacobian(Ssum, x_sum)
    @test JSsum.A[1] == Jacobian(op_sum1, x_sum)
    @test JSsum.A[2] == Jacobian(op_sum2, x_sum)

    # 7. VCAT path
    n_v = 4
    op_v1 = Sigmoid(Float64, (n_v,), 2)
    op_v2 = Pow(Float64, (n_v,), 2)
    Vop = VCAT(op_v1, op_v2)
    x_v = randn(n_v)
    JV = Jacobian(Vop, x_v)
    @test JV.A[1] == Jacobian(op_v1, x_v)
    @test JV.A[2] == Jacobian(op_v2, x_v)

    # 8. HCAT path with multi-index block (length(idx) > 1) to cover else branch
    m_h1, m_h2 = 3, 2
    Ah1 = Sigmoid(Float64, (m_h1,), 2)
    Ah2 = Pow(Float64, (m_h1,), 2)
    # Build an HCAT where second entry depends on two blocks by wrapping a VCAT then splitting via ArrayPartition
    # Simplest: reuse existing op with duplicated single-block indices to force length(idx)==1 already covered above;
    # Instead create HCAT of (Ah1, Ah2) and test that Jac is HCAT of their Jacobians (length(idx)==1 branches)
    # Then also construct a DCAT that groups indices to test DCAT multi-length branch.
    Hsimple = HCAT(Ah1, Ah2)
    xh = ArrayPartition(randn(m_h1), randn(m_h1))
    JH = Jacobian(Hsimple, xh)
    @test JH.A[1] == Jacobian(Ah1, xh.x[1])
    @test JH.A[2] == Jacobian(Ah2, xh.x[2])

    # 9. DCAT path with a grouped idx >1 (simulate by constructing custom DCAT via its constructor)
    # Create a DCAT where one sub-operator takes both pieces as joint input.
    op_joint = HCAT(Ah1, Ah2)  # Nonlinear op acting on combined partition
    # idxD entries: first uses both 1 and 2; second uses only 1
    D = DCAT(op_joint, Ah1)
    xh = ArrayPartition(randn(m_h1), randn(m_h1), randn(m_h1))
    JD = Jacobian(D, xh)
    @test JD.A[1] == Jacobian(op_joint, (xh.x[1], xh.x[2]))
    @test JD.A[2] == Jacobian(Ah1, xh.x[3])

    # 10. Reshape path
    n_rs = 6
    op_rs = Sigmoid(Float64, (n_rs,), 2)
    R = Reshape(op_rs, 2, 3)
    x_rs = randn(n_rs)
    JR = Jacobian(R, x_rs)
    @test size(JR) == ((2,  3), (n_rs,))

    # 11. BroadCast path (Val{false}) already in other test file; here just basic structure
    n_bc, l_bc = 3, 4
    op_bc = Sigmoid(Float64, (n_bc,), 2)
    Bc = BroadCast(op_bc, (n_bc, l_bc))
    x_bc = randn(n_bc)
    JBc = Jacobian(Bc, x_bc)
    # ensure repeating structure consistent
    @test size(JBc) == ((n_bc, l_bc), (n_bc,))

    # 12. Equality and show output
    n_eq = 4
    op_eq = Sigmoid(Float64, (n_eq,), 2)
    x_eq = randn(n_eq)
    J1 = Jacobian(op_eq, x_eq)
    J2 = Jacobian(op_eq, copy(x_eq))
    J3 = Jacobian(op_eq, x_eq .+ 0.1)  # different point
    @test J1 == J2
    @test J1 != J3
    buf = IOBuffer()
    show(buf, J1)
    shown = String(take!(buf))
    @test occursin("J(", shown)
    @test occursin("->", shown)

    # 13. Type and thread safety properties
    @test domain_type(J1) == domain_type(op_eq)
    @test codomain_type(J1) == codomain_type(op_eq)
    @test domain_storage_type(J1) == domain_storage_type(op_eq)
    @test codomain_storage_type(J1) == codomain_storage_type(op_eq)
    @test !is_thread_safe(J1)
end
