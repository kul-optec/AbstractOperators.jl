if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("./utils.jl")
end
using AbstractOperators: can_be_combined, combine

@testset "Combination Rules" begin
    
    @testset "AffineAdd Combinations" begin
        n = 5
        A = randn(n, n)
        b1 = randn(n)
        b2 = randn(n)
        
        # Create affine operators
        op1 = AffineAdd(MatrixOp(A), b1)
        op2 = AffineAdd(DiagOp(ones(n)), b2)
        
        @test can_be_combined(op1, op2)
        
        # Test combination result
        combined = combine(op1, op2)
        @test combined isa AffineAdd
        @test combined.d ≈ A * b2 .+ b1
        
        # Test with different signs
        op3 = AffineAdd(MatrixOp(A), b1, false)
        combined_diff_signs = combine(op1, op3)
        @test combined_diff_signs.d ≈ A * b1 .- b1

        # Test matrix-op + affine-op
        A2 = randn(n, n)
        op4 = MatrixOp(A2)

        @test can_be_combined(op4, op1)

        combined_mat_affine = combine(op4, op1)
        @test combined_mat_affine isa AffineAdd
        @test combined_mat_affine.A isa MatrixOp
        @test combined_mat_affine.d ≈ A2 * b1
        @test combined_mat_affine.A.A ≈ A2 * A

    end

    @testset "Compose Combinations" begin
        n = 3
        A = randn(n, n)
        B = randn(n, n)
        
        # Create composed operators
        op1 = Compose(MatrixOp(A), DiagOp(ones(n)))
        op2 = Compose(MatrixOp(B), Eye(n))
        
        @test can_be_combined(op1, op2)
        
        combined = combine(op1, op2)
        @test combined isa MatrixOp
        @test combined.A ≈ A * Diagonal(ones(n)) * B
        
        # Test with single operator
        single_op = MatrixOp(A)
        composed_op = Compose(MatrixOp(B), Eye(n))
        
        @test can_be_combined(single_op, composed_op)
        combined_single = combine(single_op, composed_op)
        @test combined_single isa MatrixOp
        @test combined_single.A ≈ A * B

        # Compose of Composes with Matrices
        n = 4
        A = randn(n, n)
        B = randn(n, n)
        comp1 = Compose(MatrixOp(A), FiniteDiff((n+1,)))    # A*(diff(x))
        comp2 = Compose(FiniteDiff((n,)), MatrixOp(B))      # diff(B*x)
        @test can_be_combined(comp2, comp1)
        combined_cc = combine(comp2, comp1)
        x = randn(n+1)
        @test combined_cc * x ≈ comp2 * (comp1 * x)

        # Left matrix with Compose (generic combine(L, R::Compose) path)
        C = randn(n, n)
        left_mat = MatrixOp(C)
        @test can_be_combined(left_mat, comp1)
        combined_lc = combine(left_mat, comp1)
        x = randn(n+1)
        @test combined_lc * x ≈ left_mat * (comp1 * x)
    end

    @testset "DCAT Combinations" begin
        n = 4
        A = [randn(n, n) for _ in 1:2]
        d = [randn(n) for _ in 1:2]
        
        # Create DCAT operator
        matrix_ops = DCAT(MatrixOp.(A)...)
        diag_ops = DCAT(DiagOp.(d)...)
        
        @test can_be_combined(diag_ops, matrix_ops)
        combined_dcat = combine(diag_ops, matrix_ops)
        @test combined_dcat isa DCAT
        @test combined_dcat.A[1].A ≈ Diagonal(d[1]) * A[1]
        @test combined_dcat.A[2].A ≈ Diagonal(d[2]) * A[2]
    end

    @testset "HCAT Combinations" begin
        n = 3
        A = randn(n, n)
        B = randn(n, n)
        d = randn(n)
        
        # Create HCAT operator
        hcat_op = HCAT(MatrixOp(A), MatrixOp(B))
        diag_op = DiagOp(d)
        
        @test can_be_combined(diag_op, hcat_op)
        combined_hcat = combine(diag_op, hcat_op)
        @test combined_hcat isa HCAT
        @test size(combined_hcat) == ((3,), ((3,), (3,)))
        @test combined_hcat.A[1].A ≈ d .* A
        @test combined_hcat.A[2].A ≈ d .* B
    end

    @testset "Scale Combinations" begin
        n = 4
        A = randn(n, n)
        B = randn(n, n)
        
        # Create scale operators
        matrix_op = MatrixOp(A)
        scale_op = Scale(3.0, MatrixOp(B))
        
        @test can_be_combined(scale_op, matrix_op)
        combined = combine(scale_op, matrix_op)
        @test combined isa MatrixOp
        @test combined.A ≈ 3.0 * B * A

        @test can_be_combined(matrix_op, scale_op)
        combined = combine(matrix_op, scale_op)
        @test combined isa MatrixOp
        @test combined.A ≈ 3 * A * B
        
        # Test with adjoint
        @test can_be_combined(matrix_op, scale_op')
        combined = combine(matrix_op, scale_op')
        @test combined isa MatrixOp
        @test combined.A ≈ 3 * A * B'

        @test can_be_combined(scale_op', matrix_op)
        combined = combine(scale_op', matrix_op)
        @test combined isa MatrixOp
        @test combined.A ≈ 3 * B' * A

        @test can_be_combined(matrix_op', scale_op)
        combined = combine(matrix_op', scale_op)
        @test combined isa MatrixOp
        @test combined.A ≈ 3 * A' * B

        @test can_be_combined(scale_op, matrix_op')
        combined = combine(scale_op, matrix_op')
        @test combined isa MatrixOp
        @test combined.A ≈ 3 * B * A'

        @test can_be_combined(scale_op', matrix_op')
        combined = combine(scale_op', matrix_op')
        @test combined isa MatrixOp
        @test combined.A ≈ 3 * B' * A'

        @test can_be_combined(matrix_op', scale_op')
        combined = combine(matrix_op', scale_op')
        @test combined isa MatrixOp
        @test combined.A ≈ 3 * A' * B'
    end

    @testset "Sum Combinations" begin
        n = 3
        A = randn(n, n)
        B = randn(n, n)
        d = randn(n)
        
        diag_op = DiagOp(d)
        sum_op = Sum(MatrixOp(A), MatrixOp(B))
        
        @test can_be_combined(diag_op, sum_op)
        combined = combine(diag_op, sum_op)
        @test combined isa Sum
        @test combined.A[1].A ≈ Diagonal(d) * A
        @test combined.A[2].A ≈ Diagonal(d) * B
    end

    @testset "MatrixOp Combinations" begin
        n = 4
        m = 3
        A = randn(n, m)
        B = randn(m, n)
        
        # Basic MatrixOp combinations
        op1 = MatrixOp(A)
        op2 = MatrixOp(B)
        
        @test can_be_combined(op2, op1)
        combined = combine(op2, op1)
        @test combined isa MatrixOp
        @test combined.A ≈ B * A
        
        # MatrixOp with adjoint
        C = randn(n, m)
        op3 = MatrixOp(C)
        adj_op = op3'
        @test can_be_combined(op1, adj_op)
        combined_adj = combine(op1, adj_op)
        @test combined_adj.A ≈ A * C'
        
        # MatrixOp with DiagOp
        d = randn(m)
        diag_op = DiagOp(d)
        @test can_be_combined(op1, diag_op)
        combined_diag = combine(op1, diag_op)
        @test combined_diag.A ≈ A * Diagonal(d)
    end

    @testset "Scale Combinations" begin
        n = 3
        α = 2.0
        β = 3.0
        
        # Scale with Eye
        scale1 = Scale(α, Eye(n))
        scale2 = Scale(β, Eye(n))
        
        @test can_be_combined(scale1, scale2)
        combined = combine(scale1, scale2)
        @test combined isa Scale
        @test combined.coeff ≈ α * β
        
        # Scale with DiagOp
        d = randn(n)
        diag_op = DiagOp(d)
        @test can_be_combined(scale1, diag_op)
        combined_diag = combine(scale1, diag_op)
        @test combined_diag isa DiagOp
        @test combined_diag.d ≈ α * d
        
        # Scale with adjoint
        adj_scale = scale2'
        @test can_be_combined(scale1, adj_scale)
        combined = combine(scale1, adj_scale)
        x = randn(n)
        y = combined * x
        @test y ≈ α * β * x
    end

    @testset "Zeros Combinations" begin
        n = 4
        m = 3
        
        zeros_op1 = Zeros(Float64, (n,), Float64, (m,))
        zeros_op2 = Zeros(Float64, (n,), Float64, (m,))
        
        # Zeros with Zeros
        @test can_be_combined(zeros_op1, zeros_op2)
        combined = combine(zeros_op1, zeros_op2)
        @test combined isa Zeros
        @test size(combined) == ((m,), (n,))
        
        # Zeros with DiagOp
        d = randn(n)
        diag_op = DiagOp(d)
        @test can_be_combined(zeros_op1, diag_op)
        combined_diag = combine(zeros_op1, diag_op)
        @test combined_diag isa Zeros
        @test size(combined_diag) == ((m,), (n,))

        # type-changing Zeros with DiagOp
        zeros_op3 = Zeros(Float64, (n,), Float32, (m,))
        @test can_be_combined(zeros_op3, diag_op)
        combined_diag2 = combine(zeros_op3, diag_op)
        @test combined_diag2 isa Zeros
        @test size(combined_diag2) == ((m,), (n,))
        @test domain_type(combined_diag2) == Float64
        @test codomain_type(combined_diag2) == Float32

        # type-changing DiagOp with Zeros
        diag_op2 = DiagOp(Float64, (m,), rand(ComplexF64, m))
        @test can_be_combined(diag_op2, zeros_op3)
        combined_diag3 = combine(diag_op2, zeros_op3)
        @test combined_diag3 isa Zeros
        @test size(combined_diag3) == ((m,), (n,))
        @test domain_type(combined_diag3) == Float64
        @test codomain_type(combined_diag3) == ComplexF64
    end

    @testset "DiagOp Combinations" begin
        n = 5
        d1 = randn(n)
        d2 = randn(n)
        
        # DiagOp with DiagOp
        diag1 = DiagOp(d1)
        diag2 = DiagOp(d2)
        
        @test can_be_combined(diag1, diag2)
        combined = combine(diag1, diag2)
        @test combined isa DiagOp
        @test combined.d ≈ d1 .* d2
        
        # DiagOp with adjoint
        adj_diag = diag2'
        @test can_be_combined(diag1, adj_diag)
        combined_adj = combine(diag1, adj_diag)
        @test combined_adj isa DiagOp
        @test combined_adj.d ≈ d1 .* conj.(d2)
        
        # DiagOp with Scale
        α = 2.0
        scale_op = Scale(α, Eye(n))
        @test can_be_combined(diag1, scale_op)
        combined_scale = combine(diag1, scale_op)
        @test combined_scale isa DiagOp
        @test combined_scale.d ≈ α * d1
    end

    @testset "Eye Combinations" begin
        n = 4
        eye_op = Eye(n)
        
        # Eye with any linear operator
        A = randn(n, n)
        mat_op = MatrixOp(A)
        
        @test can_be_combined(mat_op, eye_op)
        combined = combine(mat_op, eye_op)
        @test combined isa MatrixOp
        @test combined.A ≈ A
        
        # Eye with Scale
        α = 2.0
        scale_op = Scale(α, Eye(n))
        @test can_be_combined(eye_op, scale_op)
        combined_scale = combine(eye_op, scale_op)
        @test combined_scale isa Scale
        @test combined_scale.coeff ≈ α
        
        # Eye with adjoint
        adj_eye = eye_op'
        @test can_be_combined(eye_op, adj_eye)
        combined_adj = combine(eye_op, adj_eye)
        @test combined_adj isa Eye
    end

    @testset "Mixed Combinations" begin
        # Helper square matrices
        n = 4
        A = randn(n, n)
        B = randn(n, n)
        C = randn(n, n)
        D = randn(n, n)
        S = randn(n, n)
        α = 1.7
        β = -0.9

        comp1 = Compose(MatrixOp(A), FiniteDiff((n+1,)))    # A*(diff(x))
        comp2 = Compose(FiniteDiff((n,)), MatrixOp(B))      # diff(B*x)

        # Scale with Compose (combine(L::Scale, R::Compose))
        scale_left = Scale(α, FiniteDiff((n,)))
        @test can_be_combined(scale_left, comp1)
        combined_sc = combine(scale_left, comp1)
        x = randn(n+1)
        @test combined_sc * x ≈ scale_left * (comp1 * x)

        # Adjoint Scale with Compose (combine(L::AdjointOperator{<:Scale}, R::Compose))
        scale_left_adj = scale_left'
        @test can_be_combined(scale_left_adj, comp2)
        combined_sac = combine(scale_left_adj, comp2)
        x = randn(n)
        @test combined_sac * x ≈ scale_left_adj * (comp2 * x)

        # Compose with Scale (combine(L::Compose, R::Scale))
        scale_right = Scale(β, FiniteDiff((n+1,)))
        @test can_be_combined(comp2, scale_right)
        combined_cs = combine(comp2, scale_right)
        x = randn(n+1)
        @test combined_cs * x ≈ comp2 * (scale_right * x)

        # Compose with Adjoint Scale (combine(L::Compose, R::AdjointOperator{<:Scale}))
        scale_right = Scale(β, FiniteDiff((n,)))
        scale_right_adj = scale_right'
        @test can_be_combined(comp2, scale_right_adj)
        combined_csa = combine(comp2, scale_right_adj)
        @test combined_csa * x ≈ comp2 * (scale_right_adj * x)

        # Adjoint Scale with DiagOp (combine(T1::AdjointOperator{<:Scale}, T2::DiagOp))
        scale_left = Scale(α, FiniteDiff((n+1,)))
        scale_left_adj = scale_left'
        dvec = randn(n)
        diag_op = DiagOp(dvec)
        @test can_be_combined(scale_left_adj, diag_op)
        combined_asd = combine(scale_left_adj, diag_op)
        x = randn(n)
        @test combined_asd * x ≈ scale_left_adj * (diag_op * x)

        # DiagOp' with Scale (combine(T1::AdjointOperator{<:DiagOp}, T2::Scale))
        scale_right = Scale(β, FiniteDiff((n+1,)))
        diag_op_adj = diag_op'
        @test can_be_combined(diag_op_adj, scale_right)
        combined_das = combine(diag_op_adj, scale_right)
        x = randn(n+1)
        @test combined_das * x ≈ diag_op_adj * (scale_right * x)

        # Scale' with DiagOp' (combine(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:DiagOp}))
        scale_left = Scale(α, FiniteDiff((n+1,)))
        scale_left_adj = scale_left'
        @test can_be_combined(scale_left_adj, diag_op_adj)
        combined_sad = combine(scale_left_adj, diag_op_adj)
        x = randn(n)
        @test combined_sad * x ≈ scale_left_adj * (diag_op_adj * x)

        # Adjoint DiagOp with MatrixOp (both orders to exercise can_be_combined variants)
        mat_op = MatrixOp(A)
        @test can_be_combined(diag_op_adj, mat_op)
        combined_dm = combine(diag_op_adj, mat_op)
        @test combined_dm * x ≈ diag_op_adj * (mat_op * x)
        @test can_be_combined(mat_op, diag_op_adj)
        combined_md = combine(mat_op, diag_op_adj)
        @test combined_md * x ≈ mat_op * (diag_op_adj * x)

        # Scale' with MatrixOp' (combine(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:MatrixOp}))
        mat_op_adj = mat_op'
        scale_left = Scale(α, FiniteDiff((n+1,)))
        scale_left_adj = scale_left'
        @test can_be_combined(scale_left_adj, mat_op_adj)
        combined_sma = combine(scale_left_adj, mat_op_adj)
        x = randn(n)
        @test combined_sma * x ≈ scale_left_adj * (mat_op_adj * x)

        # Branch: combine(L, R::Compose) -> if branch (combined isa Compose)
        M = MatrixOp(rand(n,n-2))
        S = Scale(α, FiniteDiff((n-1,)) * FiniteDiff((n,)))
        C = S * FiniteDiff((n+1,))
        @test can_be_combined(M, C)
        combined = combine(M, C)
        x = randn(n+1)
        @test combined * x ≈ M.A * (α * diff(diff(diff(x))))

        # Branch: combine(L::Compose, R::Compose) -> if branch (combined isa Compose)
        C1 = Compose(FiniteDiff((n,)), MatrixOp(A))
        C2 = Scale(α, FiniteDiff((n+1,))) * FiniteDiff((n+2,))
        @test can_be_combined(C1, C2)
        combined_cc = combine(C1, C2)
        x = randn(n+2)
        @test combined_cc * x ≈ C1 * (C2 * x)

        # Branch: combine(L::Scale, R::Compose) -> if branch (can_be_combined(L.A, R.A[end]))
        C3 = Scale(α, FiniteDiff((n+1,))) * FiniteDiff((n+2,))
        @test can_be_combined(S, C3)
        combined_sc = combine(S, C3)
        x = randn(n+2)
        @test combined_sc * x ≈ S * (C3 * x)

        # Branch: combine(L::Scale, R::Compose) -> if branch (can_be_combined(L.A, R.A[end]))
        n2 = 5
        A1 = randn(n2, n2)
        A2 = randn(n2, n2)
        A3 = randn(n2, n2)
        comp_tail = Compose(MatrixOp(A2), FiniteDiff((n2+1,))) # last is MatrixOp so combinable with MatrixOp(A1)
        scl = Scale(2.3, MatrixOp(A1))
        comb_sc_if = combine(scl, comp_tail)
        x2 = randn(n2+1)
        @test comb_sc_if * x2 ≈ scl * (comp_tail * x2)

        # Branch: combine(L::AdjointOperator{<:Scale}, R::Compose) -> if branch
        scl_adj = scl'
        comp_tail2 = Compose(MatrixOp(A2), FiniteDiff((n2+1,)))
        comb_sac_if = combine(scl_adj, comp_tail2)
        @test comb_sac_if * x2 ≈ scl_adj * (comp_tail2 * x2)

        # Branch: combine(L::Compose, R::Scale) -> if branch
        comp_head = Compose(FiniteDiff((n2,)), MatrixOp(A2))
        scl_r = Scale(-1.1, FiniteDiff((n2+1,)))
        comb_cs_if = combine(comp_head, scl_r)
        @test comb_cs_if * x2 ≈ comp_head * (scl_r * x2)

        # Branch: combine(L::Compose, R::AdjointOperator{<:Scale}) -> if branch
        scl_r = Scale(-1.1, FiniteDiff((n2,)))
        scl_r_adj = scl_r'
        comb_csa_if = combine(comp_head, scl_r_adj)
        x3 = randn(n2-1)
        @test comb_csa_if * x3 ≈ comp_head * (scl_r_adj * x3)

        # Branch: combine(T1::Scale, T2::MatrixOp) -> if branch (can_be_combined)
        scl_mat_if = Scale(0.7, FiniteDiff((n2,)))
        mat2 = MatrixOp(A2)
        comb_scale_mat_if = combine(scl_mat_if, mat2)
        x3 = randn(n2)
        @test comb_scale_mat_if * x3 ≈ scl_mat_if * (mat2 * x3)

        # Branch: combine(T1::Scale, T2::DiagOp) -> else branch (non-combinable path)
        # Use FiniteDiff so scaled_diagop cannot be combined further
        fd = FiniteDiff((n2+1,))
        scl_fd = Scale(1.5, fd)
        dvec = randn(n2+1)
        diag_long = DiagOp(dvec)
        # Adjust x dimension for FiniteDiff input (n2+1)
        comb_scale_diag_else = combine(scl_fd, diag_long)
        @test comb_scale_diag_else * x2 ≈ scl_fd * (diag_long * x2)

        # Branch: combine(T1::AdjointOperator{<:Scale}, T2::DiagOp) -> if branch
        sclA = Scale(1.2, FiniteDiff((n2+1,)))
        sclA_adj = sclA'
        diagA = DiagOp(randn(n2))
        comb_asd_if = combine(sclA_adj, diagA)
        @test comb_asd_if * x3 ≈ sclA_adj * (diagA * x3)

        # Branch: combine(T1::DiagOp, T2::Scale) else branch (non-combinable)
        # Make second scale wrap FiniteDiff so scaled_diagop cannot be combined
        scl_fd2 = Scale(-0.4, FiniteDiff((n2+1,)))
        diag_diff = DiagOp(randn(n2))
        comb_diag_scale_else = combine(diag_diff, scl_fd2)
        @test comb_diag_scale_else * x2 ≈ diag_diff * (scl_fd2 * x2)

        # Branch: combine(T1::AdjointOperator{<:DiagOp}, T2::Scale) -> if branch
        diag_if = DiagOp(randn(n2))
        diag_if_adj = diag_if'
        scl_lin = Scale(0.9, FiniteDiff((n2+1,)))
        comb_adjdiag_scale_if = combine(diag_if_adj, scl_lin)
        @test comb_adjdiag_scale_if * x2 ≈ diag_if_adj * (scl_lin * x2)

        # Branch: combine(T1::AdjointOperator{<:Scale}, T2::AdjointOperator{<:DiagOp}) -> if branch
        scl_lin_adj = scl_lin'
        diag_if_adj2 = diag_if'
        comb_sad_if = combine(scl_lin_adj, diag_if_adj2)
        @test comb_sad_if * x3 ≈ scl_lin_adj * (diag_if_adj2 * x3)

        # Exercise can_be_combined variants for matrix/diag adjoint combinations
        mat_op2 = MatrixOp(A2)
        diag_base = DiagOp(randn(n2))
        @test can_be_combined(diag_base, mat_op2)
        @test can_be_combined(mat_op2, diag_base)
        @test can_be_combined(diag_base', mat_op2)
        @test can_be_combined(mat_op2, diag_base')
        @test can_be_combined(mat_op2', diag_base)
        @test can_be_combined(diag_base, mat_op2')
        @test can_be_combined(diag_base', mat_op2')
        @test can_be_combined(mat_op2', diag_base')
    end
end