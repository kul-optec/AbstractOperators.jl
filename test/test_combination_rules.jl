using Test
using LinearAlgebra
using AbstractOperators
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

    @testset "BroadCast Combinations" begin
        n = 4
        A = randn(n, n)
        
        # Create operators
        diag_op = MatrixOp(A)
        broad_op = BroadCast(DiagOp(ones(n)), (n, 2n))
        
        @test can_be_combined(diag_op, broad_op)
        @test !can_be_combined(broad_op, diag_op)  # Size mismatch
        
        # Test combination
        combined = combine(diag_op, broad_op)
        @test combined isa BroadCast
        @test size(combined) == ((4, 8), (4,))
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

    @testset "Transform Combinations" begin
        n = 8  # Power of 2 for DCT
        
        # Test DCT combinations
        dct_op = DCT(n)
        idct_op = IDCT(n)
        
        @test can_be_combined(dct_op, idct_op)
        @test can_be_combined(idct_op, dct_op)
        
        combined_dct = combine(dct_op, idct_op)
        @test combined_dct isa Eye
        
        # Test DFT combinations
        dft_op = DFT(n)
        idft_op = IDFT(n)
        
        @test can_be_combined(dft_op, idft_op)
        @test can_be_combined(idft_op, dft_op)
        
        combined_dft = combine(dft_op, idft_op)
        @test combined_dft isa Scale{<:Number,<:Eye}
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
        @test domainType(combined_diag2) == Float64
        @test codomainType(combined_diag2) == Float32

        # type-changing DiagOp with Zeros
        diag_op2 = DiagOp(Float64, (m,), rand(ComplexF64, m))
        @test can_be_combined(diag_op2, zeros_op3)
        combined_diag3 = combine(diag_op2, zeros_op3)
        @test combined_diag3 isa Zeros
        @test size(combined_diag3) == ((m,), (n,))
        @test domainType(combined_diag3) == Float64
        @test codomainType(combined_diag3) == ComplexF64
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
end