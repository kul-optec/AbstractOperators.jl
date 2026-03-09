if !isdefined(Main, :verb)
    verb = false
end
if !isdefined(Main, :test_op)
    include("./utils.jl")
end
Random.seed!(0)

@testset "Coverage gap tests" begin
    ## SimpleBatchOp: scalar diag paths (Eye returns scalar 1.0 for diag)
    @testset "SimpleBatchOp scalar diag" begin
        op = Eye(Float64, (3,))
        for threaded in (false, true)
            (threaded && Threads.nthreads() == 1) && continue
            batch_op = BatchOp(op, (2,); threaded)
            @test diag(batch_op) == 1.0
            @test diag_AcA(batch_op) == 1.0
            @test diag_AAc(batch_op) == 1.0
        end
    end

    ## utils.jl: argument validation via check() directly
    @testset "check() error paths" begin
        op = DiagOp(rand(3))
        y = zeros(3)
        x = rand(3)
        # non-array input
        @test_throws ArgumentError AbstractOperators.check(y, op, "bad")
        # non-array output
        @test_throws ArgumentError AbstractOperators.check("bad", op, x)
        # ArrayPartition input for single-domain op
        x_ap = ArrayPartition(rand(3), rand(3))
        @test_throws ArgumentError AbstractOperators.check(y, op, x_ap)
        # ArrayPartition output for single-codomain op
        y_ap = ArrayPartition(zeros(3), zeros(3))
        @test_throws ArgumentError AbstractOperators.check(y_ap, op, x)
    end

    ## Compose: storage type mismatch error
    @testset "Compose storage type mismatch" begin
        opA = MatrixOp(randn(3, 4))
        @test_throws DomainError Compose(opA, MatrixOp(randn(3, 3)))  # size mismatch: A is 3×4, B is 3×3
    end

    ## syntax.jl: getindex of Compose with diagonal intermediates
    @testset "Compose getindex with diagonal intermediates" begin
        d = randn(5)
        opD = DiagOp(d)
        opM = MatrixOp(randn(5, 5))
        comp = opM * opD  # Compose with is_diagonal on all intermediates
        sliced = comp[1:3]
        x = randn(5)
        y_full = comp * x
        @test norm(sliced * x - y_full[1:3]) < 1e-9
    end

    ## combination_rules: Scale * Compose paths
    @testset "combination_rules Scale∘Compose" begin
        n = 4
        d1 = randn(n)
        d2 = randn(n)
        opD1 = DiagOp(d1)
        opD2 = DiagOp(d2)
        opM = MatrixOp(randn(n, n))
        x = randn(n)

        # Scale(coeff, DiagOp) * Compose(DiagOp, MatrixOp) → combine path
        scaled = 2.0 * opD1
        comp = opD2 * opM
        result = scaled * comp
        @test norm(result * x - 2.0 * d1 .* (d2 .* (opM.A * x))) < 1e-9

        # AdjointScale * Compose path
        result2 = scaled' * comp
        @test norm(result2 * x - conj(2.0) * conj.(d1) .* (d2 .* (opM.A * x))) < 1e-9

        # Compose * Scale path
        result3 = comp * scaled
        @test norm(result3 * x - d2 .* (opM.A * (2.0 * d1 .* x))) < 1e-9

        # Compose * AdjointScale path
        result4 = comp * scaled'
        @test norm(result4 * x - d2 .* (opM.A * (conj(2.0) * conj.(d1) .* x))) < 1e-9
    end

    ## combination_rules: Scale+DiagOp combinations
    @testset "combination_rules Scale∘DiagOp" begin
        n = 4
        d = randn(n)
        opD = DiagOp(d)
        opM = MatrixOp(randn(n, n))
        x = randn(n)

        # AdjointScale * DiagOp
        result = (2.0 * opD)' * opD
        @test norm(result * x - conj(2.0) * conj.(d) .* (d .* x)) < 1e-9
        # DiagOp * Scale
        result2 = opD * (2.0 * opD)
        @test norm(result2 * x - d .* (2.0 * d .* x)) < 1e-9
        # AdjointDiagOp * Scale
        result3 = opD' * (2.0 * opD)
        @test norm(result3 * x - conj.(d) .* (2.0 * d .* x)) < 1e-9
        # AdjointScale * AdjointDiagOp
        result4 = (2.0 * opD)' * opD'
        @test norm(result4 * x - conj(2.0) * conj.(d) .* conj.(d) .* x) < 1e-9
    end

    ## combination_rules: MatrixOp∘Scale combos
    @testset "combination_rules MatrixOp∘Scale" begin
        n = 4
        A = randn(n, n)
        opA = MatrixOp(A)
        opD = DiagOp(randn(n))
        x = randn(n)

        # Scale(coeff, MatrixOp) with combinable inner
        result = (2.0 * opA) * opD
        @test norm(result * x - 2.0 * A * (opD * x)) < 1e-9

        # AdjointScale(MatrixOp) * MatrixOp
        result2 = (2.0 * opA)' * opA
        ref = conj(2.0) * A' * A * x
        @test norm(result2 * x - ref) < 1e-9

        # AdjointScale(MatrixOp) * AdjointMatrixOp
        result3 = (2.0 * opA)' * opA'
        ref2 = conj(2.0) * A' * A' * x
        @test norm(result3 * x - ref2) < 1e-9
    end

    ## properties.jl: combine fallback error
    @testset "combine fallback error" begin
        opS = Sigmoid(Float64, (3,))
        opP = Pow(Float64, (3,), 2.0)
        @test_throws ErrorException AbstractOperators.combine(opS, opP)
    end

    ## properties.jl: combine with null operator
    @testset "combine with null operators" begin
        z = Zeros(Float64, (3,), Float64, (4,))
        opM = MatrixOp(randn(4, 3))
        # Zeros * MatrixOp → Zeros
        result = AbstractOperators.combine(z, opM)
        @test is_null(result)
        # MatrixOp * Zeros → Zeros
        result2 = AbstractOperators.combine(opM, z)
        @test is_null(result2)
    end

    ## MatrixOp: complex matrix, real input adjoint mul!
    @testset "MatrixOp complex matrix real adjoint" begin
        A = randn(ComplexF64, 4, 3)
        opA = MatrixOp(Float64, (3,), A)
        x = randn(3)
        y = opA * x
        # adjoint: should give real output even with complex matrix
        z = opA' * y
        @test eltype(z) <: Real
        @test norm(z - real.(A' * y)) < 1e-9
    end

    ## HCAT: fun_name with reversed indices
    @testset "HCAT fun_name reversed" begin
        opA = MatrixOp(randn(3, 3))
        opB = MatrixOp(randn(3, 4))
        opH = HCAT(opB, opA)  # create with reversed order
        opH_perm = opH[[2, 1]]  # permute to trigger the swap branch
        io = IOBuffer()
        show(io, opH_perm)
        s = String(take!(io))
        @test occursin("[", s)
    end

    ## Sum: filter zeros to single operator
    @testset "Sum with Zeros filtering" begin
        opA = MatrixOp(randn(3, 3))
        z = Zeros(Float64, (3,), Float64, (3,))
        result = Sum(opA, z)
        @test result == opA  # Sum should return opA directly when Zeros are filtered
    end

    ## Compose: remove_slicing with length(A) > 2 (3-operator Compose)
    @testset "Compose remove_slicing" begin
        # Build a 3-operator Compose with GetIndex at A[1] (input side)
        n = 10
        d = randn(5)
        G = GetIndex((n,), 1:5)       # (10,) → (5,)
        D = DiagOp(d)                  # (5,) → (5,)
        F = FiniteDiff((5,), 1)        # (5,) → (4,)
        sliced3 = F * D * G            # A = (G, D, F), 3 operators
        @test sliced3 isa AbstractOperators.Compose
        @test length(sliced3.A) == 3
        @test AbstractOperators.is_sliced(sliced3)
        unsliced = AbstractOperators.remove_slicing(sliced3)
        @test unsliced isa AbstractOperators.Compose
        @test length(unsliced.A) == 2
        @test !AbstractOperators.is_sliced(unsliced)
    end

    ## DCAT: stacked operator (ArrayPartition) codomain path
    @testset "DCAT stacked operators" begin
        opV = VCAT(MatrixOp(randn(3, 2)), MatrixOp(randn(4, 2)))
        opB = MatrixOp(randn(4, 5))
        dc = DCAT(opV, opB)
        x = ArrayPartition(randn(2), randn(5))
        y = dc * x
        @test length(y.x) == 3  # opV produces 2 parts, opB produces 1

        z = dc' * y
        @test length(z.x) == 2
    end

    ## Jacobian: DCAT with ArrayPartition domain
    @testset "Jacobian DCAT ArrayPartition" begin
        opD1 = DiagOp(randn(3))
        opD2 = DiagOp(randn(4))
        dc = DCAT(opD1, opD2)
        jac = jacobian(dc, ArrayPartition(randn(3), randn(4)))
        @test size(jac, 1) == size(dc, 1)
        @test size(jac, 2) == size(dc, 2)
    end

    ## GetIndex: SubArray constructor path
    @testset "GetIndex from SubArray" begin
        x = randn(5, 5)
        sv = @view x[1:3, :]
        g = GetIndex(sv, 1:2)
        @test size(g, 1) == (2,)
    end

    ## HCAT: slicing expression paths (get_slicing_expr, get_slicing_mask, remove_slicing)
    @testset "HCAT slicing expressions" begin
        n = 10
        d1, d2 = randn(5), randn(5)
        D1 = DiagOp(d1) * GetIndex((n,), 1:5)     # sliced Compose
        D2 = DiagOp(d2) * GetIndex((n,), 6:10)     # sliced Compose
        H = HCAT(D1, D2)
        @test AbstractOperators.is_sliced(H)
        expr = AbstractOperators.get_slicing_expr(H)
        @test expr == ((1:5,), (6:10,))
        mask = AbstractOperators.get_slicing_mask(H)
        @test length(collect(mask)) == 2
        unsliced = AbstractOperators.remove_slicing(H)
        @test !AbstractOperators.is_sliced(unsliced)
    end

    ## properties.jl: domain_storage_type for ArrayPartition
    @testset "ArrayPartition storage types" begin
        opA = MatrixOp(randn(3, 2))
        opB = MatrixOp(randn(3, 4))
        opH = HCAT(opA, opB)
        dst = domain_storage_type(opH)
        @test dst <: ArrayPartition
    end
end
