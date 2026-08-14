@testitem "concurrent batched mul! restores thread counts" tags = [
        :batching, :SimpleBatchOp, :SpreadingBatchOp,
    ] begin
    using AbstractOperators, LinearAlgebra, Random

    # Regression test for the save/restore race that the NestedThreading extraction fixed.
    # Before it, two batched `mul!` calls running concurrently captured each other's
    # already-restricted thread counts and left BLAS/FFTW permanently pinned at 1 with no
    # budget scope active. The refcounted registry snapshots once and restores once.
    Random.seed!(0)

    op = MatrixOp(randn(32, 32))
    x = randn(32, 8)

    blas_before = BLAS.get_num_threads()
    expected = zeros(32, 8)

    for threaded in (true, false)
        batch_op = BatchOp(op, (8,); threaded)
        mul!(expected, batch_op, x)

        results = map(1:8) do _
            Threads.@spawn begin
                y = zeros(32, 8)
                for _ in 1:20
                    mul!(y, batch_op, x)
                end
                y
            end
        end
        for r in results
            @test fetch(r) ≈ expected
        end

        @test BLAS.get_num_threads() == blas_before
    end
end

@testitem "nested batch operators do not widen the inner thread budget" tags = [
        :batching, :SpreadingBatchOp,
    ] begin
    using AbstractOperators, LinearAlgebra, Random
    import NestedThreading

    # A small inner loop would compute a budget above 1 on its own; nested inside an outer
    # loop that already saturates the machine it must stay clamped.
    Random.seed!(0)

    observed = Ref(0)
    NestedThreading.with_thread_budget(1) do
        NestedThreading.with_thread_budget(Threads.nthreads()) do
            observed[] = BLAS.get_num_threads()
        end
    end
    @test observed[] == 1
end
