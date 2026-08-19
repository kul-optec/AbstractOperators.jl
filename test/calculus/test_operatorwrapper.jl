@testitem "OperatorWrapper: CPU mul! forward and adjoint" tags = [
    :calculus, :OperatorWrapper,
] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(42)

    n = 8
    cpu_op = FiniteDiff(Float64, (n,), 1)           # n-1 output
    wrapper = OperatorWrapper(cpu_op)

    x = randn(n)
    y = zeros(n - 1)
    mul!(y, wrapper, x)
    y_ref = zeros(n - 1)
    mul!(y_ref, cpu_op, x)
    @test y ≈ y_ref

    b = randn(n - 1)
    z = zeros(n)
    mul!(z, wrapper', b)
    z_ref = zeros(n)
    mul!(z_ref, cpu_op', b)
    @test z ≈ z_ref

    @test size(wrapper) == size(cpu_op)
    @test domain_type(wrapper) == domain_type(cpu_op)
    @test codomain_type(wrapper) == codomain_type(cpu_op)
    @test is_linear(wrapper) == is_linear(cpu_op)
    @test domain_array_type(wrapper) == Array{Float64}
    @test codomain_array_type(wrapper) == Array{Float64}

    io = IOBuffer()
    show(io, wrapper)
    @test occursin("CPU[", String(take!(io)))
end

@testitem "OperatorWrapper: properties forwarded" tags = [:calculus, :OperatorWrapper] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(42)

    n = 6
    op = MatrixOp(randn(n, n))
    wrapper = OperatorWrapper(op)
    @test is_full_row_rank(wrapper) == is_full_row_rank(op)
    @test is_full_column_rank(wrapper) == is_full_column_rank(op)
    @test is_thread_safe(wrapper) == false  # always false regardless of inner op
end

@testitem "OperatorWrapper: CPU mul! via test_op" tags = [:calculus, :OperatorWrapper] setup = [
    TestUtils,
] begin
    using Random, AbstractOperators
    Random.seed!(42)

    n = 6
    op = MatrixOp(randn(n, n))
    wrapper = OperatorWrapper(op)

    test_op(wrapper, randn(n), randn(n), verb)
end

@testitem "OperatorWrapper: copy_operator" tags = [:calculus, :OperatorWrapper] setup = [TestUtils] begin
    using Random, LinearAlgebra, AbstractOperators
    Random.seed!(43)

    n = 8
    cpu_op = FiniteDiff(Float64, (n,), 1)
    wrapper = OperatorWrapper(cpu_op)
    wrapper2 = copy_operator(wrapper; threaded = true)
    @test wrapper2 isa OperatorWrapper

    x = randn(n)
    y1 = zeros(n - 1)
    y2 = zeros(n - 1)
    mul!(y1, wrapper, x)
    mul!(y2, wrapper2, x)
    @test y1 ≈ y2
end

@testitem "OperatorWrapper (GPU)" tags = [:gpu, :calculus, :OperatorWrapper] setup = [TestUtils, GpuEnvSetup] begin
    using Random, AbstractOperators, GPUEnv

    for backend in gpu_backends()
        Random.seed!(42)
        n = 32
        op = FiniteDiff(Float32, (n,), 1)
        storage_type = gpu_wrapper(backend, Float32, n)
        wrapper = OperatorWrapper(op; array_type = storage_type)
        @test domain_array_type(wrapper) <: backend.array_type
        @test codomain_array_type(wrapper) <: backend.array_type

        x = gpu_randn(backend, Float32, n)
        y = gpu_zeros(backend, Float32, n - 1)
        mul!(y, wrapper, x)
        @test y isa storage_type
        @test collect(y) ≈ op * collect(x)

        r = gpu_randn(backend, Float32, n - 1)
        z = gpu_zeros(backend, Float32, n)
        mul!(z, wrapper', r)
        @test z isa storage_type
        ref = zeros(Float32, n)
        mul!(ref, op', collect(r))
        @test collect(z) ≈ ref
    end
end

@testitem "OperatorWrapper: threading traits" tags = [:calculus, :OperatorWrapper] setup = [TestUtils] begin
    using Random, AbstractOperators
    Random.seed!(1)

    n = 1 << 16   # FiniteDiff's threshold is the "arithmetic" class (2^15)
    threaded_leaf = FiniteDiff(Float64, (n,); threaded = true)
    serial_leaf = FiniteDiff(Float64, (n,); threaded = false)
    @test is_threaded(OperatorWrapper(threaded_leaf)) == true
    @test is_threaded(OperatorWrapper(serial_leaf)) == false
    @test supports_threading(OperatorWrapper(serial_leaf)) == true
end
