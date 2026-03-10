# JET static analysis for NFFTOperators mul! methods
@testitem "@test_opt NFFTOperators mul!" tags = [:jet, :nfft] begin
    using JET, NFFTOperators, AbstractOperators
    image_size = (16, 16)
    D = 2
    trajectory = (rand(D, 32, 8) .- 0.5f0)  # Float32 k-space trajectory
    trajectory = Float64.(trajectory)
    dcf = ones(32, 8)
    op = NFFTOp(image_size, trajectory, dcf)
    img = zeros(ComplexF64, image_size)
    ksp = zeros(ComplexF64, 32, 8)
    @test_opt target_modules = (NFFTOperators,) mul!(ksp, op, img)
    @test_opt target_modules = (NFFTOperators,) mul!(img, AdjointOperator(op), ksp)
end

# JET static analysis for NfftNormalOp mul! methods
@testitem "@test_opt NfftNormalOp mul!" tags = [:jet, :nfft] begin
    using JET, NFFTOperators, AbstractOperators
    image_size = (16, 16)
    D = 2
    trajectory = Float64.(rand(D, 32, 8) .- 0.5f0)
    dcf = ones(32, 8)
    op = NFFTOp(image_size, trajectory, dcf)
    normal_op = AbstractOperators.get_normal_op(op)
    img = zeros(ComplexF64, image_size)
    @test_opt target_modules = (NFFTOperators,) mul!(img, normal_op, img)
end

# JET call analysis for NFFTOperators constructors
@testitem "@test_call NFFTOperators" tags = [:jet, :nfft] begin
    using JET, NFFTOperators
    image_size = (16, 16)
    trajectory = Float64.(rand(2, 32, 8) .- 0.5f0)
    dcf = ones(32, 8)

    @test_call target_modules = (NFFTOperators,) NFFTOp(image_size, trajectory, dcf)
end

# JET package-level analysis for NFFTOperators
@testitem "JET test_package NFFTOperators" tags = [:jet, :nfft] begin
    using JET, NFFTOperators
    JET.test_package(NFFTOperators; target_modules = (NFFTOperators,))
end
