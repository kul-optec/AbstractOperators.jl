function test_nufft_op(op, plan, image, dcf)
	ksp₁ = similar(image, ComplexF64, size(op, 1))
	mul!(vec(ksp₁), plan, image)
	ksp₂ = similar(ksp₁)
	mul!(ksp₂, op, image)
	@test ksp₂ == ksp₁
	image₂ = similar(image)
	if dcf === nothing
		mul!(image₂, plan', vec(ksp₂ .* op.dcf))
		@test norm(image₂ .- image) / norm(image) < 0.5
	else
		image₁ = similar(image)
		mul!(image₁, plan', vec(ksp₁ .*= dcf))
		mul!(image₂, op', ksp₂)
		@test image₂ ≈ image₁
	end
	normal_op = AbstractOperators.get_normal_op(op)
	image₃ = similar(image)
	mul!(image₃, normal_op, image)
	@test image₃ ≈ image₂
end

function test_2D_nufft_op(threaded)
	trajectory = rand(2, 128, 50) .- 0.5
	dcf = rand(128, 50)
	image_size = (128, 128)
	image = rand(ComplexF64, image_size)
	plan = plan_nfft(reshape(trajectory, 2, :), image_size)
	op = NfftOp(image_size, trajectory, dcf; threaded)
	return test_nufft_op(op, plan, image, dcf)
end

function test_3D_nufft_op(threaded)
	trajectory = rand(3, 128, 50) .- 0.5
	dcf = rand(128, 50)
	image_size = (64, 64, 64)
	image = rand(ComplexF64, image_size)
	plan = plan_nfft(reshape(trajectory, 3, :), image_size)
	op = NfftOp(image_size, trajectory, dcf; threaded)
	return test_nufft_op(op, plan, image, dcf)
end

function test_realistic_2D_nufft_op(threaded)
	trajectory = Array{Float64}(undef, 2, 256, 201)
	ϕₛₜₑₚ = 2π / 201
	for i in 1:201
		ϕ = i * ϕₛₜₑₚ
		trajectory[1, :, i] = cos(ϕ) .* ((-64:0.5:63.5) ./ 128)
		trajectory[2, :, i] = sin(ϕ) .* ((-64:0.5:63.5) ./ 128)
	end
	image_size = (128, 128)
	image = zeros(ComplexF64, image_size)
	for idx in CartesianIndices(image)
		d = norm([idx[1] - 64, idx[2] - 64])
		if d < 15
			image[idx] = 1.0
		end
	end
	plan = plan_nfft(reshape(trajectory, 2, :), image_size)
	op = NfftOp(image_size, trajectory; threaded)
	return test_nufft_op(op, plan, image, nothing)
end

@testset "NfftOp" begin
	@testset "2D" begin
		@testset "single-threaded" begin
			test_2D_nufft_op(false)
		end
		@testset "multi-threaded" begin
			test_2D_nufft_op(true)
		end
	end
	@testset "realistic 2D" begin
		@testset "single-threaded" begin
			test_realistic_2D_nufft_op(false)
		end
		@testset "multi-threaded" begin
			test_realistic_2D_nufft_op(true)
		end
	end
	@testset "3D" begin
		@testset "single-threaded" begin
			test_3D_nufft_op(false)
		end
		@testset "multi-threaded" begin
			test_3D_nufft_op(true)
		end
	end
end
