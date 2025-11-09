using AbstractOperators
using Test
using Documenter
using Aqua
using Pkg
using LinearMaps

verb = false

@testset "AbstractOperators" begin
	@testset "Linear operators" begin
		for f in (
			"linearoperators/test_diagop.jl",
			"linearoperators/test_eye.jl",
			"linearoperators/test_finitediff.jl",
			"linearoperators/test_getindex.jl",
			"linearoperators/test_lbfgs.jl",
			"linearoperators/test_lmatrixop.jl",
			"linearoperators/test_matrixop.jl",
			"linearoperators/test_mylinop.jl",
			"linearoperators/test_variation.jl",
			"linearoperators/test_zeropad.jl",
			"linearoperators/test_zerosop.jl"
		)
			include(f)
		end
	end

	@testset "Non-Linear operators" begin
		include("test_nonlinear_operators.jl")
	end

	@testset "Linear Calculus rules" begin
		   for f in (
			   "calculus/test_adjointoperator.jl",
			   "calculus/test_affineadd.jl",
			   "calculus/test_Ax_mul_Bx.jl",
			   "calculus/test_Ax_mul_Bxt.jl",
			   "calculus/test_Axt_mul_Bx.jl",
			   "calculus/test_broadcast.jl",
			   "calculus/test_combinations.jl",
			   "calculus/test_compose.jl",
			   "calculus/test_dcat.jl",
			   "calculus/test_hadamardprod.jl",
			   "calculus/test_hcat.jl",
			   "calculus/test_Jacobian.jl",
			   "calculus/test_reshape.jl",
			   "calculus/test_scale.jl",
			   "calculus/test_sum.jl",
			   "calculus/test_vcat.jl"
		   )
			   include(f)
		   end
		   include("test_combination_rules.jl")
	end

	@testset "Batch operators" begin
		include("batching/test_SimpleBatchOp.jl")
		include("batching/test_SpreadingBatchOp.jl")
	end

	include("test_syntax.jl")

	@testset "Documentation" begin
		DocMeta.setdocmeta!(
			AbstractOperators,
			:DocTestSetup,
			:(using AbstractOperators);
			recursive=true,
		)
		doctest(AbstractOperators; fix=false)
	end

	Aqua.test_all(AbstractOperators)
end

include("test_LinearMapsExt.jl")

for sub in ("FFTWOperators", "NFFTOperators", "WaveletOperators", "DSPOperators")
	pkgdir = normpath(@__DIR__, "..", sub)
	Pkg.activate(pkgdir)
	Pkg.test()
end
