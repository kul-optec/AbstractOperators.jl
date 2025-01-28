using AbstractOperators
using LinearAlgebra, FFTW, DSP, SparseArrays, RecursiveArrayTools, CUDA
using Printf
using Random
using Test
using Documenter
using Aqua
using BenchmarkTools
using Base.Threads: nthreads, @threads

include("utils.jl")
Random.seed!(0)

verb = true

@testset "AbstractOperators" begin
	@testset "Linear operators" begin
		include("test_linear_operators.jl")
	end

	@testset "Non-Linear operators" begin
		include("test_nonlinear_operators.jl")
	end

	@testset "Linear Calculus rules" begin
		include("test_linear_operators_calculus.jl")
	end

	@testset "Nonlinear Calculus rules" begin
		include("test_nonlinear_operators_calculus.jl")
	end

	@testset "L-BFGS" begin
		include("test_lbfgs.jl")
	end

	@testset "Batch operators" begin
		include("test_SimpleBatchOp.jl")
		include("test_SpreadingBatchOp.jl")
	end

	@testset "Syntax shorthands" begin
		include("test_syntax.jl")
	end

	@testset "Documentation" begin
		DocMeta.setdocmeta!(
			AbstractOperators, :DocTestSetup, :(using AbstractOperators); recursive=true
		)
		doctest(AbstractOperators; fix=false)
	end

	Aqua.test_all(AbstractOperators)
end
