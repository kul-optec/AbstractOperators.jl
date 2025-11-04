module FFTWOperators

using AbstractOperators, FFTW, LinearAlgebra
using Base.Cartesian: @ncall
using Polyester: @batch
import LinearAlgebra: mul!
import Base: size, ndims

import AbstractOperators:
	domain_type,
	codomain_type,
	fun_name,
	get_normal_op,
	allocate_in_domain,
	allocate_in_codomain,
	domain_storage_type,
	codomain_storage_type,
	can_be_combined,
	combine,
	is_thread_safe,
	is_AcA_diagonal,
	is_AAc_diagonal,
	diag_AcA,
	diag_AAc,
	is_orthogonal,
	is_invertible,
	is_full_row_rank,
	is_full_column_rank,
	is_symmetric,
	has_fast_opnorm,
	check

function __init__()
	push!(
		AbstractOperators.thread_count_functions[],
		FFTW.get_num_threads => FFTW.set_num_threads,
	)
end

include("DFT.jl")
include("RDFT.jl")
include("IRDFT.jl")
include("DCT.jl")
include("Shift.jl")
include("combination_rules.jl")

end # module FFTWOperators
