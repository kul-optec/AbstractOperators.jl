module DSPOperators

using AbstractOperators, FFTW
import LinearAlgebra: mul!
import Base: size, ndims
using DSP: xcorr, conv

import AbstractOperators:
	domain_type,
	codomain_type,
	fun_name,
	get_normal_op,
	allocate_in_domain,
	allocate_in_codomain,
	domain_storage_type,
	codomain_storage_type,
	is_full_column_rank,
	is_full_row_rank,
	is_thread_safe


include("Conv.jl")
include("Filt.jl")
include("MIMOFilt.jl")
include("Xcorr.jl")

end # module DSPOperators
