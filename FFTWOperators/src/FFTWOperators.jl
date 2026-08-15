module FFTWOperators

using AbstractOperators, FFTW, LinearAlgebra
using Base.Cartesian: @ncall
using Polyester: @batch
import LinearAlgebra: mul!
import Base: size, ndims

import AbstractOperators:
    _normalize_array_type,
    _array_wrapper_type,
    domain_type,
    codomain_type,
    fun_name,
    get_normal_op,
    allocate_in_domain,
    allocate_in_codomain,
    domain_array_type,
    codomain_array_type,
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
    check,
    is_threaded,
    supports_threading,
    _copy_operator_impl

"""
	_fftw_num_threads(num_threads, threaded) -> Int

Resolve the plan-time FFTW thread count from the two spellings a caller may use.

`num_threads` is FFTW's own vocabulary and wins when given. `threaded` is the vocabulary
used uniformly across AbstractOperators, and maps onto it: `true` means "use the available
Julia threads", `false` means one thread. FFTW is a *counted* pool in NestedThreading, so
this is a plan property rather than a loop property -- which is why it is fixed at
construction and `is_threaded` merely reads it back.
"""
function _fftw_num_threads(num_threads, threaded)
    num_threads !== nothing && return Int(num_threads)
    threaded === nothing && return Threads.nthreads()
    return threaded ? Threads.nthreads() : 1
end

include("DFT.jl")
include("RDFT.jl")
include("IRDFT.jl")
include("DCT.jl")
include("Shift.jl")
include("combination_rules.jl")

end # module FFTWOperators
