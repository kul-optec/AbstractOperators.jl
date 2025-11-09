module NFFTOperators

export NfftOp

using LinearAlgebra
using AbstractOperators
using FastBroadcast
using Polyester: disable_polyester_threads
import LinearAlgebra: mul!
import Base: size
import NFFT: NFFT
import NFFTTools: NFFTTools
import AbstractOperators:
	set_thread_counts_expr,
	domain_type,
	codomain_type,
	fun_name,
	get_normal_op,
	allocate_in_domain,
	allocate_in_codomain,
	domain_storage_type,
	codomain_storage_type,
	AdjointOperator
import Base.Threads: nthreads
import FFTW: FFTW

include("NfftOp.jl")
include("NormalNfftOp.jl")

end # module NFFTOperators