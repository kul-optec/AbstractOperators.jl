module NFFTOperators

export NFFTOp

using LinearAlgebra
using AbstractOperators
using FastBroadcast
using NestedThreading: with_full_threads, with_restricted_threads
import LinearAlgebra: mul!
import Base: size
import NFFT: NFFT
import NFFTTools: NFFTTools
import AbstractOperators:
    domain_type,
    codomain_type,
    fun_name,
    get_normal_op,
    allocate_in_domain,
    allocate_in_codomain,
    domain_array_type,
    codomain_array_type,
    _array_wrapper_type,
    is_threaded,
    supports_threading,
    is_thread_safe,
    _copy_operator_impl,
    AdjointOperator

import FFTW: FFTW

include("NFFTOp.jl")
include("NormalNfftOp.jl")

end # module NFFTOperators
