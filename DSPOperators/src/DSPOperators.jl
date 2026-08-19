module DSPOperators

using AbstractOperators, FFTW
using AbstractFFTs: AbstractFFTs
import LinearAlgebra: mul!
import Base: size, ndims

import AbstractOperators:
    domain_type,
    codomain_type,
    check,
    fun_name,
    get_normal_op,
    allocate_in_domain,
    allocate_in_codomain,
    domain_array_type,
    codomain_array_type,
    is_full_column_rank,
    is_full_row_rank,
    is_thread_safe,
    supports_threading,
    is_threaded

include("Conv.jl")
include("Filt.jl")
include("MIMOFilt.jl")
include("Xcorr.jl")

# ─── Threading ────────────────────────────────────────────────────────────────
#
# None of the DSP operators has a Julia-level threaded path: the work happens inside
# DSP.jl/FFTW, which manage their own parallelism. Declaring `supports_threading = false`
# is what lets a threaded batch operator wrap them -- `threaded = false` then asks nothing
# of them, so `copy_operator`'s fallback can answer it with a plain deepcopy instead of
# refusing. Without these declarations, batching any of them raises.
for OpType in (:Conv, :Xcorr, :AbstractFilt, :AbstractMIMOFilt)
    @eval AbstractOperators.is_threaded(::$OpType) = false
    @eval AbstractOperators.supports_threading(::$OpType) = false
end

end # module DSPOperators
