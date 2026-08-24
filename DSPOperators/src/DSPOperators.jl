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
    is_threaded,
    _resolve_threaded,
    _copy_operator_impl

# ─── Threading ────────────────────────────────────────────────────────────────
#
# `Conv` and `Xcorr` plan their own FFTW transforms (this package does not depend on
# DSP.jl), and FFTW is a *counted* thread pool: threading is a property of the plan,
# fixed at construction, exactly as for the FFTWOperators subpackage's DFT/DCT/etc. See
# `_dsp_fftw_num_threads` below and each operator's `num_threads`/`threaded` constructor
# keywords. `Filt` and `MIMOFilt`, by contrast, genuinely have no threaded path at any
# level: IIR filtering is a sequential recursion (each output sample depends on the
# previous one), not an FFT, so there is nothing for either Julia or FFTW to parallelize
# -- their `is_threaded`/`supports_threading = false` declarations are right after their
# `include`s below.

const THRESHOLD_C2C = 2^13

"""
	_dsp_fftw_num_threads(num_threads, threaded, n) -> Int

Resolve the plan-time FFTW thread count for `Conv`/`Xcorr`, mirroring
`FFTWOperators._fftw_num_threads`: `num_threads` is FFTW's own vocabulary and an explicit
command (wins outright if given); `threaded` is the package-wide keyword and follows the
package-wide rule (`false` vetoes, `true` enables subject to the size policy).

PROVENANCE: provisional. `THRESHOLD_C2C` is borrowed from FFTWOperators'
`fftw_threading_threshold(:c2c) == 2^13` rather than independently measured -- both
operators plan and execute a `c2c`/`r2c` FFT of the padded convolution length internally,
so the same cost class applies, but this package has no benchmark sweep of its own yet.
"""
function _dsp_fftw_num_threads(num_threads, threaded::Bool, n::Int)
    num_threads !== nothing && return Int(num_threads)
    use = _resolve_threaded(threaded) do
        Threads.nthreads() > 1 && n >= THRESHOLD_C2C
    end
    return use ? Threads.nthreads() : 1
end

function _dsp_with_fftw_threads(f, num_threads::Int)
    prev = FFTW.get_num_threads()
    FFTW.set_num_threads(num_threads)
    try
        return f()
    finally
        FFTW.set_num_threads(prev)
    end
end

include("Conv.jl")
include("Filt.jl")
include("MIMOFilt.jl")
include("Xcorr.jl")

# `Filt`/`MIMOFilt` have no threaded path at any level -- see the module docstring above.
# Declaring `supports_threading = false` is what lets a threaded batch operator wrap them
# (`threaded = false` then asks nothing of them, so `copy_operator`'s fallback can answer
# with a plain deepcopy instead of refusing); without it, batching either of them raises.
for OpType in (:AbstractFilt, :AbstractMIMOFilt)
    @eval AbstractOperators.is_threaded(::$OpType) = false
    @eval AbstractOperators.supports_threading(::$OpType) = false
end

end # module DSPOperators
