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
    _resolve_threaded,
    _elementwise_threaded,
    threading_threshold,
    THRESHOLD_MEMORY_BOUND,
    _copy_operator_impl

"""
	fftw_threading_threshold(kind::Symbol) -> Int

Element count at which threading an FFTW transform of this `kind` starts to pay.

PROVENANCE: measured (AMD EPYC 7352, 8 threads, OPENBLAS_NUM_THREADS=1, 2026-08-15),
sweeping `mul!` against a 1-thread plan of the same size:

| kind | first sustained win | speedup there | at n = 2^22 |
|---|---|---|---|
| `:c2c` (DFT/IDFT) | 2^13 | 1.88x | 5.04x |
| `:r2r` (DCT/IDCT) | 2^15 | 1.44x | 2.00x |
| `:r2c` (RDFT/IRDFT) | 2^15 | 1.25x | 3.69x |

Below these sizes threading an FFT is a *large* pessimisation, not a wash -- a 256-point
c2c transform measures 0.02x -- which is why the policy applies here rather than trusting
FFTW's planner to sort it out.
"""
fftw_threading_threshold(kind::Symbol) = kind === :c2c ? 2^13 : 2^15

"""
	_fftw_num_threads(kind, num_threads, threaded, n) -> Int

Resolve the plan-time FFTW thread count.

`num_threads` is FFTW's own vocabulary and is an explicit **command**: given, it wins
outright, which keeps an escape hatch for callers who know what they want. `threaded` is the
package-wide keyword and follows the package-wide rule -- `false` vetoes, `true`/`nothing`
enable subject to the policy above. See `AbstractOperators._resolve_threaded`.
"""
function _fftw_num_threads(kind::Symbol, num_threads, threaded::Bool, n::Int)
    num_threads !== nothing && return Int(num_threads)
    use = _resolve_threaded(threaded) do
        Threads.nthreads() > 1 && n >= fftw_threading_threshold(kind)
    end
    return use ? Threads.nthreads() : 1
end

"""
	_with_fftw_threads(f, num_threads)

Run the planning callable `f` with FFTW's global thread count temporarily set to
`num_threads`, restoring the previous value afterwards.

FFTW's thread count is process-global state consulted at *plan* time, so it has to be set
around planning and put back; forgetting the restore would silently change the thread count
of every plan built later in the session.
"""
function _with_fftw_threads(f, num_threads::Int)
    prev = FFTW.get_num_threads()
    FFTW.set_num_threads(num_threads)
    try
        return f()
    finally
        FFTW.set_num_threads(prev)
    end
end

include("DFT.jl")
include("RDFT.jl")
include("IRDFT.jl")
include("DCT.jl")
include("Shift.jl")
include("combination_rules.jl")

end # module FFTWOperators
