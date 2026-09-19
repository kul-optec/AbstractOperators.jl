export Xcorr

"""
	Xcorr([domain_type=Float64::Type,] dim_in::Tuple, h::AbstractVector; num_threads, threaded)
	Xcorr(x::AbstractVector, h::AbstractVector; num_threads, threaded)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractVector`, returns the cross correlation between `x` and `h`. Uses FFT-based implementation.

- `domain_type`: the element type of the operator's domain. Defaults to `Float64`, or, when
  `x` is given instead of `dim_in`, to `eltype(x)`. Must match `eltype(h)`.
- `dim_in::Tuple`: the size of the input array `x` the operator acts on. Must be
  one-dimensional (SISO); see `Filt`/`MIMOFilt` for MIMO filtering.
- `x::AbstractVector`: an input array whose `size` and `eltype` are used in place of
  `dim_in`/`domain_type`.
- `h::AbstractVector`: the filter kernel to cross-correlate with.
- `num_threads`: the number of FFTW threads to plan with. Defaults to the number of Julia
  threads available. `num_threads` wins if both it and `threaded` are given.
- `threaded`: the package-wide spelling of the same choice: `true` (default) uses the
  available Julia threads subject to the size policy, `false` forces one. FFTW is a counted
  thread pool, so this is fixed when the plan is built and reported back by `is_threaded`.

Examples
```jldoctest
julia> using DSPOperators

julia> Xcorr(Float64, (10,), [1.0, 0.5, 0.2])
◎  ℝ^10 -> ℝ^19
```
"""
struct Xcorr{
        T, H <: AbstractVector{T}, Hc <: AbstractVector,
        P1 <: AbstractFFTs.Plan, P2 <: AbstractFFTs.Plan,
        P3 <: AbstractFFTs.Plan, P4 <: AbstractFFTs.Plan,
    } <: LinearOperator
    dim_in::Tuple{Int}
    h::H
    # Forward pass (xcorr)
    fftlen_fwd::Int
    padlen::Int
    h_fft_conj::Hc   # conj(rfft(h padded to fftlen_fwd))
    buf_fwd::H       # scratch buffer size fftlen_fwd
    buf_fwd_c::Hc    # complex scratch buffer
    R_fwd::P1        # rfft plan, fftlen_fwd
    I_fwd::P2        # irfft plan, fftlen_fwd
    # Adjoint pass (conv(b, h) and slice)
    fftlen_adj::Int
    h_fft_adj::Hc    # rfft(h padded to fftlen_adj)
    buf_adj::H       # scratch buffer size fftlen_adj
    buf_adj_c::Hc    # complex scratch buffer
    R_adj::P3        # rfft plan, fftlen_adj
    I_adj::P4        # irfft plan, fftlen_adj
    # The thread count the FFTW plans were built with -- see `DFT.num_threads` in
    # FFTWOperators for why this is recorded rather than switchable.
    num_threads::Int
end

# FFT planning flags: FFTW.MEASURE only for CPU Arrays; no flags for GPU backends.
_xcorr_plan_kwargs(::Type{<:Array}) = (flags = FFTW.MEASURE,)
_xcorr_plan_kwargs(::Type) = (;)

# Constructors
function Xcorr(
        domain_type::Type, DomainDim::NTuple{N, Int}, h::H;
        num_threads = nothing, threaded::Bool = true,
    ) where {H <: AbstractVector, N}
    eltype(h) != domain_type && error("eltype(h) is $(eltype(h)), should be $(domain_type)")
    N != 1 && error("Xcorr treats only SISO, check Filt and MIMOFilt for MIMO")

    n = DomainDim[1]
    m = length(h)
    padlen = max(n, m)
    outlen = 2 * padlen - 1

    plan_kw = _xcorr_plan_kwargs(H)

    fftlen_fwd = nextpow(2, outlen)
    fftlen_adj = fftlen_fwd
    nthr = _dsp_fftw_num_threads(num_threads, threaded, fftlen_fwd)

    # Only the `plan_*` calls themselves consult FFTW's global thread count, so only they
    # need to run inside `_dsp_with_fftw_threads` -- keeping the closures to that (rather
    # than the surrounding buffer allocation and branching) matches every other FFTW-based
    # operator in this codebase (see FFTWOperators' RDFT/DCT) and keeps JET's `@test_call`
    # able to infer each closure's return type precisely.

    # Forward pass plans
    buf_fwd = similar(h, fftlen_fwd)
    if domain_type <: Real
        complex_type = Complex{domain_type}
        buf_fwd_c = similar(h, complex_type, fftlen_fwd ÷ 2 + 1)
        R_fwd, I_fwd = _dsp_with_fftw_threads(nthr) do
            plan_rfft(buf_fwd; plan_kw...), plan_irfft(buf_fwd_c, fftlen_fwd; plan_kw...)
        end
    else
        buf_fwd_c = similar(buf_fwd)
        R_fwd = _dsp_with_fftw_threads(() -> plan_fft(buf_fwd; plan_kw...), nthr)
        I_fwd = inv(R_fwd)
    end

    # Adjoint pass: CPU uses tiled FIR — no FFT state needed.
    # GPU backends allocate FFT plans; same fftlen as forward pass is correct.
    buf_adj = similar(h, fftlen_adj)
    if domain_type <: Real
        buf_adj_c = similar(h, Complex{domain_type}, fftlen_adj ÷ 2 + 1)
        R_adj, I_adj = _dsp_with_fftw_threads(nthr) do
            plan_rfft(buf_adj; plan_kw...), plan_irfft(buf_adj_c, fftlen_adj; plan_kw...)
        end
    else
        buf_adj_c = similar(buf_adj)
        R_adj = _dsp_with_fftw_threads(() -> plan_fft(buf_adj; plan_kw...), nthr)
        I_adj = inv(R_adj)
    end

    fill!(buf_fwd, zero(domain_type))
    copyto!(view(buf_fwd, 1:m), h)
    h_fft_conj = conj.(R_fwd * buf_fwd)
    fill!(buf_fwd, zero(domain_type))

    fill!(buf_adj, zero(domain_type))
    copyto!(view(buf_adj, 1:m), h)
    h_fft_adj = R_adj * buf_adj
    fill!(buf_adj, zero(domain_type))

    return Xcorr{
        domain_type, typeof(h), typeof(buf_fwd_c),
        typeof(R_fwd), typeof(I_fwd), typeof(R_adj), typeof(I_adj),
    }(
        DomainDim, h,
        fftlen_fwd, padlen, h_fft_conj, buf_fwd, buf_fwd_c, R_fwd, I_fwd,
        fftlen_adj, h_fft_adj, buf_adj, buf_adj_c, R_adj, I_adj,
        nthr,
    )
end

Xcorr(x::H, h::H; kwargs...) where {H} = Xcorr(eltype(x), size(x), h; kwargs...)

# Mappings

function mul!(y, A::Xcorr{T}, b) where {T}
    check(y, A, b)
    n = length(b)
    # Forward: xcorr(b, h; padmode=:longest)
    # = irfft(rfft(b_padded) .* conj(rfft(h_padded)), fftlen)[fftlen-padlen+2:fftlen, 1:padlen]
    fill!(A.buf_fwd, zero(T))
    copyto!(view(A.buf_fwd, 1:n), b)
    mul!(A.buf_fwd_c, A.R_fwd, A.buf_fwd)
    A.buf_fwd_c .*= A.h_fft_conj
    mul!(A.buf_fwd, A.I_fwd, A.buf_fwd_c)
    # Gather: DSP.xcorr format = [neg lags ascending, non-neg lags ascending]
    # neg lags -(padlen-1) to -1 are at positions fftlen-padlen+2 to fftlen
    # pos lags 0 to padlen-1 are at positions 1 to padlen
    fftlen, padlen = A.fftlen_fwd, A.padlen
    neg_start = fftlen - padlen + 2
    copyto!(view(y, 1:(padlen - 1)), view(A.buf_fwd, neg_start:fftlen))
    copyto!(view(y, padlen:length(y)), view(A.buf_fwd, 1:padlen))
    return y
end

# CPU adjoint: tiled 8-wide FIR (fast for cache-resident accumulators)
function mul!(y, L::AdjointOperator{<:Xcorr{T, <:Array{T}}}, b) where {T}
    check(y, L, b)
    A = L.A
    _xcorr_fir_adj!(y, b, A.h, A.padlen)
    return y
end

# GPU adjoint: FFT-based conv
function mul!(y, L::AdjointOperator{<:Xcorr{T}}, b) where {T}
    check(y, L, b)
    A = L.A
    n = length(y)
    outlen = length(b)
    fill!(A.buf_adj, zero(T))
    copyto!(view(A.buf_adj, 1:outlen), b)
    mul!(A.buf_adj_c, A.R_adj, A.buf_adj)
    A.buf_adj_c .*= A.h_fft_adj
    mul!(A.buf_adj, A.I_adj, A.buf_adj_c)
    padlen = A.padlen
    y .= @view(A.buf_adj[padlen:(padlen + n - 1)])
    return y
end

# Tiled FIR adjoint: y[j] = Σ_k h[k] * b[padlen+j-k]  (k = 1..m)
# Processes 8 output samples per outer iteration to keep accumulators in
# registers and avoid repeated reads/writes of y.
function _xcorr_fir_adj!(y, b, h, padlen)
    m = length(h); n = length(y); T = eltype(y)
    j = 1
    @inbounds while j ≤ n - 7
        a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = zero(T)
        for k in 1:m
            hk = h[k]
            base = padlen + j - k
            a0 = muladd(hk, b[base], a0)
            a1 = muladd(hk, b[base + 1], a1)
            a2 = muladd(hk, b[base + 2], a2)
            a3 = muladd(hk, b[base + 3], a3)
            a4 = muladd(hk, b[base + 4], a4)
            a5 = muladd(hk, b[base + 5], a5)
            a6 = muladd(hk, b[base + 6], a6)
            a7 = muladd(hk, b[base + 7], a7)
        end
        y[j] = a0; y[j + 1] = a1; y[j + 2] = a2; y[j + 3] = a3
        y[j + 4] = a4; y[j + 5] = a5; y[j + 6] = a6; y[j + 7] = a7
        j += 8
    end
    @inbounds while j ≤ n
        acc = zero(T)
        for k in 1:m
            acc = muladd(h[k], b[padlen + j - k], acc)
        end
        y[j] = acc
        j += 1
    end
    return
end

# Properties

domain_type(::Xcorr{T}) where {T} = T
codomain_type(::Xcorr{T}) where {T} = T
domain_array_type(::Xcorr{T, H}) where {T, H} = H
codomain_array_type(::Xcorr{T, H}) where {T, H} = H
is_thread_safe(::Xcorr) = false
is_threaded(op::Xcorr) = op.num_threads > 1
supports_threading(::Xcorr) = true

function _copy_operator_impl(
        op::Xcorr{T, H, Hc, P1, P2, P3, P4}; storage_type = nothing, threaded = nothing
    ) where {T, H, Hc, P1, P2, P3, P4}
    new_threaded = threaded === nothing ? is_threaded(op) : threaded
    if storage_type === nothing && new_threaded == is_threaded(op)
        # Plans are read-only during execution and safe to share; only the per-call
        # scratch buffers (mutated by `mul!`) need a fresh allocation.
        return Xcorr{T, H, Hc, P1, P2, P3, P4}(
            op.dim_in, op.h,
            op.fftlen_fwd, op.padlen, op.h_fft_conj, similar(op.buf_fwd), similar(op.buf_fwd_c), op.R_fwd, op.I_fwd,
            op.fftlen_adj, op.h_fft_adj, similar(op.buf_adj), similar(op.buf_adj_c), op.R_adj, op.I_adj,
            op.num_threads,
        )
    end
    # `h` is the operator's actual filter data, not a scratch buffer, so a storage-type
    # change must carry its values over with `copyto!` rather than allocate uninitialized.
    new_h = storage_type === nothing ? op.h : copyto!(similar(storage_type{T}, size(op.h)), op.h)
    return Xcorr(T, op.dim_in, new_h; threaded = new_threaded)
end

is_full_row_rank(L::Xcorr) = true
is_full_column_rank(L::Xcorr) = true

size(L::Xcorr) = (2 * max(L.dim_in[1], length(L.h)) - 1,), L.dim_in

fun_name(A::Xcorr) = "◎"
