export Conv

abstract type AbstractConv{T, N, H <: AbstractArray} <: LinearOperator end

"""
	Conv([domain_type=Float64::Type,] dim_in::Tuple, h::AbstractVector; num_threads, threaded)
	Conv(x::AbstractVector, h::AbstractVector; num_threads, threaded)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractVector`, returns the convolution between `x` and `h`. Uses `conv` and hence FFT algorithm.

- `num_threads`: the number of FFTW threads to plan with. Defaults to the number of Julia
  threads available. `num_threads` wins if both it and `threaded` are given.
- `threaded`: the package-wide spelling of the same choice: `true` (default) uses the
  available Julia threads subject to the size policy, `false` forces one. FFTW is a counted
  thread pool, so this is fixed when the plan is built and reported back by `is_threaded`.

```jldoctest
julia> using DSPOperators

julia> Conv((10,),randn(5))
★  ℝ^10 -> ℝ^14

```
"""
struct Conv{T, N, H <: AbstractArray{T}, Hc, P1 <: AbstractFFTs.Plan, P2 <: AbstractFFTs.Plan} <: AbstractConv{T, N, H}
    dim_in::NTuple{N, Int}
    h::H
    buf::H
    buf_c1::Hc
    buf_c2::Hc
    R::P1
    I::P2
    # The thread count the FFTW plans were built with -- see `DFT.num_threads` in
    # FFTWOperators for why this is recorded rather than switchable.
    num_threads::Int
end

# Constructors

isTypeReal(::Type{T}) where {T} = T <: Real

###standard constructor
function Conv(
        domain_type::Type, dim_in::NTuple{N, Int}, h::H;
        num_threads = nothing, threaded::Bool = true,
    ) where {N, H <: AbstractArray}
    eltype(h) != domain_type && error("eltype(h) is $(eltype(h)), should be $(domain_type)")

    buf = similar(h, domain_type, dim_in .+ size(h) .- 1)
    nthr = _dsp_fftw_num_threads(num_threads, threaded, length(buf))
    # Only the `plan_*` calls themselves consult FFTW's global thread count, so only they
    # need to run inside `_dsp_with_fftw_threads` -- keeping the closure to that (rather
    # than the surrounding buffer allocation and branching) matches every other FFTW-based
    # operator in this codebase (see FFTWOperators' RDFT/DCT) and keeps JET's `@test_call`
    # able to infer the closure's return type precisely.
    if isTypeReal(domain_type)
        buf_size = ntuple(d -> d == 1 ? size(buf, d) >> 1 + 1 : size(buf, d), Val(N))
        buf_c1 = similar(h, Complex{domain_type}, buf_size)
        R, I = _dsp_with_fftw_threads(nthr) do
            plan_rfft(buf), plan_irfft(buf_c1, size(buf, 1))
        end
    else
        buf_c1 = similar(buf)
        R = _dsp_with_fftw_threads(() -> plan_fft(buf), nthr)
        I = inv(R)
    end
    buf_c2 = similar(buf_c1)
    return Conv{domain_type, N, H, typeof(buf_c1), typeof(R), typeof(I)}(
        dim_in, h, buf, buf_c1, buf_c2, R, I, nthr
    )
end

Conv(dim_in::NTuple{N, Int}, h::H; kwargs...) where {H <: AbstractVector, N} = Conv(eltype(h), dim_in, h; kwargs...)
Conv(x::H, h::H; kwargs...) where {H} = Conv(eltype(x), size(x), h; kwargs...)
Conv(dim_in::NTuple{N, Int}, h::H; kwargs...) where {H <: AbstractArray, N} = Conv(eltype(h), dim_in, h; kwargs...)
Conv(x::H, h::H; kwargs...) where {H <: AbstractArray} = Conv(eltype(x), size(x), h; kwargs...)

# Mappings
function mul!(
        y::AbstractArray{T, N}, A::AbstractConv{T, N}, b::AbstractArray{T, N}
    ) where {T, N}
    check(y, A, b)
    fill!(A.buf, zero(T))
    view(A.buf, axes(A.h)...) .= A.h
    mul!(A.buf_c1, A.R, A.buf)
    fill!(A.buf, zero(T))
    view(A.buf, axes(b)...) .= b
    mul!(A.buf_c2, A.R, A.buf)
    A.buf_c2 .*= A.buf_c1
    return mul!(y, A.I, A.buf_c2)
end

function mul!(
        y::AbstractArray{T, N}, L::AdjointOperator{C}, b::AbstractArray{T, N}
    ) where {T, N, C <: AbstractConv{T, N}}
    check(y, L, b)
    fill!(L.A.buf, zero(T))
    view(L.A.buf, axes(L.A.h)...) .= L.A.h
    mul!(L.A.buf_c1, L.A.R, L.A.buf)
    fill!(L.A.buf, zero(T))
    view(L.A.buf, axes(b)...) .= b
    mul!(L.A.buf_c2, L.A.R, L.A.buf)
    L.A.buf_c2 .*= conj.(L.A.buf_c1)
    mul!(L.A.buf, L.A.I, L.A.buf_c2)
    y .= view(L.A.buf, axes(y)...)
    return y
end

# Properties

domain_type(::AbstractConv{T}) where {T} = T
codomain_type(::AbstractConv{T}) where {T} = T
domain_array_type(::AbstractConv{T, N, H}) where {T, N, H} = H
codomain_array_type(::AbstractConv{T, N, H}) where {T, N, H} = H
is_thread_safe(::Conv) = false
is_threaded(op::Conv) = op.num_threads > 1
supports_threading(::Conv) = true

function _copy_operator_impl(
        op::Conv{T, N, H, Hc, P1, P2}; storage_type = nothing, threaded = nothing
    ) where {T, N, H, Hc, P1, P2}
    new_threaded = threaded === nothing ? is_threaded(op) : threaded
    if storage_type === nothing && new_threaded == is_threaded(op)
        # Plans are read-only during execution and safe to share; only the per-call
        # scratch buffers (mutated by `mul!`) need a fresh allocation.
        return Conv{T, N, H, Hc, P1, P2}(
            op.dim_in, op.h, similar(op.buf), similar(op.buf_c1), similar(op.buf_c2), op.R, op.I, op.num_threads
        )
    end
    # `h` is the operator's actual filter data, not a scratch buffer, so a storage-type
    # change must carry its values over with `copyto!` rather than allocate uninitialized.
    new_h = storage_type === nothing ? op.h : copyto!(similar(storage_type{T}, size(op.h)), op.h)
    return Conv(op.dim_in, new_h; threaded = new_threaded)
end

#TODO find out a way to verify this,
is_full_row_rank(L::Conv) = true
is_full_column_rank(L::Conv) = true
is_full_row_rank(::AbstractConv) = true
is_full_column_rank(::AbstractConv) = true

size(L::Conv) = (L.dim_in[1] + length(L.h) - 1,), L.dim_in
size(L::AbstractConv) = (L.dim_in[1] + length(L.h) - 1,), L.dim_in

fun_name(A::Conv) = "★"
fun_name(::AbstractConv) = "★"
