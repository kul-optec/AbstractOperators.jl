export Xcorr
#TODO make more efficient

"""
	Xcorr([domain_type=Float64::Type,] dim_in::Tuple, h::AbstractVector)
	Xcorr(x::AbstractVector, h::AbstractVector)

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractVector`, returns the cross correlation between `x` and `h`. Uses `xcross`.

Examples
```jldoctest
julia> using DSPOperators

julia> Xcorr(Float64, (10,), [1.0, 0.5, 0.2])
◎  ℝ^10 -> ℝ^19
```
"""
struct Xcorr{T,H<:AbstractVector{T}} <: LinearOperator
	dim_in::Tuple{Int}
	h::H
end

# Constructors
function Xcorr(domain_type::Type, DomainDim::NTuple{N,Int}, h::H) where {H<:AbstractVector,N}
	eltype(h) != domain_type && error("eltype(h) is $(eltype(h)), should be $(domain_type)")
	N != 1 && error("Xcorr treats only SISO, check Filt and MIMOFilt for MIMO")
	return Xcorr{domain_type,H}(DomainDim, h)
end
Xcorr(x::H, h::H) where {H} = Xcorr(eltype(x), size(x), h)

# Mappings

function mul!(y::H, A::Xcorr{T,H}, b::H) where {T,H}
	return y .= xcorr(b, A.h; padmode=:longest)
end

function mul!(y::H, L::AdjointOperator{Xcorr{T,H}}, b::H) where {T,H}
	A = L.A
	l = floor(Int64, size(A, 1)[1] / 2)
	idx = (l + 1):(l + length(y))
	return y .= conv(b, A.h)[idx]
end

# Properties

domain_type(::Xcorr{T}) where {T} = T
codomain_type(::Xcorr{T}) where {T} = T
is_thread_safe(::Xcorr) = false

#TODO find out a way to verify this,
is_full_row_rank(L::Xcorr) = true
is_full_column_rank(L::Xcorr) = true

size(L::Xcorr) = (2 * max(L.dim_in[1], length(L.h)) - 1,), L.dim_in

fun_name(A::Xcorr) = "◎"
