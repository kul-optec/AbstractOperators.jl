export Filt

"""
`Filt([domainType=Float64::Type,] dim_in::Tuple, b::AbstractVector, [a::AbstractVector,])`

`Filt(x::AbstractVector, b::AbstractVector, [a::AbstractVector,])`

Creates a `LinearOperator` which, when multiplied with an array `x::AbstractVector`, returns a vector `y` filtered by an IIR filter of coefficients `b` and `a`. If only `b` is provided a FIR is used to comute `y` instead. 

"""
struct Filt{T,N} <: LinearOperator
	dim_in::NTuple{N,Int}
	b::AbstractVector{T}
	a::AbstractVector{T}
	si::AbstractVector{T}
	function Filt{T,N}(dim_in,b,a) where {T,N}
		isempty(b) && throw(ArgumentError("filter vector b must be non-empty"))
		isempty(a) && throw(ArgumentError("filter vector a must be non-empty"))
		a[1] == 0  && throw(ArgumentError("filter vector a[1] must be nonzero"))
		as = length(a)
		bs = length(b)
		sz = max(as, bs)
		silen = sz - 1

		# Filter coefficient normalization
		if a[1] != 1
			norml = a[1]
			a ./= norml
			b ./= norml
		end
		# Pad the coefficients with zeros if needed
		bs<sz   && (b = copyto!(zeros(eltype(b), sz), b))
		1<as<sz && (a = copyto!(zeros(eltype(a), sz), a))

		si = zeros(promote_type(eltype(b), eltype(a)), max(length(a), length(b))-1)
		new(dim_in,b,a,si)
	end
end

# Constructors

#default constructor
function Filt(T::Type, dim_in::NTuple{N,Int}, b::AbstractVector, a::AbstractVector) where {N}
	eltype(b) != T && error("eltype of b is $(eltype(b)), should be $T")
	eltype(a) != T && error("eltype of a is $(eltype(b)), should be $T")
	Filt{T,N}(dim_in,b,a)
end

function Filt(T::Type, dim_in::NTuple{N,Int}, b::AbstractVector) where {N}
	eltype(b) != T && error("eltype of b is $(eltype(b)), should be $T")
	Filt{T,N}(dim_in,b,[convert(T,1.0)])
end

Filt(dim_in::Int,  b::AbstractVector, a::AbstractVector) =
Filt(eltype(b),(dim_in,), b, a)

Filt(dim_in::Tuple,  b::AbstractVector, a::AbstractVector) =
Filt(eltype(b), dim_in, b, a)

Filt(dim_in::Int,  b::AbstractVector) =
Filt(eltype(b),(dim_in,), b)

Filt(dim_in::Tuple,  b::AbstractVector) =
Filt(eltype(b), dim_in, b)

Filt(x::AbstractArray, b::AbstractVector, a::AbstractVector) =
Filt(eltype(x),size(x), b, a)

Filt(x::AbstractArray, b::AbstractVector) =
Filt(size(x), b)

# Mappings

function mul!(y::AbstractArray{T},L::Filt{T,N},x::AbstractArray{T}) where {T,N}
	for col = 1:size(x,2)
		length(L.a) != 1 ? iir!(y,L.b,L.a,x,L.si,col,col) : fir!(y,L.b,x,L.si,col,col)
	end
end

function mul!(y::AbstractArray{T},L::AdjointOperator{Filt{T,N}},x::AbstractArray{T}) where {T,N}
    A = L.A
	for col = 1:size(x,2)
		length(A.a) != 1 ? iir_rev!(y,A.b,A.a,x,A.si,col,col) : fir_rev!(y,A.b,x,A.si,col,col)
	end
end

function iir!(y, b, a, x, si, coly, colx)
    silen = length(si)
    @inbounds for i=1:size(x, 1)
        xi = x[i,colx]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi - a[j+1]*val
        end
        si[silen] = b[silen+1]*xi - a[silen+1]*val
        y[i,coly] = val
    end
    si .= 0. #reset state
end

# Utilities

function iir_rev!(y, b, a, x, si, coly, colx)
    silen = length(si)
    @inbounds for i=size(x, 1):-1:1
        xi = x[i,colx]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi - a[j+1]*val
        end
        si[silen] = b[silen+1]*xi - a[silen+1]*val
        y[i,coly] = val
    end
    si .= 0.
end

function fir!(y, b, x, si, coly, colx)
    silen = length(si)
    @inbounds for i=1:size(x, 1)
        xi = x[i,colx]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi
        end
        si[silen] = b[silen+1]*xi
        y[i,coly] = val
    end
    si .= 0.
end

function fir_rev!(y, b, x, si, coly, colx)
    silen = length(si)
    @inbounds for i=size(x, 1):-1:1
        xi = x[i,colx]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi
        end
        si[silen] = b[silen+1]*xi
        y[i,coly] = val
    end
    si .= 0.
end

# Properties

domainType(L::Filt{T, N}) where {T, N} = T
codomainType(L::Filt{T, N}) where {T, N} = T

size(L::Filt) = L.dim_in, L.dim_in

fun_name(L::Filt)  = size(L.a,1) != 1 ? "IIR" : "FIR"

#TODO find out a way to verify this, 
# probably for IIR it means zeros inside unit circle
is_full_row_rank(L::Filt)    = true
is_full_column_rank(L::Filt) = true
