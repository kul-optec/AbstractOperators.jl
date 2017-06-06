export MIMOFilt

immutable MIMOFilt{T, A<:AbstractVector{T}} <: LinearOperator
	dim_out::Tuple{Int,Int}
	dim_in::Tuple{Int,Int}
	B::Vector{A}
	A::Vector{A}
	SI::Vector{A}
end

# Constructors

#default constructor
function MIMOFilt{N}(domainType::Type, dim_in::NTuple{N,Int}, b::Vector, a::Vector)

	N != 2 && error("length(dim_in) must be equal to 2")
	eltype(b) != eltype(a) && error("eltype(b) must be equal to eltype(a)")
	typeof(b[1][1]) != domainType && error("filter coefficient of b must be $domainType")
	typeof(a[1][1]) != domainType && error("filter coefficient of a must be $domainType")

	length(b) != length(a) && error("filter vectors b must be as many as a")

	mod(length(b),dim_in[2]) !=0 && error("wrong number of filters")
	dim_out = (dim_in[1], div(length(b),dim_in[2]) )

	B,A,SI = Array{typeof(b[1]),1}(length(b)),Array{typeof(b[1]),1}(length(b)),Array{typeof(b[1]),1}(length(b))

	for i = 1:length(b)
		a[i][1] == 0  && error("filter vector a[$i][1] must be nonzero")

		B[i]  = b[i]
		A[i]  = a[i]

		as = length(A[i])
		bs = length(B[i])
		sz = max(as, bs)
		silen = sz - 1

		# Filter coefficient normalization
		if A[i][1] != 1
			norml = A[i][1]
			A[i] ./= norml
			B[i] ./= norml
		end

		# Pad the coefficients with zeros if needed
		bs<sz   && (B[i] = copy!(zeros(domainType, sz), B[i]))
		1<as<sz && (A[i] = copy!(zeros(domainType, sz), A[i]))

		SI[i] = zeros(domainType, max(length(a[i]), length(b[i]))-1)

	end
	MIMOFilt{domainType, typeof(B[1])}(dim_out, dim_in, B, A, SI)
end

MIMOFilt{D1<:AbstractVector}(dim_in::Tuple,  b::Vector{D1}, a::Vector{D1}) =
MIMOFilt(eltype(b[1]), dim_in, b, a)

MIMOFilt{D1<:AbstractVector}(dim_in::Tuple,  b::Vector{D1}) =
MIMOFilt(eltype(b[1]), dim_in, b, [[1.0] for i in eachindex(b)])

MIMOFilt{D1<:AbstractVector}(x::AbstractMatrix,  b::Vector{D1}, a::Vector{D1}) =
MIMOFilt(eltype(x), size(x), b, a)

MIMOFilt{D1<:AbstractVector}(x::AbstractMatrix,  b::Vector{D1}) =
MIMOFilt(eltype(x), size(x), b, [[1.0] for i in eachindex(b)])

# Mappings

function A_mul_B!{T,A}(y::AbstractArray{T},L::MIMOFilt{T,A},x::AbstractArray{T})
	cnt = 0
	cx  = 0
	y .= 0. #TODO avoid this?
	for cy = 1:L.dim_out[2]
		cnt += 1
		cx  += 1
		length(L.A[cnt]) != 1 ? add_iir!(y,L.B[cnt],L.A[cnt],x,L.SI[cnt],cy,cx) :
		add_fir!(y,L.B[cnt],x,L.SI[cnt],cy,cx)

		for c2 = 2:L.dim_in[2]
			cnt += 1
			cx  += 1
			length(L.A[cnt]) != 1 ? add_iir!(y,L.B[cnt],L.A[cnt],x,L.SI[cnt],cy,cx) :
			add_fir!(y,L.B[cnt],x,L.SI[cnt],cy,cx)
		end
		cx = 0
	end
end

function Ac_mul_B!{T,A}(y::AbstractArray{T},L::MIMOFilt{T,A},x::AbstractArray{T})
	cnt = 0
	cx  = 0
	y .= 0. #TODO avoid this?
	for cy = 1:L.dim_out[2]
		cnt += 1
		cx  += 1
		length(L.A[cnt]) != 1 ? add_iir_rev!(y,L.B[cnt],L.A[cnt],x,L.SI[cnt],cx,cy) :
		add_fir_rev!(y,L.B[cnt],x,L.SI[cnt],cx,cy)

		for c2 = 2:L.dim_in[2]
			cnt += 1
			cx  += 1
			length(L.A[cnt]) != 1 ? add_iir_rev!(y,L.B[cnt],L.A[cnt],x,L.SI[cnt],cx,cy) :
			add_fir_rev!(y,L.B[cnt],x,L.SI[cnt],cx,cy)
		end
		cx = 0
	end
end

# Properties

domainType{T, M}(L::MIMOFilt{T, M}) = T
codomainType{T, M}(L::MIMOFilt{T, M}) = T

size(L::MIMOFilt) = L.dim_out, L.dim_in

fun_name(L::MIMOFilt)  = "※"

# Utilities

function add_iir!(y, b, a, x, si, coly, colx)
    silen = length(si)
    @inbounds for i=1:size(x, 1)
        xi = x[i,colx]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi - a[j+1]*val
        end
        si[silen] = b[silen+1]*xi - a[silen+1]*val
        y[i,coly] += val
    end
    si .= 0. #reset state
end

function add_iir_rev!(y, b, a, x, si, coly, colx)
    silen = length(si)
    @inbounds for i=size(x, 1):-1:1
        xi = x[i,colx]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi - a[j+1]*val
        end
        si[silen] = b[silen+1]*xi - a[silen+1]*val
        y[i,coly] += val
    end
    si .= 0.
end

function add_fir!(y, b, x, si, coly, colx)
    silen = length(si)
    @inbounds for i=1:size(x, 1)
        xi = x[i,colx]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi
        end
        si[silen] = b[silen+1]*xi
        y[i,coly] += val
    end
    si .= 0.
end

function add_fir_rev!(y, b, x, si, coly, colx)
    silen = length(si)
    @inbounds for i=size(x, 1):-1:1
        xi = x[i,colx]
        val = si[1] + b[1]*xi
        for j=1:(silen-1)
            si[j] = si[j+1] + b[j+1]*xi
        end
        si[silen] = b[silen+1]*xi
        y[i,coly] += val
    end
    si .= 0.
end
