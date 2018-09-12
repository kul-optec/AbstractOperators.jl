export MIMOFilt

"""
`MIMOFilt([domainType=Float64::Type,] dim_in::Tuple, B::Vector{AbstractVector}, [A::Vector{AbstractVector},])`

`MIMOFilt(x::AbstractMatrix, b::Vector{AbstractVector}, [a::Vector{AbstractVector},])`

Creates a `LinearOperator` which, when multiplied with a matrix `X`, returns a matrix `Y`. Here a Multiple Input Multiple Output system is evaluated: the columns of `X` and `Y` represent the input signals and output signals respectively. 

```math
\\mathbf{y}_i = \\sum_{j = 1}^{M} \\mathbf{h}_{i,j} * \\mathbf{x}_j 
```
where ``\\mathbf{y}_i`` and ``\\mathbf{x}_j`` are the ``i``-th and ``j``-th columns of the output `Y` and input `X` matrices respectively.

The filters ``\\mathbf{h}_{i,j}`` can be represented either by providing coefficients `B` and `A` (IIR) or `B` alone (FIR). These coefficients must be given in a `Vector` of `Vector`s. 

For example for a `3` by `2` MIMO system (i.e. `size(X,2) == 3` inputs and `size(Y,2) == 2` outputs) `B` must be:

`B = [b11, b12, b13, b21, b22, b23]`

where `bij` are vector containing the filter coeffients of `h_{i,j}`.

```julia
julia> m,n = 10,3; #time samples, number of inputs

julia> B  = [[1.;0.;1.],[1.;0.;1.],[1.;0.;1.],[1.;0.;1.],[1.;0.;1.],[1.;0.;1.], ];
      #B = [   b11   ,     b12   ,    b13   ,   b21    ,   b22,       b23    , ]

julia> A  = [[1.;1.;1.],[2.;2.;2.],[      3.],[      4.],[      5.],[      6.], ];
      #A = [   a11   ,     a12   ,    a13   ,   a21    ,   a22,       a23    , ]

julia> op = MIMOFilt(Float64, (m,n), B, A)
※  ℝ^(10, 3) -> ℝ^(10, 2) 

julia> X = randn(m,n); #input signals

julia> Y = op*X;       #output signals

julia> Y[:,1] ≈ filt(B[1],A[1],X[:,1])+filt(B[2],A[2],X[:,2])+filt(B[3],A[3],X[:,3])
true

julia> Y[:,2] ≈ filt(B[4],A[4],X[:,1])+filt(B[5],A[5],X[:,2])+filt(B[6],A[6],X[:,3])
true

```

"""
struct MIMOFilt{T, A<:AbstractVector{T}} <: LinearOperator
	dim_out::Tuple{Int,Int}
	dim_in::Tuple{Int,Int}
	B::Vector{A}
	A::Vector{A}
	SI::Vector{A}
end

# Constructors

#default constructor
function MIMOFilt(domainType::Type, dim_in::NTuple{N,Int}, b::Vector, a::Vector) where {N}

	N != 2 && error("length(dim_in) must be equal to 2")
	eltype(b) != eltype(a) && error("eltype(b) must be equal to eltype(a)")
	typeof(b[1][1]) != domainType && error("filter coefficient of b must be $domainType")
	typeof(a[1][1]) != domainType && error("filter coefficient of a must be $domainType")

	length(b) != length(a) && error("filter vectors b must be as many as a")

	mod(length(b),dim_in[2]) !=0 && error("wrong number of filters")
	dim_out = (dim_in[1], div(length(b),dim_in[2]) )

    B,A,SI = similar(b),similar(b),similar(b)

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
		bs<sz   && (B[i] = copyto!(zeros(domainType, sz), B[i]))
		1<as<sz && (A[i] = copyto!(zeros(domainType, sz), A[i]))

		SI[i] = zeros(domainType, max(length(a[i]), length(b[i]))-1)

	end
	MIMOFilt{domainType, typeof(B[1])}(dim_out, dim_in, B, A, SI)
end

MIMOFilt(dim_in::Tuple,  b::Vector{D1}, a::Vector{D1}) where {D1<:AbstractVector} =
MIMOFilt(eltype(b[1]), dim_in, b, a)

MIMOFilt(dim_in::Tuple,  b::Vector{D1}) where {D1<:AbstractVector} =
MIMOFilt(eltype(b[1]), dim_in, b, [[1.0] for i in eachindex(b)])

MIMOFilt(x::AbstractMatrix,  b::Vector{D1}, a::Vector{D1}) where {D1<:AbstractVector} =
MIMOFilt(eltype(x), size(x), b, a)

MIMOFilt(x::AbstractMatrix,  b::Vector{D1}) where {D1<:AbstractVector} =
MIMOFilt(eltype(x), size(x), b, [[1.0] for i in eachindex(b)])

# Mappings

function mul!(y::AbstractArray{T},L::MIMOFilt{T,A},x::AbstractArray{T}) where {T,A}
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

function mul!(y::AbstractArray{T},M::AdjointOperator{MIMOFilt{T,A}},x::AbstractArray{T}) where {T,A}
    L = M.A
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

domainType(L::MIMOFilt{T, M}) where {T, M} = T
codomainType(L::MIMOFilt{T, M}) where {T, M} = T

size(L::MIMOFilt) = L.dim_out, L.dim_in

#TODO find out a way to verify this, 
# probably for IIR it means zeros inside unit circle
is_full_row_rank(L::MIMOFilt)    = true
is_full_column_rank(L::MIMOFilt) = true

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
