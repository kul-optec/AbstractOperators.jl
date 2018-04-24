#NonLinearCompose

export NonLinearCompose

"""
`NonLinearCompose(A::AbstractOperator,B::AbstractOperator)`

Compose opeators in such fashion:

`A(⋅)*B(⋅)`

# Example: Matrix multiplication

```julia
julia> n1,m1,n2,m2 = 3,4,4,6 

julia> x = (randn(n1,m1),randn(n2,m2)); #inputs

julia> C = NonLinearCompose( Eye(n1,n2), Eye(m1,m2) )
# i.e. `I(⋅)*I(⋅)`

julia> Y = x[1]*x[2]

julia> C*x ≈ Y
true

```
"""

struct NonLinearCompose{N,
			L1 <: HCAT{1},
			L2 <: HCAT{1},
			C <: Tuple{AbstractArray,AbstractArray},
			D <: NTuple{N,Union{AbstractArray,Tuple}}
			} <: NonLinearOperator
	A::L1
	B::L2
	buf::C
	bufx::D
	function NonLinearCompose(A::L1, B::L2, buf::C, bufx::D) where {L1,L2,C,D}
		if ( (ndoms(A,1) > 1 || ndoms(B,1) > 1) || 
             (ndims(A,1) > 2 || ndims(B,1) > 2) ||
             (size(B,1)[1] == 1 ? (length(size(A,1)) == 1 ? false : true) : # outer product case
              ndims(A,1) == 1 ? true : (size(A,1)[2] != size(B,1)[1]))
            ) 
                throw(DimensionMismatch("cannot compose operators"))
        end
		N = length(bufx)
		new{N,L1,L2,C,D}(A,B,buf,bufx)
	end
end

struct NonLinearComposeJac{N,
			   L1 <: HCAT{1},
			   L2 <: HCAT{1},
			   C <: Tuple{AbstractArray,AbstractArray},
			   D <: NTuple{N,Union{AbstractArray,Tuple}}
			   } <: LinearOperator
	A::L1
	B::L2
	buf::C
	bufx::D
	function NonLinearComposeJac{N}(A::L1, B::L2, buf::C, bufx::D) where {N,L1,L2,C,D}
		new{N,L1,L2,C,D}(A,B,buf,bufx)
	end
end

# Constructors
function NonLinearCompose(L1::AbstractOperator,L2::AbstractOperator)

	A = HCAT(L1, Zeros( domainType(L2), size(L2,2), codomainType(L1), size(L1,1) ))
	B = HCAT(Zeros( domainType(L1), size(L1,2), codomainType(L2), size(L2,1) ), L2 )

	buf  = zeros(codomainType(A),size(A,1)),zeros(codomainType(B),size(B,1))
	bufx = zeros(codomainType(L1),size(L1,1)), zeros(codomainType(L2),size(L2,1))

	NonLinearCompose(A,B,buf,bufx)
end

# Jacobian
function Jacobian(P::NonLinearCompose{N,L,C,D},x::DD) where  {M,N,L,C,
							      D<: NTuple{N,Union{AbstractArray,Tuple}},
							      DD<: NTuple{M,AbstractArray},
							      }
	NonLinearComposeJac{N}(Jacobian(P.A,x),Jacobian(P.B,x),P.buf,P.bufx)
end

# Mappings
function A_mul_B!(y, P::NonLinearCompose{N,L,C,D}, b) where {N,L,C,D}
	A_mul_B_skipZeros!(P.buf[1],P.A,b)
	A_mul_B_skipZeros!(P.buf[2],P.B,b)
	A_mul_B!(y,P.buf[1],P.buf[2])
end

function Ac_mul_B!(y, P::NonLinearComposeJac{N,L,C,D}, b) where {N,L,C,D}

	A_mul_Bc!(P.bufx[1],b,P.buf[2])
	Ac_mul_B_skipZeros!(y,P.A,P.bufx[1])

	Ac_mul_B!(P.bufx[2],P.buf[1],b)
	Ac_mul_B_skipZeros!(y,P.B,P.bufx[2])

end

# special case outer product  
function Ac_mul_B!(y, P::NonLinearComposeJac{N,L,C,D}, b) where {N,L,C,D <: Tuple{AbstractVector,AbstractArray}}

    p = reshape(P.bufx[1], length(P.bufx[1]),1)
    A_mul_Bc!(p,b,P.buf[2])
	Ac_mul_B_skipZeros!(y,P.A,P.bufx[1])

	Ac_mul_B!(P.bufx[2],P.buf[1],b)
	Ac_mul_B_skipZeros!(y,P.B,P.bufx[2])

end

# Properties
function size(P::NonLinearCompose) 
	size_out = ndims(P.B,1) == 1 ? (size(P.A,1)[1],) :
	(size(P.A,1)[1], size(P.B,1)[2])
	size_out, size(P.A,2)
end

function size(P::NonLinearComposeJac) 
	size_out = ndims(P.B,1) == 1 ? (size(P.A,1)[1],) :
	(size(P.A,1)[1], size(P.B,1)[2])
	size_out, size(P.A,2)
end

fun_name(L::NonLinearCompose) = fun_name(L.A.A[1])"*"*fun_name(L.B.A[2]) 
fun_name(L::NonLinearComposeJac) = fun_name(L.A.A[1])"*"*fun_name(L.B.A[2]) 

domainType(L::NonLinearCompose)   = domainType.(L.A)
codomainType(L::NonLinearCompose) = codomainType(L.A)

domainType(L::NonLinearComposeJac)   = domainType.(L.A)
codomainType(L::NonLinearComposeJac) = codomainType(L.A)

# utils
import Base: permute

function permute(P::NonLinearCompose{N,L,C,D}, p::AbstractVector{Int}) where {N,L,C,D}
	NonLinearCompose(permute(P.A,p),permute(P.B,p),P.buf,P.bufx)
end

remove_displacement(N::NonLinearCompose) = NonLinearCompose(remove_displacement(N.A), remove_displacement(N.B), N.buf, N.bufx)
