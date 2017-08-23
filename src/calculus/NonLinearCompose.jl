#NonLinearCompose

export NonLinearCompose

immutable NonLinearCompose{N,
			   L1 <: HCAT{1},
			   L2 <: HCAT{1},
			   C <: Tuple{AbstractArray,AbstractArray},
			   D <: NTuple{N,Union{AbstractArray,Tuple}}
			   } <: NonLinearOperator
	A::L1
	B::L2
	mid::C
	midx::D
	function NonLinearCompose(A::L1, B::L2, mid::C, midx::D) where {L1,L2,C,D}
		(ndoms(A,1) > 1 || ndoms(B,1) > 1) ||
		(ndims(A,1) > 2 || ndims(B,1) > 2) ||
		(size(A,1)[2] != size(B,1)[1]) && 
		throw(DimensionMismatch("cannot compose operators"))
		N = length(midx)
		new{N,L1,L2,C,D}(A,B,mid,midx)
	end
end

immutable NonLinearComposeJac{N,
			      L1 <: HCAT{1},
			      L2 <: HCAT{1},
			      C <: Tuple{AbstractArray,AbstractArray},
			      D <: NTuple{N,Union{AbstractArray,Tuple}}
			      } <: LinearOperator
	A::L1
	B::L2
	mid::C
	midx::D
	function NonLinearComposeJac{N}(A::L1, B::L2, mid::C, midx::D) where {N,L1,L2,C,D}
		new{N,L1,L2,C,D}(A,B,mid,midx)
	end
end

# Constructors
function NonLinearCompose(L1::AbstractOperator,L2::AbstractOperator)

	A = HCAT(L1, Zeros( domainType(L2), size(L2,2), codomainType(L1), size(L1,1) ))
	B = HCAT(Zeros( domainType(L1), size(L1,2), codomainType(L2), size(L2,1) ), L2 )

	mid  = zeros(codomainType(A),size(A,1)),zeros(codomainType(B),size(B,1))
	midx = zeros(codomainType(L1),size(L1,1)), zeros(codomainType(L2),size(L2,1))

	NonLinearCompose(A,B,mid,midx)
end

# Jacobian
function Jacobian{M,N,L,C,
		  D  <: NTuple{N,Union{AbstractArray,Tuple}},
		  DD <: NTuple{M,AbstractArray},
		  }(P::NonLinearCompose{N,L,C,D},x::DD)  
	NonLinearComposeJac{N}(Jacobian(P.A,x),Jacobian(P.B,x),P.mid,P.midx)
end

# Mappings
function A_mul_B!{N,L,C,D}(y, P::NonLinearCompose{N,L,C,D}, b)
	A_mul_B_skipZeros!(P.mid[1],P.A,b)
	A_mul_B_skipZeros!(P.mid[2],P.B,b)
	A_mul_B!(y,P.mid[1],P.mid[2])
end

function Ac_mul_B!{N,L,C,D}(y, P::NonLinearComposeJac{N,L,C,D}, b)

	A_mul_Bc!(P.midx[1],b,P.mid[2])
	Ac_mul_B_skipZeros!(y,P.A,P.midx[1])

	Ac_mul_B!(P.midx[2],P.mid[1],b)
	Ac_mul_B_skipZeros!(y,P.B,P.midx[2])

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

function permute{N,L,C,D}(P::NonLinearCompose{N,L,C,D}, p::AbstractVector{Int})
	NonLinearCompose(permute(P.A,p),permute(P.B,p),P.mid,P.midx)
end



