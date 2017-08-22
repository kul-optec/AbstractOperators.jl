#NonLinearCompose

export NonLinearCompose

immutable NonLinearCompose{N,
			   L <: Tuple{HCAT{1},HCAT{1}},
			   C <: Tuple{AbstractArray,AbstractArray},
			   D <: NTuple{N,Union{AbstractArray,Tuple}}
			   } <: NonLinearOperator
	A::L
	mid::C
	midx::D
	function NonLinearCompose(A::L, mid::C, midx::D) where {L,C,D}
		(ndoms(A[1],1) > 1 || ndoms(A[2],1) > 1) ||
		(ndims(A[1],1) > 2 || ndims(A[2],1) > 2) ||
		(size(A[1],1)[2] != size(A[2],1)[1]) && 
		throw(DimensionMismatch("cannot compose operators"))
		N = length(midx)
		new{N,L,C,D}(A,mid,midx)
	end
end

immutable NonLinearComposeJac{N,
			      L <: Tuple{HCAT{1},HCAT{1}},
			      C <: Tuple{AbstractArray,AbstractArray},
			      D <: NTuple{N,Union{AbstractArray,Tuple}}
			      } <: LinearOperator
	A::L
	mid::C
	midx::D
end

# Constructors
function NonLinearCompose(L1::AbstractOperator,L2::AbstractOperator)

	A = HCAT(L1, Zeros( domainType(L2), size(L2,2), codomainType(L1), size(L1,1) ))
	B = HCAT(Zeros( domainType(L1), size(L1,2), codomainType(L2), size(L2,1) ), L2 )

	mid  = zeros(codomainType(A),size(A,1)),zeros(codomainType(B),size(B,1))
	midx = zeros(codomainType(L1),size(L1,1)), zeros(codomainType(L2),size(L2,1))

	NonLinearCompose((A,B),mid,midx)
end

# Jacobian
function Jacobian{M,N,L,C,
		  D  <: NTuple{N,Union{AbstractArray,Tuple}},
		  DD <: NTuple{M,AbstractArray},
		  }(P::NonLinearCompose{N,L,C,D},x::DD)  
	J = ([Jacobian(a,x) for a in P.A]...)
	NonLinearComposeJac{N,typeof(J),C,D}(J,P.mid,P.midx)
end

# Mappings
function A_mul_B!{N,L,C,D}(y, P::NonLinearCompose{N,L,C,D}, b)
	A_mul_B_skipZeros!(P.mid[1],P.A[1],b)
	A_mul_B_skipZeros!(P.mid[2],P.A[2],b)
	A_mul_B!(y,P.mid[1],P.mid[2])
end

function Ac_mul_B!{N,L,C,D}(y, P::NonLinearComposeJac{N,L,C,D}, b)

	A_mul_Bc!(P.midx[1],b,P.mid[2])
	Ac_mul_B_skipZeros!(y,P.A[1],P.midx[1])

	Ac_mul_B!(P.midx[2],P.mid[1],b)
	Ac_mul_B_skipZeros!(y,P.A[2],P.midx[2])

end

# Properties
function size(P::NonLinearCompose) 
	size_out = ndims(P.A[end],1) == 1 ? (size(P.A[1],1)[1],) :
	(size(P.A[1],1)[1], size(P.A[end],1)[2])
	size_out, size(P.A[1],2)
end

function size(P::NonLinearComposeJac) 
	size_out = ndims(P.A[end],1) == 1 ? (size(P.A[1],1)[1],) :
	(size(P.A[1],1)[1], size(P.A[end],1)[2])
	size_out, size(P.A[1],2)
end

fun_name(L::NonLinearCompose) = fun_name(L.A[1].A[1])"*"*fun_name(L.A[2].A[2]) 
fun_name(L::NonLinearComposeJac) = fun_name(L.A[1].A[1])"*"*fun_name(L.A[2].A[2]) 

domainType(L::NonLinearCompose)   = domainType.(L.A[1])
codomainType(L::NonLinearCompose) = codomainType(L.A[1])

domainType(L::NonLinearComposeJac)   = domainType.(L.A[1])
codomainType(L::NonLinearComposeJac) = codomainType(L.A[1])

