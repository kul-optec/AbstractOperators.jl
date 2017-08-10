#NonLinearCompose

export NonLinearCompose

immutable NonLinearCompose{M,N} <: NonLinearOperator
	A::NTuple{M,HCAT{1,N}}
	mid::NTuple{M,AbstractArray}
	midx::NTuple{N,AbstractArray}
	midx2::NTuple{N,AbstractArray}
end

immutable NonLinearComposeJac{M,N} <: LinearOperator
	A::NTuple{M,HCAT{1,N}}
	mid::NTuple{M,AbstractArray}
	midx::NTuple{N,AbstractArray}
	midx2::NTuple{N,AbstractArray}
end

# Constructors
function NonLinearCompose(L1::AbstractOperator,L2::AbstractOperator)
	A = HCAT(L1, Zeros( domainType(L2), size(L2,2), codomainType(L1), size(L1,1) ))
	B = HCAT(Zeros( domainType(L1), size(L1,2), codomainType(L2), size(L2,1) ), L2 )
	mid  = zeros(codomainType(A),size(A,1)),zeros(codomainType(B),size(B,1))
	midx = zeros(codomainType(L1),size(L1,1)),zeros(codomainType(L2),size(L2,1))
	midx2 = zeros(domainType(L1),size(L1,2)),zeros(domainType(L2),size(L2,2))
	NonLinearCompose{2,2}((A,B),mid,midx,midx2)
end

# Jacobian
function Jacobian{M,N,L<:NonLinearCompose{M,N}}(P::L,x::NTuple{N,AbstractArray})  
	NonLinearComposeJac{M,N}(([Jacobian(a,x) for a in P.A]...),P.mid,P.midx,P.midx2)
end

# Mappings
function A_mul_B!{M,N}(y, P::NonLinearCompose{M,N}, b)
	A_mul_B!(P.mid[1],P.A[1],b)
	A_mul_B!(P.mid[2],P.A[2],b)
	A_mul_B!(y,P.mid[1],P.mid[2])
end

function Ac_mul_B!{M,N}(y, P::NonLinearComposeJac{M,N}, b)

	A_mul_Bc!(P.midx[1],b,P.mid[2])
	Ac_mul_B!(y,P.A[1],P.midx[1])

	Ac_mul_B!(P.midx[2],P.mid[1],b)
	Ac_mul_B!(P.midx2,P.A[2],P.midx[2])

	y[1] .+= P.midx2[1]
	y[2] .+= P.midx2[2]

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

fun_name{M,N}(L::NonLinearCompose{M,N}) = N == 2 ? fun_name(L.A[1])"*"*fun_name(L.A[2]) : "*"
fun_name{M,N}(L::NonLinearComposeJac{M,N}) = N == 2 ? fun_name(L.A[1])"*"*fun_name(L.A[2]) : "*"

domainType(L::NonLinearCompose)   = domainType.(L.A[1])
codomainType(L::NonLinearCompose) = codomainType(L.A[1])

domainType(L::NonLinearComposeJac)   = domainType.(L.A[1])
codomainType(L::NonLinearComposeJac) = codomainType(L.A[1])

