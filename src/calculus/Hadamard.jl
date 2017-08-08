
#Hadamard
export Hadamard

immutable Hadamard{N,
		  L  <:NTuple{N,AbstractOperator},
		  L2 <:NTuple{N,LinearOperator},
		  T1 <:NTuple{N,AbstractArray},
		  T2 <:NTuple{N,AbstractArray}
		  } <: NonLinearOperator
	A::L
	J::L2
	mid::T1
	mid2::T2
end

# Constructors
function Hadamard{N}(L1::AbstractOperator,L2::AbstractOperator,x::NTuple{N,AbstractArray})
	A = (L1,L2)
	J = Jacobian.(A,x)
	mid  = zeros.(codomainType.(A), size.(A,1))
	mid2 = zeros.(  domainType.(A), size.(A,1))
	Hadamard{2,typeof(A),typeof(J),typeof(mid),typeof(mid2)}(A,J,mid,mid2)
end

# Mappings
function A_mul_B!{N,L,T}(y::AbstractArray,P::Hadamard{N,L,T},b::NTuple{N,AbstractArray})
	A_mul_B!(P.mid[1],P.A[1],b[1])
	A_mul_B!(P.mid[2],P.A[2],b[2])
	y .= P.mid[1].*P.mid[2]
end

# Jacobian
function Ac_mul_B!{N,
		   L  <: NTuple{N,AbstractOperator},
		   L2 <: NTuple{N,LinearOperator},
		   T1 <: NTuple{N,AbstractArray},
		   T2 <: NTuple{N,AbstractArray},
		   A  <: Hadamard{N,L,L2,T1,T2}
		   }(y, J::Jacobian{A}, b)

        J.A.mid2[1] .= J.A.mid[2].*b 
	Ac_mul_B!(y[1], J.A.J[1], J.A.mid2[1])

        J.A.mid2[2] .= J.A.mid[1].*b 
	Ac_mul_B!(y[2], J.A.J[2], J.A.mid2[2])

end


# Properties

size(P::Hadamard) = size(P.A[1],1), size.(P.A,2)

fun_name(L::Hadamard) = fun_name(L.A[1])"⊙"*fun_name(L.A[2]) 

domainType(L::Hadamard)   = domainType.(L.A)
codomainType(L::Hadamard) = codomainType(L.A[end])
