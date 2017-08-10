
#Hadamard
export Hadamard

immutable Hadamard{M,N,
		   C <: NTuple{M,AbstractArray},
		   D <: NTuple{N,AbstractArray},
		   L <: NTuple{M,AbstractOperator},
		   V <:VCAT{M,N,C,D,L},
		   } <: NonLinearOperator
	A::V
	mid::C
	mid2::C
	function Hadamard(A::V,mid::C,mid2::C) where {M,N,C,D,L,V<:VCAT{M,N,C,D,L}}
		any([ai != size(A,1)[1] for ai in size(A,1)]) && 
		throw(DimensionMismatch("cannot compose operators"))
		any(any(    
			sum([!is_null(A[m][n]) for n = 1:N, m = 1:M],2) .> 1
			)) &&
		throw(DimensionMismatch("cannot compose operators"))

		new{M,N,C,D,L,V}(A,mid,mid2)
	end
end

immutable HadamardJacobian{M,N,
			   C <: NTuple{M,AbstractArray},
			   D <: Union{NTuple{N,AbstractArray}, AbstractArray},
			   L <: NTuple{M,AbstractOperator},
			   V <:VCAT{M,N,C,D,L},
			   } <: LinearOperator
	A::V
	mid::C
	mid2::C
	function HadamardJacobian(A::V,mid::C,mid2::C) where {M,N,C,D,L,V<:VCAT{M,N,C,D,L}}
		new{M,N,C,D,L,V}(A,mid,mid2)
	end
end

# Constructors
function Hadamard{N}(L::Vararg{HCAT{1,N}})
	A = VCAT(L...)
	mid  = zeros.(codomainType(A), size(A,1))
	mid2 = zeros.(codomainType(A), size(A,1))
	Hadamard(A,mid,mid2)
end

function Hadamard(L::Vararg{AbstractOperator})

	Z  = Zeros.(domainType.(L),size.(L,2),codomainType.(L),size.(L,1))
	Op = [(Z[1:i-1]...,L[i], Z[i+1:end]...) for i in eachindex(L)]
	hcats = [HCAT(op...) for op in Op ]
	Hadamard(hcats...)
	
end

# Mappings
function A_mul_B!{M,N,C,D,L,V}(y, P::Hadamard{M,N,C,D,L,V}, b::D)
	A_mul_B!(P.mid,P.A,b)
		
	y .= P.mid[1].*P.mid[2]
	for i = 3:M
		y .*= P.mid[i]
	end
end

# Jacobian
Jacobian{M,N,C,D<: NTuple{N,AbstractArray},L,V,H<:Hadamard{M,N,C,D,L,V}}(P::H,x::D) = 
HadamardJacobian(Jacobian(P.A,x),P.mid,P.mid2)

function Ac_mul_B!{M,N,C,D,L,V}(y::D, J::HadamardJacobian{M,N,C,D,L,V}, b)
	for i = 1:M
		c = (J.mid[1:i-1]...,J.mid[i+1:end]...,b)
		J.mid2[i] .= (.*)(c...)
	end
	Ac_mul_B!(y, J.A, J.mid2)

end

# Properties

size(P::Hadamard) = size(P.A[1],1), size(P.A[1],2)
size(P::HadamardJacobian) = size(P.A[1],1), size(P.A[1],2)

fun_name{M,N}(L::Hadamard{M,N}) = N == 2 ? fun_name(L.A[1])"⊙"*fun_name(L.A[2]) : "⊙"
fun_name{M,N}(L::HadamardJacobian{M,N}) = N == 2 ? fun_name(L.A[1])"⊙"*fun_name(L.A[2]) :"⊙"

domainType(L::Hadamard)   = domainType.(L.A[1])
codomainType(L::Hadamard) = codomainType(L.A[1])

domainType(L::HadamardJacobian)   = domainType.(L.A[1])
codomainType(L::HadamardJacobian) = codomainType(L.A[1])
