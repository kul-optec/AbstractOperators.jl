#Hadamard
export Hadamard

"""
`Hadamard(A::AbstractOperator,B::AbstractOperator)`

Compose opeators such that their output is multiplied elementwise:

`A(⋅).*B(⋅)`

# Example

```julia
julia> n,m = 5,10

julia> x = (randn(n),randn(m)); #inputs

julia> A = randn(m,n); #A matrix

julia> C = Hadamard( MatrixOp(A), Eye(m) )
# i.e. `A(⋅).*I(⋅)`

julia> Y = (A*x[1]).*x[2]

julia> C*x ≈ Y
true

```
"""

struct Hadamard{M, C, V <: VCAT{M}} <: NonLinearOperator
	A::V
	mid::C
	mid2::C
	function Hadamard(A::V, mid::C, mid2::C) where {M, C, V <: VCAT{M}}
		any([ai != size(A,1)[1] for ai in size(A,1)]) &&
		throw(DimensionMismatch("cannot compose operators"))

		new{M, C, V}(A,mid,mid2)
	end
end

struct HadamardJacobian{M, C, V <: VCAT{M}} <: LinearOperator
	A::V
	mid::C
	mid2::C
	function HadamardJacobian(A::V,mid::C,mid2::C) where {M, C, V <: VCAT{M}}
		new{M, C, V}(A,mid,mid2)
	end
end

# Constructors
function Hadamard(L1::AbstractOperator,L2::AbstractOperator)

	A = HCAT(L1, Zeros( domainType(L2), size(L2,2), codomainType(L1), size(L1,1) ))
	B = HCAT(Zeros( domainType(L1), size(L1,2), codomainType(L2), size(L2,1) ), L2 )

    V = VCAT(A,B)

	mid  = zeros.(codomainType(V), size(V,1))
	mid2 = zeros.(codomainType(V), size(V,1))

	Hadamard(V,mid,mid2)
end

# Mappings
function A_mul_B!(y, H::Hadamard{M,C,V}, b) where {M,C,V}
	A_mul_B!(H.mid,H.A,b)

	y .= H.mid[1]
    for i = 2:length(H.mid)
		y .*= H.mid[i]
	end
end

# Jacobian
Jacobian(A::H, x::D) where {M, D<:Tuple, C, V, H <: Hadamard{M,C,V}} =
HadamardJacobian(Jacobian(A.A,x),A.mid,A.mid2)

function Ac_mul_B!(y, J::HadamardJacobian{M,C,V}, b) where {M,C,V}
    for i = 1:length(J.mid)
		c = (J.mid[1:i-1]...,J.mid[i+1:end]...,b)
		J.mid2[i] .= (.*)(c...)
	end
	Ac_mul_B!(y, J.A, J.mid2)

end

# Properties
size(P::Hadamard) = size(P.A[1],1), size(P.A[1],2)
size(P::HadamardJacobian) = size(P.A[1],1), size(P.A[1],2)

fun_name(L::Hadamard) = "⊙"
fun_name(L::HadamardJacobian) = "J(⊙)"

domainType(L::Hadamard)   = domainType.(L.A[1])
codomainType(L::Hadamard) = codomainType(L.A[1])

domainType(L::HadamardJacobian)   = domainType.(L.A[1])
codomainType(L::HadamardJacobian) = codomainType(L.A[1])

# utils
import Base: permute

function permute(H::Hadamard, p::AbstractVector{Int})
    A = VCAT([permute(a,p) for a in H.A.A]...)
    Hadamard(A,H.mid,H.mid2)
end
