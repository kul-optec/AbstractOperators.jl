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

struct Hadamard{M, # number of domains  
                N, # number of AbstractOperator 
                L <: NTuple{N,AbstractOperator},
                P <: NTuple{N,Union{Int,Tuple}},
                C <: Union{NTuple{M,AbstractArray}, AbstractArray},
                V <: VCAT{M,N,L,P,C}
               } <: NonLinearOperator
	A::V
	mid::C
	mid2::C
	function Hadamard(A::V, mid::C, mid2::C) where {M, N, L, P, C, V <: VCAT{M,N,L,P,C}}
		any([ai != size(A,1)[1] for ai in size(A,1)]) &&
		throw(DimensionMismatch("cannot compose operators"))
		any(any(
			sum([!is_null(A[m][n]) for n = 1:N, m = 1:M],2) .> 1
			)) &&
		throw(DimensionMismatch("cannot compose operators"))

		new{M, N, L, P, C, V}(A,mid,mid2)
	end
end

struct HadamardJacobian{M, # number of domains  
                        N, # number of AbstractOperator 
                        L <: NTuple{N,AbstractOperator},
                        P <: NTuple{N,Union{Int,Tuple}},
                        C <: Union{NTuple{M,AbstractArray}, AbstractArray},
                        V <: VCAT{M,N,L,P,C}
                       } <: LinearOperator
	A::V
	mid::C
	mid2::C
	function HadamardJacobian(A::V,mid::C,mid2::C) where {M, N, L, P, C, V <: VCAT{M,N,L,P,C}}
		new{M, N, L, P, C, V}(A,mid,mid2)
	end
end

# Constructors
function Hadamard(L::Vararg{HCAT{1,N}}) where {N}
	A = VCAT(L...)
	mid  = zeros.(codomainType(A), size(A,1))
	mid2 = zeros.(codomainType(A), size(A,1))
	Hadamard(A,mid,mid2)
end

function Hadamard(L::Vararg{AbstractOperator})

	M = sum(ndoms.(L,2))
	Z  = Zeros.(domainType.(L),size.(L,2),codomainType.(L),size.(L,1))
	Op = [(Z[1:i-1]...,L[i], Z[i+1:end]...) for i in eachindex(L)]
	hcats = [HCAT(op...) for op in Op ]
	Hadamard(hcats...)

end

# Mappings
function A_mul_B!(y, H::Hadamard{M,N,L,P,C,V}, b) where {M,N,L,P,C,V}
	A_mul_B!(H.mid,H.A,b)

	y .= H.mid[1].*H.mid[2]
	for i = 3:M
		y .*= H.mid[i]
	end
end

# Jacobian
Jacobian(A::H, x::D) where {M, N, L, P, C, V, H <: Hadamard{M,N,L,P,C,V}, D <: Tuple } =
HadamardJacobian(Jacobian(A.A,x),A.mid,A.mid2)

function Ac_mul_B!(y, J::HadamardJacobian{M,N,L,P,C,V}, b) where {M,N,L,P,C,V}
	for i = 1:M
		c = (J.mid[1:i-1]...,J.mid[i+1:end]...,b)
		J.mid2[i] .= (.*)(c...)
	end
	Ac_mul_B!(y, J.A, J.mid2)

end

# Properties
size(P::Hadamard) = size(P.A[1],1), size(P.A[1],2)
size(P::HadamardJacobian) = size(P.A[1],1), size(P.A[1],2)

fun_name(L::Hadamard{M,N})         where {M,N} = N == 2 ? 
fun_name(L.A[1].A[L.A.A[1].idxs[1]])"⊙"*fun_name(L.A[2].A[L.A.A[2].idxs[2]]) : "⊙"
fun_name(L::HadamardJacobian{M,N}) where {M,N} = N == 2 ?
"J("*fun_name(L.A[1].A[L.A.A[1].idxs[1]])"⊙"*fun_name(L.A[2].A[L.A.A[2].idxs[2]])*")" : "J(⊙)"

domainType(L::Hadamard)   = domainType.(L.A[1])
codomainType(L::Hadamard) = codomainType(L.A[1])

domainType(L::HadamardJacobian)   = domainType.(L.A[1])
codomainType(L::HadamardJacobian) = codomainType(L.A[1])

# utils
import Base: permute

function permute(H::Hadamard{M,N,L,P,C,V}, p::AbstractVector{Int}) where {M,N,L,P,C,V}
    A = VCAT([permute(a,p) for a in H.A.A][p]...)
    Hadamard(A,H.mid,H.mid2)
end
