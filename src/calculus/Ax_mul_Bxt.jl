#Ax_mul_Bxt

export Ax_mul_Bxt

"""
`Ax_mul_Bxt(A::AbstractOperator,B::AbstractOperator)`

Compose opeators such that:

`(Ax)'*(Bx)`

# Example: Matrix multiplication

```julia
julia> A,B = randn(4,4),randn(4,4);

julia> P = Ax_mul_Bxt(MatrixOp(A),MatrixOp(B))
▒*▒  ℝ^4 -> ℝ^1

julia> x = randn(4);

julia> P*x ≈ [(A*x)'*(B*x)]
true

```
"""
struct Ax_mul_Bxt{N,
                   L1 <: AbstractOperator,
                   L2 <: AbstractOperator,
                   C <: AbstractArray,
                   D <: AbstractArray,
                  } <: NonLinearOperator
  A::L1
  B::L2
  bufA::C
  bufB::C
  bufC::C
  bufD::D
  function Ax_mul_Bxt(A::L1, B::L2, bufA::C, bufB::C, bufC::C, bufD::D) where {L1,L2,C,D}
    if size(A) != size(B) || ndims(A,1) > 2 
      throw(CimensionMismatch("Cannot compose operators"))
    end
    N = ndims(A,1)
    new{N,L1,L2,C,D}(A,B,bufA,bufB,bufC,bufD)
  end
end

struct Ax_mul_BxtJac{N,
                      L1 <: AbstractOperator,
                      L2 <: AbstractOperator,
                      C <: AbstractArray,
                      D <: AbstractArray,
                     } <: LinearOperator
  A::L1
  B::L2
  bufA::C
  bufB::C
  bufC::C
  bufD::D
end

# Constructors
function Ax_mul_Bxt(A::AbstractOperator,B::AbstractOperator)
  bufA = zeros(codomainType(A),size(A,1)) 
  bufB = zeros(codomainType(A),size(A,1)) 
  bufC = zeros(codomainType(A),size(A,1)) 
  bufD = zeros(domainType(A),size(A,2)) 
  Ax_mul_Bxt(A,B,bufA,bufB,bufC,bufD)
end

# Jacobian
function Jacobian(P::Ax_mul_Bxt{N,L1,L2,C,D}, x::AbstractArray) where {N,L1,L2,C,D}
  JA, JB = Jacobian(P.A, x), Jacobian(P.B, x)
  Ax_mul_BxtJac{N,typeof(JA),typeof(JB),C,D}(JA,JB,P.bufA,P.bufB,P.bufC,P.bufD)
end

# Mappings
# N == 1 input is a vector
function mul!(y, P::Ax_mul_Bxt{1,L1,L2,C,D}, b) where {L1,L2,C,D}
  mul!(P.bufA,P.A,b)
  mul!(P.bufB,P.B,b)
  mul!(y,P.bufA, P.bufB')
end

function mul!(y, J::AdjointOperator{Ax_mul_BxtJac{1,L1,L2,C,D}}, b) where {L1,L2,C,D}
  #y .= conj(J.A.A'*J.A.bufB+J.A.B'*J.A.bufA).*b[1]
  mul!(y, J.A.A', J.A.bufB)
  mul!(J.A.bufD, J.A.B', J.A.bufA)
  y .= conj.( y .+  J.A.bufD ) .* b[1]
  return y
end

# N == 2 input is a matrix
function mul!(y, P::Ax_mul_Bxt{2,L1,L2,C,D}, b) where {L1,L2,C,D}
  mul!(P.bufA,P.A,b)
  mul!(P.bufB,P.B,b)
  mul!(y,P.bufA',P.bufB)
  return y
end

function mul!(y, J::AdjointOperator{Ax_mul_BxtJac{2,L1,L2,C,D}}, b) where {L1,L2,C,D}
  # y = A'*((B*x)*b') + B'*((A*x)*b)
  mul!(J.A.bufC, J.A.bufB, b')
  mul!(y, J.A.A', J.A.bufC)
  mul!(J.A.bufC, J.A.bufA, b)
  mul!(J.A.bufD, J.A.B', J.A.bufC)
  y .+= J.A.bufD
  return y
end

size(P::Ax_mul_Bxt{1}) = ((size(P.A,2)[1],size(P.A,2)[1]),size(P.A,2))
size(P::Ax_mul_Bxt{2}) = ((size(P.A,1)[1],size(P.A,1)[1]),size(P.A,2))

size(P::Ax_mul_BxtJac{1}) = ((size(P.A,2)[1],size(P.A,2)[1]),size(P.A,2))
size(P::Ax_mul_BxtJac{2}) = ((size(P.A,1)[1],size(P.A,1)[1]),size(P.A,2))

fun_name(L::Ax_mul_Bxt) = fun_name(L.A)*"*"*fun_name(L.B) 
fun_name(L::Ax_mul_BxtJac) = fun_name(L.A)*"*"*fun_name(L.B) 

domainType(L::Ax_mul_Bxt)   = domainType(L.A)
codomainType(L::Ax_mul_Bxt) = codomainType(L.A)

domainType(L::Ax_mul_BxtJac)   = domainType(L.A)
codomainType(L::Ax_mul_BxtJac) = codomainType(L.A)

# utils
function permute(P::Ax_mul_Bxt, p::AbstractVector{Int})
  Ax_mul_Bxt(permute(P.A,p),permute(P.B,p),P.buf,P.bufx)
end

remove_displacement(N::Ax_mul_Bxt) = Ax_mul_Bxt(remove_displacement(N.A), remove_displacement(N.B), N.bufA, N.bufB)
