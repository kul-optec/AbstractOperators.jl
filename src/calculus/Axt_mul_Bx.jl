#Axt_mul_Bx

export Axt_mul_Bx

"""
`Axt_mul_Bx(A::AbstractOperator,B::AbstractOperator)`

Compose opeators such that:

`(Ax)'*(Bx)`

# Example: Matrix multiplication

```julia
julia> A,B = randn(4,4),randn(4,4);

julia> P = Axt_mul_Bx(MatrixOp(A),MatrixOp(B))
▒*▒  ℝ^4 -> ℝ^1

julia> x = randn(4);

julia> P*x ≈ [(A*x)'*(B*x)]
true

```
"""
struct Axt_mul_Bx{N,
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
  function Axt_mul_Bx(A::L1, B::L2, bufA::C, bufB::C, bufC::C, bufD::D) where {L1,L2,C,D}
    if size(A) != size(B) || ndims(A,1) > 2 
      throw(CimensionMismatch("Cannot compose operators"))
    end
    N = ndims(A,1)
    new{N,L1,L2,C,D}(A,B,bufA,bufB,bufC,bufD)
  end
end

struct Axt_mul_BxJac{N,
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
function Axt_mul_Bx(A::AbstractOperator,B::AbstractOperator)
  bufA = zeros(codomainType(A),size(A,1)) 
  bufB = zeros(codomainType(A),size(A,1)) 
  bufC = zeros(codomainType(A),size(A,1)) 
  bufD = zeros(domainType(A),size(A,2)) 
  Axt_mul_Bx(A,B,bufA,bufB,bufC,bufD)
end

# Jacobian
function Jacobian(P::Axt_mul_Bx{N,L1,L2,C,D}, x::AbstractArray) where {N,L1,L2,C,D}
  JA, JB = Jacobian(P.A, x), Jacobian(P.B, x)
  Axt_mul_BxJac{N,typeof(JA),typeof(JB),C,D}(JA,JB,P.bufA,P.bufB,P.bufC,P.bufD)
end

# Mappings
# N == 1 input is a vector
function mul!(y, P::Axt_mul_Bx{1,L1,L2,C,D}, b) where {L1,L2,C,D}
  mul!(P.bufA,P.A,b)
  mul!(P.bufB,P.B,b)
  y[1] = dot(P.bufA,P.bufB)
end

function mul!(y, J::AdjointOperator{Axt_mul_BxJac{1,L1,L2,C,D}}, b) where {L1,L2,C,D}
  #y .= conj(J.A.A'*J.A.bufB+J.A.B'*J.A.bufA).*b[1]
  mul!(y, J.A.A', J.A.bufB)
  mul!(J.A.bufD, J.A.B', J.A.bufA)
  y .= conj.( y .+  J.A.bufD ) .* b[1]
  return y
end

# N == 2 input is a matrix
function mul!(y, P::Axt_mul_Bx{2,L1,L2,C,D}, b) where {L1,L2,C,D}
  mul!(P.bufA,P.A,b)
  mul!(P.bufB,P.B,b)
  mul!(y,P.bufA',P.bufB)
  return y
end

function mul!(y, J::AdjointOperator{Axt_mul_BxJac{2,L1,L2,C,D}}, b) where {L1,L2,C,D}
  # y = A'*((B*x)*b') + B'*((A*x)*b)
  mul!(J.A.bufC, J.A.bufB, b')
  mul!(y, J.A.A', J.A.bufC)
  mul!(J.A.bufC, J.A.bufA, b)
  mul!(J.A.bufD, J.A.B', J.A.bufC)
  y .+= J.A.bufD
  return y
end

size(P::Axt_mul_Bx{1}) = ((1,),size(P.A,2))
size(P::Axt_mul_Bx{2}) = ((size(P.A,1)[2],size(P.A,1)[2]),size(P.A,2))

size(P::Axt_mul_BxJac{1}) = ((1,),size(P.A,2))
size(P::Axt_mul_BxJac{2}) = ((size(P.A,1)[2],size(P.A,1)[2]),size(P.A,2))

fun_name(L::Axt_mul_Bx) = fun_name(L.A)*"*"*fun_name(L.B) 
fun_name(L::Axt_mul_BxJac) = fun_name(L.A)*"*"*fun_name(L.B) 

domainType(L::Axt_mul_Bx)   = domainType(L.A)
codomainType(L::Axt_mul_Bx) = codomainType(L.A)

domainType(L::Axt_mul_BxJac)   = domainType(L.A)
codomainType(L::Axt_mul_BxJac) = codomainType(L.A)

# utils
function permute(P::Axt_mul_Bx, p::AbstractVector{Int})
  Axt_mul_Bx(permute(P.A,p),permute(P.B,p),P.buf,P.bufx)
end

remove_displacement(N::Axt_mul_Bx) = Axt_mul_Bx(remove_displacement(N.A), remove_displacement(N.B), N.bufA, N.bufB)
