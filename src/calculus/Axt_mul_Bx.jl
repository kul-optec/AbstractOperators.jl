#Axt_mul_Bx

export Axt_mul_Bx

"""
`Axt_mul_Bx(A::AbstractOperator,B::AbstractOperator)`

Create an operator `P` such that:

`P*x == (Ax)'*(Bx)`

# Example

```julia
julia> A,B = randn(4,4),randn(4,4);

julia> P = Axt_mul_Bx(MatrixOp(A),MatrixOp(B))
▒*▒  ℝ^4 -> ℝ^1

julia> x = randn(4);

julia> P*x == [(A*x)'*(B*x)]
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
    if ndims(A,1) == 1
      if size(A) != size(B)
        throw(DimensionMismatch("Cannot compose operators"))
      end
    elseif ndims(A,1) == 2 && ndims(B,1) == 2 && size(A,2) == size(B,2)
      if size(A,1)[1] != size(B,1)[1]
        throw(DimensionMismatch("Cannot compose operators"))
      end
    else
      throw(DimensionMismatch("Cannot compose operators"))
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
  bufA = allocateInCodomain(A)
  bufB = allocateInCodomain(B)
  bufC = allocateInCodomain(A)
  bufD = allocateInDomain(A)
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
  # y .= J.A.A'*((J.A.bufB)*b') + J.A.B'*((J.A.bufA)*b)
  mul!(J.A.bufC, J.A.bufB, b')
  mul!(y, J.A.A', J.A.bufC)
  mul!(J.A.bufB, J.A.bufA, b)
  mul!(J.A.bufD, J.A.B', J.A.bufB)
  y .+= J.A.bufD
  return y
end

size(P::Union{Axt_mul_Bx{1},Axt_mul_BxJac{1}}) = ((1,),size(P.A,2))
size(P::Union{Axt_mul_Bx{2},Axt_mul_BxJac{2}}) = ((size(P.A,1)[2],size(P.B,1)[2]),size(P.A,2))

fun_name(L::Union{Axt_mul_Bx,Axt_mul_BxJac}) = fun_name(L.A)*"*"*fun_name(L.B)

domainType(L::Union{Axt_mul_Bx,Axt_mul_BxJac}) = domainType(L.A)
codomainType(L::Union{Axt_mul_Bx,Axt_mul_BxJac}) = codomainType(L.A)

# utils
function permute(P::Axt_mul_Bx{N,L1,L2,C,D},
                 p::AbstractVector{Int}) where {N,L1,L2,C,D <:ArrayPartition}
  Axt_mul_Bx(permute(P.A,p),permute(P.B,p),P.bufA,P.bufB,P.bufC,ArrayPartition(P.bufD.x[p]) )
end

remove_displacement(P::Axt_mul_Bx) =
Axt_mul_Bx(remove_displacement(P.A), remove_displacement(P.B), P.bufA, P.bufB, P.bufC, P.bufD)
