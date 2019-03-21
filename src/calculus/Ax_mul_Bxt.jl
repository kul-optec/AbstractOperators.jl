#Ax_mul_Bxt

export Ax_mul_Bxt

"""
`Ax_mul_Bxt(A::AbstractOperator,B::AbstractOperator)`

Create an operator `P` such that:

`P == (Ax)*(Bx)'`

# Example: Matrix multiplication

```julia
julia> A,B = randn(4,4),randn(4,4);

julia> P = Ax_mul_Bxt(MatrixOp(A),MatrixOp(B))
▒*▒  ℝ^4 -> ℝ^(4, 4)

julia> x = randn(4);

julia> P*x == (A*x)*(B*x)'
true

```
"""
struct Ax_mul_Bxt{
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
    if ndims(A,1) == 1
      if size(A) != size(B)   
        throw(DimensionMismatch("Cannot compose operators"))
      end
    elseif ndims(A,1) == 2 && ndims(B,1) == 2 && size(A,2) == size(B,2)
      if size(A,1)[2] != size(B,1)[2]   
        throw(DimensionMismatch("Cannot compose operators"))
      end
    else
      throw(DimensionMismatch("Cannot compose operators"))
    end
    new{L1,L2,C,D}(A,B,bufA,bufB,bufC,bufD)
  end
end

struct Ax_mul_BxtJac{
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
  s,t = size(A,1), codomainType(A)
  bufA = eltype(s) <: Int ? zeros(t,s) : ArrayPartition(zeros.(t,s)...)
  bufC = eltype(s) <: Int ? zeros(t,s) : ArrayPartition(zeros.(t,s)...)
  s,t = size(B,1), codomainType(B)
  bufB = eltype(s) <: Int ? zeros(t,s) : ArrayPartition(zeros.(t,s)...)
  s,t = size(A,2), domainType(A)
  bufD = eltype(s) <: Int ? zeros(t,s) : ArrayPartition(zeros.(t,s)...)
  Ax_mul_Bxt(A,B,bufA,bufB,bufC,bufD)
end

# Jacobian
function Jacobian(P::Ax_mul_Bxt{L1,L2,C,D}, x::AbstractArray) where {L1,L2,C,D}
  JA, JB = Jacobian(P.A, x), Jacobian(P.B, x)
  Ax_mul_BxtJac{typeof(JA),typeof(JB),C,D}(JA,JB,P.bufA,P.bufB,P.bufC,P.bufD)
end

# Mappings
function mul!(y, P::Ax_mul_Bxt{L1,L2,C,D}, b) where {L1,L2,C,D}
  mul!(P.bufA,P.A,b)
  mul!(P.bufB,P.B,b)
  mul!(y,P.bufA, P.bufB')
end

function mul!(y, J::AdjointOperator{Ax_mul_BxtJac{L1,L2,C,D}}, b) where {L1,L2,C,D}
  #y .= J.A.A'*(b*(J.A.bufB)) + J.A.B'*(b'*(J.A.bufA))
  mul!(J.A.bufC, b, J.A.bufB)
  mul!(y, J.A.A', J.A.bufC)
  mul!(J.A.bufB, b', J.A.bufA)
  mul!(J.A.bufD, J.A.B', J.A.bufB)
  y .+= J.A.bufD
  return y
end

size(P::Union{Ax_mul_Bxt,Ax_mul_BxtJac}) = ((size(P.A,1)[1],size(P.B,1)[1]),size(P.A,2))

fun_name(L::Union{Ax_mul_Bxt,Ax_mul_BxtJac}) = fun_name(L.A)*"*"*fun_name(L.B) 

domainType(L::Union{Ax_mul_Bxt,Ax_mul_BxtJac})   = domainType(L.A)
codomainType(L::Union{Ax_mul_Bxt,Ax_mul_BxtJac}) = codomainType(L.A)

# utils
function permute(P::Ax_mul_Bxt{L1,L2,C,D}, 
                 p::AbstractVector{Int}) where {L1,L2,C,D <:ArrayPartition}
  Ax_mul_Bxt(permute(P.A,p),permute(P.B,p),P.bufA,P.bufB,P.bufC,ArrayPartition(P.bufD.x[p]) )
end

remove_displacement(P::Ax_mul_Bxt) = 
Ax_mul_Bxt(remove_displacement(P.A), remove_displacement(P.B), P.bufA, P.bufB, P.bufC, P.bufD)
