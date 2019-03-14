#SelfHadamard

export SelfHadamard

"""
`SelfHadamard(A::AbstractOperator,B::AbstractOperator)`

Compose opeators such that:

`(Ax).*(Bx)`

# Example: Matrix multiplication

```julia

```
"""
struct SelfHadamard{
                    L1 <: AbstractOperator,
                    L2 <: AbstractOperator,
                    C <: AbstractArray,
                    D <: AbstractArray,
                   } <: NonLinearOperator
  A::L1
  B::L2
  bufA::C
  bufB::C
  bufD::D
  function SelfHadamard(A::L1, B::L2, bufA::C, bufB::C, bufD::D) where {L1,L2,C,D}
    if size(A) != size(B)  
      throw(DimensionMismatch("Cannot compose operators"))
    end
    new{L1,L2,C,D}(A,B,bufA,bufB,bufD)
  end
end

struct SelfHadamardJac{
                       L1 <: AbstractOperator,
                       L2 <: AbstractOperator,
                       C <: AbstractArray,
                       D <: AbstractArray,
                      } <: LinearOperator
  A::L1
  B::L2
  bufA::C
  bufB::C
  bufD::D
end

# Constructors
function SelfHadamard(A::AbstractOperator,B::AbstractOperator)
  s,t = size(A,1), codomainType(A)
  bufA = eltype(s) <: Int ? zeros(t,s) : ArrayPartition(zeros.(t,s)...)
  bufB = eltype(s) <: Int ? zeros(t,s) : ArrayPartition(zeros.(t,s)...)
  s,t = size(A,2), domainType(A)
  bufD = eltype(s) <: Int ? zeros(t,s) : ArrayPartition(zeros.(t,s)...)
  SelfHadamard(A,B,bufA,bufB,bufD)
end

# Jacobian
function Jacobian(P::SelfHadamard{L1,L2,C,D}, x::AbstractArray) where {L1,L2,C,D}
  JA, JB = Jacobian(P.A, x), Jacobian(P.B, x)
  SelfHadamardJac{typeof(JA),typeof(JB),C,D}(JA,JB,P.bufA,P.bufB,P.bufD)
end

# Mappings
function mul!(y, P::SelfHadamard{L1,L2,C,D}, b) where {L1,L2,C,D}
  mul!(P.bufA,P.A,b)
  mul!(P.bufB,P.B,b)
  y .= P.bufA .* P.bufB
  return y
end

function mul!(y, J::AdjointOperator{SelfHadamardJac{L1,L2,C,D}}, b) where {L1,L2,C,D}
  # y .= J.A.B' * ( J.A.bufA .*b ) + J.A.A' * ( J.A.bufB .* b )
  J.A.bufA .*= b
  mul!(y, J.A.B', J.A.bufA)
  J.A.bufB .*= b
  mul!(J.A.bufD, J.A.A', J.A.bufB)
  y .+= J.A.bufD
  return y
end

size(P::Union{SelfHadamard,SelfHadamardJac}) = (size(P.A,1),size(P.A,2))

fun_name(L::Union{SelfHadamard,SelfHadamardJac}) = fun_name(L.A)*".*"*fun_name(L.B) 

domainType(L::Union{SelfHadamard,SelfHadamardJac})   = domainType(L.A)
codomainType(L::Union{SelfHadamard,SelfHadamardJac}) = codomainType(L.A)

# utils
function permute(P::SelfHadamard, p::AbstractVector{Int})
  SelfHadamard(permute(P.A,p),permute(P.B,p),P.buf,P.bufx)
end

remove_displacement(N::SelfHadamard) = SelfHadamard(remove_displacement(N.A), remove_displacement(N.B), N.bufA, N.bufB)
