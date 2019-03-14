#SelfHadamard

export SelfHadamard

"""
`SelfHadamard(A::AbstractOperator,B::AbstractOperator)`

Compose opeators such that:

`(Ax).*(Bx)`

# Example

```julia
julia> A,B = Sin(3), Cos(3);

julia> P = SelfHadamard(A,B)
sin.*cos  ℝ^3 -> ℝ^3

julia> x = randn(3);

julia> P*x ≈ (sin.(x).*cos.(x))
true


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
  bufC::C
  bufD::D
  function SelfHadamard(A::L1, B::L2, bufA::C, bufB::C, bufC::C, bufD::D) where {L1,L2,C,D}
    if size(A) != size(B)  
      throw(DimensionMismatch("Cannot compose operators"))
    end
    new{L1,L2,C,D}(A,B,bufA,bufB,bufC,bufD)
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
  bufC::C
  bufD::D
end

# Constructors
function SelfHadamard(A::AbstractOperator,B::AbstractOperator)
  bufA = zeros(codomainType(A),size(A,1)) 
  bufB = zeros(codomainType(B),size(B,1)) 
  bufC = zeros(codomainType(B),size(B,1)) 
  bufD = zeros(domainType(A),size(A,2)) 
  buf = eltype(s) <: Int ? zeros(t,s) : ArrayPartition(zeros.(t,s))
  SelfHadamard(A,B,bufA,bufB,bufC,bufD)
end

# Jacobian
function Jacobian(P::SelfHadamard{L1,L2,C,D}, x::AbstractArray) where {L1,L2,C,D}
  JA, JB = Jacobian(P.A, x), Jacobian(P.B, x)
  SelfHadamardJac{typeof(JA),typeof(JB),C,D}(JA,JB,P.bufA,P.bufB,P.bufC,P.bufD)
end

# Mappings
function mul!(y, P::SelfHadamard{L1,L2,C,D}, b) where {L1,L2,C,D}
  mul!(P.bufA,P.A,b)
  mul!(P.bufB,P.B,b)
  y .= P.bufA .* P.bufB
  return y
end

function mul!(y, J::AdjointOperator{SelfHadamardJac{L1,L2,C,D}}, b) where {L1,L2,C,D}
  y .= J.A.B' * ( J.A.bufA .*b ) + J.A.A' * ( J.A.bufB .* b )
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
