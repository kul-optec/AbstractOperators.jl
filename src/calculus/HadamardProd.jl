#HadamardProd

export HadamardProd

"""
`HadamardProd(A::AbstractOperator,B::AbstractOperator)`

Create an operator `P` such that:

`P*x == (Ax).*(Bx)`

# Example

```julia
julia> A,B = Sin(3), Cos(3);

julia> P = HadamardProd(A,B)
sin.*cos  ℝ^3 -> ℝ^3

julia> x = randn(3);

julia> P*x == (sin.(x).*cos.(x))
true


```
"""
struct HadamardProd{
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
  function HadamardProd(A::L1, B::L2, bufA::C, bufB::C, bufD::D) where {L1,L2,C,D}
    if size(A) != size(B)
      throw(DimensionMismatch("Cannot compose operators"))
    end
    new{L1,L2,C,D}(A,B,bufA,bufB,bufD)
  end
end

struct HadamardProdJac{
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
function HadamardProd(A::AbstractOperator,B::AbstractOperator)
  bufA = allocateInCodomain(A)
  bufB = allocateInCodomain(B)
  bufD = allocateInDomain(A)
  HadamardProd(A,B,bufA,bufB,bufD)
end

# Jacobian
function Jacobian(P::HadamardProd{L1,L2,C,D}, x::AbstractArray) where {L1,L2,C,D}
  JA, JB = Jacobian(P.A, x), Jacobian(P.B, x)
  HadamardProdJac{typeof(JA),typeof(JB),C,D}(JA,JB,P.bufA,P.bufB,P.bufD)
end

# Mappings
function mul!(y, P::HadamardProd{L1,L2,C,D}, b) where {L1,L2,C,D}
  mul!(P.bufA,P.A,b)
  mul!(P.bufB,P.B,b)
  y .= P.bufA .* P.bufB
  return y
end

function mul!(y, J::AdjointOperator{HadamardProdJac{L1,L2,C,D}}, b) where {L1,L2,C,D}
  #y .= J.A.B' * ( J.A.bufA .*b ) + J.A.A' * ( J.A.bufB .* b )
  J.A.bufA .*= b
  mul!(y, J.A.B', J.A.bufA)
  J.A.bufB .*= b
  mul!(J.A.bufD, J.A.A', J.A.bufB)
  y .+= J.A.bufD
  return y
end

size(P::Union{HadamardProd,HadamardProdJac}) = (size(P.A,1),size(P.A,2))

fun_name(L::Union{HadamardProd,HadamardProdJac}) = fun_name(L.A)*".*"*fun_name(L.B)

domainType(L::Union{HadamardProd,HadamardProdJac})   = domainType(L.A)
codomainType(L::Union{HadamardProd,HadamardProdJac}) = codomainType(L.A)

# utils
function permute(P::HadamardProd{L1,L2,C,D},
                 p::AbstractVector{Int}) where {L1,L2,C,D <:ArrayPartition}
  HadamardProd(permute(P.A,p),permute(P.B,p),P.bufA,P.bufB,ArrayPartition(P.bufD.x[p]) )
end

remove_displacement(P::HadamardProd) =
HadamardProd(remove_displacement(P.A), remove_displacement(P.B), P.bufA, P.bufB, P.bufD)
