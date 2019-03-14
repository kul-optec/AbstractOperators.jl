#Ax_mul_Bx

export Ax_mul_Bx

"""
`Ax_mul_Bx(A::AbstractOperator,B::AbstractOperator)`

Compose opeators such that:

`(Ax)*(Bx)`

# Example: Matrix multiplication

```julia
julia> A,B = randn(4,4),randn(4,4);

julia> P = Ax_mul_Bx(MatrixOp(A),MatrixOp(B))
▒*▒  ℝ^4 -> ℝ^(4, 4)

julia> x = randn(4);

julia> P*x ≈ (A*x)*(B*x)'
true

```
"""
struct Ax_mul_Bx{
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
  function Ax_mul_Bx(A::L1, B::L2, bufA::C, bufB::C, bufC::C, bufD::D) where {L1,L2,C,D}
    if size(A,2) != size(B,2) || size(A,1)[2] != size(B,1)[1] || ndims(A,1) != 2 
      throw(DimensionMismatch("Cannot compose operators"))
    end
    new{L1,L2,C,D}(A,B,bufA,bufB,bufC,bufD)
  end
end

struct Ax_mul_BxJac{
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
function Ax_mul_Bx(A::AbstractOperator,B::AbstractOperator)
  bufA = zeros(codomainType(A),size(A,1)) 
  bufB = zeros(codomainType(B),size(B,1)) 
  bufC = zeros(codomainType(B),size(B,1)) 
  bufD = zeros(domainType(A),size(A,2)) 
  Ax_mul_Bx(A,B,bufA,bufB,bufC,bufD)
end

# Jacobian
function Jacobian(P::Ax_mul_Bx{L1,L2,C,D}, x::AbstractArray) where {L1,L2,C,D}
  JA, JB = Jacobian(P.A, x), Jacobian(P.B, x)
  Ax_mul_BxJac{typeof(JA),typeof(JB),C,D}(JA,JB,P.bufA,P.bufB,P.bufC,P.bufD)
end

# Mappings
function mul!(y, P::Ax_mul_Bx{L1,L2,C,D}, b) where {L1,L2,C,D}
  mul!(P.bufA,P.A,b)
  mul!(P.bufB,P.B,b)
  mul!(y,P.bufA, P.bufB)
end

function mul!(y, J::AdjointOperator{Ax_mul_BxJac{L1,L2,C,D}}, b) where {L1,L2,C,D}
  #y .= J.A.B' * ( J.A.bufA'*b ) + J.A.A' * ( b*J.A.bufB' )
  mul!(J.A.bufC, J.A.bufA', b)
  mul!(y, J.A.B', J.A.bufC)
  mul!(J.A.bufA, b, J.A.bufB')
  mul!(J.A.bufD, J.A.A', J.A.bufA)
  y .+= J.A.bufD
  return y
end

size(P::Union{Ax_mul_Bx,Ax_mul_BxJac}) = ((size(P.A,1)[1],size(P.B,1)[2]),size(P.A,2))

fun_name(L::Union{Ax_mul_Bx,Ax_mul_BxJac}) = fun_name(L.A)*"*"*fun_name(L.B) 

domainType(L::Union{Ax_mul_Bx,Ax_mul_BxJac})   = domainType(L.A)
codomainType(L::Union{Ax_mul_Bx,Ax_mul_BxJac}) = codomainType(L.A)

# utils
function permute(P::Ax_mul_Bx, p::AbstractVector{Int})
  Ax_mul_Bx(permute(P.A,p),permute(P.B,p),P.buf,P.bufx)
end

remove_displacement(N::Ax_mul_Bx) = Ax_mul_Bx(remove_displacement(N.A), remove_displacement(N.B), N.bufA, N.bufB)
