#NonLinearCompose

export NonLinearCompose

"""
`NonLinearCompose(A::AbstractOperator,B::AbstractOperator)`

Compose opeators such that:

`A(x1)*B(x2)`

where `x1` and `x2` are two independent inputs.

# Example: Matrix multiplication

```julia
julia> n1,m1,n2,m2 = 3,4,4,6 

julia> x = ArrayPartition(randn(n1,m1),randn(n2,m2)); #inputs

julia> C = NonLinearCompose( Eye(n1,n2), Eye(m1,m2) )
# i.e. `I(⋅)*I(⋅)`

julia> Y = x.x[1]*x.x[2]

julia> C*x ≈ Y
true

```
"""
struct NonLinearCompose{
                        L1 <: HCAT,
                        L2 <: HCAT,
                        C <: AbstractArray,
                        D <: AbstractArray
                        } <: NonLinearOperator
  A::L1
  B::L2
  buf::C
  bufx::D
  function NonLinearCompose(A::L1, B::L2, buf::C, bufx::D) where {L1,L2,C,D}
    if ( (ndoms(A,1) > 1 || ndoms(B,1) > 1) || 
        (ndims(A,1) > 2 || ndims(B,1) > 2) ||
      (size(B,1)[1] == 1 ? (length(size(A,1)) == 1 ? false : true) : # outer product case
       ndims(A,1) == 1 ? true : (size(A,1)[2] != size(B,1)[1]))
     ) 
      throw(DimensionMismatch("cannot compose operators"))
    end
    @warn "`NonLinearCompose` will be substituted by `Ax_mul_Bx` in future versions of AbstractOperators"
    new{L1,L2,C,D}(A,B,buf,bufx)
  end
end

struct NonLinearComposeJac{
                           L1 <: HCAT,
                           L2 <: HCAT,
                           C <: AbstractArray,
                           D <: AbstractArray 
                           } <: LinearOperator
  A::L1
  B::L2
  buf::C
  bufx::D
end

# Constructors
function NonLinearCompose(L1::AbstractOperator,L2::AbstractOperator)

  A = HCAT(L1, Zeros( domainType(L2), size(L2,2), codomainType(L1), size(L1,1) ))
  B = HCAT(Zeros( domainType(L1), size(L1,2), codomainType(L2), size(L2,1) ), L2 )

  buf  = ArrayPartition(zeros(codomainType(A),size(A,1)),   zeros(codomainType(B),size(B,1)))
  bufx = ArrayPartition(zeros(codomainType(L1),size(L1,1)), zeros(codomainType(L2),size(L2,1)))

  NonLinearCompose(A,B,buf,bufx)
end

# Jacobian
function Jacobian(P::NonLinearCompose{L1,L2,C,D}, x::AbstractArray) where  {L1,L2,C,D}
  NonLinearComposeJac(Jacobian(P.A,x),Jacobian(P.B,x),P.buf,P.bufx)
end

# Mappings
function mul!(y, P::NonLinearCompose{L1,L2,C,D}, b) where {L1,L2,C,D}
  mul_skipZeros!(P.buf.x[1],P.A,b)
  mul_skipZeros!(P.buf.x[2],P.B,b)
  mul!(y,P.buf.x[1],P.buf.x[2])
end

function mul!(y, J::AdjointOperator{NonLinearComposeJac{L1,L2,C,D}}, b) where {L1,L2,C,D}
  P = J.A
  mul!(P.bufx.x[1],b,P.buf.x[2]')
  mul_skipZeros!(y,P.A',P.bufx.x[1])

  mul!(P.bufx.x[2],P.buf.x[1]',b)
  mul_skipZeros!(y,P.B',P.bufx.x[2])
end

# special case outer product  
function mul!(y, 
              J::AdjointOperator{NonLinearComposeJac{L1,L2,C,D}}, 
              b) where {
                        T, L1,L2,C, 
                        A <: Tuple{AbstractVector,AbstractArray}, 
                        D <: ArrayPartition{T,A} }
  P = J.A
  p = reshape(P.bufx.x[1], length(P.bufx.x[1]),1)
  mul!(p,b,P.buf.x[2]')
  mul_skipZeros!(y,P.A',P.bufx.x[1])

  mul!(P.bufx.x[2],P.buf.x[1]',b)
  mul_skipZeros!(y,P.B',P.bufx.x[2])
end

# Properties
function size(P::NonLinearCompose) 
  size_out = ndims(P.B,1) == 1 ? (size(P.A,1)[1],) :
  (size(P.A,1)[1], size(P.B,1)[2])
  size_out, size(P.A,2)
end

function size(P::NonLinearComposeJac) 
  size_out = ndims(P.B,1) == 1 ? (size(P.A,1)[1],) :
  (size(P.A,1)[1], size(P.B,1)[2])
  size_out, size(P.A,2)
end

fun_name(L::NonLinearCompose) = fun_name(L.A.A[1])*"*"*fun_name(L.B.A[2]) 
fun_name(L::NonLinearComposeJac) = fun_name(L.A.A[1])*"*"*fun_name(L.B.A[2]) 

domainType(L::NonLinearCompose)   = domainType.(Ref(L.A))
codomainType(L::NonLinearCompose) = codomainType(L.A)

domainType(L::NonLinearComposeJac)   = domainType.(Ref(L.A))
codomainType(L::NonLinearComposeJac) = codomainType(L.A)

# utils
function permute(P::NonLinearCompose{L,C,D}, p::AbstractVector{Int}) where {L,C,D}
  NonLinearCompose(permute(P.A,p),permute(P.B,p),P.buf,P.bufx)
end

remove_displacement(N::NonLinearCompose) = NonLinearCompose(remove_displacement(N.A), remove_displacement(N.B), N.buf, N.bufx)
