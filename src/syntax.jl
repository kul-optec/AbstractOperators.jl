import Base: transpose, *, +, -

###### ' ######
transpose{T <: LinearOperator}(L::T) = Transpose(L)

######+,-######
(+){T <: LinearOperator}(L::T) = L
(-){T <: LinearOperator}(L::T) = Scale(-1.0, L)
(-)(L::Sum) = Sum((-).(L.A), L.midC, L.midD)
(+)(L1::LinearOperator, L2::LinearOperator) = Sum((L1,  L2 ))
(-)(L1::LinearOperator, L2::LinearOperator) = Sum((L1, -L2 ))
(+)(L1::LinearOperator, L2::Sum           ) = Sum((L1,L2.A...))
(-)(L1::LinearOperator, L2::Sum           ) = Sum((L1,((-).(L2.A))...))
(+)(L1::Sum,            L2::LinearOperator) = L2+L1
(-)(L1::Sum,            L2::LinearOperator) = Sum((L1.A..., -L2))

###### * ######
function (*){T <: Union{AbstractArray, Tuple}}(L::LinearOperator, b::T)
  y = deepzeros(codomainType(L), size(L, 1))
	A_mul_B!(y, L, b)
  return y
end

*(L1::LinearOperator, L2::LinearOperator) = Compose(L1,L2)

*{E<:Eye}(L1::E, L2::LinearOperator) = L2
*{E<:Eye}(L1::LinearOperator, L2::E) = L1
*{E1<:Eye,E2<:Eye}(L1::E1, L2::E2) = L1

*{S<:Scale}(L1::S, L2::LinearOperator) = L1.coeff*(L1.A*L2)
*{S<:Scale}(L1::LinearOperator, L2::S) = L2.coeff*(L1*L2.A)
*{S1<:Scale,S2<:Scale}(L1::S1, L2::S2) = (L1.coeff*L2.coeff)*(L1.A*L2.A)

# redefine .*
Base.broadcast(::typeof(*), d::AbstractArray, L::LinearOperator) = DiagOp(codomainType(L), d)*L
Base.broadcast(::typeof(*), d::AbstractArray, L::Scale)          = DiagOp(L.coeff*d)
