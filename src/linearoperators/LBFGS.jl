export LBFGS, update!, reset!

"""
`LBFGS(domainType::Type,dim_in::Tuple, M::Integer)`

`LBFGS(dim_in::Tuple, M::Integer)`

`LBFGS(x::AbstractArray, M::Integer)`

Construct a Limited-Memory BFGS `LinearOperator` with memory `M`. The memory of `LBFGS` can be updated using the function `update!`, where the current iteration variable and gradient (`x`, `grad`) and the previous ones (`x_prev` and `grad_prev`) are needed:

```
julia> L = LBFGS(Float64,(4,),5)
LBFGS  ℝ^4 -> ℝ^4

julia> update!(L,x,x_prev,grad,grad_prev); # update memory

julia> d = L*grad; # compute new direction

```

Use  `reset!(L)` to cancel the memory of the operator.

"""
mutable struct LBFGS{R, T <: AbstractArray, M, I <: Integer} <: LinearOperator
  currmem::I
  curridx::I
  s::T
  y::T
  s_M::AbstractArray{T, 1}
  y_M::AbstractArray{T, 1}
  ys_M::AbstractArray{R, 1}
  alphas::AbstractArray{R, 1}
  H::R
end

#default constructor

function LBFGS(x::T, M::I) where {T <: AbstractArray, I <: Integer}
  s_M = [zero(x) for i = 1:M]
  y_M = [zero(x) for i = 1:M]
  s = zero(x)
  y = zero(x)
  R = real(eltype(x))
  ys_M = similar(x, R, M)
  fill!(ys_M, one(R))
  alphas = similar(x, R, M)
  fill!(alphas, one(R))
  LBFGS{R, T, M, I}(0, 0, s, y, s_M, y_M, ys_M, alphas, one(R))
end

"""
`update!(L::LBFGS, x, x_prex, grad, grad_prev)`

See the documentation for `LBFGS`.
"""
function update!(L::LBFGS{R, T, M, I}, x::T, x_prev::T, gradx::T, gradx_prev::T) where {R, T, M, I}
  L.s .= x .- x_prev
  L.y .= gradx .- gradx_prev
  ys = real(dot(L.s, L.y))
  if ys > 0
    L.curridx += 1
    if L.curridx > M L.curridx = 1 end
    L.currmem += 1
    if L.currmem > M L.currmem = M end
    L.ys_M[L.curridx] = ys
    L.s_M[L.curridx] .= L.s
    L.y_M[L.curridx] .= L.y
    yty = real(dot(L.y, L.y))
    L.H = ys/yty
  end
  return L
end

"""
`reset!(L::LBFGS)`

Cancels the memory of `L`.
"""
function reset!(L::LBFGS)
  L.currmem, L.curridx = 0, 0
  L.H = 1.0
end

# LBFGS operators are symmetric

mul!(x::T, L::AdjointOperator{LBFGS{R, T, M, I}}, y::T) where {R, T, M, I} = mul!(x, L.A, y)

# Two-loop recursion

function mul!(d::T, L::LBFGS{R, T, M, I}, gradx::T) where {R, T, M, I}
  d .= gradx
  idx = loop1!(d,L)
  d .*= L.H
  loop2!(d,idx,L)
  return d
end

function loop1!(d::T, L::LBFGS{R, T, M, I}) where {R, T, M, I}
  idx = L.curridx
  for i = 1:L.currmem
    L.alphas[idx] = real(dot(L.s_M[idx], d))/L.ys_M[idx]
    d .-= L.alphas[idx] .* L.y_M[idx]
    idx -= 1
    if idx == 0 idx = M end
  end
  return idx
end

function loop2!(d::T, idx, L::LBFGS{R, T, M, I}) where {R, T, M, I}
  for i = 1:L.currmem
    idx += 1
    if idx > M idx = 1 end
    beta = real(dot(L.y_M[idx], d))/L.ys_M[idx]
    d .+= (L.alphas[idx] - beta) .* L.s_M[idx]
  end
  return d
end

# Properties

domainType(L::LBFGS)   =   eltype(L.y_M[1])
domainType(L::LBFGS{R,T}) where { R, T <: ArrayPartition } = eltype.(L.y_M[1].x)
codomainType(L::LBFGS) = eltype(L.y_M[1])
codomainType(L::LBFGS{R,T}) where { R, T <: ArrayPartition } = eltype.(L.y_M[1].x)

size(A::LBFGS{R,T}) where { R, T <: ArrayPartition } = (size.(A.s.x), size.(A.s.x))
size(A::LBFGS) = (size(A.s), size(A.s))

fun_name(A::LBFGS) = "LBFGS"
