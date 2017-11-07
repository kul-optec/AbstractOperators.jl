export Scale

"""
`Scale(α::Number,A::AbstractOperator)`

Shorthand constructor: 

`*(α::Number,A::AbstractOperator)` 

Scale an `AbstractOperator` by a factor of `α`.

```julia
julia> A = FiniteDiff((10,2))
δx  ℝ^(10, 2) -> ℝ^(9, 2)

julia> S = Scale(10,A)
αδx  ℝ^(10, 2) -> ℝ^(9, 2)

julia> 10*A         #shorthand 
αℱc  ℝ^10 -> ℝ^10

```

"""
struct Scale{T <: RealOrComplex, L <: AbstractOperator} <: AbstractOperator
  coeff::T
  coeff_conj::T
  A::L
end

# Constructors

Scale(coeff::T, L::R) where {T <: RealOrComplex, R <: AbstractOperator} = 
Scale{T, R}(coeff, conj(coeff), L)

# Special Constructors
# scale of scale
Scale(coeff::T2, L::S) where {T1 <: RealOrComplex, 
			      T2 <: RealOrComplex, 
			      R <: AbstractOperator, 
			      S <: Scale{T1, R}}= 
Scale(*(promote(coeff,L.coeff)...), L.A)
# scale of DiagOp
Scale(coeff::T,L::DiagOp) where {T<:RealOrComplex} = DiagOp(coeff*diag(L))

# Mappings

function A_mul_B!(y::C, L::Scale{T, A}, x::D) where {T, C <: AbstractArray, D, A <: AbstractOperator}
  A_mul_B!(y, L.A, x)
  y .*= L.coeff
end

function A_mul_B!(y::C, L::Scale{T, A}, x::D) where {T, C <: Tuple, D, A <: AbstractOperator}
  A_mul_B!(y, L.A, x)
  for k in eachindex(y)
    y[k] .*= L.coeff
  end
end

function Ac_mul_B!(y::D, L::Scale{T, A}, x::C) where {T, C, D <: AbstractArray, A <: AbstractOperator}
  Ac_mul_B!(y, L.A, x)
  y .*= L.coeff_conj
end

function Ac_mul_B!(y::D, L::Scale{T, A}, x::C) where {T, C, D <: Tuple, A <: AbstractOperator}
  Ac_mul_B!(y, L.A, x)
  for k in eachindex(y)
    y[k] .*= L.coeff_conj
  end
end

# Properties

size(L::Scale) = size(L.A)

domainType(L::Scale) = domainType(L.A)
codomainType(L::Scale) = codomainType(L.A)

is_linear(L::Scale) = is_linear(L.A)
is_null(L::Scale) = is_null(L.A)
is_eye(L::Scale) = is_diagonal(L.A)
is_diagonal(L::Scale) = is_diagonal(L.A)
is_invertible(L::Scale) = L.coeff == 0 ? false : is_invertible(L.A)
is_AcA_diagonal(L::Scale) = is_AcA_diagonal(L.A) 
is_AAc_diagonal(L::Scale) = is_AAc_diagonal(L.A)
is_full_row_rank(L::Scale) = is_full_row_rank(L.A)
is_full_column_rank(L::Scale) = is_full_column_rank(L.A)

fun_name(L::Scale) = "α$(fun_name(L.A))"
fun_type(L::Scale) = fun_type(L.A)

diag(L::Scale) = L.coeff*diag(L.A)
diag_AcA(L::Scale) = (L.coeff)^2*diag_AcA(L.A)
diag_AAc(L::Scale) = (L.coeff)^2*diag_AAc(L.A)



