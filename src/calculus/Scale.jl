export Scale

immutable Scale{T <: RealOrComplex, L <: LinearOperator} <: LinearOperator
  coeff::T
  coeff_conj::T
  A::L
end

# Constructors

Scale{T <: RealOrComplex, R <: LinearOperator}(coeff::T, L::R) = Scale{T, R}(coeff, conj(coeff), L)

# Special Constructors
# scale of scale
Scale{T1 <: RealOrComplex, T2 <: RealOrComplex, R <: LinearOperator, S <: Scale{T1, R}}(coeff::T2, L::S) = 
Scale(*(promote(coeff,L.coeff)...), L.A)


# Mappings

function A_mul_B!{T, C <: AbstractArray, D, A <: LinearOperator}(y::C, L::Scale{T, A}, x::D)
  A_mul_B!(y, L.A, x)
  y .*= L.coeff
end

function A_mul_B!{T, C <: Tuple, D, A <: LinearOperator}(y::C, L::Scale{T, A}, x::D)
  A_mul_B!(y, L.A, x)
  for k in eachindex(y)
    y[k] .*= L.coeff
  end
end

function Ac_mul_B!{T, C, D <: AbstractArray, A <: LinearOperator}(y::D, L::Scale{T, A}, x::C)
  Ac_mul_B!(y, L.A, x)
  y .*= L.coeff_conj
end

function Ac_mul_B!{T, C, D <: Tuple, A <: LinearOperator}(y::D, L::Scale{T, A}, x::C)
  Ac_mul_B!(y, L.A, x)
  for k in eachindex(y)
    y[k] .*= L.coeff_conj
  end
end

# Properties

size(L::Scale) = size(L.A)

domainType(L::Scale) = domainType(L.A)
codomainType(L::Scale) = codomainType(L.A)

is_diagonal(L::Scale) = is_diagonal(L.A)
is_gram_diagonal(L::Scale) = is_gram_diagonal(L.A)
is_invertible(L::Scale) = L.coeff == 0 ? false : is_invertible(L.A)
is_full_row_rank(L::Scale) = is_full_row_rank(L.A)
is_full_column_rank(L::Scale) = is_full_column_rank(L.A)

fun_name(L::Scale) = "α$(fun_name(L.A))"
fun_type(L::Scale) = fun_type(L.A)
