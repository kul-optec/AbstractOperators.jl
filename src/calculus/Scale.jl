export Scale

immutable Scale{T <: RealOrComplex, L <: AbstractOperator} <: AbstractOperator
  coeff::T
  coeff_conj::T
  A::L
end

# Constructors

Scale{T <: RealOrComplex, R <: AbstractOperator}(coeff::T, L::R) = Scale{T, R}(coeff, conj(coeff), L)

# Special Constructors
# scale of scale
Scale{T1 <: RealOrComplex, T2 <: RealOrComplex, R <: AbstractOperator, S <: Scale{T1, R}}(coeff::T2, L::S) = 
Scale(*(promote(coeff,L.coeff)...), L.A)
# scale of DiagOp
Scale{T<:RealOrComplex}(coeff::T,L::DiagOp) = DiagOp(coeff*diag(L))

# Mappings

function A_mul_B!{T, C <: AbstractArray, D, A <: AbstractOperator}(y::C, L::Scale{T, A}, x::D)
  A_mul_B!(y, L.A, x)
  y .*= L.coeff
end

function A_mul_B!{T, C <: Tuple, D, A <: AbstractOperator}(y::C, L::Scale{T, A}, x::D)
  A_mul_B!(y, L.A, x)
  for k in eachindex(y)
    y[k] .*= L.coeff
  end
end

function Ac_mul_B!{T, C, D <: AbstractArray, A <: AbstractOperator}(y::D, L::Scale{T, A}, x::C)
  Ac_mul_B!(y, L.A, x)
  y .*= L.coeff_conj
end

function Ac_mul_B!{T, C, D <: Tuple, A <: AbstractOperator}(y::D, L::Scale{T, A}, x::C)
  Ac_mul_B!(y, L.A, x)
  for k in eachindex(y)
    y[k] .*= L.coeff_conj
  end
end

# jacobian
jacobian{T,L}(S::Scale{T,L},x::AbstractArray) = Scale(S.coeff,jacobian(S.A,x)) 

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



