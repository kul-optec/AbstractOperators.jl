export Zeros

"""
	Zeros(domain_type::Type, dim_in::Tuple, [codomain_type::Type,] dim_out::Tuple)

Create a `LinearOperator` which, when multiplied with an array `x` of size `dim_in`, returns an array `y` of size `dim_out` filled with zeros.

For convenience `Zeros` can be constructed from any `AbstractOperator`.

```jldoctest
julia> Zeros(Eye(10,20))
0  ℝ^(10, 20) -> ℝ^(10, 20)

julia> Zeros([Eye(10,20) Eye(10,20)])
[0,0]  ℝ^(10, 20)  ℝ^(10, 20) -> ℝ^(10, 20)
	
```
"""
struct Zeros{C,N,D,M} <: LinearOperator
	dim_out::NTuple{N,Int}
	dim_in::NTuple{M,Int}
end

# Constructors
#default
function Zeros(
	domain_type::Type, dim_in::NTuple{M,Int}, codomain_type::Type, dim_out::NTuple{N,Int}
) where {N,M}
	return Zeros{codomain_type,N,domain_type,M}(dim_out, dim_in)
end

function Zeros(domain_type::Type, dim_in::NTuple{M,Int}, dim_out::NTuple{N,Int}) where {N,M}
	return Zeros{domain_type,N,domain_type,M}(dim_out, dim_in)
end

function Zeros(
	domain_type::NTuple{NN,Type},
	dim_in::NTuple{NN,Tuple},
	codomain_type::Type,
	dim_out::Tuple,
) where {NN}
	return HCAT([Zeros(domain_type[i], dim_in[i], codomain_type, dim_out) for i in 1:NN]...)
end

function Zeros(
	domain_type::Type,
	dim_in::Tuple,
	codomain_type::NTuple{NN,Type},
	dim_out::NTuple{NN,Tuple},
) where {NN}
	return VCAT([Zeros(domain_type, dim_in, codomain_type[i], dim_out[i]) for i in 1:NN]...)
end

Zeros(A::AbstractOperator) = Zeros(domain_type(A), size(A, 2), codomain_type(A), size(A, 1))

# Mappings

function mul!(
	y::AbstractArray{C,N}, A::Zeros{C,N,D,M}, b::AbstractArray{D,M}
) where {C,N,D,M}
	return fill!(y, zero(C))
end
function mul!(
	y::AbstractArray{D,M}, A::AdjointOperator{Zeros{C,N,D,M}}, b::AbstractArray{C,N}
) where {C,N,D,M}
	return fill!(y, zero(D))
end

# Properties

domain_type(::Zeros{C,N,D,M}) where {C,N,D,M} = D
codomain_type(::Zeros{C,N,D,M}) where {C,N,D,M} = C
is_thread_safe(::Zeros) = true

size(L::Zeros) = (L.dim_out, L.dim_in)

fun_name(A::Zeros) = "0"

is_null(L::Zeros) = true
is_AAc_diagonal(L::Zeros) = true
is_AcA_diagonal(L::Zeros) = true
is_diagonal(L::Zeros) = true

diag_AAc(L::Zeros) = 0
diag_AcA(L::Zeros) = 0

has_optimized_normalop(::Zeros) = true
get_normal_op(L::Zeros) = Zeros(domain_type(L), size(L, 2), domain_type(L), size(L, 2))

has_fast_opnorm(::Zeros) = true
LinearAlgebra.opnorm(L::Zeros) = zero(real(domain_type(L)))
