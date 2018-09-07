export Reshape


"""
`Reshape(A::AbstractOperator, dim_out...)`

Shorthand constructor: 

`reshape(A, idx...)` 

Reshape the codomain dimensions of an `AbstractOperator`.

```julia
julia> A = Reshape(DFT(10),2,5)
¶ℱ  ℝ^10 -> ℂ^(2, 5)

julia> R = reshape(Conv((19,),randn(10)),7,2,2)
¶★  ℝ^19 -> ℝ^(7, 2, 2)

```

"""
struct Reshape{N,L<:AbstractOperator} <: AbstractOperator
	A::L
	dim_out::NTuple{N,Int}

	function Reshape(A::L,dim_out::NTuple{N,Int}) where {N, L<:AbstractOperator}
		if prod(size(A,1)) != prod(dim_out)
			throw(DimensionMismatch("new dimensions $(dim_out) must be consistent with AbstractOperator codomain size $(size(A,1))"))
		end
		new{N,L}(A,dim_out)
	end
end

# Constructors

Reshape(A::L, dim_out::Vararg{Int,N}) where {N,L<:AbstractOperator} =
Reshape(A, dim_out)

# Mappings

function mul!(y::C, R::Reshape{N,L}, b::D) where {N,L,C,D}
	y_res = reshape(y,size(R.A,1))
	mul!(y_res, R.A, b)
end

function mul!(y::D, A::AdjointOperator{Reshape{N,L}}, b::C) where {N,L,C,D}
    R = A.A
	b_res = reshape(b,size(R.A,1))
	mul!(y, R.A', b_res)
end

# Properties

size(R::Reshape) = (R.dim_out, size(R.A,2))

  domainType(  R::Reshape) =   domainType(R.A)
codomainType(  R::Reshape) = codomainType(R.A)

is_linear(      R::Reshape) = is_linear(R.A)
is_null(        R::Reshape) = is_null(R.A)
is_eye(        R::Reshape)  = is_eye(R.A)
is_diagonal(    R::Reshape) = is_diagonal(R.A)
is_AcA_diagonal(R::Reshape) = is_AcA_diagonal(R.A) 
is_AAc_diagonal(R::Reshape) = is_AAc_diagonal(R.A)
is_orthogonal(  R::Reshape) = is_orthogonal(  R.A)
is_invertible(  R::Reshape) = is_invertible(R.A)
is_full_row_rank(  R::Reshape)    = is_full_row_rank(     R.A)   
is_full_column_rank(  R::Reshape) = is_full_column_rank(  R.A)

fun_name(R::Reshape) = "¶"*fun_name(R.A)
remove_displacement(R::Reshape) = Reshape(remove_displacement(R.A), R.dim_out)
