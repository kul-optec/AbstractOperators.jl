export DiagOp

"""
	DiagOp(domain_type::Type, dim_in::Tuple, d::AbstractArray)
	DiagOp(d::AbstractArray)

Creates a `LinearOperator` which, when multiplied with an array `x`, returns the elementwise product `d.*x`.

```jldoctest
julia> D = DiagOp(Float64, (2, 2,), [1. 2.; 3. 4.])
╲  ℝ^(2, 2) -> ℝ^(2, 2)

julia> D*ones(2,2)
2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0
	
```
"""
struct DiagOp{B, D, C <: Number, N, dS, cS, T <: Union{AbstractArray{<:Number, N}, Number}} <: LinearOperator
    dim_in::NTuple{N, Int}
    d::T
end

# Constructors

###standard constructor Operator{N}(D::Type, domain_dim::NTuple{N,Int})
function DiagOp(D::Type, domain_dim::NTuple{N, Int}, d::T; threaded::Bool = true) where {N, T <: AbstractArray}
    size(d) != domain_dim && error("dimension of d must coincide with domain_dim")
    C = promote_type(eltype(d), D)
    threaded = threaded && Threads.nthreads() > 1 && length(d) * sizeof(D) > 2^16
    B = threaded ? FastBroadcast.True() : FastBroadcast.False()
    dS = typeof(d isa SubArray ? parent(d) : d).name.wrapper{D}
    cS = typeof(d isa SubArray ? parent(d) : d).name.wrapper{C}
    return DiagOp{B, D, C, N, dS, cS, T}(domain_dim, d)
end

###standard constructor with Scalar
function DiagOp(D::Type, domain_dim::NTuple{N, Int}, d::T; threaded::Bool = true) where {N, T <: Number}
    C = promote_type(eltype(d), D)
    threaded = threaded && Threads.nthreads() > 1 && length(d) > 2^16
    B = threaded ? FastBroadcast.True() : FastBroadcast.False()
    return DiagOp{B, D, C, N, Array{D}, Array{C}, T}(domain_dim, d)
end

# other constructors
function DiagOp(d::AbstractArray{T, N}; threaded::Bool = true) where {N, T <: Number}
    C = eltype(d)
    B = (threaded && length(d) > 2^16) ? FastBroadcast.True() : FastBroadcast.False()
    S = typeof(d isa SubArray ? parent(d) : d).name.wrapper{T}
    return DiagOp{B, eltype(d), C, N, S, S, typeof(d)}(size(d), d)
end
DiagOp(domain_dim::NTuple{N, Int}, d::A; threaded::Bool = true) where {N, A <: Number} = DiagOp(Float64, domain_dim, d; threaded)

# scale of DiagOp
function Scale(coeff::T, L::DiagOp{B, D, C, N, dS, cS}) where {T <: Number, B, D, C, N, dS, cS}
    if coeff == 1
        return L
    end
    new_d = coeff * diag(L)
    new_C = promote_type(eltype(new_d), D)
    return DiagOp{B, D, new_C, N, dS, cS, typeof(new_d)}(L.dim_in, new_d)
end

# Mappings

function mul!(y::AbstractArray, L::DiagOp{B}, b::AbstractArray) where {B}
    check(y, L, b)
    return @.. thread = B y = L.d * b
end

function mul!(y::AbstractArray, L::AdjointOperator{<:DiagOp{B}}, b::AbstractArray) where {B}
    check(y, L, b)
    return @.. thread = B y = conj(L.A.d) * b
end

function mul!(y::AbstractArray, L::AdjointOperator{<:DiagOp{B, <:Real, <:Complex}}, b::AbstractArray) where {B}
    check(y, L, b)
    return @.. thread = B y = real(conj(L.A.d) * b)
end

# Transformations (we'll see about this)
# inv(L::DiagOp) = DiagOp(L.domain_type, L.dim_in, (L.d).^(-1))

# Properties

domain_storage_type(::DiagOp{<:Any, <:Any, <:Any, <:Any, dS}) where {dS} = dS
codomain_storage_type(::DiagOp{<:Any, <:Any, <:Any, <:Any, <:Any, cS}) where {cS} = cS

diag(L::DiagOp) = L.d
diag_AAc(L::DiagOp{B}) where {B} = @.. thread = B L.d * conj(L.d)
diag_AcA(L::DiagOp{B}) where {B} = @.. thread = B conj(L.d) * L.d

domain_type(::DiagOp{<:Any, D}) where {D} = D
codomain_type(::DiagOp{<:Any, <:Any, C}) where {C} = C
is_thread_safe(::DiagOp) = true

size(L::DiagOp) = (L.dim_in, L.dim_in)

fun_name(L::DiagOp) = "╲"

is_diagonal(L::DiagOp) = true
is_symmetric(L::DiagOp) = true
is_positive_definite(L::DiagOp) = all(@. L.d > 0)
is_positive_semidefinite(L::DiagOp) = all(@. L.d >= 0)

# TODO: probably the following allows for too-close-to-singular matrices
is_invertible(L::DiagOp) = 0 ∉ L.d
is_full_row_rank(L::DiagOp) = is_invertible(L)
is_full_column_rank(L::DiagOp) = is_invertible(L)

has_optimized_normalop(L::DiagOp) = true
has_optimized_normalop(L::AdjointOperator{<:DiagOp}) = true
function get_normal_op(L::DiagOp{B, D, <:Any, N, dS}) where {B, D, N, dS}
    new_d = @.. thread = B L.d * conj(L.d)
    return DiagOp{B, D, D, N, dS, dS, typeof(new_d)}(L.dim_in, new_d)
end

has_fast_opnorm(::DiagOp) = true
LinearAlgebra.opnorm(L::DiagOp) = maximum(abs, L.d)
