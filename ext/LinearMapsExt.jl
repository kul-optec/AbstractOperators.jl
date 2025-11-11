module LinearMapsExt

using LinearMaps
using AbstractOperators
import Base: size
using LinearAlgebra
import OperatorCore:
    is_linear,
    is_null,
    is_eye,
    is_symmetric,
    is_diagonal,
    is_AcA_diagonal,
    is_AAc_diagonal,
    diag_AcA,
    diag_AAc,
    is_orthogonal,
    is_invertible,
    is_full_row_rank,
    is_full_column_rank,
    is_positive_definite,
    is_positive_semidefinite

struct LinearMapWrapper{T,AT<:AbstractOperator} <: LinearMap{T}
    A::AT
    function LinearMapWrapper(A::AT) where {AT<:AbstractOperator}
        if domain_type(A) != codomain_type(A)
            error(
                "LinearMapsExt.LinearMap only supports operators with matching domain and codomain types",
            )
        end
        T = domain_type(A)
        return new{T,AT}(A)
    end
end

Base.size(L::LinearMapWrapper) = (prod(size(L.A, 1)), prod(size(L.A, 2)))

function LinearMaps._unsafe_mul!(y::AbstractVector, L::LinearMapWrapper, x::AbstractVector)
    mul!(reshape(y, size(L.A, 1)), L.A, reshape(x, size(L.A, 2)))
    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector,
    L::Union{LinearMaps.TransposeMap{<:Any,<:LinearMapWrapper},LinearMaps.AdjointMap{<:Any,<:LinearMapWrapper}},
    x::AbstractVector,
)
    At = L.lmap.A'
    mul!(reshape(y, size(At, 1)), At, reshape(x, size(At, 2)))
    return y
end

LinearAlgebra.issymmetric(L::LinearMapWrapper) = is_symmetric(L.A) && domain_type(L.A) <: Real
LinearAlgebra.ishermitian(L::LinearMapWrapper) = is_symmetric(L.A)
LinearAlgebra.isposdef(L::LinearMapWrapper) = is_positive_definite(L.A)

"""
    LinearMaps.LinearMap(A::AbstractOperator)

Wraps a linear `AbstractOperator` `A` as a `LinearMap`. The operator must have matching domain
and codomain element types. The resulting `LinearMap` can be used in any algorithm working with
only vectors, while transparently leveraging the multi-dimensional array support and in-place
implementations of `AbstractOperators`.
"""
LinearMaps.LinearMap(A::AbstractOperator) = LinearMapWrapper(A)

is_linear(L::LinearMapWrapper) = is_linear(L.A)
is_null(L::LinearMapWrapper) = is_null(L.A)
is_eye(L::LinearMapWrapper) = is_eye(L.A)
is_symmetric(L::LinearMapWrapper) = is_symmetric(L.A)
is_diagonal(L::LinearMapWrapper) = is_diagonal(L.A)
is_AcA_diagonal(L::LinearMapWrapper) = is_AcA_diagonal(L.A)
is_AAc_diagonal(L::LinearMapWrapper) = is_AAc_diagonal(L.A)
diag_AcA(L::LinearMapWrapper) = diag_AcA(L.A)
diag_AAc(L::LinearMapWrapper) = diag_AAc(L.A)
is_orthogonal(L::LinearMapWrapper) = is_orthogonal(L.A)
is_invertible(L::LinearMapWrapper) = is_invertible(L.A)
is_full_row_rank(L::LinearMapWrapper) = is_full_row_rank(L.A)
is_full_column_rank(L::LinearMapWrapper) = is_full_column_rank(L.A)
is_positive_definite(L::LinearMapWrapper) = is_positive_definite(L.A)
is_positive_semidefinite(L::LinearMapWrapper) = is_positive_semidefinite(L.A)

Base.show(io::IO, L::LinearMapWrapper) = begin
    print(io, size(L, 1), "×", size(L, 2), " LinearMap{")
    print(io, strip(string(L.A)))
    print(io, "}")
end

end # module LinearMapsExt