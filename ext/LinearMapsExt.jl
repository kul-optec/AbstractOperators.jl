module LinearMapsExt

using LinearMaps
using AbstractOperators

function LinearMaps.LinearMap(A::AbstractOperator)
    if domain_type(A) != codomain_type(A)
        error("LinearMapsExt.LinearMap only supports operators with matching domain and codomain types")
    end
    T = domain_type(A)
    At = adjoint(A)
    f = (y, x) -> mul!(reshape(y, size(A, 1)), A, reshape(x, size(A, 2)))
    fc = (y, x) -> mul!(reshape(y, size(At, 1)), At, reshape(x, size(At, 2)))
    ishermitian = is_symmetric(A)
    issymmetric = ishermitian && T <: Real
    return LinearMaps.FunctionMap{T,true}(f, fc, prod(size(A, 1)), prod(size(A, 2)); issymmetric, ishermitian)
end

end # module LinearMapsExt