# Compile-time ndoms from operator type, for use in @generated constructors.
# Specializations for HCAT/VCAT/DCAT are added in their respective files.
_ndoms_from_type(::Type{<:AbstractOperator}, dim::Int) = 1

_storage_parent(a) = a
_storage_parent(a::SubArray) = _storage_parent(parent(a))
_storage_parent(a::Base.ReshapedArray) = _storage_parent(parent(a))

_is_storage_compatible(a, ::Type{<:Array}) = _storage_parent(a) isa Array
_is_storage_compatible(a, ::Type{T}) where {T} = _storage_parent(a) isa T

@generated function _is_storage_compatible(a::ArrayPartition, ::Type{T}) where {T <: ArrayPartition}
    Tp = T.parameters[2]
    if !(Tp isa DataType && Tp <: Tuple)
        return :(a isa T)
    end
    types = Tp.parameters
    checks = [:(a.x[$i] isa $(types[i])) for i in 1:length(types)]
    cond = isempty(checks) ? :(true) : reduce((x, y) -> :($x && $y), checks)
    return :(length(a.x) == $(length(types)) && $cond)
end

function _check_domain_storage(domain_array, op)
    if !_is_storage_compatible(domain_array, domain_array_type(op))
        throw(
            ArgumentError(
                "Input storage type $(typeof(domain_array)) is not compatible with " *
                    "operator's expected domain storage $(domain_array_type(op))",
            ),
        )
    end
    return
end

function _check_codomain_storage(codomain_array, op)
    if !_is_storage_compatible(codomain_array, codomain_array_type(op))
        throw(
            ArgumentError(
                "Output storage type $(typeof(codomain_array)) is not compatible with " *
                    "operator's expected codomain storage $(codomain_array_type(op))",
            ),
        )
    end
    return
end

function check(codomain_array, op, domain_array)
    _check_domain_storage(domain_array, op)
    _check_codomain_storage(codomain_array, op)
    if (ndoms(op, 2) > 1) != (domain_array isa ArrayPartition)
        throw(ArgumentError("Input must be an ArrayPartition if and only if operator has multiple input domains"))
    end
    if domain_array isa ArrayPartition
        dtype = eltype.(domain_array.x)
    else
        dtype = eltype(domain_array)
    end
    # Use isequal instead of != to avoid Union{Missing,Bool} from tuple comparisons
    if !isequal(dtype, domain_type(op))
        throw(
            ArgumentError(
                "Input type $(dtype) does not match operator input type $(domain_type(op))",
            ),
        )
    end
    dim_in = domain_array isa ArrayPartition ? size.(domain_array.x) : size(domain_array)
    if !isequal(dim_in, size(op, 2))
        throw(
            DimensionMismatch(
                "Input size $(dim_in) does not match operator input size $(size(op, 2))",
            ),
        )
    end
    if (ndoms(op, 1) > 1) != (codomain_array isa ArrayPartition)
        throw(
            ArgumentError(
                "Output must be an ArrayPartition if and only if operator has multiple output domains",
            ),
        )
    end
    if codomain_array isa ArrayPartition
        dtype = eltype.(codomain_array.x)
    else
        dtype = eltype(codomain_array)
    end
    if !isequal(dtype, codomain_type(op))
        throw(
            ArgumentError(
                "Output type $(dtype) does not match operator output type $(codomain_type(op))",
            ),
        )
    end
    dim_out = codomain_array isa ArrayPartition ? size.(codomain_array.x) : size(codomain_array)
    if !isequal(dim_out, size(op, 1))
        throw(
            DimensionMismatch(
                "Output size $(dim_out) does not match operator output size $(size(op, 1))",
            ),
        )
    end
    return nothing
end
