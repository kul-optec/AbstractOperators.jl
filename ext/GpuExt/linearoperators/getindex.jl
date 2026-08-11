function _to_gpu_indices(ref_array::AbstractGPUArray, cpu_idx::AbstractVector{<:Integer})
    ArrayT = Base.typename(typeof(ref_array)).wrapper
    return ArrayT(Vector{Int}(cpu_idx))
end

function _to_gpu_indices(array_type::Type, cpu_idx::AbstractVector{<:Integer})
    ArrayT = Base.typename(array_type).wrapper
    return ArrayT(Vector{Int}(cpu_idx))
end

function _mask_to_linear_indices(mask::AbstractArray{Bool})
    return findall(vec(mask))
end

function _mask_to_linear_indices(mask::AbstractGPUArray{Bool})
    return findall(vec(Array(mask)))
end

function AbstractOperators._prepare_getindex_intvec(
        idx::Vector{Int}, array_type::Type{<:AbstractGPUArray}
    )
    return _to_gpu_indices(array_type, idx)
end

function AbstractOperators._prepare_getindex_boolmask(
        mask::AbstractArray{Bool}, array_type::Type{<:AbstractGPUArray}
    )
    return _to_gpu_indices(array_type, _mask_to_linear_indices(mask))
end

function GetIndex(x::AbstractGPUArray, idx::AbstractVector{Int})
    dim_in = size(x)
    dim_out = AbstractOperators.get_dim_out(dim_in, idx)
    if dim_out == dim_in
        return AbstractOperators.Eye(eltype(x), dim_in; array_type = typeof(x))
    end
    S = AbstractOperators._array_wrapper(x){eltype(x)}
    gpu_idx = _to_gpu_indices(x, idx)
    return AbstractOperators.GetIndex(eltype(x), S, dim_out, dim_in, gpu_idx)
end

function GetIndex(x::AbstractGPUArray, mask::AbstractArray{Bool})
    dim_in = size(x)
    dim_out = AbstractOperators.get_dim_out(dim_in, mask)
    if dim_out[1] == prod(dim_in)
        return reshape(AbstractOperators.Eye(eltype(x), dim_in; array_type = typeof(x)), dim_out)
    end
    S = AbstractOperators._array_wrapper(x){eltype(x)}
    gpu_idx = _to_gpu_indices(x, _mask_to_linear_indices(mask))
    return AbstractOperators.GetIndex(eltype(x), S, dim_out, dim_in, gpu_idx)
end

function mul!(
        y::AbstractGPUArray, L::GetIndex{I}, b::AbstractGPUArray
    ) where {K, I <: NTuple{K, Any}}
    check(y, L, b)
    y .= view(b, L.idx...)
    return y
end

function mul!(
        y::AbstractGPUArray, Lc::AdjointOperator{<:GetIndex{I}}, b::AbstractGPUArray
    ) where {K, I <: NTuple{K, Any}}
    check(y, Lc, b)
    fill!(y, zero(eltype(y)))
    view(y, Lc.A.idx...) .= b
    return y
end
