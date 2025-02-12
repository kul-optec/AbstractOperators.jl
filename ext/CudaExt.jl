module CudaExt

using CUDA
import AbstractOperators: storage_type_display_string

storage_type_display_string(::Type{T}) where {T<:CuArray} = "ᶜᵘ"

end # module CudaExt
