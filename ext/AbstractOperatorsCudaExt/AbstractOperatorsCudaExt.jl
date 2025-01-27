module AbstractOperatorsCudaExt

using CUDA
import AbstractOperators: storageTypeDisplayString

storage_type_display_string(::Type{T}) where {T<:CuArray} = "ᶜᵘ"

end # module AbstractOperatorsCudaExt
