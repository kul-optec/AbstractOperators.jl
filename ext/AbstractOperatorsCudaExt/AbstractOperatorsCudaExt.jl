module AbstractOperatorsCudaExt

using CUDA
import AbstractOperators: storageTypeDisplayString

storageTypeDisplayString(::Type{T}) where {T<:CuArray} = "ᶜᵘ"

end # module AbstractOperatorsCudaExt
