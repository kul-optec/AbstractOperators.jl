module GpuExt

using GPUArrays
import AbstractOperators: storage_type_display_string

storage_type_display_string(::Type{<:AbstractGPUArray}) = "ᵍᵖᵘ"

end # module GpuExt
