# Define block-arrays
module BlockArrays

using LinearAlgebra

export RealOrComplex,
       BlockArray,
       blocksize,
       blockeltype,
       blocklength,
       blockvecnorm,
       blockmaxabs,
       blocksimilar,
       blockcopy,
       blockcopy!,
       blockset!,
       blockvecdot,
       blockzeros,
       blockones,
       blockaxpy!,
       blockscale!,
       blockcumscale!,
       blockiszero


const RealOrComplex{R} = Union{R, Complex{R}}
const BlockArray{R} = Union{
	AbstractArray{C, N} where {C <: RealOrComplex{R}, N},
	Tuple{Vararg{AbstractArray{C, N} where {C <: RealOrComplex{R}, N}}}
}

# Operations on block-arrays

blocksize(x::Tuple) = blocksize.(x)
blocksize(x::AbstractArray) = size(x)

blockeltype(x::Tuple) = blockeltype.(x)
blockeltype(x::AbstractArray) = eltype(x)

blocklength(x::Tuple) = sum(blocklength.(x))
blocklength(x::AbstractArray) = length(x)

blockvecnorm(x::Tuple) = sqrt(real(blockvecdot(x, x)))
blockvecnorm(x::AbstractArray{R}) where {R <: Number} = norm(x)

blockmaxabs(x::Tuple) = maximum(blockmaxabs.(x))
blockmaxabs(x::AbstractArray{R}) where {R <: Number}= maximum(abs, x)

blocksimilar(x::Tuple) = blocksimilar.(x)
blocksimilar(x::AbstractArray) = similar(x)

blockcopy(x::Tuple) = blockcopy.(x)
blockcopy(x::AbstractArray) = copy(x)

blockcopy!(y::Tuple, x::Tuple) = blockcopy!.(y, x)
blockcopy!(y::AbstractArray, x::AbstractArray) = copyto!(y, x)

blockset!(y::Tuple, x) = blockset!.(y, x)
blockset!(y::AbstractArray, x) = (y .= x)

blockvecdot(x::T1, y::T2) where {T1 <: Tuple, T2 <: Tuple} = sum(blockvecdot.(x,y))
blockvecdot(x::AbstractArray{R1}, y::AbstractArray{R2}) where {R1 <: Number, R2 <: Number} = dot(x, y)

blockzeros(t::Tuple, s::Tuple) = blockzeros.(t, s)
blockzeros(t::Type, n::NTuple{N, Integer} where {N}) = zeros(t, n)
blockzeros(t::Tuple) = blockzeros.(t)
blockzeros(n::NTuple{N, Integer} where {N}) = zeros(n)
blockzeros(n::Integer) = zeros(n)
blockzeros(a::AbstractArray) = fill!(a,zero(eltype(a)))

blockones(t::Tuple, s::Tuple) = blockones.(t, s)
blockones(t::Type, n::NTuple{N, Integer} where {N}) = ones(t, n)
blockones(t::Tuple) = blockones.(t)
blockones(n::NTuple{N, Integer} where {N}) = ones(n)
blockones(n::Integer) = ones(n)
blockones(a::AbstractArray) = fill!(a,one(eltype(a)))

blockscale!(z::Tuple, alpha::Real, y::Tuple) = blockscale!.(z, alpha, y)
blockscale!(z::AbstractArray, alpha::Real, y::AbstractArray) = (z .= alpha.*y)

blockcumscale!(z::Tuple, alpha::Real, y::Tuple) = blockcumscale!.(z, alpha, y)
blockcumscale!(z::AbstractArray, alpha::Real, y::AbstractArray) = (z .+= alpha.*y)

blockaxpy!(z::Tuple, x::Tuple, alpha::Real, y::Tuple) = blockaxpy!.(z, x, alpha, y)
blockaxpy!(z::AbstractArray, x::AbstractArray, alpha::Real, y::AbstractArray) = (z .= x .+ alpha.*y)

blockiszero(x::AbstractArray) = iszero(x)
blockiszero(x::Tuple) = all(iszero.(x))

# Define broadcast

import Base: broadcast!

function broadcast!(f::Any, dest::Tuple, op1::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], op1[k])
   end
   return dest
end

function broadcast!(f::Any, dest::Tuple, op1::Tuple, op2::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], op1[k], op2[k])
   end
   return dest
end

function broadcast!(f::Any, dest::Tuple, coef::Number, op2::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], coef, op2[k])
   end
   return dest
end

function broadcast!(f::Any, dest::Tuple, op1::Tuple, coef::Number, op2::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], op1[k], coef, op2[k])
   end
   return dest
end

function broadcast!(f::Any, dest::Tuple, op1::Tuple, coef::Number, op2::Tuple, op3::Tuple)
   for k = eachindex(dest)
       broadcast!(f, dest[k], op1[k], coef, op2[k], op3[k])
   end
   return dest
end

end
