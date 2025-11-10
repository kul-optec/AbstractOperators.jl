# Custom Operators

This guide explains how to define custom operators in AbstractOperators.jl by extending the necessary interface functions.

## Linear Operators

To create a custom linear operator, define a struct that inherits from `LinearOperator` and implement the mandatory functions.

### Minimal Example

```julia
using AbstractOperators

# Define the operator struct
struct MyCustomLinOp{N,M,D,C} <: LinearOperator
    dim_in::NTuple{M,Int}
    dim_out::NTuple{N,Int}
    # Add any additional fields needed for your operator
end

# Mandatory functions

# 1. size: Return (codomain_size, domain_size)
Base.size(L::MyCustomLinOp) = (L.dim_out, L.dim_in)

# 2. domain_type: Return the element type of the input
AbstractOperators.domain_type(::MyCustomLinOp{N,M,D,C}) where {N,M,D,C} = D

# 3. codomain_type: Return the element type of the output
AbstractOperators.codomain_type(::MyCustomLinOp{N,M,D,C}) where {N,M,D,C} = C

# 4. fun_name: Return a string/symbol for display purposes
AbstractOperators.fun_name(::MyCustomLinOp) = "MyOp"

# 5. mul!: Forward operator (output, operator, input)
function LinearAlgebra.mul!(y::AbstractArray, L::MyCustomLinOp, x::AbstractArray)
    # Utility function to check if
    #   - eltype(x) == domain_type(L)
    #   - eltype(y) == codomain_type(L)
    #   - size(x) == size(L, 2)
    #   - size(y) == size(L, 1)
    #   - x isa domain_storage_type(L)
    #   - y isa codomain_storage_type(L)
    AbstractOperators.check(y, L, x)
    # Implement your forward operation here
    # Example: y .= some_function(x)
    return y
end

# 6. mul! for adjoint: Adjoint operator (output, adjoint_operator, input)
function LinearAlgebra.mul!(y::AbstractArray, L::AdjointOperator{<:MyCustomLinOp}, x::AbstractArray)
    AbstractOperators.check(y, L, x) # Utility function to check types and sizes
    # Implement your adjoint operation here
    # Example: y .= adjoint_function(x)
    return y
end
```

### Complete Example: Diagonal Scaling Operator

Here's a complete example of a custom diagonal operator:

```julia
using AbstractOperators, LinearAlgebra

struct DiagonalScaling{N,D,C} <: LinearOperator
    dim::NTuple{N,Int}
    diagonal::Vector{D}
end

# Constructor
function DiagonalScaling(d::Vector{T}) where {T}
    N = 1
    D = T
    C = T
    return DiagonalScaling{N,D,C}((length(d),), d)
end

# Mandatory interface
Base.size(L::DiagonalScaling) = (L.dim, L.dim)

AbstractOperators.domain_type(::DiagonalScaling{N,D,C}) where {N,D,C} = D
AbstractOperators.codomain_type(::DiagonalScaling{N,D,C}) where {N,D,C} = C

AbstractOperators.fun_name(::DiagonalScaling) = "D"

function LinearAlgebra.mul!(
    y::AbstractArray{C,N}, 
    L::DiagonalScaling{N,D,C}, 
    x::AbstractArray{D,N}
) where {N,D,C}
    y .= L.diagonal .* x
    return y
end

function LinearAlgebra.mul!(
    y::AbstractArray{D,N}, 
    L::AdjointOperator{DiagonalScaling{N,D,C}}, 
    x::AbstractArray{C,N}
) where {N,D,C}
    y .= conj.(L.A.diagonal) .* x
    return y
end

# Usage
d = [1.0, 2.0, 3.0]
D = DiagonalScaling(d)
x = ones(3)
y = D * x  # Returns [1.0, 2.0, 3.0]
```

## Nonlinear Operators

For nonlinear operators, inherit from `AbstractOperator` instead. The interface is similar but you typically don't need to implement the adjoint.

### Minimal Example

```julia
using AbstractOperators

struct MyNonlinearOp{T,N} <: AbstractOperator
    dim::NTuple{N,Int}
    # Add any parameters needed
end

# Mandatory functions

Base.size(L::MyNonlinearOp) = (L.dim, L.dim)

AbstractOperators.domain_type(::MyNonlinearOp{T,N}) where {T,N} = T
AbstractOperators.codomain_type(::MyNonlinearOp{T,N}) where {T,N} = T

AbstractOperators.fun_name(::MyNonlinearOp) = "NonlinOp"

function LinearAlgebra.mul!(
    y::AbstractArray{T,N}, 
    L::MyNonlinearOp{T,N}, 
    x::AbstractArray{T,N}
) where {T,N}
    # Implement your nonlinear function
    y .= sin.(x)  # Example
    return y
end
```

### Jacobian for Nonlinear Operators

If you want to support automatic differentiation with your nonlinear operator, you need to implement the adjoint of the Jacobian:

```julia
function LinearAlgebra.mul!(
    y::AbstractArray, 
    J::AdjointOperator{Jacobian{A,TT}}, 
    b::AbstractArray
) where {T,N,A<:MyNonlinearOp{T,N},TT<:AbstractArray{T,N}}
    L = J.A
    # Implement the Jacobian transpose multiplication
    # L.x contains the point at which the Jacobian is evaluated
    y .= cos.(L.x) .* b  # Example for sin operator
    return y
end
```

## Using MyLinOp for Quick Prototyping

For quick prototyping without defining a new struct, you can use the built-in [`MyLinOp`](@ref) constructor:

```julia
n, m = 5, 4
A = randn(n, m)

# Define operator with just the forward and adjoint functions
op = MyLinOp(
    Float64,           # domain type
    (m,),              # input dimensions
    (n,),              # output dimensions
    (y, x) -> mul!(y, A, x),      # forward function
    (y, x) -> mul!(y, A', x)      # adjoint function
)
```

## Mandatory Functions Summary

All custom operators must implement:

1. **`size(L::YourOperator)`**: Returns a tuple `(codomain_dims, domain_dims)` where each element is itself a tuple of dimensions.

2. **`domain_type(L::YourOperator)`**: Returns the element type (`Float64`, `Complex{Float64}`, etc.) of the operator's input.

3. **`codomain_type(L::YourOperator)`**: Returns the element type of the operator's output.

4. **`fun_name(L::YourOperator)`**: Returns a string used for displaying the operator. Can be a simple name or a Unicode symbol.

5. **`mul!(y, L, x)`**: In-place multiplication implementing the forward operation. Must modify `y` and return it.

6. **`mul!(y, L', x)`** (for `LinearOperator` only): In-place multiplication for the adjoint operator, where `L'` is of type `AdjointOperator{YourOperator}`.

## Optional Properties and Traits

Beyond the mandatory functions, you can optionally define various properties and traits to enable optimizations and provide additional information about your operator. These include:

- **Thread safety**: `is_thread_safe(L::YourOperator) = true`
- **Storage types**: `domain_storage_type`, `codomain_storage_type`
- **Algebraic properties**: `is_diagonal`, `is_symmetric`, `is_invertible`, etc.
- **Operator norm**: `opnorm(L::YourOperator)`
- And many more...

[StructuredOptimization.jl](https://juliafirstorder.github.io/StructuredOptimization.jl/stable/) uses the following properties for optimization, so implementing them is recommended if they apply to your operator:
- [`is_linear`](@ref) -> to identify linear operators (it is already defined for `LinearOperator` types)
- [`is_diagonal`](@ref) -> queried when the operator is used in least squares term
- [`is_AAc_diagonal`](@ref) -> queried when the term involving the operator is checked if it is proximable

For a complete list of available properties and traits, see the [Properties and Traits](@ref) page.

## Tips for Implementation

1. **Type parameters**: Use type parameters in your struct to store compile-time information (dimensions, element types) for better performance.

2. **In-place operations**: Always use in-place operations (`.=`, `mul!`) when possible to minimize memory allocations.

3. **Thread safety**: If your operator's `mul!` implementation doesn't use mutable state (e.g. buffers for intermediate results in mul! stored in the struct), mark it as thread-safe with `is_thread_safe(::YourOperator) = true`.

4. **Consistent types**: Ensure that the types returned by `domain_type` and `codomain_type` match the actual element types used in your `mul!` implementations.

5. **Testing**: Test your operator with various input sizes and types, and verify that the adjoint is correct (for linear operators) by checking that `⟨Ax, y⟩ = ⟨x, A'y⟩`.

## Example: Testing Your Custom Operator

```julia
using Test, LinearAlgebra

# Create your operator
L = MyCustomLinOp(...)

# Test basic functionality
x = randn(size(L, 2)...)
y = L * x
@test size(y) == size(L, 1)
@test eltype(y) == codomain_type(L)

# Test adjoint (for linear operators)
x = randn(domain_type(L), size(L, 2)...)
y = randn(codomain_type(L), size(L, 1)...)
@test dot(L * x, y) ≈ dot(x, L' * y)

# Test in-place operations
y_out = similar(y)
mul!(y_out, L, x)
@test y_out ≈ L * x
```
