# AbstractOperators.jl

[![Build Status](https://travis-ci.org/kul-forbes/AbstractOperators.jl.svg?branch=master)](https://travis-ci.org/kul-forbes/AbstractOperators.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/lfrmkg2s1awyxtk8/branch/master?svg=true)](https://ci.appveyor.com/project/nantonel/abstractoperators-jl/branch/master)
[![codecov](https://codecov.io/gh/kul-forbes/AbstractOperators.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kul-forbes/AbstractOperators.jl)

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://kul-forbes.github.io/AbstractOperators.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kul-forbes.github.io/AbstractOperators.jl/latest)

## Description

Abstract operators extend the syntax typically used for matrices to linear mappings of arbitrary dimensions and nonlinear functions. Unlike matrices however, abstract operators apply the mappings with specific efficient algorithms that minimize memory requirements. 
This is particularly useful in iterative algorithms and in first order large-scale optimization algorithms.

## Installation

To install the package, hit `]` from the Julia command line to enter the package manager, then

```julia
pkg> add AbstractOperators
```

## Usage

With `using AbstractOperators` the package imports several methods like multiplication `*`  and adjoint transposition `'` (and their in-place methods `mul!`).

For example, one can create a 2-D Discrete Fourier Transform as follows:

```julia
julia> A = DFT(3,4)
ℱ  ℝ^(3, 4) -> ℂ^(3, 4)
```
Here, it can be seen that `A` has a domain of dimensions `size(A,2) = (3,4)` and of type `domainType(A) = Float64` and a codomain of dimensions `size(A,1) = (3,4)` and type `codomainType(A) = Complex{Float64}`.

This linear transformation can be evaluated as follows: 

```julia
julia> x = randn(3,4); #input matrix

julia> y = A*x
3×4 Array{Complex{Float64},2}:
  -1.11412+0.0im       3.58654-0.724452im  -9.10125+0.0im       3.58654+0.724452im
 -0.905575+1.98446im  0.441199-0.913338im  0.315788+3.29666im  0.174273+0.318065im
 -0.905575-1.98446im  0.174273-0.318065im  0.315788-3.29666im  0.441199+0.913338im

julia> mul!(y, A, x) == A*x #in-place evaluation
true

julia> all(A'*y - *(size(x)...)*x .< 1e-12) 
true

julia> mul!(x, A',y) #in-place evaluation
3×4 Array{Float64,2}:
  -2.99091   9.45611  -19.799     1.6327 
 -11.1841   11.2365   -26.3614   11.7261 
   5.04815   7.61552   -6.00498   6.25586

```

Notice that inputs and outputs are not necessarily `Vectors`.

It is also possible to combine multiple `AbstractOperators` using different calculus rules. 

For example `AbstractOperators` can be concatenated horizontally: 

```julia
julia> B = Eye(Complex{Float64},(3,4))
I  ℂ^(3, 4) -> ℂ^(3, 4)

julia> H = [A B]
[ℱ,I]  ℝ^(3, 4)  ℂ^(3, 4) -> ℂ^(3, 4)
```

In this case `H` has a domain of dimensions `size(H,2) = ((3, 4), (3, 4))` and type `domainType(H) = (Float64, Complex{Float64})`.

When an `AbstractOperators` have multiple domains, this must be multiplied using a `Tuple`s of `AbstractArray`s with corresponding `size(H,2)` and `domainType(H)`, for example: 

```julia
julia> H*(x, complex(x))
3×4 Array{Complex{Float64},2}:
 -16.3603+0.0im      52.4946-8.69342im  -129.014+0.0im      44.6712+8.69342im
  -22.051+23.8135im  16.5309-10.9601im  -22.5719+39.5599im  13.8174+3.81678im
 -5.81874-23.8135im  9.70679-3.81678im  -2.21552-39.5599im  11.5502+10.9601im
```

Similarly, when an `AbstractOperators` have multiple codomains, this will return a `Tuple` of `AbstractArray`s with corresponding `size(H,1)` and `codomainType(H)`, for example: 
```julia
julia> V = VCAT(Eye(3,3),FiniteDiff((3,3)))
[I;δx]  ℝ^(3, 3) -> ℝ^(3, 3)  ℝ^(2, 3)

julia> V*ones(3,3)
([1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0], [0.0 0.0 0.0; 0.0 0.0 0.0])

```

A list of the available `AbstractOperators` and calculus rules can be found in the [documentation](https://kul-forbes.github.io/AbstractOperators.jl/latest).

## Related packages

* [ProximalOperators.jl](https://github.com/kul-forbes/ProximalOperators.jl)
* [ProximalAlgorithms.jl](https://github.com/kul-forbes/ProximalAlgorithms.jl)
* [StructuredOptimization.jl](https://github.com/kul-forbes/StructuredOptimization.jl)

## Credits

AbstractOperators.jl is developed by
[Niccolò Antonello](https://nantonel.github.io)
and [Lorenzo Stella](https://lostella.github.io)
at [KU Leuven, ESAT/Stadius](https://www.esat.kuleuven.be/stadius/),
