# AbstractOperators.jl

Abstract operators package for Julia.

## Description

Abstract operators generalize matrices by including linear mappings of arrays of any dimensions and nonlinear functions. Unlike matrices however, abstract operators apply the linear mappings with specific algorithms that minimize the memory requirements while maximizing their efficiency. 
This is particularly useful in iterative algorithms and in first order large-scale optimization algorithms.

## Installation

To install the package, use the following in the Julia command line:

```julia
Pkg.clone("https://github.com/kul-forbes/AbstractOperators.jl")
```

Remember to `Pkg.update()` to keep the package up to date.

## Usage

With `using AbstractOperators` the package imports several methods like multiplication `*`  and transposition `'` (and their in-place version `A_mul_B!`).

For example, one can create a 2-D Discrete Fourier Transform as follows:

```julia
julia> A = DFT(3,4)
ℱ  ℝ^(3, 4) -> ℂ^(3, 4)
```

This linear transformation can be evaluated with the same syntax used for matrices: 

```julia
julia> x = randn(3,4); #input matrix

julia> y = A*x
3×4 Array{Complex{Float64},2}:
  -1.11412+0.0im       3.58654-0.724452im  -9.10125+0.0im       3.58654+0.724452im
 -0.905575+1.98446im  0.441199-0.913338im  0.315788+3.29666im  0.174273+0.318065im
 -0.905575-1.98446im  0.174273-0.318065im  0.315788-3.29666im  0.441199+0.913338im

julia> A_mul_B!(y,A,x) == A*x #in-place evaluation
true

julia> all(A'*y - *(size(x)...)*x .< 1e-12) 
true

julia> Ac_mul_B!(x,A,y) #in-place evaluation
3×4 Array{Float64,2}:
  -2.99091   9.45611  -19.799     1.6327 
 -11.1841   11.2365   -26.3614   11.7261 
   5.04815   7.61552   -6.00498   6.25586

```

Notice that inputs and outputs are not necessarily `AbstractVectors`.

It is also possible to combine multiple `AbstractOperators` using different calculus rules. 

For example `AbstractOperators` can be concatenated horizontally: 

```
julia> B = Eye(Complex{Float64},(3,4))
I  ℂ^(3, 4) -> ℂ^(3, 4)

julia> H = [A B]
[ℱ,I]  ℝ^(3, 4)  ℂ^(3, 4) -> ℂ^(3, 4)
```

Evaluation of `AbstractOperators` that have multiple domains is performed using `Tuple`s of `AbstractArray`s, for example: 

```
julia> H*(x, complex(x))
3×4 Array{Complex{Float64},2}:
 -16.3603+0.0im      52.4946-8.69342im  -129.014+0.0im      44.6712+8.69342im
  -22.051+23.8135im  16.5309-10.9601im  -22.5719+39.5599im  13.8174+3.81678im
 -5.81874-23.8135im  9.70679-3.81678im  -2.21552-39.5599im  11.5502+10.9601im
```

## Credits

AbstractOperators.jl is developed by
[Niccolò Antonello](https://nantonel.github.io)
and [Lorenzo Stella](https://lostella.github.io)
at [KU Leuven, ESAT/Stadius](https://www.esat.kuleuven.be/stadius/),
