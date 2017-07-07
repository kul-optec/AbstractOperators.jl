# AbstractOperators.jl

Abstract operators package for Julia.

## Description

Abstract operators are linear mappings like matrices. Unlike matrices however, abstract operators apply the linear mappings with specific algorithms that minimize the memory requirements while maximizing their efficiency.

> #### Example: Convolution
>
> Convolution operator can be represented by a [Toeplitz matrix](https://en.wikipedia.org/wiki/Toeplitz_matrix):
> ```julia
> julia> Nh,Nx = 1000,2000; #filter and input taps
>
> julia> h = randn(Nh);   #impulse response
>
> julia> x = randn(Nx);   #input signal
>
> julia> T = hcat([
>            [zeros(i);h;zeros(Nx-1-i)] for i = 0:Nx-1
>                 ]...); #Toeplitz matrix
>
> julia> y = T*x; #convolution
> ```
>  However, it is well known the convolution operation can be efficiently performed using `fft` or `fir`/`iir` filters.
> Abstract operators exploits such operators to avoid unnecessary memory allocations and perform convolution efficiently.
> ```julia
> julia> using AbstractOperators
>
> julia> A = Conv(x,h) # Abstract convolution
> ★ ℝ^200 -> ℝ^299
>
> julia> y = A*x
> ```
> While [matrix multiplication has `O(Nh*Nx)` complexity](https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations#Matrix_algebra), `Conv` utilizes `fft` based algorithm with `O(Nx log Nx)` with evident speed-ups:
> ```julia
>julia> @elapsed A*x
> 0.000819312
> 
> julia> @elapsed T*x
> 0.002508488
> ```
> and reduced memory requirements:
>```julia
> julia> sizeof(A)+sizeof(x)
> 16016
>
> julia> sizeof(T)
> 47984000
>```

These are particularly useful in iterative (optimization) algorithms where the direct and adjoint application of linear mappings are needed at every iteration.

## Installation

To install the package, use the following in the Julia command line

```julia
Pkg.add("AbstractOperators")
```

Remember to `Pkg.update()` to keep the package up to date.
