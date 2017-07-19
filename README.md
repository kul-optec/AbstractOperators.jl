# AbstractOperators.jl

Abstract operators package for Julia.

## Description

Abstract operators are linear mappings like matrices. Unlike matrices however, abstract operators apply the linear mappings with specific algorithms that minimize the memory requirements while maximizing their efficiency.

> #### Example I: Convolution
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
> julia> sizeof(A)+sizeof(h)
> 8016
>
> julia> sizeof(T)
> 47984000
>```

These are particularly useful in iterative (optimization) algorithms where the direct and adjoint application of linear mappings are needed at every iteration.

> #### Example II: Deconvolution
>
> Deconvolution seeks to find the input signal `x` given the impulse response `h` and the output `y` and can be formulated as the following optimization problem:
> 
> > x* = argmin_x | T*x - y |_2^2               (1)
>
> This is the well-known Least Squares (LS) problem and can be solved using a pseudo-inverse:
>
> ```julia
> julia> x0 = T\y;
> ```
>
> When the size of `T` and `y` are large, solving the LS problem in such a manner can become intractable. 
> A simple alternative is the Gradient Descent algorithm which uses the gradient of the cost function of (1) and solves the optimization problem iteratively:
> 
> > x^k+1 = x^k - gamma* T^t ( T*x^k - y )      (2)
> 
> A _trivial_ (sub-optimal choice of step-size `gamma` and inefficient memory management) implementation of the Gradient Descent in Julia is the following: 
>
> ```julia
> julia> x0, gamma = zeros(x), 1e-4; # initialization, step-size gamma
> 
> julia> for i = 1:4000 x0 =  x0 -1e-4 * A'*(A*x0-y) end;
>
> ```
>
 
In this example it can be seen that the transpose of `A` can be performed using the same syntax of matrices. Nevertheless the transpose still uses a specific efficient algorithm (in this case the transpose of convolution is cross-correlation) with minimal memory requirements. 

Additionally, abstract operators can be applied using _in-place_ functions as `A_mul_B` and `Ac_mul_B`. 


## Installation

To install the package, use the following in the Julia command line:

```julia
Pkg.clone("https://github.com/kul-forbes/AbstractOperators.jl")
```

Remember to `Pkg.update()` to keep the package up to date.
