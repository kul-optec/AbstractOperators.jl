# ContourletOperators.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://kul-optec.github.io/AbstractOperators.jl/stable/operators/#Contourlet)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kul-optec.github.io/AbstractOperators.jl/latest/operators/#Contourlet)

Contourlet transform operators for the AbstractOperators.jl framework.

## Overview

ContourletOperators.jl is a specialized extension package for [AbstractOperators.jl](../README.md) that provides linear operators for the (Nonsubsampled) Contourlet Transform. It wraps [Contourlets.jl](https://github.com/hakkelt/Contourlets.jl) to offer the discrete Contourlet Transform (CT, nearly critically sampled) and the Nonsubsampled Contourlet Transform (NSCT, fully shift-invariant) as seamlessly integrated `LinearOperator` instances.

## Relationship to AbstractOperators.jl

ContourletOperators.jl is a **subpackage** of the AbstractOperators.jl ecosystem. While AbstractOperators.jl provides the core abstract operator framework, ContourletOperators.jl extends it with domain-specific functionality for directional multiresolution image analysis. This modular design allows users to access advanced contourlet capabilities only when needed.

## Installation

```julia
pkg> add ContourletOperators
```

## GPU Support

`ContourletOp`/`NSCTOp` accept GPU input/output via the `array_type` constructor keyword, e.g.
`ContourletOp(params, dim_in; array_type = CuArray{Float64})`. The underlying Contourlet Transform
itself still runs on the CPU (FFTW plans, scalar filter-bank loops) — this package does not yet
call into Contourlets.jl's own CUDA/generic GPU extensions — so `mul!` stages `x`/`y` through an
internal CPU buffer (`copyto!` in, run the CPU transform, `copyto!` out) rather than executing the
transform on-device.

## Usage Example

```julia
using ContourletOperators

# Contourlet decomposition parameters: 3 pyramid levels,
# with a parabolically increasing number of directions per level.
params = ContourletParams(J = 3, L_array = parabolic_levels(3))

# Create an input image
img = randn(256, 256)

# Discrete Contourlet Transform (nearly critically sampled)
C = ContourletOp(params, size(img))
coeffs = C * img          # ArrayPartition: coarse band + directional subbands
img_rec = C' * coeffs     # inverse Contourlet Transform

# Nonsubsampled Contourlet Transform (shift-invariant)
N = NSCTOp(params, size(img))
nsct_coeffs = N * img
img_rec2 = N' * nsct_coeffs
```

## Main Features

### 1. **ContourletOp** - Discrete Contourlet Transform Operator
Computes the nearly critically sampled Contourlet decomposition: a Laplacian Pyramid stage followed by a Directional Filter Bank at each pyramid level.

- **Multiscale + multidirectional**: `J` pyramid levels, each further split into `2^L_array[j]` directional subbands
- **Ragged subbands**: subband sizes shrink from scale to scale, following the pyramid decimation
- **Invertible operation**: forward CT and adjoint (inverse CT) for analysis and synthesis

### 2. **NSCTOp** - Nonsubsampled Contourlet Transform Operator
Computes the fully shift-invariant Contourlet decomposition: no downsampling at any stage, so every subband has the same spatial size as the input.

- **Shift-invariant**: robust to circular shifts of the input, useful for denoising and detection
- **Uniform subbands**: all subbands share the input's spatial size
- **Invertible operation**: forward NSCT and adjoint (inverse NSCT)

Both operators return their coefficients as a flat `RecursiveArrayTools.ArrayPartition` (coarse band followed by each directional subband, scale by scale), following the same multi-component codomain convention used elsewhere in AbstractOperators.jl (e.g. `HCAT`/`DCAT`).

## License

See LICENSE.md for details.
