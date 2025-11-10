# Abstract Operators

# Linear operators

## Basic Operators 

```@docs
Eye
Zeros
DiagOp
GetIndex
MatrixOp
LMatrixOp
MyLinOp
```

## Finite Differences

```@docs
FiniteDiff
Variation
```

## Optimization

```@docs
LBFGS
reset!
update!
```

# Nonlinear operators

## Basic

```@docs
Exp
Pow
Cos
Sin
Atan
Tanh
Sech
```

## Sigmoids

```@docs
Sigmoid
SoftPlus
SoftMax
```

# Subpackages

To keep package loading fast, functionalities requiring extra dependencies are separated to subpackages. These have to be added separately (e.g. `Test.add("FFTOperators")`) to access their operators.

## FFTW

!!! note
    Add package `FFTWOperators` to access the following operators.

```@docs
DFT
IDFT
RDFT
IRDFT
DCT
IDCT
FFTShift
IFFTShift
SignAlternation
fftshift_op
ifftshift_op
alternate_sign
alternate_sign!
```

## Convolution

!!! note
    Add package `DSPOperators` to access the following operators.

```@docs
Conv
Xcorr
Filt
MIMOFilt
ZeroPad
```

## NFFT

!!! note
    Add package `NFFTOperators` to access the following operators.

```@docs
NFFTOp
```

## Wavelet

!!! note
    Add package `WaveletOperators` to access the following operators.

```@docs
WaveletOp
```
