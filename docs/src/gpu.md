# GPU Support

AbstractOperators.jl provides GPU compatibility through a lightweight extension model. Most operators work transparently with GPU arrays on any of the four supported backends: CUDA.jl, AMDGPU.jl, oneAPI.jl, and OpenCL.jl.

The main exceptions are:

- `NFFTOperators.jl`, which currently supports GPU execution only with CUDA.jl
- `WaveletOperators.jl`, which currently works on CPU only
- `DCT` and `IDCT` (in FFTWOperators.jl) needs explicit loading of `AcceleratedDCTs` to activate GPU support

## Using GPU Arrays

Simply pass a GPU array to the `mul!` function or the `*` operator:

```julia
using AbstractOperators, CUDA  # or AMDGPU, oneAPI, or OpenCL

x_gpu = CuArray(randn(Float32, 10))
y_gpu = CuArray(zeros(Float32, 9))

F = FiniteDiff(Float32, (10,))  # CPU operator, GPU arrays as input
mul!(y_gpu, F, x_gpu)           # works transparently
```

For operators that carry internal arrays (like `DiagOp`), construct them from GPU arrays to get a GPU-typed operator:

```julia
d_gpu = CuArray(rand(Float32, 10))
D = DiagOp(d_gpu)                # GPU DiagOp — threading disabled automatically
y_gpu = similar(x_gpu)
mul!(y_gpu, D, x_gpu)
```

## Threading and GPU

CPU threading (via [Polyester.jl](https://github.com/JuliaSIMD/Polyester.jl)'s `@batch`) is automatically disabled when GPU arrays are detected. This happens through the `_should_thread` mechanism:

- For operators with array fields (e.g., `DiagOp`), the constructor checks `_should_thread(d)` which returns `false` for `AbstractGPUArray`s.
- For GPU arrays passed to operators at construction time (e.g., `Variation(gpu_x)`), threading is disabled automatically.
- The `GpuExt` extension overrides `_should_thread(::AbstractGPUArray) = false`.

## Storage Type Tracking

Every operator tracks its storage type via `domain_array_type` and `codomain_array_type`. This enables correct intermediate buffer allocation in composed operators:

```julia
using AbstractOperators, CUDA

# Create GPU operators
D = DiagOp(CuArray(rand(Float32, 5)))
domain_array_type(D)     # CuArray{Float32}
codomain_array_type(D)   # CuArray{Float32}

# Composition allocates GPU buffers automatically
F = FiniteDiff(CuArray{Float32}, (10,))  # array_type-aware FiniteDiff
domain_array_type(F)     # CuArray{Float32}
```

For operators without an array at construction time, pass a GPU array to the array-based constructor:

```julia
F = FiniteDiff(CuArray(ones(Float32, 5, 4)), 1)  # deduces storage type from input
V = Variation(CuArray(ones(Float32, 4, 3)))        # threaded=false automatically
```

Or use the `array_type` keyword argument (for operators that support it):

```julia
Z = Zeros(Float32, (4,), Float32, (3,); array_type=CuArray)
```

## GpuExt Extension

GPU-specific overrides live in `ext/GpuExt/`. The extension is loaded automatically when `GPUArrays.jl` is loaded (directly or via any GPU backend like CUDA.jl).

The extension provides:
- `_should_thread(::AbstractGPUArray) = false` — disables CPU threading for GPU arrays
- `array_type_display_string(::Type{<:AbstractGPUArray}) = "ᵍᵖᵘ"` — shows ᵍᵖᵘ superscript in operator display
- Variation adjoint GPU override — vectorized reshape-based stencil (no scalar indexing)
- BroadCast threading GPU override — falls back to non-threaded path

## Testing with JLArrays

The test suite includes GPU tests using [JLArrays.jl](https://github.com/JuliaGPU/JLArrays.jl), a simple CPU-backed array type that mimics GPU behavior (no scalar indexing, GPUArrays interface). GPU tests are tagged `:gpu`:

```bash
julia --project=test -e '
    using TestItemRunner
    @run_package_tests filter=ti -> :gpu in ti.tags
'
```

## Backend-Specific Notes

Most operators in AbstractOperators.jl, DSPOperators.jl, and the GPU-compatible parts of FFTWOperators.jl follow the same generic GPU-array execution model and therefore work with CUDA.jl, AMDGPU.jl, oneAPI.jl, and OpenCL.jl.

### NFFTOperators GPU Support (CUDA only)

NFFTOperators.jl supports GPU via the `array_type` keyword argument. This requires loading CUDA.jl (or any package that loads GPUArrays + Adapt):

```julia
using NFFTOperators, CUDA

image_size = (128, 128)
trajectory = rand(Float32, 2, 128, 50) .- 0.5f0
dcf = rand(Float32, 128, 50)

# Create GPU NFFT operator — ksp_buffer and dcf are CuArrays
op_gpu = NFFTOp(image_size, trajectory, dcf; array_type=CuArray, threaded=false)

x_gpu = CUDA.randn(ComplexF32, image_size)
y_gpu = op_gpu * x_gpu         # forward: image → k-space (on GPU)
img_rec = op_gpu' * y_gpu      # adjoint: k-space → image (on GPU)
```

**Note:** GPU NFFT requires CUDA.jl because it relies on cuFFT plans. JLArrays (used in tests) does not support FFT plans and cannot be used with NFFTOperators GPU.

The trajectory `k` is kept on CPU (as required by NFFT.jl's GPU plan); only the computation buffers (`ksp_buffer`, `dcf`) and input/output arrays are on GPU.

### FFTWOperators DCT/IDCT (AcceleratedDCTs)

`DCT` and `IDCT` are CPU-only unless `AcceleratedDCTs` is loaded. When `AcceleratedDCTs` is available and imported, the package activates GPU support for those operators through its extension:

```julia
using AbstractOperators, FFTWOperators, CUDA
import AcceleratedDCTs # explicitly import to activate GPU support for DCT/IDCT

x_gpu = CUDA.randn(Float32, 64)
dct_op = DCT(x_gpu)
y_gpu = dct_op * x_gpu

idct_op = IDCT(x_gpu)
x_rec_gpu = idct_op * y_gpu
```

If `AcceleratedDCTs` is not imported, `DCT` and `IDCT` continue to use the CPU FFTW implementation,
and will not work with GPU arrays. In that case, wrap them with `CpuOperatorWrapper` to use in GPU pipelines:

```julia
using AbstractOperators, FFTWOperators, CUDA

x_gpu = CUDA.randn(Float32, 64)
dct_op = CpuOperatorWrapper(DCT(Float32, (64,)); array_type = CuArray{Float32})  # CPU operator wrapped for GPU use
y_gpu = dct_op * x_gpu  # GPU in → CPU DCT → GPU out
```

## CpuOperatorWrapper

For operators that do not natively support GPU arrays (e.g., FFTWOperators DCT, custom CPU-only operators), use `CpuOperatorWrapper`. This wrapper preallocates CPU buffers for the operator's domain and codomain, allowing GPU arrays to be passed in and out while the computation happens on CPU.

```@docs
OperatorWrapper
OperatorWrapper(::AbstractOperator)
```

### Example:

```julia
using AbstractOperators, CUDA

# Any CPU operator
op = FiniteDiff(Float32, (64,))  # or any FFTWOperators, etc.

# Wrap it — preallocates CPU buffers for domain and codomain
wrapper = CpuOperatorWrapper(op)

x_gpu = CUDA.randn(Float32, 64)
y_gpu = similar(x_gpu, 63)

mul!(y_gpu, wrapper, x_gpu)   # GPU in → CPU compute → GPU out
mul!(x_gpu, wrapper', y_gpu)  # GPU in → CPU adjoint → GPU out
```

The wrapper preallocates CPU buffers (`dom_buf`, `cod_buf`) to avoid allocations during `mul!`. For parallel use, create independent copies per thread:

```julia
wrappers = [CpuOperatorWrapper(op) for _ in 1:Threads.nthreads()]
```
