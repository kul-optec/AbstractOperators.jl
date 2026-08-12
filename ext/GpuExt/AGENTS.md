# GPU Implementation

- Julia package extensions can only `import` the parent package, trigger package(s), and
  stdlib; if extension code needs a parent dependency API, expose it from the parent module
  first.
- Override `mul!` in `ext/GpuExt/` for any operator whose base implementation uses scalar
  indexing loops (`@nloops`, `@nref`, `@inbounds y[i] = b[j]`); replace with broadcast-over-view
  (`y .= view(b, idx...)`).
- When overriding a threaded operator (e.g. `Variation{..., true}`) for GPU, delegate to the
  non-threaded variant (`Variation{..., false}`) — threading strategy is CPU-only.
- For FFT plans, prefer `inv(plan)` (AbstractFFTs-generic) over backend-specific
  `FFTW.plan_inv(...)` to keep CUDA/AMDGPU compatibility.
- With JLArrays/GPUArrays, avoid `copyto!(gpu, cpu_view)` where the source is a `SubArray`;
  materialize first (e.g. `src[1:n]`), or copy from a plain array.
- Keep CPU-only implementation details out of GPU overrides unless the backend truly supports
  them.
- For GPU `GetIndex` overrides, keep boolean-mask and integer-vector fancy indexing in CPU
  paths unless the backend support is verified.
- Prefer direct `CuArray(arr)` / `CUDA.zeros(...)` / `AMDGPU.ROCArray(arr)` / `AMDGPU.zeros(...)`
  calls over intermediate conversion variables.
