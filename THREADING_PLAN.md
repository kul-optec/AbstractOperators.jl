# Unified Threading Plan for AbstractOperators.jl

Status: implemented (Phases 0-6). Branch: `nested-threading`. See the Execution record below.

## Goals

1. Every operator constructor accepts a `threaded` keyword; every `mul!` and adjoint `mul!`
   uses multithreading accordingly.
2. `copy_operator` is implemented for all operators and produces a threaded/non-threaded copy
   satisfying a requested storage type. A new `adapt_operator` avoids the copy when the input
   already matches the constraints.
3. Batching operators properly disable threading in the wrapped operators.
4. Calculus operators that benefit from multithreading are identified and implemented.

---

## 1. Current state (why this is needed)

Threading is expressed three different ways today, and coverage is sparse:

| Encoding | Operators |
|---|---|
| `Bool` type param | `Variation{T,N,Th,S}`, `AbstractBroadCast{T,N,M,Threaded}`, `SignAlternation` |
| `FastBroadcast.True/False` type param | `DiagOp{B,...}`, `Scale{Th,...}` |
| Separate struct types | `SimpleBatchOpSingleThreaded` / `SimpleBatchOpMultiThreaded`, `SpreadingBatchOp*` |
| **None** | all 10 nonlinear ops, `Eye`, `MatrixOp`, `LMatrixOp`, `GetIndex`, `ZeroPad`, `Zeros`, `FiniteDiff`, `LBFGS`, `MyLinOp`, `HCAT`, `VCAT`, `DCAT`, `Compose`, `Sum`, `AffineAdd`, `Reshape`, `Jacobian`, `AdjointOperator`, `Ax_mul_Bx`/`Ax_mul_Bxt`/`Axt_mul_Bx`, and all four subpackages except `Shift.jl` |

Threshold policy is scattered and inconsistent:

- `_should_thread` uses `length > 2^16` (`src/properties.jl:523`)
- `BroadCast` re-checks `prod(dim_in)*sizeof(T) > 2^16` (`src/calculus/BroadCast.jl:14`)
- `Scale` uses `get_output_length(L) > 1e4` (`src/calculus/Scale.jl:54`)
- `Variation` uses its own `2^16` (`src/linearoperators/Variation.jl:37`)

`_copy_operator_impl` is specialized in only 10 files; the fallback (`src/properties.jl:508`)
is `deepcopy(op)` which **silently ignores both `storage_type` and `threaded`** — a latent
correctness bug, not just missing coverage.

Two further defects to fix along the way:

- `create_BatchOp` (`src/batching/SimpleBatchOp.jl:154`) copies the wrapped operator
  **without** `threaded=false`, so a threaded batch of threaded inner operators nests Julia
  threads. Same at `src/batching/SpreadingBatchOp.jl:369` and `src/calculus/BroadCast.jl:40`.
- `FiniteDiff` `mul!` allocates (`b[idx_1] .- b[idx_2]`,
  `src/linearoperators/FiniteDiff.jl:69`); needs `@views` before it gets threaded.

---

## Phase 0 — Test runner: activate `gpu_env` only when needed — **DONE**

**Outcome.** Implemented as described, with two corrections found during execution (details
inline below): the `NEEDS_GPU` formula is computed from the *selected testitems* rather than
from the filter string, and `:gpu` testitems keep their inline `using GPUEnv`. Measured on a
single non-GPU testitem (`"Syntax: Adjoint"`): **3m20s → 8.7s**. `:jet` (224 pass) and
`:misc,:quality` (185 pass) now run with no GPU activation at all; `:calculus` (1557 pass)
still activates, correctly — see below.


**Problem.** `test/utils.jl:22-23` unconditionally runs `GPUEnv.activate(; persist=true)`
inside the `TestUtils` `@testmodule`, which is the `setup` of nearly every testitem. Any
filtered run (`:calculus`, `:jet`, a single testitem name) pays full GPU env activation.

**Design (two parts, both needed):**

1. **Split the setup module.** Move `using GPUEnv; GPUEnv.activate(; persist=true)` (and the
   `AcceleratedDCTs` install, which is FFTW/DCT-specific) out of `TestUtils` into a new
   `@testmodule GpuEnvSetup` in `test/gpu_env_setup.jl`. TestItemRunner evaluates setup
   modules lazily, only for testitems that actually run, so activation happens iff a GPU
   testitem is selected. Every `:gpu`-tagged testitem switches to
   `setup = [TestUtils, GpuEnvSetup]`. Testitems that currently do `using GPUEnv` inline drop
   that in favour of the setup module:
   - `test/test_nonlinear_operators.jl` (10 testitems, lines 142–250)
   - `test/batching/test_SimpleBatchOp.jl:231`
   - `test/batching/test_SpreadingBatchOp.jl:310`
   - `test/linearoperators/test_finitediff.jl:90`

   Part of this phase is auditing that every testitem touching a GPU backend actually carries
   the `:gpu` tag — the split makes an untagged one fail loudly rather than silently work.

2. **`runtests.jl` refactor.** Compute GPU need and expose it, so the setup module and any
   conditional code agree on one answer.

   The originally proposed formula — `:gpu in FILTER_TAGS || !isempty(FILTER_NAMES)` — is
   **wrong**: GPU testitems carry their category tags too (`Eye (GPU)` is
   `[:gpu, :linearoperator, :Eye]`), so a `:linearoperator` run selects GPU items while the
   formula says `false`, and every name filter degrades to "maybe". Instead the filter
   closure sets the flag from the items it actually accepts, which is exact:

   ```julia
   ENV["ABSTRACTOPERATORS_TEST_GPU"] = "false"   # before @run_package_tests
   # inside the filter closure, for each accepted item:
   :gpu in ti.tags && (ENV["ABSTRACTOPERATORS_TEST_GPU"] = "true")
   ```

   `GpuEnvSetup` reads the env var and errors with a clear message if it is `false` — that
   means a testitem *without* the `:gpu` tag pulled in the setup module, i.e. a tagging bug
   worth surfacing rather than papering over. The filter closure keeps the existing `VERB`
   per-item printing.

**Caveat verified.** `TestItemRunner/src/TestItemRunner.jl:191-196` applies the filter to
every discovered testitem before the first `ensure_evaled` setup evaluation at line 217, all
in one process — so the env-var handshake is well-ordered and part 2 stands.

**Two audit results.**

- `:gpu` testitems keep their inline `using GPUEnv`. That import is just a name binding and
  costs nothing; only `GPUEnv.activate` is expensive, and that is what moved. Re-exporting
  GPUEnv's helpers from `GpuEnvSetup` would have been extra indirection for no gain.
- `test/test_gpu_quality.jl`'s two items (`GpuExt Quality`, `GpuExt JET`) are tagged `:gpu`
  but exercise the extension through **JLArrays only**, a direct `test/Project.toml` dep.
  They deliberately do *not* use `GpuEnvSetup`; verified by `:jet` and `:misc,:quality` runs
  passing with zero activation.

**Validation.** `:calculus` is not a valid before/after case — it selects `*(GPU)` items by
their `:calculus` tag, so it still activates, correctly. The win shows on filters that select
no `:gpu` item: a single non-GPU testitem went 3m20s → 8.7s.

---

## Execution record (Phases 1-6)

Status: implemented. Commits: 9a5a12a (Phase 0), f8025eb (Phases 1-4), f2a3e62 (Phase 5),
plus Phase 6. What follows records where reality differed from the plan; the original
phase text is kept below for reference.

### Thresholds became measurements

`benchmark/threading_sweep.jl` was built first, as the plan intended, and every constant in
`src/threading_policy.jl` now carries a PROVENANCE line. Measured on AMD EPYC 7352,
8 threads, `OPENBLAS_NUM_THREADS=1`:

| constant | plan guess | measured |
|---|---|---|
| `THRESHOLD_ELEMENTWISE_TRANSCENDENTAL` | 2^13 | **2^10** |
| `THRESHOLD_ELEMENTWISE_ARITHMETIC` | (2^16 implied) | **2^15** |
| `THRESHOLD_MEMORY_BOUND` | 2^18 | **2^18** (kernel changed, see below) |
| `THRESHOLD_BLOCK_PARALLEL` | 2^16 | **2^18** + `MIN_BLOCKS_FOR_PARALLEL = 4` |

### Where the plan's hypotheses were wrong

1. **FastBroadcast does not win or tie everywhere.** For a pure `y = x` copy, `@.. thread =
   true` is *time-identical to serial at every swept size* — FastBroadcast does not thread a
   plain copy. `@batch` does, crossing over at 2^15. So memory-bound operators are marked
   `@batch`, not FastBroadcast.

2. **Aggregate work does not predict block parallelism.** 2 blocks x 2^18 is a 2^19
   aggregate but measures 1.02x, while the same 2^19 across 8 blocks measures 1.9x. Hence
   the second constant `MIN_BLOCKS_FOR_PARALLEL`, which the plan did not anticipate.

3. **`require_thread_safe` cannot do what the plan asked of it.** The plan has all three
   nesting sites call `adapt_operator(op; threaded=false, require_thread_safe=true)`. But
   thread safety is a property of an operator's *type* — copying an operator that owns
   scratch buffers yields another operator that owns scratch buffers. So the flag can only
   withhold the sharing fast path, never manufacture safety. The nesting sites instead
   branch on `is_thread_safe` themselves (`_per_thread_operators`), and the kwarg's
   docstring states the limitation.

4. **`is_threaded` on forwarders is load-bearing, not cosmetic.** Without it a forwarder
   inherits the `false` default, so `adapt_operator(op; threaded=false)` reports the
   constraint already satisfied and returns an operator whose children are still threaded —
   a silent nesting bug. Caught by the batching tests.

5. **A third trait was needed.** `threaded = true` is a *permission* ("thread where you
   can") while `threaded = false` is a *demand* (nesting safety). Without
   `supports_threading` to tell "can thread, currently isn't" from "has no threaded path",
   passing `threaded = true` down a `Compose` to a memory-bound `Eye` copies forever without
   ever satisfying the constraint — and, worse, the strict `copy_operator` fallback made
   batching *any* FFTW/DSP operator raise.

### Copy semantics

The plan's test criterion "`copy_operator` never returns `===` its input" is **not
assertable**: most operators are immutable structs holding only dimensions, and Julia makes
two such structs with identical fields `===` regardless of construction. The tests assert
the observable contract instead — mutable buffers are never shared, constraints are
honoured — which is also why such checks were removed in 4839a08.

### Not done

- **VCAT forward / HCAT adjoint** (Phase 5). Only `DCAT` was measured and shipped. The plan's
  own rule is that unmeasured variants are dropped, and these need their own sweep.
- **`DCT`/`RDFT`/`IRDFT` plan-time thread counts** (Phase 6). Only `DFT`/`IDFT` gained the
  `threaded` keyword; the other three FFTW transforms have no `num_threads` plumbing at all
  today, so they are declared `supports_threading = false` rather than half-wired.
- `Scale`/`DiagOp` keep their `FastBroadcast` singleton parameters, bridged by
  `_fbthread`/`_fbbool` as the plan specified.
- **The four legacy thresholds are not yet consolidated.** The plan says all four move into
  `src/threading_policy.jl` and get re-derived by benchmark. Operators added or reworked in
  Phases 3-5 do go through `threading_threshold`/`default_threaded`, but these four remain
  where they were and still gate the operators that predate this work:

  | site | expression |
  |---|---|
  | `src/properties.jl:550` | `_should_thread(d) = length(d) > 2^16` |
  | `src/calculus/BroadCast.jl:14` | `prod(dim_in) * sizeof(T) > 2^16` |
  | `src/calculus/Scale.jl:54` | `get_output_length(L) > 1e4` |
  | `src/linearoperators/Variation.jl:37` | `prod(dim_in) * sizeof(T) > 2^16` |

  `_should_thread` in particular is still what `create_BatchOp` and `create_BatchOp`'s
  spreading twin consult to decide whether to thread at all, so moving it is a behaviour
  change to the batch operators and wants its own benchmark + test pass rather than being
  folded into this one.

---

## Phase 1 — Threading infrastructure + benchmark harness

**API** (`src/properties.jl` + new `src/threading_policy.jl`):

```julia
is_threaded(op::AbstractOperator)::Bool                      # trait, default false
default_threaded(::Type{Op}, ::Type{T}, dims, ::Type{S})     # PER-OPERATOR policy
adapt_operator(op; storage_type=nothing, threaded=nothing, require_thread_safe=false)
```

### Per-operator thresholds

No single global constant. `default_threaded` dispatches on the operator type, and every
threshold lives in `src/threading_policy.jl` as a named constant with a comment recording the
machine, thread count, and benchmark that produced it:

```julia
# Sin/Cos/Exp/…: transcendental, compute-bound → threads pay off early
const THRESHOLD_ELEMENTWISE_TRANSCENDENTAL = 2^13   # elements, 8 threads, benchmarked <date>
# Eye/ZeroPad/GetIndex: pure memcpy, memory-bandwidth-bound → threads rarely pay off
const THRESHOLD_MEMORY_BOUND = 2^18
```

Operators with genuinely similar cost profiles share a constant; operators that differ get
their own. The four scattered thresholds listed in §1 all move here and get re-derived by
benchmark rather than inherited.

### Kernel selection is benchmarked, not assumed

For each operator the candidates are: plain broadcast (serial), `@..` with
`thread=FastBroadcast.True()`, `@budgeted_batch` (Polyester), and `@budgeted_threads`.

Rule of thumb applied as the *starting hypothesis*, then confirmed by measurement:

- `@batch` / FastBroadcast when the kernel touches elements directly;
- `@threads` when the body is a higher-level call (a nested `mul!`, a BLAS/FFT call, a block).

FastBroadcast is preferred when it wins or ties, since it also inherits NestedThreading's
Polyester guarding for free — but it does not win automatically, and a benchmark showing
serial broadcast ahead is a valid result that leaves the operator unthreaded.

### Benchmark harness

`benchmark/threading_sweep.jl`, outputs to `.temp/`: for a given operator, sweep size ×
element type × thread count × kernel variant using `BenchmarkTools`, with `Random.seed!(0)`
and `OPENBLAS_NUM_THREADS=1`, reporting both timing and allocations. Emits the crossover size
per variant so each constant in `threading_policy.jl` is a transcription of a measurement.
This harness is a prerequisite for Phases 3, 5 and 6 — building it first is what makes the
rest cheap.

### Encoding

`Th` is a trailing `Bool` **type parameter** on every threading-capable operator; `threaded`
is a **keyword argument** on every constructor (including the `x::AbstractArray` convenience
forms). `DiagOp`/`Scale` keep their `FastBroadcast` singleton params internally and gain
`is_threaded`; bridge with

```julia
@inline _fbthread(::Val{true})  = FastBroadcast.True()
@inline _fbthread(::Val{false}) = FastBroadcast.False()
```

### `copy_operator` vs `adapt_operator`

- `copy_operator` always returns a new object — immutable data shared, mutable buffers fresh
  (existing convention), traits set as requested.
- `adapt_operator` returns `op` itself (`===`) when it already satisfies the requested
  `storage_type`/`threaded` (and `is_thread_safe` if `require_thread_safe`), else delegates to
  `copy_operator`.
- The current short-circuit at `src/properties.jl:501` moves from `copy_operator` into
  `adapt_operator`.
- The `deepcopy` fallback at `src/properties.jl:508` becomes: `deepcopy` only when both
  `storage_type` and `threaded` are `nothing`, otherwise `throw(ArgumentError(...))` naming
  the operator type.

---

## Phase 2 — Nesting safety in batching (bug fix, independently shippable)

`create_BatchOp` (`src/batching/SimpleBatchOp.jl:154`), `create_threaded_SpreadingBatchOp`
(`src/batching/SpreadingBatchOp.jl:369`) and `OperatorBroadCast` (`src/calculus/BroadCast.jl:40`)
copy the wrapped operator **without** `threaded=false`, so a threaded batch over threaded
inner operators nests Julia threads. All three become
`adapt_operator(op; threaded=false, require_thread_safe=true)`.

Belt and braces: dynamically, every operator-internal parallel loop uses
`@budgeted_threads`/`@budgeted_batch`, and FastBroadcast's `thread=true` already routes
through Polyester, which NestedThreading guards — so nested `@..` self-suppresses. Raw
`Threads.@threads` does not, which is precisely today's hole. Statically, `threaded=false`
lets the inner operator skip its threaded path entirely, avoiding per-call budget lookups.

Also: `_copy_operator_impl` for all three batch-op families (none have one today).

---

## Phase 3 — Elementwise operators

All 10 nonlinear operators and their Jacobian adjoints, `AffineAdd`, `HadamardProd`,
`Ax_mul_Bx` / `Ax_mul_Bxt` / `Axt_mul_Bx`, `FiniteDiff`. Each gets: `threaded` keyword, `Th`
type param, `is_threaded`, threaded forward and adjoint `mul!`, `_copy_operator_impl`, and a
benchmarked kernel choice + threshold from the Phase 1 harness (hypothesis: FastBroadcast or
`@budgeted_batch`).

Also fix `FiniteDiff` `mul!` (`src/linearoperators/FiniteDiff.jl:69`): `b[idx_1] .- b[idx_2]`
allocates two temporaries — `@views` first, then thread.

`LBFGS` (stateful) and `SoftMax` (reduction + shared buffer) stay unthreaded with explicit
`is_threaded(op) = false`.

---

## Phase 4 — Forwarders and total `copy_operator` coverage

`threaded` keyword, forwarding to children, and `_copy_operator_impl` for `Compose`, `Sum`,
`HCAT`, `VCAT`, `DCAT`, `Reshape`, `AffineAdd`, `AdjointOperator`, `Jacobian`, plus the
missing simple linear operators (`Eye`, `MatrixOp`, `LMatrixOp`, `GetIndex`, `ZeroPad`,
`Zeros`, `MyLinOp`, `LBFGS`). After this, `copy_operator`/`adapt_operator` are total over
in-tree operators and the Phase 1 fallback error is unreachable for them.

`MatrixOp`/`LMatrixOp` are BLAS-backed and get **no** Julia threads: `threaded=false` wraps
`mul!` in `with_restricted_threads`; `threaded=true` leaves BLAS alone. Memory-bound
`Eye`/`ZeroPad`/`GetIndex`/`Zeros` get the keyword for uniformity, with
`THRESHOLD_MEMORY_BOUND` high enough that `default_threaded` normally returns `false`.

---

## Phase 5 — Block-parallel calculus

Opt-in `threaded=true` (default `false`) with per-block
`adapt_operator(...; threaded=false, require_thread_safe=true)` at construction, using
`@budgeted_threads` (higher-level bodies → `@threads`):

- **`DCAT`** — best candidate: blocks fully independent in both directions, no accumulation,
  no shared output.
- **`VCAT` forward** and **`HCAT` adjoint** — independent per-block outputs.

Not threaded (accumulate into a shared output; children already threaded): `HCAT` forward,
`VCAT` adjoint, `Sum`. `Compose` is inherently sequential — the flag is pure pass-through.

Benchmarked with the Phase 1 harness across block count × block size × block heterogeneity;
the threshold here is on *aggregate block work*, not array length. Variants that do not show a
win are dropped rather than shipped.

---

## Phase 6 — Subpackages

`FFTWOperators` (`DFT`/`DCT`/`RDFT`/`IRDFT`; `Shift.jl` already has `threaded`),
`DSPOperators` (`Conv`/`Filt`/`MIMOFilt`/`Xcorr`), `NFFTOperators`, `WaveletOperators`.
FFTW and NFFT are *counted* pools in NestedThreading, so `threaded` maps to plan-time thread
counts and `with_restricted_threads`, not to Julia loops.

---

## Testing

Shared testitem setup providing `test_threading_contract(op_ctor, args...)`, applied to every
operator, asserting:

- constructor accepts `threaded=true/false`; `is_threaded` round-trips
- threaded and non-threaded results are **numerically identical** where the algorithm is
  unchanged (inputs captured into variables first, per the stochastic-assertion rule in
  `CLAUDE.md`)
- `copy_operator` never returns `===` its input and never shares mutable buffers (per-field
  aliasing check)
- `adapt_operator` returns `===` when constraints already match, a copy otherwise
- `storage_type` conversion actually takes effect (today's fallback would fail this)
- batch ops: wrapped operators report `is_threaded == false` when the batch op is threaded

Tags: `:batching`, `:calculus`, `:linearoperator`, `:nonlinearoperator` plus exact CamelCase
operator tags; `:gpu` items go through `GpuEnvSetup`. JET in all three modes in the same
change — `default_threaded`/`adapt_operator` take `storage_type::Type{<:AbstractArray}` and
route through typed positional helpers to stay `@test_opt`-clean.

---

## Order and dependencies

- **Phase 0** is standalone and pays off immediately (every subsequent filtered test run is
  faster).
- **Phase 1**'s benchmark harness gates Phases 3, 5 and 6.
- **Phase 2** is an independent bug fix.
- Then 3 → 4 → 5 → 6.
