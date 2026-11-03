# Benchmark entry point loaded by AirspeedVelocity.
#
# Assembles the full `SUITE` from `bench_common.jl` plus every file in `suites/`. To benchmark
# one family on its own, run its suite file directly instead:
#
#     julia --project=benchmark benchmark/suites/linearoperators.jl
#
# See `bench_common.jl` for the threading gate (`BENCH_THREADED`) and the size constants.

include(joinpath(@__DIR__, "bench_common.jl"))

const BENCH_SUITE_FILES = [
    "linearoperators.jl",
    "calculus.jl",
    "nonlinearoperators.jl",
    "batching.jl",
    "dspoperators.jl",
    "fftwoperators.jl",
    "nfftoperators.jl",
    "waveletoperators.jl",
    "normaloperators.jl",
]

for f in BENCH_SUITE_FILES
    include(joinpath(@__DIR__, "suites", f))
end

finalize_suite!(SUITE)
