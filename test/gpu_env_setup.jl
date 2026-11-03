@testmodule GpuEnvSetup begin
    using Pkg

    # Activating the GPU environment is expensive, so it lives here rather than in
    # `TestUtils`: TestItemRunner evaluates setup modules lazily, only for testitems that
    # survive the filter, so a run that selects no `:gpu` testitem never pays for it.
    #
    # `runtests.jl` sets `ABSTRACTOPERATORS_TEST_GPU` from the filter *before* any setup
    # module is evaluated. Seeing "false" here means a testitem that is not tagged `:gpu`
    # pulled in this setup module — a tagging bug worth surfacing loudly.
    if get(ENV, "ABSTRACTOPERATORS_TEST_GPU", "true") == "false"
        error(
            "GpuEnvSetup was evaluated although the current test filter selected no " *
                ":gpu testitem. A testitem that uses GpuEnvSetup is missing the :gpu tag."
        )
    end

    using GPUEnv
    GPUEnv.activate(; persist = true)

    # FFTW/DCT-specific: only the "DCT/IDCT (GPU)" testitem consumes this.
    if VERSION >= v"1.11" && Base.find_package("AcceleratedDCTs") === nothing
        Pkg.add(name = "AcceleratedDCTs", version = "0.4")
    end

end  # @testmodule GpuEnvSetup
