using TestItemRunner

const FILTER_PARTS = if length(ARGS) > 0
    @assert length(ARGS) == 1
    split(ARGS[1], ",")
else
    String[]
end
const FILTER_TAGS = map(p -> Symbol(p[2:end]), filter(x -> startswith(x, ":"), FILTER_PARTS))
const FILTER_NAMES = filter(x -> !startswith(x, ":"), FILTER_PARTS)

const VERB = get(ENV, "ABSTRACTOPERATORS_TEST_VERBOSE", "false") == "true"

# Whether any selected testitem needs the GPU environment. TestItemRunner applies the
# filter to every discovered testitem before it evaluates the first setup module, so this
# is fully determined by the time `GpuEnvSetup` (test/gpu_env_setup.jl) reads it.
# Computing it from the accepted items — rather than from the filter string — is what makes
# it exact: GPU testitems also carry their category tags, so e.g. a `:linearoperator` run
# selects `Eye (GPU)` too.
ENV["ABSTRACTOPERATORS_TEST_GPU"] = "false"

function select_testitem(ti)
    run_item = if isempty(FILTER_PARTS)
        true
    else
        any(t -> t in ti.tags, FILTER_TAGS) || any(n -> n == ti.name, FILTER_NAMES)
    end
    if run_item
        :gpu in ti.tags && (ENV["ABSTRACTOPERATORS_TEST_GPU"] = "true")
        VERB && println("Running @testitem: ", ti.name)
    end
    return run_item
end

@run_package_tests filter = select_testitem
