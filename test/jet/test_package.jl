# Standalone: julia --project=test test/jet/test_package.jl
@testitem "JET test_package" tags = [:jet, :base] begin
    using JET, AbstractOperators
    JET.test_package(AbstractOperators; target_modules = (AbstractOperators,))
end
