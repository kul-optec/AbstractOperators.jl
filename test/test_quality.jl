@testitem "Documentation" tags = [:quality] begin
    using Documenter, AbstractOperators
    # Eagerly load weak-dep extensions so they're precompiled before the doctest
    # sandbox runs `using AbstractOperators, LinearMaps` (avoids stray precompile output).
    using LinearMaps
    DocMeta.setdocmeta!(
        AbstractOperators,
        :DocTestSetup,
        :(using AbstractOperators);
        recursive = true,
    )
    doctest(AbstractOperators; fix = false)
end

@testitem "Aqua" tags = [:quality] begin
    using Aqua, AbstractOperators

    # persistent_tasks is disabled: it is unreliable and not relevant for this package.
    Aqua.test_all(AbstractOperators, persistent_tasks = false)
end
