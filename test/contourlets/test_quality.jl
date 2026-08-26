@testitem "Aqua" tags = [:quality, :contourlet] begin
    using Aqua, ContourletOperators
    Aqua.test_all(ContourletOperators; persistent_tasks = VERSION >= v"1.11")
end
