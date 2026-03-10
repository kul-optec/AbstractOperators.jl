@testitem "Aqua" tags = [:quality, :dsp] begin
    using DSPOperators, Aqua
    Aqua.test_all(DSPOperators)
end
