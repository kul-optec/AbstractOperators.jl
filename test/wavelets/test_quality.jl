@testitem "Aqua" tags = [:quality, :wavelet] begin
    using Aqua, WaveletOperators
    Aqua.test_all(WaveletOperators)
end
