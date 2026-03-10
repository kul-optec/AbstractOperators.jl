@testitem "Aqua" tags = [:quality, :nfft] begin
    using Aqua, NFFTOperators
    Aqua.test_all(NFFTOperators)
end
