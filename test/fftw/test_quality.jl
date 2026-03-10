@testitem "Aqua" tags = [:quality, :fftw] begin
    using Aqua, FFTW, FFTWOperators
    Aqua.test_all(FFTWOperators; piracies = false)
    Aqua.test_piracies(FFTWOperators; treat_as_own = [FFTW, AbstractOperators])
end
