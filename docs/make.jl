using Documenter
using LinearAlgebra, OperatorCore
using AbstractOperators, ContourletOperators, DSPOperators, FFTWOperators, NFFTOperators, WaveletOperators

DocMeta.setdocmeta!(
    AbstractOperators, :DocTestSetup, :(using AbstractOperators); recursive = true
)

makedocs(;
    modules = [OperatorCore, AbstractOperators, ContourletOperators, DSPOperators, FFTWOperators, NFFTOperators, WaveletOperators],
    format = Documenter.HTML(),
    sitename = "AbstractOperators.jl",
    authors = "Niccolò Antonello, Lorenzo Stella and Tamás Hakkel",
    pages = [
        "Home" => "index.md",
        "Abstract Operators" => "operators.md",
        "Calculus rules" => "calculus.md",
        "Properties" => "properties.md",
        "Batch operators" => "batching.md",
        "GPU Support" => "gpu.md",
        "Performance" => "performance.md",
        "Custom Operators" => "custom.md",
        "LinearMaps wrapper" => "linearmaps.md",
    ],
    checkdocs = :exports
)

deploydocs(; repo = "github.com/kul-optec/AbstractOperators.jl", target = "build")
