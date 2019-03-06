using Documenter, AbstractOperators

makedocs(
  modules = [AbstractOperators],
  format = Documenter.HTML(),
  sitename = "AbstractOperators.jl",
  authors = "Niccolò Antonello and Lorenzo Stella",
  pages = [
  "Home"               => "index.md",
  "Abstract Operators" => "operators.md",
  "Calculus rules"     => "calculus.md",
  "Properties"         => "properties.md",
  ],
)

deploydocs(
  repo   = "github.com/kul-forbes/AbstractOperators.jl.git",
  target = "build",
)
