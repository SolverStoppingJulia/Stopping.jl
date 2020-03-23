using Documenter
using Stopping

makedocs(
    sitename = "Stopping.jl",
    format = Documenter.HTML(assets = ["assets/style.css"], prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Stopping],
    pages = [
             "Home" => "index.md",
             "API" => "api.md",
             "Examples and tutorials" => "tutorial.md",
             "How to State" => "howtostate.md",
             "How to State for NLPs" => "howtostate-nlp.md",
             "How to Stop" => "howtostop.md",
             "How to Stop 2" => "howtostop-2.md",
             "How to Stop for NLPs" => "howtostop-nlp.md",
             "Solve linear algebra" => "linear-algebra.md",
             "Use a buffer function" => "buffer.md",
             "A fixed point algorithm" => "fixed-point.md",
             "Backtracking linesearch algorithm" => "backls.md",
             "Unconstrained optimization algorithm" => "uncons.md",
             "Active set algorithm" => "active-set.md",
             "Quadratic penalty algorithm" => "penalty.md",
             "Run optimization algorithms" => "run-optimsolver.md",
             "Benchmark optimization algorithms" => "benchmark.md"
            ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/Goysa2/Stopping.jl")#
