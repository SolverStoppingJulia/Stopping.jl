using Documenter
using Stopping

makedocs(
    sitename = "Stopping.jl",
    format = Documenter.HTML(assets = ["assets/style.css"], prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Stopping],
    pages = [
             "Home" => "index.md",
             "API" => "api.md",
             "Examples and tutorials" => "tutorial.md"
            ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/Goysa2/Stopping.jl")#
