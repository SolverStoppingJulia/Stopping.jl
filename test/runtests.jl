@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# Should we use NLPModels?

using NLPModels
using State
using Stopping
using Printf
using LinearAlgebra

include("test-unitaire-stopping-meta.jl")
printstyled("StoppingMeta tests passed \n", color = :green)
include("test-unitaire-stopping-generic.jl")
printstyled("GenericStopping tests passed \n", color = :green)
include("test-unitaire-ls-stopping.jl")
printstyled("LineSearch stopping tests passed \n", color = :green)
include("test-unitaire-nlp-stopping.jl")
printstyled("Unconsmin test passed \n", color = :green)
include("test-unitaire-nlp-stopping_2.jl")
printstyled("Consmin test passed \n", color = :green)
