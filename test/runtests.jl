using Test

# Should we use NLPModels?

using NLPModels
using Stopping
using Printf
using LinearAlgebra

printstyled("Generic State tests... ")
include("test-state/test-unitaire-GenericStatemod.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("LSAtT tests... ")
include("test-state/test-unitaire-LSAtTmod.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("NLPAtX tests... ")
include("test-state/test-unitaire-NLPAtXmod.jl")
printstyled("passed ✓ \n", color = :green)

printstyled("StoppingMeta tests... ")
include("test-unitaire-stopping-meta.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("GenericStopping tests... ")
include("test-unitaire-generic-stopping.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("LineSearch stopping tests... ")
include("test-unitaire-ls-stopping.jl")
printstyled(" passed ✓ \n", color = :green)
printstyled("Unconsmin test... ")
include("test-unitaire-nlp-stopping.jl")
printstyled(" passed ✓ \n", color = :green)
printstyled("Consmin test... ")
include("test-unitaire-nlp-stopping_2.jl")
printstyled("passed ✓ \n", color = :green)
