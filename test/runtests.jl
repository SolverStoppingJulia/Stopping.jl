using Test

using DataFrames, LinearAlgebra, NLPModels, Printf, SparseArrays

using Stopping
using Stopping: _init_field

printstyled("Generic State tests... ")
include("test-state/unit-test-GenericStatemod.jl")
#printstyled("passed ✓ \n", color = :green)
#printstyled("LSAtT tests... ")
include("test-state/unit-test-LSAtTmod.jl")
#printstyled("passed ✓ \n", color = :green)
#printstyled("NLPAtX tests... ")
include("test-state/unit-test-NLPAtXmod.jl")
#printstyled("passed ✓ \n", color = :green)
#printstyled("ListOfStates tests... ")
include("test-state/unit-test-ListOfStates.jl")
#printstyled("passed ✓ \n", color = :green)
printstyled("UserSpecificStructure tests... ")
include("test-stopping/test-users-struct-function.jl")
printstyled("passed ✓ \n", color = :green)

include("test-stopping/unit-test-stopping-meta.jl")
include("test-stopping/unit-test-remote-control.jl")

printstyled("GenericStopping tests... ")
include("test-stopping/test-unitaire-generic-stopping.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("LineSearch stopping tests... ")
include("test-stopping/test-unitaire-ls-stopping.jl")
printstyled(" passed ✓ \n", color = :green)
printstyled("Unconsmin tests... ")
include("test-stopping/test-unitaire-nlp-stopping.jl")
include("test-stopping/test-unitaire-nlp-evals.jl")
printstyled(" passed ✓ \n", color = :green)
printstyled("Consmin tests... ")
include("test-stopping/test-unitaire-nlp-stopping_2.jl")
include("test-stopping/strong-epsilon-check.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("LAStopping tests... ")
include("test-stopping/test-unitaire-linearalgebrastopping.jl")
printstyled("passed ✓ \n", color = :green)

printstyled("HowTo tests...\n")
include("examples/runhowto.jl")
