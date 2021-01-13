using Test

using DataFrames, LinearAlgebra, NLPModels, Printf, SparseArrays

using Stopping
using Stopping: _init_field

printstyled("State tests...\n")

include("test-state/unit-test-GenericStatemod.jl")
include("test-state/unit-test-LSAtTmod.jl")
include("test-state/unit-test-NLPAtXmod.jl")
include("test-state/unit-test-ListOfStates.jl")

include("test-stopping/test-users-struct-function.jl")
include("test-stopping/unit-test-stopping-meta.jl")
include("test-stopping/unit-test-remote-control.jl")

printstyled("Stopping tests...\n")

include("test-stopping/test-unitaire-generic-stopping.jl")
#printstyled("passed ✓ \n", color = :green)
#printstyled("LineSearch stopping tests... ")
include("test-stopping/test-unitaire-ls-stopping.jl")
#printstyled(" passed ✓ \n", color = :green)
printstyled("Unconsmin tests... ")
include("test-stopping/test-unitaire-nlp-stopping.jl")
include("test-stopping/test-unitaire-nlp-evals.jl")
printstyled(" passed ✓ \n", color = :green)
#printstyled("Consmin tests... ")
include("test-stopping/test-unitaire-nlp-stopping_2.jl")
include("test-stopping/strong-epsilon-check.jl")
#printstyled("passed ✓ \n", color = :green)
#printstyled("LAStopping tests... ")
include("test-stopping/test-unitaire-linearalgebrastopping.jl")
#printstyled("passed ✓ \n", color = :green)

printstyled("HowTo tests...\n")

include("examples/runhowto.jl")
