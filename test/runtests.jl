using Test

using ADNLPModels, DataFrames, LinearAlgebra, LLSModels, NLPModels, Printf, SparseArrays
using NLPModelsModifiers

using Stopping
using Stopping: _init_field

using SolverTools: LineModel

#"State tests...\n"
include("test-state/unit-test-GenericStatemod.jl")
include("test-state/unit-test-OneDAtXmod.jl")
include("test-state/unit-test-NLPAtXmod.jl")
include("test-state/unit-test-ListOfStates.jl")

include("test-stopping/unit-test-voidstopping.jl")
include("test-stopping/test-users-struct-function.jl")
include("test-stopping/unit-test-stopping-meta.jl")
include("test-stopping/unit-test-remote-control.jl")

#"Stopping tests...\n"
include("test-stopping/test-unitaire-generic-stopping.jl")
include("test-stopping/test-unitaire-ls-stopping.jl")
include("test-stopping/unit-test-line-model.jl")
include("test-stopping/test-unitaire-nlp-stopping.jl")
include("test-stopping/test-unitaire-nlp-evals.jl") #not in an environment
include("test-stopping/test-unitaire-nlp-stopping_2.jl")
include("test-stopping/strong-epsilon-check.jl")
include("test-stopping/test-unitaire-linearalgebrastopping.jl")

#"HowTo tests..."
include("examples/runhowto.jl")

#printstyled("Run OptimSolver tests...\n")
#include("examples/run-optimsolver.jl")
