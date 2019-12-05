module Stopping

export AbstractStopping

using LinearAlgebra
using NLPModels

const Iterate           = Union{Number, Vector, Nothing}
const FloatVoid         = Union{Number, Nothing}
const MatrixType        = Any #Union{Number, AbstractArray, Nothing}

"""
AbstractStopping
Abstract type, if specialized stopping were to be implemented they would need to
be subtypes of AbstractStopping
"""
abstract type AbstractStopping end

# State
include("State/GenericStatemod.jl")
include("State/LSAtTmod.jl")
include("State/NLPAtXmod.jl")

export AbstractState, GenericState, update!
export LSAtT, copy, update!
export NLPAtX, update! #, convert_nlp, convert_ls

# Stopping
include("Stopping/StoppingMetamod.jl")
include("Stopping/GenericStoppingmod.jl")
include("Stopping/LineSearchStoppingmod.jl")
include("Stopping/NLPStoppingmod.jl")

end # end of module
