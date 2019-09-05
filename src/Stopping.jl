module Stopping

export AbstractStopping

using LinearAlgebra
using NLPModels
using State

const Iterate = Union{Float64,Vector, Nothing}

"""
AbstractStopping
Abstract type, if specialized stopping were to be implemented they would need to
be subtypes of AbstractStopping
"""
abstract type AbstractStopping end

include("StoppingMetamod.jl")
include("GenericStoppingmod.jl")
include("LineSearchStoppingmod.jl")
include("NLPStoppingmod.jl")

end # end of module
