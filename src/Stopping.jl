module Stopping

export AbstractStopping

using LinearAlgebra
using NLPModels
using State

const Iterate = Union{Float64,Vector, Nothing}

abstract type AbstractStopping end

include("StoppingMetamod.jl")
include("GenericStoppingmod.jl")
include("LineSearchStoppingmod.jl")
include("NLPStoppingmod.jl")

end # end of module
