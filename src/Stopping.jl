module Stopping
export AbstractStopping

using NLPModels
using State
using LinearAlgebra

abstract type AbstractStopping end

const StoppingOrNothing = Union{AbstractStopping, Nothing}
const Iterate           = Union{Float64,Vector, Nothing}

include("StoppingMetamod.jl")
include("GenericStoppingmod.jl")
include("LineSearchStoppingmod.jl")
include("NLPStoppingmod.jl")



end # module
