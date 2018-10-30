module Stopping
export AbstractStopping

using NLPModels
using State

abstract type AbstractStopping end

const StoppingOrNothing = Union{AbstractStopping, Void}
const Iterate           = Union{Float64,Vector, Void}

include("StoppingMetamod.jl")
include("GenericStoppingmod.jl")
include("LineSearchStoppingmod.jl")
include("NLPStoppingmod.jl")



end # module
