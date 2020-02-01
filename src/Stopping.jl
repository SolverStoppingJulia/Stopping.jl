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

"""
Type: AbstractState
Abstract type, if specialized state were to be implemented they would need to
be subtypes of AbstractState
"""
abstract type AbstractState end

# State
include("State/GenericStatemod.jl")
include("State/LSAtTmod.jl")
include("State/NLPAtXmod.jl")

export AbstractState, GenericState, update!
export LSAtT, copy, update!
export NLPAtX, update! #, convert_nlp, convert_ls

"""
AbstractStoppingMeta
Abstract type, if specialized meta for stopping were to be implemented they
would need to be subtypes of AbstractStoppingMeta
"""
abstract type AbstractStoppingMeta end
include("Stopping/StoppingMetamod.jl")

export AbstractStoppingMeta, StoppingMeta

# Stopping
include("Stopping/GenericStoppingmod.jl")
include("Stopping/LineSearchStoppingmod.jl")
include("Stopping/NLPStoppingmod.jl")

export GenericStopping, start!, stop!, update_and_start!, update_and_stop!
export fill_in!, reinit!, status
export LS_Stopping
export NLPStopping, unconstrained_check, optim_check_bounded, KKT

end # end of module
