"""
Module Stopping:

## Purpose

Tools to ease the uniformization of stopping criteria in iterative solvers.

When a solver is called on an optimization model, four outcomes may happen:

1. the approximate solution is obtained, the problem is considered solved
2. the problem is declared unsolvable (unboundedness, infeasibility ...)
3. the maximum available resources are not sufficient to compute the solution
4. some algorithm dependent failure happens

This tool eases the first three items above. It defines a type

    mutable struct GenericStopping <: AbstractStopping
        problem       :: Any          # an arbitrary instance of a problem
        meta          :: AbstractStoppingMeta # contains the used parameters
        current_state :: AbstractState        # the current state

The *StoppingMeta* provides default tolerances, maximum resources, ...  as well as (boolean) information on the result.

### Your Stopping your way

The *GenericStopping* (with *GenericState*) provides a complete structure to handle stopping criteria.
Then, depending on the problem structure, you can specialize a new Stopping by
redefining a State and some functions specific to your problem.

See also *NLPStopping*, *NLPAtX*, *LS_Stopping*, *LSAtT*

In these examples, the function `optimality_residual` computes the residual of the optimality conditions is an additional attribute of the types.

## Functions

The tool provides two main functions:
* `start!(stp :: AbstractStopping)` initializes the time and the tolerance at the starting point and check wether the initial guess is optimal.
* `stop!(stp :: AbstractStopping)` checks optimality of the current guess as well as failure of the system (unboundedness for instance) and maximum resources (number of evaluations of functions, elapsed time ...)

Stopping uses the informations furnished by the State to evaluate its functions. Communication between the two can be done through the following functions:
* `update_and_start!(stp :: AbstractStopping; kwargs...)` updates the states with informations furnished as kwargs and then call start!.
* `update_and_stop!(stp :: AbstractStopping; kwargs...)` updates the states with informations furnished as kwargs and then call stop!.
* `fill_in!(stp :: AbstractStopping, x :: Iterate)` a function that fill in all the State with all the informations required to correctly evaluate the stopping functions. This can reveal useful, for instance, if the user do not trust the informations furnished by the algorithm in the State.
* `reinit!(stp :: AbstractStopping)` reinitialize the entries of
the Stopping to reuse for another call.
"""
module Stopping

using LinearAlgebra, NLPModels

const Iterate           = Union{Number, AbstractVector, Nothing}
const FloatVoid         = Union{Number, Nothing}
const MatrixType        = Union{Number, AbstractArray, AbstractMatrix, Nothing}

"""
AbstractStopping

Abstract type, if specialized stopping were to be implemented they would need to
be subtypes of AbstractStopping
"""
abstract type AbstractStopping end

export AbstractStopping

"""
AbstractState

Abstract type, if specialized state were to be implemented they would need to
be subtypes of AbstractState
"""
abstract type AbstractState end

# State
include("State/GenericStatemod.jl")
include("State/LSAtTmod.jl")
include("State/NLPAtXmod.jl")

export AbstractState, GenericState, update!, copy, compress_state!, copy_compress_state
export LSAtT, update!
export NLPAtX, update!

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
export NLPStopping, unconstrained_check, unconstrained2nd_check
export optim_check_bounded, KKT

using LinearAlgebra, SparseArrays, LinearOperators #v.1.0.1

include("Stopping/LinearAlgebraStopping.jl")

export LAStopping, LinearSystem, linear_system_check, normal_equation_check


end # end of module
