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

The `StoppingMeta` provides default tolerances, maximum resources, ...  as well as (boolean) information on the result.

### Your Stopping your way

The `GenericStopping` (with `GenericState`) provides a complete structure to handle stopping criteria.
Then, depending on the problem structure, you can specialize a new Stopping by
redefining a State and some functions specific to your problem.

See also `NLPStopping`, `NLPAtX`, `LS_Stopping`, `OneDAtX`

In these examples, the function `optimality_residual` computes the residual of the optimality conditions is an additional attribute of the types.

## Functions

The tool provides two main functions:
- `start!(stp :: AbstractStopping)` initializes the time and the tolerance at the starting point and check wether the initial guess is optimal.
- `stop!(stp :: AbstractStopping)` checks optimality of the current guess as well as failure of the system (unboundedness for instance) and maximum resources (number of evaluations of functions, elapsed time ...)

Stopping uses the informations furnished by the State to evaluate its functions. Communication between the two can be done through the following functions:
- `update_and_start!(stp :: AbstractStopping; kwargs...)` updates the states with informations furnished as kwargs and then call start!.
- `update_and_stop!(stp :: AbstractStopping; kwargs...)` updates the states with informations furnished as kwargs and then call stop!.
- `fill_in!(stp :: AbstractStopping, x :: T)` a function that fill in all the State with all the informations required to correctly evaluate the stopping functions. This can reveal useful, for instance, if the user do not trust the informations furnished by the algorithm in the State.
- `reinit!(stp :: AbstractStopping)` reinitialize the entries of
the Stopping to reuse for another call.
"""
module Stopping

  using LinearAlgebra, SparseArrays, DataFrames, Printf
  using LinearOperators, NLPModels

  const MatrixType = Union{Number, 
                           AbstractArray, 
                           AbstractMatrix, 
                           Nothing, 
                           AbstractLinearOperator} 
                           #Krylov.PreallocatedLinearOperator,

  """
  AbstractState: 

  Abstract type, if specialized state were to be implemented they would need to
  be subtypes of `AbstractState`.
  """
  abstract type AbstractState{S,T} end

  # State
  include("State/GenericStatemod.jl")
  include("State/OneDAtXmod.jl")
  include("State/NLPAtXmod.jl")

  export scoretype, xtype
  export AbstractState, GenericState, update!, copy, compress_state!, copy_compress_state
  export OneDAtX, update!
  export NLPAtX, update!

  include("State/ListOfStates.jl")

  export AbstractListofStates, ListofStates, VoidListofStates
  export add_to_list!, length, print, getindex, state_type


  function _instate(stt :: Symbol, es :: Symbol)
    for t in fieldnames(GenericState)
      if es == t
        es = esc(Symbol(stt,".$t"))
      end
    end
    es
  end

  function _instate(state :: Symbol, a::Any)
    a
  end

  function _instate(state :: Symbol, ex :: Expr)
    for i=1:length(ex.args)
      ex.args[i] = _instate(state, ex.args[i])
    end
    ex
  end

  """
  `@instate state expression`

  Macro that set the prefix state. to all the variables whose name belong to the 
  field names of the state.
  """
  macro instate(state :: Symbol, ex)
    if typeof(ex) == Expr
      ex = _instate(state, ex)
    end
    ex
  end

  export @instate

  include("Stopping/StopRemoteControl.jl")
  export AbstractStopRemoteControl, StopRemoteControl, cheap_stop_remote_control

  """
  AbstractStoppingMeta

  Abstract type, if specialized meta for stopping were to be implemented they
  would need to be subtypes of AbstractStoppingMeta
  """
  abstract type AbstractStoppingMeta end

  """
  AbstractStopping

  Abstract type, if specialized stopping were to be implemented they would need to
  be subtypes of AbstractStopping
  """
  abstract type AbstractStopping{Pb   <: Any, 
                                M    <: AbstractStoppingMeta, 
                                SRC  <: AbstractStopRemoteControl,
                                T    <: AbstractState,
                                MStp <: Any, #AbstractStopping
                                LoS  <: AbstractListofStates} end

  include("Stopping/StoppingMetamod.jl")

  export AbstractStoppingMeta, StoppingMeta, tol_check, update_tol!, OK_check

  struct VoidStopping{Pb, M, SRC, T, MStp, LoS} <: AbstractStopping{Pb, M, SRC, T, MStp, LoS} end
  function VoidStopping() 
    return VoidStopping{Any, StoppingMeta, StopRemoteControl, 
                        GenericState, Nothing, VoidListofStates}() 
  end

  export AbstractStopping, VoidStopping

  import Base.show
  function show(io :: IO, stp :: VoidStopping)
    println(io, typeof(stp))
  end
  function show(io :: IO, stp :: AbstractStopping)
    println(io, typeof(stp))
    #print(io, stp.meta) #we can always print stp.meta
    #print(io, stp.stop_remote) #we can always print stp.stop_remote
    #print(io, stp.current_state) #we can always print stp.current_state
    if !(typeof(stp.main_stp) <: VoidStopping)
      println(io, "It has a main_stp $(typeof(stp.main_stp))")
    else
      println(io, "It has no main_stp.")
    end
    if typeof(stp.listofstates) != VoidListofStates
      nmax = stp.listofstates.n == -1 ? Inf : stp.listofstates.n
      println(io, "It handles a list of states $(typeof(stp.listofstates)) of maximum length $(nmax)")
    else
      println(io, "It doesn't keep track of the state history.")
    end
    try
      print("Problem is ")
      show(io, stp.pb)
      print(" ")
    catch
      print("Problem is $(typeof(stp.pb)). ")
    end
    if !isnothing(stp.stopping_user_struct)
      try
        print("The user-defined structure is ")
        show(io, stp.stopping_user_struct)
      catch
        print("The user-defined structure is  of type $(typeof(stp.stopping_user_struct)).\n")
      end
    else
      print(io, "No user-defined structure is furnished.\n")
    end
  end

  # Stopping
  include("Stopping/GenericStoppingmod.jl")
  include("Stopping/NLPStoppingmod.jl")

  export GenericStopping, start!, stop!, cheap_stop!, update_and_start!
  export update_and_stop!, cheap_update_and_stop!, cheap_update_and_start!
  export fill_in!, reinit!, status, elapsed_time
  export NLPStopping, unconstrained_check, unconstrained2nd_check, max_evals!
  export optim_check_bounded, KKT

  using LinearAlgebra, SparseArrays, LinearOperators #v.1.0.1

  include("Stopping/LinearAlgebraStopping.jl")

  export LAStopping, LinearSystem, LACounters, linear_system_check, normal_equation_check

end # end of module
