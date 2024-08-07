"""
Type: LAStopping

Methods: `start!`, `stop!`, `update_and_start!`, `update_and_stop!`, `fill_in!`, `reinit!`, `status`, 
`linear_system_check`, `normal_equation_check`

Specialization of GenericStopping. Stopping structure for linear algebra
solving either

``Ax = b``

or

```math
min\\_{x} \\tfrac{1}{2}\\|Ax - b\\|^2
```

Attributes:
- `pb`         : a problem using, for instance, either `LLSModel` (designed for linear least square problem, see https://github.com/JuliaSmoothOptimizers/LLSModels.jl ) or `LinearSystem`.
- `current_state`      : The information relative to the problem, see `GenericState`.
- (opt) `meta` : Metadata relative to stopping criteria, see `StoppingMeta`.
- (opt) `main_stp` : Stopping of the main loop in case we consider a Stopping
                          of a subproblem.
                          If not a subproblem, then `VoidStopping`.
- (opt) `listofstates` : ListofStates designed to store the history of States.
- (opt) `stopping_user_struct` : Contains a structure designed by the user.

Constructors: 
- `LAStopping(pb, meta::AbstractStoppingMeta, stop_remote::AbstractStopRemoteControl, state::AbstractState; main_stp::AbstractStopping=VoidStopping(), list::AbstractListofStates = VoidListofStates(), user_struct::AbstractDict = Dict(), zero_start::Bool = false)`
     The default constructor.
- `LAStopping(pb, meta::AbstractStoppingMeta, state::AbstractState; main_stp::AbstractStopping=VoidStopping(), list::AbstractListofStates = VoidListofStates(), user_struct::AbstractDict = Dict(), zero_start::Bool = false, kwargs...)`
     The one passing the `kwargs` to the `stop_remote`.
- `LAStopping(pb, state::AbstractState; stop_remote::AbstractStopRemoteControl = StopRemoteControl(), main_stp::AbstractStopping=VoidStopping(), list::AbstractListofStates = VoidListofStates(), user_struct::AbstractDict = Dict(), zero_start::Bool = false, kwargs...)`
     The one passing the `kwargs` to the `meta`.
- `LAStopping(:: Union{AbstractLinearOperator, AbstractMatrix}, :: Union{AbstractGPUVector, AbstractVector}; sparse::Bool = true, n_listofstates::Int = 0, kwargs...)`
     The one setting up a default problem (`sparse ? LLSModel(A, b) : LinearSystem(A, b)`), a default `GenericState` using x, and initializing the list of states if `n_listofstates>0`. 
- `LAStopping(:: Union{AbstractLinearOperator, AbstractMatrix}, :: Union{AbstractGPUVector, AbstractVector}, :: AbstractState; sparse::Bool = true, kwargs...)`
     The one setting up a default problem (`sparse ? LLSModel(A, b) : LinearSystem(A, b)`). 

Notes:
- No specific State targeted
- State don't necessarily keep track of evals
- Evals are checked only for `pb.A` being a LinearOperator
- `zero_start` is true if 0 is the initial guess (not check automatically)
- `LLSModel` counter follow `NLSCounters` (see `init_max_counters_NLS`)
- By default, `meta.max_cntrs` is initialized with an NLSCounters

See also `GenericStopping`, `NLPStopping`, `linear_system_check`, `normal_equation_check`
 """
mutable struct LAStopping{Pb, M, SRC, T, MStp, LoS} <: AbstractStopping{Pb, M, SRC, T, MStp, LoS}

  # problem
  pb::Pb
  # Common parameters
  meta::M
  stop_remote::SRC
  # current state of the problem
  current_state::T
  # Stopping of the main problem, or nothing
  main_stp::MStp
  # History of states
  listofstates::LoS
  # User-specific structure
  stopping_user_struct::AbstractDict

  #zero is initial point
  zero_start::Bool
end

get_pb(stp::LAStopping) = stp.pb
get_meta(stp::LAStopping) = stp.meta
get_remote(stp::LAStopping) = stp.stop_remote
get_state(stp::LAStopping) = stp.current_state
get_main_stp(stp::LAStopping) = stp.main_stp
get_list_of_states(stp::LAStopping) = stp.listofstates
get_user_struct(stp::LAStopping) = stp.stopping_user_struct

function LAStopping(
  pb::Pb,
  meta::M,
  stop_remote::SRC,
  current_state::T;
  main_stp::AbstractStopping = VoidStopping(),
  list::AbstractListofStates = VoidListofStates(),
  user_struct::AbstractDict = Dict(),
  zero_start::Bool = false,
) where {Pb <: Any, M <: AbstractStoppingMeta, SRC <: AbstractStopRemoteControl, T <: AbstractState}
  return LAStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct, zero_start)
end

function LAStopping(
  pb::Pb,
  meta::M,
  current_state::T;
  main_stp::AbstractStopping = VoidStopping(),
  list::AbstractListofStates = VoidListofStates(),
  user_struct::AbstractDict = Dict(),
  zero_start::Bool = false,
  kwargs...,
) where {Pb <: Any, M <: AbstractStoppingMeta, T <: AbstractState}
  stop_remote = StopRemoteControl(; kwargs...) #main_stp == VoidStopping() ? StopRemoteControl() : cheap_stop_remote_control()

  return LAStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct, zero_start)
end

function LAStopping(
  pb::Pb,
  current_state::T;
  stop_remote::AbstractStopRemoteControl = StopRemoteControl(), #main_stp == VoidStopping() ? StopRemoteControl() : cheap_stop_remote_control()
  main_stp::AbstractStopping = VoidStopping(),
  list::AbstractListofStates = VoidListofStates(),
  user_struct::AbstractDict = Dict(),
  zero_start::Bool = false,
  kwargs...,
) where {Pb <: Any, T <: AbstractState}
  if :max_cntrs in keys(kwargs)
    mcntrs = kwargs[:max_cntrs]
  elseif Pb <: LLSModel
    mcntrs = init_max_counters_NLS()
  else
    mcntrs = init_max_counters_linear_operators()
  end

  if :optimality_check in keys(kwargs)
    oc = kwargs[:optimality_check]
  else
    oc = linear_system_check
  end

  meta = StoppingMeta(; max_cntrs = mcntrs, optimality_check = oc, kwargs...)

  return LAStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct, zero_start)
end

function LAStopping(
  A::TA,
  b::Tb;
  x::Tb = zeros(eltype(Tb), size(A, 2)),
  sparse::Bool = true,
  n_listofstates::Int = 0,
  kwargs...,
) where {TA <: Any, Tb <: Union{AbstractGPUVector, AbstractVector}}
  pb = sparse ? LLSModel(A, b) : LinearSystem(A, b)
  state = GenericState(x)

  mcntrs = sparse ? init_max_counters_NLS() : init_max_counters_linear_operators()

  if n_listofstates > 0 && :list ∉ keys(kwargs)
    list = ListofStates(n_listofstates, Val{typeof(state)}())
    return LAStopping(pb, state, max_cntrs = mcntrs, list = list; kwargs...)
  end

  return LAStopping(pb, state, max_cntrs = mcntrs; kwargs...)
end

function LAStopping(
  A::TA,
  b::Tb,
  state::S;
  sparse::Bool = true,
  kwargs...,
) where {TA <: Any, Tb <: Union{AbstractGPUVector, AbstractVector}, S <: AbstractState}
  pb = sparse ? LLSModel(A, b) : LinearSystem(A, b)

  mcntrs = sparse ? init_max_counters_NLS() : init_max_counters_linear_operators()

  return LAStopping(pb, state, max_cntrs = mcntrs; kwargs...)
end

"""
Type: LACounters
"""
mutable struct LACounters{T <: Int}
  nprod::T
  ntprod::T
  nctprod::T
  sum::T

  function LACounters(nprod::T, ntprod::T, nctprod::T, sum::T) where {T <: Int}
    return new{T}(nprod, ntprod, nctprod, sum)
  end
end

function LACounters(; nprod::Int64 = 0, ntprod::Int64 = 0, nctprod::Int64 = 0, sum::Int64 = 0)
  return LACounters(nprod, ntprod, nctprod, sum)
end

"""
init\\_max\\_counters\\_linear\\_operators: counters for LinearOperator

`init_max_counters_linear_operators(; allevals :: T = 20000, nprod = allevals, ntprod = allevals, nctprod = allevals, sum = 11 * allevals)`
"""
function init_max_counters_linear_operators(;
  allevals::T = 20000,
  nprod::T = allevals,
  ntprod::T = allevals,
  nctprod::T = allevals,
  sum::T = allevals * 11,
) where {T <: Int}
  cntrs =
    Dict{Symbol, T}([(:nprod, nprod), (:ntprod, ntprod), (:nctprod, nctprod), (:neval_sum, sum)])

  return cntrs
end

"""
LinearSystem: Minimal structure to store linear algebra problems

`LinearSystem(:: Union{AbstractLinearOperator, AbstractMatrix}, :: Union{AbstractGPUVector, AbstractVector})`

Note:
Another option is to convert the `LinearSystem` as an `LLSModel`.
"""
mutable struct LinearSystem{
  TA <: Union{AbstractLinearOperator, AbstractMatrix},
  Tb <: Union{AbstractGPUVector, AbstractVector},
}
  A::TA
  b::Tb

  counters::LACounters

  function LinearSystem(
    A::TA,
    b::Tb;
    counters::LACounters = LACounters(),
    kwargs...,
  ) where {
    TA <: Union{AbstractLinearOperator, AbstractMatrix},
    Tb <: Union{AbstractGPUVector, AbstractVector},
  }
    return new{TA, Tb}(A, b, counters)
  end
end

function LAStopping(
  A::TA,
  b::Tb;
  x::Tb = zeros(eltype(Tb), size(A, 2)),
  kwargs...,
) where {TA <: AbstractLinearOperator, Tb <: Union{AbstractGPUVector, AbstractVector}}
  return LAStopping(A, b, GenericState(x), kwargs...)
end

function LAStopping(
  A::TA,
  b::Tb,
  state::AbstractState;
  kwargs...,
) where {TA <: AbstractLinearOperator, Tb <: Union{AbstractGPUVector, AbstractVector}}
  return LAStopping(
    LinearSystem(A, b),
    state,
    max_cntrs = init_max_counters_linear_operators(),
    kwargs...,
  )
end

"""
 \\_resources\\_check!: check if the optimization algorithm has 
exhausted the resources. This is the Linear Algebra specialized version.

 Note:
 - function does _not_ keep track of the evals in the state
 - check `:nprod`, `:ntprod`, and `:nctprod` in the `LinearOperator` entries
 """
function _resources_check!(stp::LAStopping, x::T) where {T}

  #GenericState has no field evals.
  #_smart_update!(stp.current_state, evals = cntrs)

  # check all the entries in the counter
  # global user limit diagnostic
  stp.meta.resources = _counters_loop!(stp.pb.counters, stp.meta.max_cntrs)

  return stp.meta.resources
end

function _counters_loop!(cntrs::LACounters{T}, max_cntrs::Dict{Symbol, T}) where {T}
  sum, max_f = 0, false

  for f in (:nprod, :ntprod, :nctprod)
    ff = getfield(cntrs, f)
    max_f = max_f || (ff > max_cntrs[f])
    sum += ff
  end

  return max_f || (sum > max_cntrs[:neval_sum])
end

function _counters_loop!(cntrs::NLSCounters, max_cntrs::Dict{Symbol, T}) where {T}
  sum, max_f = 0, false

  for f in intersect(fieldnames(NLSCounters), keys(max_cntrs))
    max_f = f != :counters ? (max_f || (getfield(cntrs, f) > max_cntrs[f])) : max_f
  end
  for f in intersect(fieldnames(Counters), keys(max_cntrs))
    max_f = max_f || (getfield(cntrs.counters, f) > max_cntrs[f])
  end

  return max_f || (sum > max_cntrs[:neval_sum])
end

"""
linear\\_system\\_check: return ||Ax-b||_p

`linear_system_check(:: Union{LinearSystem, LLSModel}, :: AbstractState; pnorm :: Real = Inf, kwargs...)`

Note:
- Returns the p-norm of state.res
- state.res is filled in if nothing.
"""
function linear_system_check(pb::LinearSystem, state::AbstractState; pnorm::Real = Inf, kwargs...)
  pb.counters.nprod += 1
  if length(state.res) == 0
    update!(state, res = pb.A * state.x - pb.b)
  end

  return norm(state.res, pnorm)
end

function linear_system_check(pb::LLSModel, state::AbstractState; pnorm::Real = Inf, kwargs...)
  if length(state.res) == 0
    Axmb = if xtype(state) <: SparseVector
      sparse(residual(pb, state.x))
    else
      residual(pb, state.x)
    end
    update!(state, res = Axmb)
  end

  return norm(state.res, pnorm)
end

"""
normal\\_equation\\_check: return ||A'Ax-A'b||_p

`normal_equation_check(:: Union{LinearSystem, LLSModel}, :: AbstractState; pnorm :: Real = Inf, kwargs...)`

Note: pb must have A and b entries
"""
function normal_equation_check(pb::LinearSystem, state::AbstractState; pnorm::Real = Inf, kwargs...)
  pb.counters.nprod += 1
  pb.counters.ntprod += 1
  return norm(pb.A' * (pb.A * state.x) - pb.A' * pb.b, pnorm)
end

function normal_equation_check(pb::LLSModel, state::AbstractState; pnorm::Real = Inf, kwargs...)
  nres = jtprod_residual(pb, state.x, residual(pb, state.x))
  return norm(nres, pnorm)
end
