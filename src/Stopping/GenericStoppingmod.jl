"""
 Type: `GenericStopping`

 Methods: `start!`, `stop!`, `update_and_start!`, `update_and_stop!`, `fill_in!`, `reinit!`, `status`

 A generic Stopping to solve instances with respect to some
 optimality conditions. Optimality is decided by computing a score, which is then
 tested to zero.

 Tracked data include:
- `pb`         : A problem
- `current_state` : The information relative to the problem, see `GenericState`.
- (opt) `meta` : Metadata relative to a stopping criteria, see `StoppingMeta`.
- (opt) `main_stp` : Stopping of the main loop in case we consider a Stopping
                       of a subproblem.
                       If not a subproblem, then `VoidStopping`.
- (opt) `listofstates` : `ListofStates` designed to store the history of States.
- (opt) `stopping_user_struct` : Contains a structure designed by the user.

 Constructors: 
- `GenericStopping(pb, meta::AbstractStoppingMeta, stop_remote::AbstractStopRemoteControl, state::AbstractState; main_stp::AbstractStopping=VoidStopping(), list::AbstractListofStates = VoidListofStates(), user_struct::AbstractDict = Dict(), kwargs...)`
     The default constructor.
- `GenericStopping(pb, meta::AbstractStoppingMeta, state::AbstractState; main_stp::AbstractStopping=VoidStopping(), list::AbstractListofStates = VoidListofStates(), user_struct::AbstractDict = Dict(), kwargs...)`
     The one passing the `kwargs` to the `stop_remote`.
- `GenericStopping(pb, state::AbstractState; stop_remote::AbstractStopRemoteControl = StopRemoteControl(), main_stp::AbstractStopping=VoidStopping(), list::AbstractListofStates = VoidListofStates(), user_struct::AbstractDict = Dict(), kwargs...)`
     The one passing the `kwargs` to the `meta`.
- `GenericStopping(pb, stop_remote::AbstractStopRemoteControl, state::AbstractState; main_stp::AbstractStopping=VoidStopping(), list::AbstractListofStates = VoidListofStates(), user_struct::AbstractDict = Dict(), kwargs...)`
     The one passing the `kwargs` to the `meta`.
- `GenericStopping(pb, x; n_listofstates=, kwargs...)`
     The one setting up a default state using x, and initializing the list of states if `n_listofstates>0`. 


 Note: Metadata can be provided by the user or created with the Stopping
       constructor via kwargs. If a specific StoppingMeta is given and
       kwargs are provided, the kwargs have priority.

 Examples:
 `GenericStopping(pb, GenericState(ones(2)), rtol = 1e-1)`

 Besides optimality conditions, we consider classical emergency exit:
- domain error        (for instance: NaN in x)
- unbounded problem   (not implemented)
- unbounded x         (x is too large)
- tired problem       (time limit attained)
- resources exhausted (not implemented)
- stalled problem     (not implemented)
- iteration limit     (maximum number of iteration (i.e. nb of stop) attained)
- main_pb limit       (tired or resources of main problem exhausted)

 There is an additional default constructor which creates a Stopping with a default State.

 `GenericStopping(:: Any, :: Union{Number, AbstractVector}; kwargs...)`

 Note: Keywords arguments are forwarded to the classical constructor.

 Examples:
 `GenericStopping(pb, x0, rtol = 1e-1)`
"""
mutable struct GenericStopping{Pb, M, SRC, T, MStp, LoS} <:
               AbstractStopping{Pb, M, SRC, T, MStp, LoS}

  # Problem
  pb::Pb

  # Problem stopping criterion
  meta::M
  stop_remote::SRC

  # Current information on the problem
  current_state::T

  # Stopping of the main problem, or nothing
  main_stp::MStp

  # History of states
  listofstates::LoS

  # User-specific structure
  stopping_user_struct::AbstractDict
end

get_pb(stp::GenericStopping) = stp.pb
get_meta(stp::GenericStopping) = stp.meta
get_remote(stp::GenericStopping) = stp.stop_remote
get_state(stp::GenericStopping) = stp.current_state
get_main_stp(stp::GenericStopping) = stp.main_stp
get_list_of_states(stp::GenericStopping) = stp.listofstates
get_user_struct(stp::GenericStopping) = stp.stopping_user_struct

function GenericStopping(
  pb::Pb,
  meta::M,
  stop_remote::SRC,
  current_state::T;
  main_stp::AbstractStopping = VoidStopping(),
  list::AbstractListofStates = VoidListofStates(),
  user_struct::AbstractDict = Dict(),
  kwargs...,
) where {Pb <: Any, M <: AbstractStoppingMeta, SRC <: AbstractStopRemoteControl, T <: AbstractState}
  return GenericStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct)
end

function GenericStopping(
  pb::Pb,
  meta::M,
  current_state::T;
  main_stp::AbstractStopping = VoidStopping(),
  list::AbstractListofStates = VoidListofStates(),
  user_struct::AbstractDict = Dict(),
  kwargs...,
) where {Pb <: Any, M <: AbstractStoppingMeta, T <: AbstractState}
  stop_remote = StopRemoteControl(; kwargs...) #main_stp == VoidStopping() ? StopRemoteControl() : cheap_stop_remote_control()

  return GenericStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct)
end

function GenericStopping(
  pb::Pb,
  current_state::T;
  stop_remote::AbstractStopRemoteControl = StopRemoteControl(), #main_stp == VoidStopping() ? StopRemoteControl() : cheap_stop_remote_control(),
  main_stp::AbstractStopping = VoidStopping(),
  list::AbstractListofStates = VoidListofStates(),
  user_struct::AbstractDict = Dict(),
  kwargs...,
) where {Pb <: Any, T <: AbstractState}
  meta = StoppingMeta(; kwargs...)

  return GenericStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct)
end

function GenericStopping(
  pb::Pb,
  stop_remote::SRC,
  current_state::T;
  main_stp::AbstractStopping = VoidStopping(),
  list::AbstractListofStates = VoidListofStates(),
  user_struct::AbstractDict = Dict(),
  kwargs...,
) where {Pb <: Any, SRC <: AbstractStopRemoteControl, T <: AbstractState}
  meta = StoppingMeta(; kwargs...)

  return GenericStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct)
end

"""
    `GenericStopping(pb :: Any, x :: T; n_listofstates :: Int = 0, kwargs...)`

Setting the keyword argument `n_listofstates > 0` initialize a ListofStates of length `n_listofstates`.
"""
function GenericStopping(pb::Any, x::T; n_listofstates::Int = 0, kwargs...) where {T}
  state = GenericState(x)
  if n_listofstates > 0 && :list ∉ keys(kwargs)
    list = ListofStates(n_listofstates, Val{typeof(state)}())
    return GenericStopping(pb, state, list = list; kwargs...)
  end
  return GenericStopping(pb, state; kwargs...)
end

"""
    `update!(stp::AbstractStopping; kwargs...)`

update!: generic update function for the Stopping

Shortcut for update!(stp.current_state; kwargs...)
"""
function update!(stp::AbstractStopping; kwargs...)
  return update!(stp.current_state; kwargs...)
end

"""
    `fill_in!(stp::AbstractStopping, x::T) where {T}`

fill_in!: fill in the unspecified values of the AbstractState.

Note: NotImplemented for Abstract/Generic-Stopping.
"""
function fill_in!(stp::AbstractStopping, x::T) where {T}
  return throw(error("NotImplemented function"))
end

"""
    `update_and_start!(stp::AbstractStopping; no_opt_check::Bool = false, kwargs...)`

Update values in the State and initialize the Stopping.
Returns the optimality status of the problem as a boolean.

 Note: 
  - Kwargs are forwarded to the `update!` call.  
  - `no_opt_check` skip optimality check in `start!` (`false` by default).  
"""
function update_and_start!(stp::AbstractStopping; no_opt_check::Bool = false, kwargs...)
  if stp.stop_remote.cheap_check
    _smart_update!(stp.current_state; kwargs...)
  else
    update!(stp; kwargs...)
  end
  OK = start!(stp, no_opt_check = no_opt_check)

  return OK
end

"""
    `start!(stp::AbstractStopping; no_opt_check::Bool = false, kwargs...)`

Update the Stopping and return `true` if we must stop.

Purpose is to know if there is a need to even perform an optimization algorithm
or if we are at an optimal solution from the beginning. 
Set `no_opt_check` to `true` avoid checking optimality and domain errors.

The function `start!` successively calls: `_domain_check(stp, x)`, `_optimality_check!(stp, x)`, `_null_test(stp, x)` and  `_user_check!(stp, x, true)`.

Note: - `start!` initializes `stp.meta.start_time` (if not done before),
`stp.current_state.current_time` and `stp.meta.optimality0` 
(if `no_opt_check` is false).   
       - Keywords argument are passed to the `_optimality_check!` call.   
       - Compatible with the `StopRemoteControl`.   
"""
function start!(stp::AbstractStopping; no_opt_check::Bool = false, kwargs...)
  state = stp.current_state
  src = stp.stop_remote

  #Initialize the time counter
  if src.tired_check && isnan(stp.meta.start_time)
    stp.meta.start_time = time()
  end
  #and synchornize with the State
  if src.tired_check && isnan(state.current_time)
    _update_time!(state, stp.meta.start_time)
  end

  if !no_opt_check
    stp.meta.domainerror = if src.domain_check
      #don't check current_score
      _domain_check(stp.current_state, current_score = true)
    else
      stp.meta.domainerror
    end
    if src.optimality_check
      optimality0 = _optimality_check!(stp; kwargs...)
      norm_optimality0 = norm(optimality0, Inf)
      if src.domain_check && isnan(norm_optimality0)
        stp.meta.domainerror = true
      elseif norm_optimality0 == Inf
        stp.meta.optimality0 = one(typeof(norm_optimality0))
      else
        stp.meta.optimality0 = norm_optimality0
      end

      if _null_test(stp, optimality0)
        stp.meta.optimal = true
      end
    end
  end

  src.user_start_check && _user_check!(stp, state.x, true)

  OK = OK_check(stp.meta)

  #do nothing if typeof(stp.listofstates) == VoidListofStates
  add_to_list!(stp.listofstates, stp.current_state)

  return OK
end

"""
    `reinit!(:: AbstractStopping; rstate :: Bool = false, kwargs...)`

Reinitialize the meta-data in the Stopping.

 Note:
- If `rstate` is set as `true` it reinitializes the current State
(with the kwargs).
- If `rlist` is set as true the list of states is also reinitialized, either
set as a `VoidListofStates` if `rstate` is `true` or a list containing only the current
state otherwise.
"""
function reinit!(stp::AbstractStopping; rstate::Bool = false, rlist::Bool = true, kwargs...)
  stp.meta.start_time = NaN
  stp.meta.optimality0 = 1.0

  #reinitialize the boolean status
  reinit!(stp.meta)

  #reinitialize the counter of stop
  stp.meta.nb_of_stop = 0

  #reinitialize the list of states
  if rlist && (typeof(stp.listofstates) != VoidListofStates)
    #TODO: Warning we cannot change the type of ListofStates 
    stp.listofstates = rstate ? VoidListofStates() : ListofStates(stp.current_state)
  end

  #reinitialize the state
  if rstate
    reinit!(stp.current_state; kwargs...)
  end

  return stp
end

"""
    `update_and_stop!(stp :: AbstractStopping; kwargs...)`

Update the values in the state and return the optimality status of the problem as a boolean.

Note: Kwargs are forwarded to the `update!` call.
"""
function update_and_stop!(stp::AbstractStopping; kwargs...)
  if stp.stop_remote.cheap_check
    _smart_update!(stp.current_state; kwargs...)
    OK = cheap_stop!(stp)
  else
    update!(stp; kwargs...)
    OK = stop!(stp)
  end

  return OK
end

"""
    `stop!(:: AbstractStopping; kwargs...)`

Update the Stopping and return a boolean true if we must stop.

It serves the same purpose as `start!` in an algorithm; telling us if we
stop the algorithm (because we have reached optimality or we loop infinitely,
etc).

The function `stop!` successively calls: `_domain_check`, `_optimality_check`,
`_null_test`, `_unbounded_check!`, `_tired_check!`, `_resources_check!`,
`_stalled_check!`, `_iteration_check!`, `_main_pb_check!`, `add_to_list!`

Note:
- kwargs are sent to the `_optimality_check!` call.
- If `listofstates != VoidListofStates`, call `add_to_list!`.
"""
function stop!(stp::AbstractStopping; no_opt_check::Bool = false, kwargs...)
  x = stp.current_state.x
  src = stp.stop_remote

  src.unbounded_and_domain_x_check && _unbounded_and_domain_x_check!(stp, x)
  stp.meta.domainerror = if src.domain_check
    #don't check x and current_score
    _domain_check(stp.current_state, x = true, current_score = true)
  else
    stp.meta.domainerror
  end
  if !no_opt_check
    # Optimality check
    if src.optimality_check
      score = _optimality_check!(stp; kwargs...)
      if src.domain_check && any(isnan, score)
        stp.meta.domainerror = true
      end
      if _null_test(stp, score)
        stp.meta.optimal = true
      end
    end

    src.infeasibility_check && _infeasibility_check!(stp, x)
    src.unbounded_problem_check && _unbounded_problem_check!(stp, x)
    src.tired_check && _tired_check!(stp, x)
    src.resources_check && _resources_check!(stp, x)
    src.stalled_check && _stalled_check!(stp, x)
    src.iteration_check && _iteration_check!(stp, x)
    src.main_pb_check && _main_pb_check!(stp, x)
    src.user_check && _user_check!(stp, x)
  end

  OK = OK_check(stp.meta)

  _add_stop!(stp)

  #do nothing if typeof(stp.listofstates) == VoidListofStates
  add_to_list!(stp.listofstates, stp.current_state)

  return OK
end

"""
    `cheap_stop!(:: AbstractStopping; kwargs...)`

Update the Stopping and return a boolean true if we must stop.

It serves the same purpose as `stop!`, but avoids any potentially expensive checks.
We no longer browse `x` and `res` in the State, and no check on the `main_stp`.
Check only the updated entries in the meta.

The function `cheap_stop!` successively calls:
`_null_test`, `_unbounded_check!`, `_tired_check!`, `_resources_check!`,
`_stalled_check!`, `_iteration_check!`, `add_to_list!`

Note:
- kwargs are sent to the `_optimality_check!` call.
- If `listofstates != VoidListofStates`, call `add_to_list!`.
"""
function cheap_stop!(stp::AbstractStopping; kwargs...)
  x = stp.current_state.x
  src = stp.stop_remote

  # Optimality check
  if src.optimality_check
    score = _optimality_check!(stp; kwargs...) #update state.current_score
    if _null_test(stp, score)
      stp.meta.optimal = true
    end
  end
  OK = stp.meta.optimal

  OK = OK || (src.infeasibility_check && _infeasibility_check!(stp, x))
  OK = OK || (src.unbounded_problem_check && _unbounded_problem_check!(stp, x))
  OK = OK || (src.tired_check && _tired_check!(stp, x))
  OK = OK || (src.resources_check && _resources_check!(stp, x))
  OK = OK || (src.iteration_check && _iteration_check!(stp, x))
  OK = OK || (src.user_check && _user_check!(stp, x))

  _add_stop!(stp)

  #do nothing if typeof(stp.listofstates) == VoidListofStates
  add_to_list!(stp.listofstates, stp.current_state)

  return OK
end

"""
    `_add_stop!(:: AbstractStopping)`

Increment a counter of stop.

Fonction called everytime `stop!` is called. In theory should be called once 
every iteration of an algorithm.

Note: update `meta.nb_of_stop`.
"""
function _add_stop!(stp::AbstractStopping)
  stp.meta.nb_of_stop += 1

  return stp
end

"""
    `_iteration_check!(:: AbstractStopping,  :: T)`

Check if the optimization algorithm has reached the 
iteration limit.

Note: Compare `meta.iteration_limit` with `meta.nb_of_stop`.
"""
function _iteration_check!(stp::AbstractStopping, x::T) where {T}
  max_iter = stp.meta.nb_of_stop >= stp.meta.max_iter
  if max_iter
    stp.meta.iteration_limit = true
  end

  return stp.meta.iteration_limit
end

"""
    `_stalled_check!(:: AbstractStopping, :: T)`

Check if the optimization algorithm is stalling.

Note: Do nothing by default for AbstractStopping.
"""
function _stalled_check!(stp::AbstractStopping, x::T) where {T}
  return stp.meta.stalled
end

"""
    `_tired_check!(:: AbstractStopping, :: T)`

Check if the optimization algorithm has been running for too long.

Note: 
  - Return `false` if `meta.start_time` is `NaN` (by default).  
  - Update `meta.tired`.  
"""
function _tired_check!(stp::AbstractStopping, x::T) where {T}
  stime = stp.meta.start_time #can be NaN
  ctime = time()

  #Keep the current_state updated
  _update_time!(stp.current_state, ctime)

  elapsed_time = ctime - stime
  max_time = elapsed_time > stp.meta.max_time #NaN > 1. is false

  if max_time
    stp.meta.tired = true
  end

  return stp.meta.tired
end

function _tired_check!(stp::VoidStopping, x::T) where {T}
  return false
end

"""
    `_resources_check!(:: AbstractStopping, :: T)`

Check if the optimization algorithm has exhausted the resources.

Note: Do nothing by default `meta.resources` for AbstractStopping.
"""
function _resources_check!(stp::AbstractStopping, x::T) where {T}
  return stp.meta.resources
end

function _resources_check!(stp::VoidStopping, x::T) where {T}
  return false
end

"""
    `_main_pb_check!(:: AbstractStopping, :: T)`

Check the resources and the time of the upper problem if `main_stp != VoidStopping`.

Note: - Modify the meta of the `main_stp`.   
      - return `false` for `VoidStopping`.
"""
function _main_pb_check!(stp::AbstractStopping, x::T) where {T}
  max_time = _tired_check!(stp.main_stp, x)
  resources = _resources_check!(stp.main_stp, x)
  main_main_pb = _main_pb_check!(stp.main_stp, x)

  check = max_time || resources || main_main_pb

  if check
    stp.meta.main_pb = true
  end

  return stp.meta.main_pb
end

function _main_pb_check!(stp::VoidStopping, x::T) where {T}
  return false
end

"""
    `_unbounded_and_domain_x_check!(:: AbstractStopping, :: T)`

Check if x gets too big, and if it has NaN or missing values.

Note:
- compare `||x||_∞` with `meta.unbounded_x` and update `meta.unbounded`.   
- it also checks `NaN` and `missing` and update `meta.domainerror`.    
- short-circuiting if one of the two is `true`.
"""
function _unbounded_and_domain_x_check!(stp::AbstractStopping, x::T) where {T}
  bigX(z::eltype(T)) = (abs(z) >= stp.meta.unbounded_x)
  (stp.meta.unbounded, stp.meta.domainerror) = _large_and_domain_check(bigX, x)
  return stp.meta.unbounded || stp.meta.domainerror
end

function _large_and_domain_check(f, itr)
  for x in itr
    v = f(x)
    w = ismissing(x) || isnan(x)
    if w
      return (false, true)
    elseif v
      return (true, false)
    end
  end
  return (false, false)
end

"""
    `_unbounded_problem_check!(:: AbstractStopping, :: T)`

Check if problem relative informations are unbounded

Note: Do nothing by default.
"""
function _unbounded_problem_check!(stp::AbstractStopping, x::T) where {T}
  return stp.meta.unbounded_pb
end

"""
    `_infeasibility_check!(:: AbstractStopping, :: T)`

Check if problem is infeasible.

Note: `meta.infeasible` is `false` by default.
"""
function _infeasibility_check!(stp::AbstractStopping, x::T) where {T}
  return stp.meta.infeasible
end

"""
    `_optimality_check!(:: AbstractStopping; kwargs...)`

Compute the optimality score.

"""
function _optimality_check!(
  stp::AbstractStopping{Pb, M, SRC, T, MStp, LoS};
  kwargs...,
) where {Pb, M, SRC, T, MStp, LoS}
  setfield!(
    stp.current_state,
    :current_score,
    stp.meta.optimality_check(stp.pb, stp.current_state; kwargs...),
  )

  return stp.current_state.current_score
end

"""
    `_null_test(:: AbstractStopping, :: T)`

Check if the score is close enough to zero (up to some precisions found in the meta).

Note:   
- the second argument is compared with 
`meta.tol_check(meta.atol, meta.rtol, meta.optimality0)`,
and `meta.tol_check_neg(meta.atol, meta.rtol, meta.optimality0)`.    
- Compatible sizes is not verified.
"""
function _null_test(stp::AbstractStopping, optimality::T) where {T}
  check_pos, check_neg = tol_check(stp.meta)
  optimal = _inequality_check(optimality, check_pos, check_neg)

  return optimal
end

#remove the Missing option here
_inequality_check(opt::Number, check_pos::Number, check_neg::Number) =
  (opt <= check_pos) && (opt >= check_neg)
_inequality_check(opt, check_pos::Number, check_neg::Number)::Bool =
  !any(z -> (ismissing(z) || (z > check_pos) || (z < check_neg)), opt)
function _inequality_check(opt::T, check_pos::T, check_neg::T) where {T}
  size_check = try
    n = size(opt)
    ncp, ncn = size(check_pos), size(check_neg)
    n != ncp || n != ncn
  catch
    false
  end

  if size_check
    throw(
      ErrorException(
        "Error: incompatible size in _null_test wrong size of optimality, tol_check and tol_check_neg",
      ),
    )
  end

  for (o, cp, cn) in zip(opt, check_pos, check_neg)
    v = o > cp || o < cn
    if v
      return false
    end
  end

  return true
end

"""
    `_user_check!( :: AbstractStopping, x :: T, start :: Bool)`

Nothing by default.

Call the `user_check_func!(:: AbstractStopping, :: Bool)` from the meta.
The boolean `start` is `true` when called from the `start!` function.
"""
function _user_check!(stp::AbstractStopping, x::T, start::Bool) where {T}
  #callback function furnished by the user
  stp.meta.user_check_func!(stp, start)

  return stp.meta.stopbyuser
end

function _user_check!(stp::AbstractStopping, x::T) where {T}
  return _user_check!(stp, x, false)
end

const status_meta_list = Dict([
  (:Optimal, :optimal),
  (:SubProblemFailure, :fail_sub_pb),
  (:SubOptimal, :suboptimal),
  (:Unbounded, :unbounded),
  (:UnboundedPb, :unbounded_pb),
  (:Stalled, :stalled),
  (:IterationLimit, :iteration_limit),
  (:TimeLimit, :tired),
  (:EvaluationLimit, :resources),
  (:ResourcesOfMainProblemExhausted, :main_pb),
  (:Infeasible, :infeasible),
  (:StopByUser, :stopbyuser),
  (:Exception, :exception),
  (:DomainError, :domainerror),
])

"""
    `status(:: AbstractStopping; list = false)`

Returns the status of the algorithm:

The different statuses are:
- `Optimal`: reached an optimal solution.
- `SubProblemFailure`
- `SubOptimal`: reached an acceptable solution.
- `Unbounded`: current iterate too large in norm.
- `UnboundedPb`: unbouned problem.
- `Stalled`: stalled algorithm.
- `IterationLimit`: too many iterations of the algorithm.
- `TimeLimit`: time limit.
- `EvaluationLimit`: too many ressources used,
                          i.e. too many functions evaluations.
- `ResourcesOfMainProblemExhausted`: in the case of a substopping, EvaluationLimit or TimeLimit
  for the main stopping.
- `Infeasible`: default return value, if nothing is done the problem is
               considered feasible.
- `StopByUser`: stopped by the user.
- `DomainError`: there is a NaN somewhere.
- `Exception`: unhandled exception
- `Unknwon`: if stopped for reasons unknown by Stopping.

Note:
- Set keyword argument `list` to true, to get an `Array` with all the statuses.   
- The different statuses correspond to boolean values in the meta.   
"""
function status(stp::AbstractStopping; list = false)
  if list
    list_status = findall(x -> getfield(stp.meta, x), status_meta_list)
    if list_status == zeros(0)
      list_status = [:Unknown]
    end
  else
    list_status = findfirst(x -> getfield(stp.meta, x), status_meta_list)
    if isnothing(list_status)
      list_status = :Unknown
    end
  end

  return list_status
end

"""
    `elapsed_time(:: AbstractStopping)`

Returns the elapsed time.

`current_time` and `start_time` are NaN if not initialized.
"""
function elapsed_time(stp::AbstractStopping)
  return stp.current_state.current_time - stp.meta.start_time
end
