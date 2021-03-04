"""
 Type: GenericStopping

 Methods: start!, stop!, update\\_and\\_start!, update\\_and\\_stop!, fill\\_in!, reinit!, status

 A generic Stopping to solve instances with respect to some
 optimality conditions. Optimality is decided by computing a score, which is then
 tested to zero.

 Tracked data include:
- pb         : A problem
- state      : The information relative to the problem, see *GenericState*
- (opt) meta : Metadata relative to a stopping criterion, see *StoppingMeta*.
- (opt) main_stp : Stopping of the main loop in case we consider a Stopping
                       of a subproblem.
                       If not a subproblem, then *nothing*.
- (opt) listofstates : ListofStates designed to store the history of States.
- (opt) stopping_user_struct : Contains any structure designed by the user.

 Constructor: `GenericStopping(:: Any, :: AbstractState; meta :: AbstractStoppingMeta = StoppingMeta(), main_stp :: Union{AbstractStopping, Nothing} = nothing, stopping_user_struct :: Any = nothing, kwargs...)`

 Note: Metadata can be provided by the user or created with the Stopping
       constructor via kwargs. If a specific StoppingMeta is given and
       kwargs are provided, the kwargs have priority.

 Examples:
 GenericStopping(pb, GenericState(ones(2)), rtol = 1e-1)

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
 GenericStopping(pb, x0, rtol = 1e-1)
"""
mutable struct GenericStopping{Pb, M, SRC, T, MStp, LoS, Uss} <: AbstractStopping{Pb, M, SRC, T, MStp, LoS, Uss}

  # Problem
  pb                   :: Pb

  # Problem stopping criterion
  meta                 :: M
  stop_remote          :: SRC

  # Current information on the problem
  current_state        :: T

  # Stopping of the main problem, or nothing
  main_stp             :: MStp

  # History of states
  listofstates         :: LoS

  # User-specific structure
  stopping_user_struct :: Uss

end

function GenericStopping(pb            :: Pb,
                         meta          :: M,
                         stop_remote   :: SRC,
                         current_state :: T;
                         main_stp      :: AbstractStopping = VoidStopping(),
                         list          :: AbstractListofStates = VoidListofStates(),
                         stopping_user_struct :: Any = nothing,
                         kwargs...
                         ) where {Pb  <: Any, 
                                  M   <: AbstractStoppingMeta, 
                                  SRC <: AbstractStopRemoteControl,
                                  T   <: AbstractState}

  return GenericStopping(pb, meta, stop_remote, current_state, 
                        main_stp, list, stopping_user_struct)
end

function GenericStopping(pb            :: Pb,
                         meta          :: M,
                         current_state :: T;
                         main_stp      :: AbstractStopping = VoidStopping(),
                         list          :: AbstractListofStates = VoidListofStates(),
                         stopping_user_struct :: Any = nothing,
                         kwargs...
                         ) where {Pb <: Any, 
                                  M  <: AbstractStoppingMeta,
                                  T  <: AbstractState}
                                  
  stop_remote = StopRemoteControl() #main_stp == nothing ? StopRemoteControl() : cheap_stop_remote_control()
 
  return GenericStopping(pb, meta, stop_remote, current_state, 
                        main_stp, list, stopping_user_struct)
end

function GenericStopping(pb            :: Pb,
                         current_state :: T;
                         main_stp      :: AbstractStopping = VoidStopping(),
                         list          :: AbstractListofStates = VoidListofStates(),
                         stopping_user_struct :: Any = nothing,
                         kwargs...
                         ) where {Pb <: Any, T <: AbstractState}

  meta = StoppingMeta(; kwargs...)
  stop_remote = StopRemoteControl() #main_stp == nothing ? StopRemoteControl() : cheap_stop_remote_control()

  return GenericStopping(pb, meta, stop_remote, current_state, 
                        main_stp, list, stopping_user_struct)
end

function GenericStopping(pb            :: Pb,
                         stop_remote   :: SRC,
                         current_state :: T;
                         main_stp      :: AbstractStopping = VoidStopping(),
                         list          :: AbstractListofStates = VoidListofStates(),
                         stopping_user_struct :: Any = nothing,
                         kwargs...
                         ) where {Pb  <: Any, 
                                  SRC <: AbstractStopRemoteControl,
                                  T   <: AbstractState}
                                  
  meta = StoppingMeta(; kwargs...)
 
  return GenericStopping(pb, meta, stop_remote, current_state, 
                        main_stp, list, stopping_user_struct)
end

function GenericStopping(pb :: Any, x :: T; kwargs...) where T
  return GenericStopping(pb, GenericState(x); kwargs...)
end

"""
update!: generic update function for the Stopping

`update!(:: AbstractStopping; kwargs...)`

Shortcut for update!(stp.current_state; kwargs...)
"""
function update!(stp :: AbstractStopping; kwargs...)
  return update!(stp.current_state; kwargs...)
end

"""
fill_in!: fill in the unspecified values of the AbstractState.

`fill_in!(:: AbstractStopping, x :: Union{Number, AbstractVector})`

Note: NotImplemented for Abstract/Generic-Stopping.
"""
function fill_in!(stp :: AbstractStopping, x :: T) where T
  return throw(error("NotImplemented function"))
end

"""
update\\_and\\_start!: update values in the State and initialize the Stopping.
Returns the optimality status of the problem as a boolean.

`update_and_start!(:: AbstractStopping; kwargs...)`

 Note: 
  - Kwargs are forwarded to the `update!` call.  
  - `no_opt_check` skip optimality check in `start!` (`false` by default).  
"""
function update_and_start!(stp          :: AbstractStopping; 
                           no_opt_check :: Bool = false, 
                           kwargs...)

  if stp.stop_remote.cheap_check
    _smart_update!(stp.current_state; kwargs...)
  else
    update!(stp; kwargs...)
  end
  OK = start!(stp, no_opt_check = no_opt_check)

  return OK
end

"""
 Update the Stopping and return `true` if we must stop.

 `start!(:: AbstractStopping; no_opt_check :: Bool = false, kwargs...)`

 Purpose is to know if there is a need to even perform an optimization algorithm
 or if we are at an optimal solution from the beginning. 
 Set `no_opt_check` to `true` avoid checking optimality and domain errors.

 The function `start!` successively calls: `_domain_check(stp, x)`,
 `_optimality_check!(stp, x)`, `_null_test(stp, x)` and 
 `_user_check!(stp, x, true)`.

 Note: - `start!` initializes `stp.meta.start_time` (if not done before),
 `stp.current_state.current_time` and `stp.meta.optimality0` 
 (if `no_opt_check` is false).   
       - Keywords argument are passed to the `_optimality_check!` call.   
       - Compatible with the `StopRemoteControl`.   
"""
function start!(stp :: AbstractStopping; 
                no_opt_check :: Bool = false,
                kwargs...)

  state = stp.current_state
  src   = stp.stop_remote

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
                              _domain_check(stp.current_state, 
                                            current_score = true)
                            else 
                              stp.meta.domainerror
                            end
    if !stp.meta.domainerror && src.optimality_check
      optimality0          = _optimality_check!(stp; kwargs...)
      norm_optimality0     = norm(optimality0, Inf)
      if src.domain_check && isnan(norm_optimality0)
        stp.meta.domainerror = true
      elseif norm_optimality0 == Inf
        stp.meta.optimality0 = one(typeof(norm_optimality0))
      else
        stp.meta.optimality0 = norm_optimality0
      end

      if _null_test(stp, optimality0) stp.meta.optimal = true end
    end
  end
 
  src.user_start_check && _user_check!(stp, state.x, true)

  OK = OK_check(stp.meta)

  #do nothing if typeof(stp.listofstates) == VoidListofStates
  add_to_list!(stp.listofstates, stp.current_state)

  return OK
end

"""
 reinit!: reinitialize the meta-data in the Stopping.

 `reinit!(:: AbstractStopping; rstate :: Bool = false, kwargs...)`

 Note:
- If `rstate` is set as `true` it reinitializes the current State
(with the kwargs).
- If `rlist` is set as true the list of states is also reinitialized, either
set as a `VoidListofStates` if `rstate` is `true` or a list containing only the current
state otherwise.
"""
function reinit!(stp    :: AbstractStopping;
                 rstate :: Bool = false,
                 rlist  :: Bool = true,
                 kwargs...)

  stp.meta.start_time  = NaN
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
update\\_and\\_stop!: update the values in the State and
return the optimality status of the problem as a boolean.

`update_and_stop!(stp :: AbstractStopping; kwargs...)`

Note: Kwargs are forwarded to the `update!` call.
"""
function update_and_stop!(stp :: AbstractStopping; kwargs...)
    
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
stop!: update the Stopping and return a boolean true if we must stop.

`stop!(:: AbstractStopping; kwargs...)`

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
function stop!(stp          :: AbstractStopping; 
               no_opt_check :: Bool = false, 
               kwargs...)

  x   = stp.current_state.x
  src = stp.stop_remote

  src.unbounded_and_domain_x_check && _unbounded_and_domain_x_check!(stp, x)
  stp.meta.domainerror = if src.domain_check
                            #don't check x and current_score
                            _domain_check(stp.current_state, x = true, 
                                                          current_score = true)
                          else 
                            stp.meta.domainerror
                          end
  if !no_opt_check && !stp.meta.domainerror
    # Optimality check
    if src.optimality_check
      score = _optimality_check!(stp; kwargs...)
      if any(isnan, score)
        stp.meta.domainerror = true
      end
      if _null_test(stp, score) stp.meta.optimal = true end
    end

    src.infeasibility_check     && _infeasibility_check!(stp, x)
    src.unbounded_problem_check && _unbounded_problem_check!(stp, x)
    src.tired_check             && _tired_check!(stp, x)
    src.resources_check         && _resources_check!(stp, x)
    src.stalled_check           && _stalled_check!(stp, x)
    src.iteration_check         && _iteration_check!(stp, x)
    src.main_pb_check           && _main_pb_check!(stp, x)
    src.user_check              && _user_check!(stp, x)
  end

  OK = OK_check(stp.meta)
 
  _add_stop!(stp)

  #do nothing if typeof(stp.listofstates) == VoidListofStates
  add_to_list!(stp.listofstates, stp.current_state)

  return OK
end

"""
stop!: update the Stopping and return a boolean true if we must stop.

`cheap_stop!(:: AbstractStopping; kwargs...)`

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
function cheap_stop!(stp :: AbstractStopping; kwargs...)

  x   = stp.current_state.x
  src = stp.stop_remote

  # Optimality check
  if src.optimality_check
    score = _optimality_check!(stp; kwargs...) #update state.current_score
    if _null_test(stp, score) stp.meta.optimal = true end
  end
  OK = stp.meta.optimal

  OK = OK || (src.infeasibility_check     && _infeasibility_check!(stp, x))
  OK = OK || (src.unbounded_problem_check && _unbounded_problem_check!(stp, x))
  OK = OK || (src.tired_check             && _tired_check!(stp, x))
  OK = OK || (src.resources_check         && _resources_check!(stp, x))
  OK = OK || (src.iteration_check         && _iteration_check!(stp, x))
  OK = OK || (src.user_check              && _user_check!(stp, x))

  _add_stop!(stp)

  #do nothing if typeof(stp.listofstates) == VoidListofStates
  add_to_list!(stp.listofstates, stp.current_state)

  return OK
end

"""
\\_add\\_stop!: increment a counter of stop.

`_add_stop!(:: AbstractStopping)`

Fonction called everytime `stop!` is called. In theory should be called once 
every iteration of an algorithm.

Note: update `meta.nb_of_stop`.
"""
function _add_stop!(stp :: AbstractStopping)

  stp.meta.nb_of_stop += 1

  return stp
end

"""
\\_iteration\\_check!: check if the optimization algorithm has reached the 
iteration limit.

`_iteration_check!(:: AbstractStopping,  :: T)`

Note: Compare `meta.iteration_limit` with `meta.nb_of_stop`.
"""
function _iteration_check!(stp :: AbstractStopping,
                           x   :: T) where T

  max_iter = stp.meta.nb_of_stop >= stp.meta.max_iter
  if max_iter stp.meta.iteration_limit = true end

  return stp.meta.iteration_limit
end

"""
\\_stalled\\_check!: check if the optimization algorithm is stalling.

`_stalled_check!(:: AbstractStopping, :: T)`

Note: Do nothing by default for AbstractStopping.
"""
function _stalled_check!(stp :: AbstractStopping,
                         x   :: T) where T
  return stp.meta.stalled
end

"""
\\_tired\\_check!: check if the optimization algorithm has been running
 for too long.

`_tired_check!(:: AbstractStopping, :: T)`

Note: 
  - Return `false` if `meta.start_time` is `NaN` (by default).  
  - Update `meta.tired`.  
"""
function _tired_check!(stp :: AbstractStopping,
                       x   :: T) where T

  stime = stp.meta.start_time #can be NaN
  ctime = time()

  #Keep the current_state updated
  _update_time!(stp.current_state, ctime)

  elapsed_time = ctime - stime
  max_time     = elapsed_time > stp.meta.max_time #NaN > 1. is false

  if max_time stp.meta.tired = true end

  return stp.meta.tired
end

function _tired_check!(stp :: VoidStopping,
                       x   :: T) where T
  return false
end

"""
\\_resources\\_check!: check if the optimization algorithm has exhausted
 the resources.

`_resources_check!(:: AbstractStopping, :: T)`

Note: Do nothing by default `meta.resources` for AbstractStopping.
"""
function _resources_check!(stp :: AbstractStopping,
                           x   :: T) where T
  return stp.meta.resources
end

function _resources_check!(stp :: VoidStopping,
                           x   :: T) where T
  return false
end

"""
\\_main\\_pb\\_check!: check the resources and the time of the upper problem
 if `main_stp != VoidStopping`.

`_main_pb_check!(:: AbstractStopping, :: T)`

Note: - Modify the meta of the `main_stp`.   
      - return `false` for `VoidStopping`.
"""
function _main_pb_check!(stp :: AbstractStopping,
                         x   :: T) where T

  max_time     = _tired_check!(stp.main_stp, x)
  resources    = _resources_check!(stp.main_stp, x)
  main_main_pb = _main_pb_check!(stp.main_stp, x)

  check = max_time || resources || main_main_pb

  if check stp.meta.main_pb = true end

  return stp.meta.main_pb
end

function _main_pb_check!(stp :: VoidStopping,
                         x   :: T) where T
 return false
end

"""
\\_unbounded\\_and\\_domain\\_x\\_check!: check if x gets too big, 
and if it has NaN or missing values.

`_unbounded_and_domain_x_check!(:: AbstractStopping, :: T)`

Note:
- compare `||x||_∞` with `meta.unbounded_x` and update `meta.unbounded`.   
- it also checks `NaN` and `missing` and update `meta.domainerror`.    
- short-circuiting if one of the two is `true`.
"""
function _unbounded_and_domain_x_check!(stp :: AbstractStopping,
                                        x   :: T) where T

  bigX(z :: eltype(T)) = (abs(z) >= stp.meta.unbounded_x)
  (stp.meta.unbounded, stp.meta.domainerror) = _large_and_domain_check(bigX, x)
  return stp.meta.unbounded || stp.meta.domainerror
end

function _large_and_domain_check(f, itr)
    for x in itr
        v   = f(x)
        w   = ismissing(x) || isnan(x)
        if w
          return (false, true)
        elseif v
          return (true, false)
        end
    end
    return (false, false)
end

"""
\\_unbounded\\_problem!: check if problem relative informations are unbounded

`_unbounded_problem_check!(:: AbstractStopping, :: T)`

Note: Do nothing by default.
"""
function _unbounded_problem_check!(stp :: AbstractStopping,
                                   x   :: T) where T

 return stp.meta.unbounded_pb
end

"""
\\_infeasibility\\_check!: check if problem is infeasible

`_infeasibility_check!(:: AbstractStopping, :: T)`

Note: `meta.infeasible` is `false` by default.
"""
function _infeasibility_check!(stp :: AbstractStopping,
                               x   :: T) where T

 return stp.meta.infeasible
end

"""
\\_optimality\\_check: compute the optimality score.

`_optimality_check!(:: AbstractStopping; kwargs...)`

"""
function _optimality_check!(stp :: AbstractStopping{Pb, M, SRC, T, 
                                                    MStp, LoS, Uss};
                           kwargs...) where {Pb, M, SRC, T, MStp, LoS, Uss}

  setfield!(stp.current_state, :current_score,
            stp.meta.optimality_check(stp.pb, stp.current_state; kwargs...))

  return stp.current_state.current_score
end

"""
\\_null\\_test: check if the score is close enough to zero
(up to some precisions found in the meta).

`_null_test(:: AbstractStopping, :: T)`

Note:   
- the second argument is compared with 
`meta.tol_check(meta.atol, meta.rtol, meta.optimality0)`,
and `meta.tol_check_neg(meta.atol, meta.rtol, meta.optimality0)`.    
- Compatible sizes is not verified.
"""
function _null_test(stp  :: AbstractStopping, optimality :: T) where T

  check_pos, check_neg = tol_check(stp.meta)
  optimal = _inequality_check(optimality, check_pos, check_neg)

  return optimal
end

#remove the Missing option here
_inequality_check(opt :: Number, check_pos :: Number, check_neg :: Number) = (opt <= check_pos) && (opt >= check_neg)
_inequality_check(opt, check_pos :: Number, check_neg :: Number) :: Bool = !any(z->(ismissing(z) || (z > check_pos) || (z < check_neg)), opt)
function _inequality_check(opt :: T, check_pos :: T, check_neg :: T) where T

  size_check = try
                 n = size(opt)
                 ncp, ncn = size(check_pos), size(check_neg)
                 n != ncp || n != ncn
               catch
                 false
               end

  if size_check
    throw(ErrorException("Error: incompatible size in _null_test wrong size of optimality, tol_check and tol_check_neg"))
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
\\_user\\_check: nothing by default.

`_user_check!( :: AbstractStopping, x :: T, start :: Bool)`

Call the `user_check_func!(:: AbstractStopping, :: Bool)` from the meta.
The boolean `start` is `true` when called from the `start!` function.
"""
function _user_check!(stp :: AbstractStopping, x :: T, start :: Bool) where T
  #callback function furnished by the user
  stp.meta.user_check_func!(stp, start)
 
  return stp.meta.stopbyuser
end

function _user_check!(stp :: AbstractStopping, x :: T) where T
  return _user_check!(stp, x, false)
end

const status_meta_list = Dict([(:Optimal, :optimal),
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
             (:DomainError, :domainerror)])

"""
status: returns the status of the algorithm:

`status(:: AbstractStopping; list = false)`

The different statuses are:
- Optimal: reached an optimal solution.
- SubProblemFailure
- SubOptimal: reached an acceptable solution.
- Unbounded: current iterate too large in norm.
- UnboundedPb: unbouned problem.
- Stalled: stalled algorithm.
- IterationLimit: too many iterations of the algorithm.
- TimeLimit: time limit.
- EvaluationLimit: too many ressources used,
                          i.e. too many functions evaluations.
- ResourcesOfMainProblemExhausted: in the case of a substopping, EvaluationLimit or TimeLimit
  for the main stopping.
- Infeasible: default return value, if nothing is done the problem is
               considered feasible.
- StopByUser: stopped by the user.
- DomainError: there is a NaN somewhere.
- Exception: unhandled exception
- Unknwon: if stopped for reasons unknown by Stopping.

Note:
- Set keyword argument `list` to true, to get an `Array` with all the statuses.   
- The different statuses correspond to boolean values in the meta.   
"""
function status(stp :: AbstractStopping; list = false)

  if list
    list_status = findall(x -> getfield(stp.meta, x), status_meta_list)
    if list_status == zeros(0) list_status = [:Unknown] end
  else
    list_status = findfirst(x -> getfield(stp.meta, x), status_meta_list)
    if isnothing(list_status) list_status = :Unknown end
  end

  return list_status
end

"""
elapsed_time: returns the elapsed time.

`elapsed_time(:: AbstractStopping)`

`current_time` and `start_time` are NaN if not initialized.
"""
function elapsed_time(stp :: AbstractStopping)
  return stp.current_state.current_time - stp.meta.start_time
end
