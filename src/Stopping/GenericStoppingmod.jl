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
- (opt) listofstates : ListStates designed to store the history of States.
- (opt) user_specific_struct : Contains any structure designed by the user.

 Constructor: `GenericStopping(:: Any, :: AbstractState; meta :: AbstractStoppingMeta = StoppingMeta(), main_stp :: Union{AbstractStopping, Nothing} = nothing, user_specific_struct :: Any = nothing, kwargs...)`

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

 `GenericStopping(:: Any, :: Iterate; kwargs...)`

 Note: Keywords arguments are forwarded to the classical constructor.

 Examples:
 GenericStopping(pb, x0, rtol = 1e-1)
"""
mutable struct GenericStopping <: AbstractStopping

    # Problem
    pb :: Any

    # Problem stopping criterion
    meta :: AbstractStoppingMeta

    # Current information on the problem
    current_state :: AbstractState

    # Stopping of the main problem, or nothing
    main_stp :: Union{AbstractStopping, Nothing}

    # History of states
    listofstates :: Union{ListStates, Nothing}

    # User-specific structure
    user_specific_struct :: Any

    function GenericStopping(pb            :: Any,
                             current_state :: AbstractState;
                             meta          :: AbstractStoppingMeta = StoppingMeta(),
                             main_stp      :: Union{AbstractStopping, Nothing} = nothing,
                             list          :: Union{ListStates, Nothing} = nothing,
                             user_specific_struct :: Any = nothing,
                             kwargs...)

     if !(isempty(kwargs))
      meta = StoppingMeta(; kwargs...)
     end

     return new(pb, meta, current_state, main_stp, list, user_specific_struct)
    end
end

function GenericStopping(pb :: Any, x :: Iterate; kwargs...)
 return GenericStopping(pb, GenericState(x); kwargs...)
end

"""
fill_in!: fill in the unspecified values of the AbstractState.

`fill_in!(:: AbstractStopping, x :: Iterate)`

Note: NotImplemented for Abstract/Generic-Stopping.
"""
function fill_in!(stp :: AbstractStopping, x :: Iterate)
 return throw(error("NotImplemented function"))
end

"""
update\\_and\\_start!: update the values in the State and initialize the Stopping.
Returns the optimality status of the problem as a boolean.

`update_and_start!(:: AbstractStopping; kwargs...)`

 Note: Kwargs are forwarded to the *update!* call.
"""
function update_and_start!(stp :: AbstractStopping; kwargs...)

    update!(stp.current_state; kwargs...)
    OK = start!(stp)

    return OK
end

"""
 start!: update the Stopping and return a boolean true if we must stop.

 `start!(:: AbstractStopping; no_start_opt_check :: Bool = false, kwargs...)`

 Purpose is to know if there is a need to even perform an optimization algorithm
 or if we are at an optimal solution from the beginning. Set *no\\_start\\_opt\\_check*
 to *true* to avoid checking optimality.

 The function *start!* successively calls: *\\_domain\\_check*, *\\_optimality\\_check*,
 *\\_null\\_test*

 Note: - *start!* initialize the start\\_time (if not done before) and *meta.optimality0*.
       - Keywords argument are sent to the *\\_optimality\\_check!* call.
"""
function start!(stp :: AbstractStopping; no_start_opt_check :: Bool = false, kwargs...)

 stt_at_x = stp.current_state
 x        = stt_at_x.x

 #Initialize the time counter
 if isnan(stp.meta.start_time)
  stp.meta.start_time = time()
 end
 #and synchornize with the State
 if stt_at_x.current_time == nothing
  update!(stt_at_x, current_time = time())
 end

 stp.meta.domainerror = _domain_check(stp.current_state)
 if !stp.meta.domainerror && !no_start_opt_check
   # Optimality check
   optimality0          = _optimality_check(stp; kwargs...)
   stp.meta.optimality0 = norm(optimality0, Inf)
   if (true in isnan.(optimality0))
     stp.meta.domainerror = true
   end

   stp.meta.optimal     = _null_test(stp, optimality0)
 end

 OK = stp.meta.optimal || stp.meta.domainerror

 return OK
end

"""
 reinit!: reinitialize the MetaData in the Stopping.

 `reinit!(:: AbstractStopping; rstate :: Bool = false, kwargs...)`

 Note:
- If *rstate* is set as true it reinitializes the current State
(with the kwargs).
- If *rlist* is set as true the list of states is also reinitialized, either
set as nothing if *rstate* is true, and a list containing only the current
state if *rstate* is false.
"""
function reinit!(stp    :: AbstractStopping;
                 rstate :: Bool = false,
                 rlist  :: Bool = true,
                 kwargs...)

 stp.meta.start_time  = NaN
 stp.meta.optimality0 = 1.0

 #reinitialize the boolean status
 stp.meta.fail_sub_pb     = false
 stp.meta.unbounded       = false
 stp.meta.unbounded_pb    = false
 stp.meta.tired           = false
 stp.meta.stalled         = false
 stp.meta.iteration_limit = false
 stp.meta.resources       = false
 stp.meta.optimal         = false
 stp.meta.suboptimal      = false
 stp.meta.main_pb         = false
 stp.meta.domainerror     = false

 #reinitialize the counter of stop
 stp.meta.nb_of_stop = 0

 #reinitialize the list of states
 if rlist
  list = rstate ? nothing : ListStates(stp.current_state)
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

Note: Kwargs are forwarded to the *update!* call.
"""
function update_and_stop!(stp :: AbstractStopping; kwargs...)

 update!(stp.current_state; kwargs...)
 OK = stop!(stp)

 return OK
end

"""
stop!: update the Stopping and return a boolean true if we must stop.

`stop!(:: AbstractStopping; kwargs...)`

It serves the same purpose as *start!* in an algorithm; telling us if we
stop the algorithm (because we have reached optimality or we loop infinitely,
etc).

The function *stop!* successively calls: *\\_domain\\_check*, *\\_optimality\\_check*,
*\\_null\\_test*, *\\_unbounded\\_check!*, *\\_tired\\_check!*, *\\_resources\\_check!*,
*\\_stalled\\_check!*, *\\_iteration\\_check!*, *\\_main\\_pb\\_check!*, add\\_to\\_list!


Note:
- Kwargs are sent to the *\\_optimality\\_check!* call.
- If listofstates != nothing, call add\\_to\\_list! to update the list of State.
"""
function stop!(stp :: AbstractStopping; kwargs...)

 x        = stp.current_state.x
 time     = stp.meta.start_time

 stp.meta.domainerror = _domain_check(stp.current_state)
 if !stp.meta.domainerror
   # Optimality check
   score = _optimality_check(stp; kwargs...)
   if true in isnan.(score)
    stp.meta.domainerror = true
   end
   stp.meta.optimal = _null_test(stp, score)

   _unbounded_check!(stp, x)
   _unbounded_problem_check!(stp, x)
   _tired_check!(stp, x, time_t = time)
   _resources_check!(stp, x)
   _stalled_check!(stp, x)
   _iteration_check!(stp, x)

   if stp.main_stp != nothing
       _main_pb_check!(stp, x)
   end

   _user_check!(stp, x)
 end

 OK = stp.meta.optimal || stp.meta.tired || stp.meta.stalled
 OK = OK || stp.meta.iteration_limit || stp.meta.unbounded || stp.meta.resources
 OK = OK || stp.meta.unbounded_pb || stp.meta.main_pb || stp.meta.domainerror
 OK = OK || stp.meta.suboptimal || stp.meta.fail_sub_pb


 _add_stop!(stp)

 if stp.listofstates != nothing
  add_to_list!(stp.listofstates, stp.current_state)
 end

 return OK
end

"""
\\_add\\_stop!: increment a counter of stop.

`_add_stop!(:: AbstractStopping)`

Fonction called everytime *stop!* is called. In theory should be called once every
iteration of an algorithm.

Note: update *meta.nb\\_of\\_stop*.
"""
function _add_stop!(stp :: AbstractStopping)

 stp.meta.nb_of_stop += 1

 return stp
end

"""
\\_iteration\\_check!: check if the optimization algorithm has reached the iteration
limit.

`_iteration_check!(:: AbstractStopping,  :: Iterate)`

Note: Compare *meta.iteration_limit* with *meta.nb\\_of\\_stop*.
"""
function _iteration_check!(stp :: AbstractStopping,
                           x   :: Iterate)

 max_iter = stp.meta.nb_of_stop >= stp.meta.max_iter

 stp.meta.iteration_limit = max_iter

 return stp
end

"""
\\_stalled\\_check!: check if the optimization algorithm is stalling.

`_stalled_check!(:: AbstractStopping, :: Iterate)`

Note: By default *meta.stalled* is false for Abstract/Generic Stopping.
"""
function _stalled_check!(stp :: AbstractStopping,
                         x   :: Iterate)

 stp.meta.stalled = false

 return stp
end

"""
\\_tired\\_check!: check if the optimization algorithm has been running for too long.

`_tired_check!(:: AbstractStopping, :: Iterate; time_t :: Number = NaN)`

Note: - Return false if *time_t* is NaN (by default).
  - Update *meta.tired*.
"""
function _tired_check!(stp    :: AbstractStopping,
                       x      :: Iterate;
                       time_t :: Number = NaN)

 # Time check
 if !isnan(time_t)
    update!(stp.current_state, current_time = time())
    elapsed_time = stp.current_state.current_time - time_t
    max_time     = elapsed_time > stp.meta.max_time
 else
    max_time = false
 end

 stp.meta.tired = max_time

 return stp
end

"""
\\_resources\\_check!: check if the optimization algorithm has exhausted the resources.

`_resources_check!(:: AbstractStopping, :: Iterate)`

Note: By default *meta.resources* is false for Abstract/Generic Stopping.
"""
function _resources_check!(stp    :: AbstractStopping,
                           x      :: Iterate)

 max_evals = false
 max_f     = false

 stp.meta.resources = max_evals || max_f

 return stp
end

"""
\\_main\\_pb\\_check!: check the resources and the time of the upper problem (if main_stp != nothing).

`_main_pb_check!(:: AbstractStopping, :: Iterate)`

Note: - Modify the meta of the *main_stp*.
      - By default `meta.main_pb = false`.
"""
function _main_pb_check!(stp    :: AbstractStopping,
                         x      :: Iterate)

 # Time check
 time = stp.main_stp.meta.start_time
 _tired_check!(stp.main_stp, x, time_t = time)
 max_time = stp.main_stp.meta.tired

 # Resource check
 _resources_check!(stp.main_stp, x)
 resources = stp.main_stp.meta.resources

 if stp.main_stp.main_stp != nothing
   _main_pb_check!(stp.main_stp, x)
   main_main_pb = stp.main_stp.meta.main_pb
 else
   main_main_pb = false
 end

 stp.meta.main_pb = max_time || resources || main_main_pb

 return stp
end

"""
\\_unbounded\\_check!: check if x gets too big.

`_unbounded_check!(:: AbstractStopping, :: Iterate)`

Note: compare ||x|| with *meta.unbounded_x* and update *meta.unbounded*.
"""
function _unbounded_check!(stp  :: AbstractStopping,
                           x    :: Iterate)

 pnorm = stp.meta.norm_unbounded_x
 x_too_large = norm(x, pnorm) >= stp.meta.unbounded_x

 stp.meta.unbounded = x_too_large

 return stp
end

"""
\\_unbounded\\_problem!: check if problem relative informations are unbounded

`_unbounded_problem_check!(:: AbstractStopping, :: Iterate)`

Note: *meta.unbounded_pb* is false by default.
"""
function _unbounded_problem_check!(stp  :: AbstractStopping,
                                   x    :: Iterate)

 stp.meta.unbounded_pb = false

 return stp
end

"""
\\_optimality\\_check: compute the optimality score.

`_optimality_check(:: AbstractStopping; kwargs...)`

Note: By default returns Inf for Abstract/Generic Stopping.
"""
function _optimality_check(stp  :: AbstractStopping; kwargs...)

 optimality = stp.meta.optimality_check(stp.pb, stp.current_state; kwargs...)
 stp.current_state.current_score = optimality

 return optimality
end

"""
\\_null\\_test: check if the score is close enough to zero
(up to some precisions found in the meta).

`_null_test(:: AbstractStopping, :: Union{Number,AbstractVector})`

Note:
- the second argument is compared with `meta.tol_check(meta.atol, meta.rtol, meta.optimality0)`
and `meta.tol_check_neg(meta.atol, meta.rtol, meta.optimality0)`.
- Compatible size is not verified.
"""
function _null_test(stp  :: AbstractStopping, optimality :: Union{Number,AbstractVector})

    atol, rtol, opt0 = stp.meta.atol, stp.meta.rtol, stp.meta.optimality0
    check_pos = stp.meta.tol_check(atol, rtol, opt0)
    check_neg = stp.meta.tol_check_neg(atol, rtol, opt0)

    optimal  =  !(false in (optimality .<= check_pos))
    optimal &=  !(false in (optimality .>= check_neg))

    return optimal
end

"""
\\_user\\_check: nothing by default.

`_user_check!( :: AbstractStopping, x :: Iterate)`
"""
function _user_check!(stp :: AbstractStopping, x :: Iterate)
 nothing
end

"""
status: returns the status of the algorithm:

`status(:: AbstractStopping; list = false)`

The different status are:
- Optimal: reached an optimal solution.
- Unbounded: current iterate too large in norm.
- UnboundedPb: unbouned problem.
- Stalled: stalled algorithm.
- IterationLimit: too many iterations of the algorithm.
- Tired: algorithm too slow.
- ResourcesExhausted: too many ressources used,
                          i.e. too many functions evaluations.
- ResourcesOfMainProblemExhausted: in the case of a substopping, ResourcesExhausted or Tired
  for the main stopping.
- Infeasible: default return value, if nothing is done the problem is
               considered feasible.
- DomainError: there is a NaN somewhere.

Note:
  - Set keyword argument *list* to true, to get an Array with all the status.
  - The different status correspond to boolean values in the MetaData, see *StoppingMeta*.
"""
function status(stp :: AbstractStopping; list = false)

 tt = Dict([(:Optimal, :optimal),
            (:SubProblemFailure, :fail_sub_pb),
            (:SubOptimal, :suboptimal),
            (:Unbounded, :unbounded),
            (:UnboundedPb, :unbounded_pb),
            (:Stalled, :stalled),
            (:IterationLimit, :iteration_limit),
            (:Tired, :tired),
            (:ResourcesExhausted, :resources),
            (:ResourcesOfMainProblemExhausted, :main_pb),
            (:Infeasible, :infeasible),
            (:DomainError, :domainerror)])

 if list
  list_status = findall(x -> getfield(stp.meta, x), tt)
  if list_status == zeros(0) list_status = [:Unknown] end
 else
  list_status = findfirst(x -> getfield(stp.meta, x), tt)
  if list_status == nothing list_status = :Unknown end
 end

 return list_status
end
