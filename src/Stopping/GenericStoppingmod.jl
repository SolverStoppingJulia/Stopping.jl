export GenericStopping,  start!, reinit!, stop!, update_and_start!, update_and_stop!
export fill_in!, status

"""
 Type : GenericStopping
 Methods : start!, stop!

 A generic stopping criterion to solve instances (pb) with respect to some
 optimality conditions (optimality_check)
 Besides optimality conditions, we consider classical emergency exit:
 - unbounded problem
 - stalled problem
 - tired problem (measured by the number of evaluations of functions and time)

 Input :
    - pb         : An problem
    - state      : The information relative to the problem
    - (opt) meta : Metadata relative to stopping criterion. Can be provided by
				   the user or created with the Stopping constructor with kwargs
				   If a specific StoppingMeta is given as well as kwargs are
				   provided, the kwargs have priority.
    - (opt) main_stp : Stopping of the main loop in case we consider a Stopping
                       of a subproblem.
                       If not a subproblem, then of type Nothing.
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

    function GenericStopping(pb               :: Any,
                             current_state    :: AbstractState;
                             meta             :: AbstractStoppingMeta = StoppingMeta(),
                             main_stp         :: Union{AbstractStopping, Nothing} = nothing,
                             kwargs...)

     if !(isempty(kwargs))
      meta = StoppingMeta(; kwargs...)
     end

     return new(pb, meta, current_state, main_stp)
    end
end

"""
GenericStopping(pb, x): additional default constructor
The function creates a Stopping where the State is by default.
Keywords arguments are forwarded to the classical constructor.
"""
function GenericStopping(pb :: Any, x :: Iterate; kwargs...)
 return GenericStopping(pb, GenericState(x); kwargs...)
end

"""
update_and_start!: Update the values in the State and initializes the Stopping
Returns the optimity status of the problem.
"""
function update_and_start!(stp :: AbstractStopping; kwargs...)

    update!(stp.current_state; kwargs...)
    OK = start!(stp)

    return OK
end

"""
fill_in!: A function that fill in the unspecified values of the AbstractState.
"""
function fill_in!(stp :: AbstractStopping, x :: Iterate)
 return throw(error("NotImplemented function"))
end

"""
 start!:
 Input: Stopping.
 Output: optimal or not.
 Purpose is to know if there is a need to even perform an optimization algorithm or if we are
 at an optimal solution from the beginning.

 Note: start! initialize the start_time (if not done before) and meta.optimality0.

 Keywords argument are sent to the _optimality_check!
"""
function start!(stp :: AbstractStopping; kwargs...)

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

 # Optimality check
 optimality0          = _optimality_check(stp; kwargs...)
 stp.meta.optimality0 = optimality0
 if isnan(optimality0)
   #printstyled("DomainError: optimality0 is NaN\n", color = :red)
   stp.meta.domainerror = true
 end

 stp.meta.optimal     = _null_test(stp, optimality0)

 OK = stp.meta.optimal || stp.meta.domainerror

 return OK
end

"""
 reinit!:
 Input: Stopping.
 Output: Stopping modified.
 Reinitialize the meta data filled in by the start!

 If rstate is set as true it reinitialize the current_state
 (with the kwargs)
"""
function reinit!(stp :: AbstractStopping; rstate :: Bool = false, kwargs...)

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

 #reinitialize the state
 if rstate
  reinit!(stp.current_state; kwargs...)
 end

 return stp
end

"""
update_and_stop!: Update the values in the State and returns the optimity status
of the problem.
"""
function update_and_stop!(stp :: AbstractStopping; kwargs...)

 update!(stp.current_state; kwargs...)
 OK = stop!(stp)

 return OK
end

"""
stop!:
Inputs: Interface Stopping. Output: optimal or not.
Serves the same purpose as start! When in an algorithm, tells us if we
stop the algorithm (because we have reached optimality or we loop infinitely,
etc).

Keywords argument are sent to the _optimality_check!
"""
function stop!(stp :: AbstractStopping; kwargs...)

 x        = stp.current_state.x
 time     = stp.meta.start_time

 # Optimality check
 score = _optimality_check(stp; kwargs...)
 if isnan(score)
  #printstyled("DomainError: score is NaN\n", color = :red)
  stp.meta.domainerror = true
 end
 stp.meta.optimal = _null_test(stp, score)

 # global user limit diagnostic
 _unbounded_check!(stp, x)
 _unbounded_problem_check!(stp, x)
 _tired_check!(stp, x, time_t = time)
 _resources_check!(stp, x)
 _stalled_check!(stp, x)
 _iteration_check!(stp, x)

 if stp.main_stp != nothing
     _main_pb_check!(stp, x)
 end

 OK = stp.meta.optimal || stp.meta.tired || stp.meta.stalled
 OK = OK || stp.meta.iteration_limit || stp.meta.unbounded
 OK = OK || stp.meta.unbounded_pb || stp.meta.main_pb || stp.meta.domainerror
 OK = OK || stp.meta.suboptimal || stp.meta.fail_sub_pb


 _add_stop!(stp)

 return OK
end

"""
_add_stop!:
Fonction called everytime stop! is called. In theory should be called once every
iteration of an algorithm
"""
function _add_stop!(stp :: AbstractStopping)

 stp.meta.nb_of_stop += 1

 return stp
end

"""
_iteration_check!: Checks if the optimization algorithm is reached the iteration
limit.
"""
function _iteration_check!(stp :: AbstractStopping,
                           x   :: Iterate)

 max_iter = stp.meta.nb_of_stop >= stp.meta.max_iter

 stp.meta.iteration_limit = max_iter

 return stp
end

"""
_stalled_check!: Checks if the optimization algorithm is stalling.

false by default.
"""
function _stalled_check!(stp :: AbstractStopping,
                         x   :: Iterate)

 stp.meta.stalled = false

 return stp
end

"""
_tired_check!: Checks if the optimization algorithm is "tired" (i.e.
been running too long)
"""
function _tired_check!(stp    :: AbstractStopping,
                       x      :: Iterate;
                       time_t :: Number = NaN)

 # Time check
 if !isnan(time_t)
    elapsed_time = time() - time_t
    max_time     = elapsed_time > stp.meta.max_time
 else
    max_time = false
 end

 # global user limit diagnostic
 stp.meta.tired = max_time

 return stp
end

"""
_resources_check!: Checks if the optimization algorithm has exhausted the resources.
"""
function _resources_check!(stp    :: AbstractStopping,
                           x      :: Iterate)

 max_evals = false
 max_f     = false

 # global limit diagnostic
 stp.meta.resources = max_evals || max_f

 return stp
end

"""
_main_pb_check!: Checks the resources of the upper problem
(if main_stp != nothing)
!! By default stp.meta.main_pb = false
!! Modify the meta of the main_stp
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

 # global user limit diagnostic
 stp.meta.main_pb = max_time || resources || main_main_pb

 return stp
end

"""
_unbounded_check!: If x gets too big it is likely that the problem is unbounded
"""
function _unbounded_check!(stp  :: AbstractStopping,
                           x    :: Iterate)

 # check if x is too large
 x_too_large = norm(x,Inf) >= stp.meta.unbounded_x

 stp.meta.unbounded = x_too_large

 return stp
end

"""
_unbounded_problem!: check if problem relative informations are unbounded

not implemented.
"""
function _unbounded_problem_check!(stp  :: AbstractStopping,
                                   x    :: Iterate)

 stp.meta.unbounded_pb = false

 return stp
end

"""
_optimality_check: If we reached a good approximation of an optimum to our
problem.
"""
function _optimality_check(stp  :: AbstractStopping; kwargs...)
 return Inf
end

"""
_null_test:
check if the optimality value is null (up to some precisions found in the meta).
"""
function _null_test(stp  :: AbstractStopping, optimality :: Number)

    atol, rtol, opt0 = stp.meta.atol, stp.meta.rtol, stp.meta.optimality0

    #optimal = optimality < atol || optimality < (rtol * opt0)
    optimal = optimality <= stp.meta.tol_check(atol, rtol, opt0)

    return optimal
end

"""
status:
Takes an AbstractStopping as input. Returns the status of the algorithm:
    - Optimal : if we reached an optimal solution
    - Unbounded : if the problem doesn't have a lower bound
    - Stalled : if algorithm is stalling
    - IterationLimit : if we did too  many iterations of the algorithm
    - Tired : if the algorithm takes too long
    - ResourcesExhausted: if we used too many ressources,
                          i.e. too many functions evaluations
    - ResourcesOfMainProblemExhausted: in the case of a substopping, ResourcesExhausted or Tired
    for the main stopping.
    - Infeasible : default return value, if nothing is done the problem is
                   considered infeasible
    - DomainError : there is a NaN somewhere

 set list to true, to get an Array with all the valid status.
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
