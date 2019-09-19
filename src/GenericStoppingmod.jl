export GenericStopping,  start!, stop!, update_and_start!, update_and_stop!
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
"""
mutable struct GenericStopping <: AbstractStopping

	# Problem
	pb :: Any

	# Problem stopping criterion
	meta :: StoppingMeta

	# Current information on the problem
	current_state :: AbstractState

	function GenericStopping(pb               :: Any,
				 current_state    :: AbstractState;
                                 meta             :: StoppingMeta = StoppingMeta(),
				 kwargs...)

	 if !(isempty(kwargs))
	  meta = StoppingMeta(; kwargs...)
	 end

         return new(pb, meta, current_state)
	end
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
"""
function start!(stp :: AbstractStopping)

 stt_at_x = stp.current_state
 x        = stt_at_x.x

 #Initialize the time counter
 if isnan(stp.meta.start_time)
  stp.meta.start_time = time()
 end
 #and synchornize with the State
 if isnan(stt_at_x.start_time)
  stt_at_x.start_time = time()
 end

 # Optimality check
 optimality0          = _optimality_check(stp)
 stp.meta.optimality0 = optimality0
 stp.meta.optimal     = _null_test(stp, optimality0)

 OK = stp.meta.optimal

 return OK
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
etc)."""
function stop!(stp :: AbstractStopping)

 stt_at_x = stp.current_state
 x        = stt_at_x.x
 time     = stp.meta.start_time #stt_at_x.start_time

 # Optimality check
 stp.meta.optimal = _null_test(stp,_optimality_check(stp))

 # global user limit diagnostic
 _unbounded_check!(stp, x)
 _tired_check!(stp, x, time_t = time)
 _resources_check!(stp, x)
 _stalled_check!(stp, x)

 OK = stp.meta.optimal || stp.meta.tired || stp.meta.stalled || stp.meta.unbounded

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
_stalled_check!: Checks if the optimization algorithm is stalling.
"""
function _stalled_check!(stp :: AbstractStopping,
                         x   :: Iterate)

 max_iter = stp.meta.nb_of_stop >= stp.meta.max_iter

 stp.meta.stalled = max_iter || stp.meta.optimal_sub_pb

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

 ##############
 ##############
  # Maximum number of function and derivative(s) computation
  # temporaire, fonctionne seulement pour les NLPModels
  if typeof(stp.pb) <: AbstractNLPModel
	  max_evals = (neval_obj(stp.pb) + neval_grad(stp.pb) + neval_hprod(stp.pb) + neval_hess(stp.pb)) > stp.meta.max_eval
	  max_f = neval_obj(stp.pb) > stp.meta.max_f
  else
	  max_evals = false
	  max_f = false
  end
  ##############
  ##############

 # global user limit diagnostic
 stp.meta.tired = max_time || max_evals || max_f

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
_optimality_check: If we reached a good approximation of an optimum to our
problem. In it's basic form only checks the norm of the gradient.
"""
function _optimality_check(stp  :: AbstractStopping)

 optimality = Inf

 return optimality
end

"""
_null_test:
check if the optimality value is null (up to some precisions found in the meta).
"""
function _null_test(stp  :: AbstractStopping, optimality :: Number)

	atol, rtol, opt0 = stp.meta.atol, stp.meta.rtol, stp.meta.optimality0

	optimal = optimality < atol || optimality < (rtol * opt0)

	return optimal
end

"""
status:
Takes an AbstractStopping as input. Returns the status of the algorithm:
 	- Optimal : if we reached an optimal solution
	- Unbounded : if the problem doesn't have a lower bound
	- Stalled : if we did too  many iterations of the algorithm
	- Tired : if the algorithm takes too long
	- ResourcesExhausted: if we used too many ressources,
                          i.e. too many functions evaluations
	- Unfeasible : default return value, if nothing is done the problem is
				   considered unfeasible
"""
function status(stp :: AbstractStopping)
	if stp.meta.optimal
		return :Optimal
	elseif stp.meta.unbounded
		return :Unbounded
	elseif stp.meta.stalled
		return :Stalled
	elseif stp.meta.tired
		return :Tired
	elseif stp.meta.resources
		return :ResourcesExhausted
	elseif !stp.meta.feasible
		return :Unfeasible
	end
end
