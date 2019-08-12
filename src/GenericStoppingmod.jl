export GenericStopping,  start!, stop!, update_and_start!, update_and_stop!
export _stalled_check!, fill_in!

"""
 Type : GenericStopping
 Methods : start!, stop!

 A generic stopping criterion to solve instances (pb) with respect to some
 optimality conditions (optimality_check)
 Besides optimality conditions, we consider classical emergency exit:
 - unbounded problem
 - stalled problem
 - tired problem (measured by the number of evaluations of functions and time)
"""
mutable struct GenericStopping <: AbstractStopping
	# problem
	pb :: Any

	# Problem stopping criterion
	meta :: StoppingMeta

	# information courante sur le Line search
	current_state :: AbstractState

	function GenericStopping(pb               :: Any,
							 current_state    :: AbstractState;
                             meta             :: StoppingMeta = StoppingMeta())

        return new(pb, meta, current_state)
	end
end

"""
update_and_start!: TO DO
"""
function update_and_start!(stp :: AbstractStopping; kwargs...)
	update!(stp.current_state; kwargs...)
	OK = start!(stp)
end

"""
fill_in! : A function that fill in the unspecified values of the AbstractState.
"""
function fill_in!(stp :: AbstractStopping, x :: Iterate)
 return throw(error("NotImplemented function"))
end

"""
 start! Inputs: Interface Stopping. Output: optimal or not. Purpose is to
 know if there is a need to even perform an optimization algorithm or if we are
 at an optimal solution from the beginning.
"""
function start!(stp      :: AbstractStopping)

 rst_at_x = stp.current_state
 x = rst_at_x.x
 rst_at_x.start_time = time()

 # Optimality check
 stp.meta.optimal = _null_test(stp,_optimality_check(stp))

 OK = stp.meta.optimal

 return OK
end

"""
update_and_stop!: TO DO
"""
function update_and_stop!(stp :: AbstractStopping; kwargs...)
 update!(stp.current_state; kwargs...)
 OK = stop!(stp)
 return OK
end

""" stop! Inputs: Interface Stopping. Output: optimal or not.
Serves the same purpose as start! When in an algorithm, tells us if we
stop the algorithm (because we have reached optimality or we loop infinitely,
etc)."""
function stop!(stp      :: AbstractStopping)
 rst_at_x = stp.current_state
 x = rst_at_x.x
 time = rst_at_x.start_time

 # Optimality check
 stp.meta.optimal = _null_test(stp,_optimality_check(stp))

 # global user limit diagnostic
 _unbounded_check!(stp, x)
 _tired_check!(stp, x, time_t = time)
 _stalled_check!(stp, x)

 OK = stp.meta.optimal || stp.meta.tired || stp.meta.stalled || stp.meta.unbounded
 add_stop!(stp.meta)
 return OK
end

"""_stalled_check. Checks if the optimization algorithm is stalling."""
function _stalled_check!(stp    :: AbstractStopping,
                         x      :: Iterate;
                         dx     :: Iterate = Number[],
                         df     :: Iterate = Number[])

 # Stalled iterates
 stalled_x = norm(stp.current_state.dx,Inf) <= norm(x, Inf)*stp.meta.rtol_x
 stalled_f = norm(stp.current_state.df,Inf) <= norm(x, Inf)*stp.meta.rtol_f

 max_iter = stp.meta.nb_of_stop >= stp.meta.max_iter

 stp.meta.stalled = stalled_x || stalled_f || max_iter || stp.meta.stalled_linesearch

 return stp
end

"""_tired_check. Checks if the optimization algorithm is "tired" (i.e.
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

  # Maximum number of function and derivative(s) computation
  # temporaire, fonctionne seulement pour les NLPModels
  if typeof(stp.pb) <: AbstractNLPModel
	  max_evals = (neval_obj(stp.pb) + neval_grad(stp.pb) + neval_hprod(stp.pb) + neval_hess(stp.pb)) > stp.meta.max_eval
  else
	  max_evals = false
  end
 # global user limit diagnostic
 stp.meta.tired = max_time || max_evals

 # print_with_color(:yellow, "tired = $(stp.meta.tired) \n")
 return stp
end

"""_unbounded_check! If x gets too big it is likely that the problem is unbounded"""
function _unbounded_check!(stp  :: AbstractStopping,
                           x    :: Iterate)
printstyled("unbounded  ✓ \n", color=:yellow
 # check if x is too large
 x_too_large = norm(x,Inf) >= stp.meta.unbounded_x

 stp.meta.unbounded = x_too_large
 # print_with_color(:yellow, "unbounded = $(stp.meta.unbounded) \n")

 return stp
end

"""_optimality_check. If we reached a good approximation of an optimum to our
problem. In it's basic form only checks the norm of the gradient."""
function _optimality_check(stp  :: AbstractStopping)

 # print_with_color(:red, "on passe dans le _optimality_check de GenericStoppingmod \n")
 optimality = Inf
 # print_with_color(:yellow, "optimal = $optimality \n")
 return optimality
end

"""
check if the optimality value is null (up to some precisions found in the meta).
"""
function _null_test(stp  :: AbstractStopping, optimality :: Number)

 atol, rtol = stp.meta.atol, stp.meta.rtol
 # print_with_color(:blue, "optimality = $optimality atol = $atol rtol = $rtol \n")
 optimal = optimality < atol || optimality < (rtol * optimality)

 return optimal
end
