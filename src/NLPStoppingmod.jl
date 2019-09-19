export NLPStopping, unconstrained, fill_in!

################################################################################
# Specific stopping module for non linear problems
################################################################################

"""
Stopping structure for non-linear programming problems.
Inputs:
 - pb : An AbstractNLPModel
 - main_pb : An AbstractNLPModel or nothing
 - optimality_check : a stopping criterion through an admissibility function
 - meta : StoppingMeta
 - max_cntrs :: Dict contains the max number of evaluations
 - current_state : the current state of the problem (i.e an NLPAtX)

 * The main_pb entry is designed to handle the case where the Stopping
 is used to solve a problem as a subproblem of a main problem.
 If main_pb = nothing, then pb acts as the main problem.
 This is used in the fill_in! (TODO) and can be used in optimality_check (TODO).

 * optimality_check : takes two inputs (AbstractNLPModel, NLPAtX)
 and returns a Float64 to be compared at 0.
 (Idée: ajouter une nouvelle input main_pb dans le optimality_check?)
 """
mutable struct NLPStopping <: AbstractStopping

	# problem
	pb :: AbstractNLPModel

    # main problem
    main_pb :: Union{AbstractNLPModel, Nothing}

	# stopping criterion
	optimality_check :: Function # will be put in optimality_check

	# Common parameters
	meta      :: StoppingMeta
    # Parameters specific to the NLPModels
    max_cntrs :: Dict #contains the max number of evaluations

	# current state of the line search Algorithm
	current_state :: AbstractState

	function NLPStopping(pb         	:: AbstractNLPModel,
						 admissible 	:: Function,
						 current_state 	:: AbstractState;
						 meta       	:: StoppingMeta = StoppingMeta(),
                         max_cntrs      :: Dict = _init_max_counters(),
                         main_pb        :: Union{AbstractNLPModel, Nothing} = nothing,
						 kwargs...)

		if !(isempty(kwargs))
			meta = StoppingMeta(;kwargs...)
		end

		return new(pb, main_pb, admissible, meta, max_cntrs, current_state)
	end

end

"""
_init_max_counters(): initialize the maximum number of evaluations on each of
                        the functions present in the Counters (NLPModels).
"""
function _init_max_counters()

    cntrs = Dict([(:neval_obj,    20000), (:neval_grad,   20000),
                  (:neval_cons,   20000), (:neval_jcon,   20000),
                  (:neval_jgrad,  20000), (:neval_jac,    20000),
                  (:neval_jprod,  20000), (:neval_jtprod, 20000),
                  (:neval_hess,   20000), (:neval_hprod,  20000),
                  (:neval_jhprod, 20000), (:neval_sum,    20000*11)])

 return cntrs
end

"""
fill_in! : A function that fill in the required values in the State

TO DO: constraint version
"""
function fill_in!(stp  :: NLPStopping,
                  x    :: Iterate;
                  fx   :: Iterate    = nothing,
                  gx   :: Iterate    = nothing)

 obfx = fx == nothing  ? obj(stp.pb, x)   : fx
 grgx = gx == nothing  ? grad(stp.pb, x)  : gx

 return update!(stp.current_state, x=x, fx = obfx, gx = grgx)
end

"""
_resources_check!: Checks if the optimization algorithm has exhausted the resources.
                    This is the NLP specialized version that takes into account
                    the evaluation of the functions following the sum_counters
                    structure from NLPModels.
"""
function _resources_check!(stp    :: NLPStopping,
                           x      :: Iterate)

  cntrs = stp.current_state.evals #Counters in the state
  max_cntrs = stp.max_cntrs

  # check all the entries in the counter
  max_f = false
  for f in fieldnames(Counters)
      max_f = max_f && (max_cntrs[f] > getfield(cntrs, f))
  end

 # Maximum number of function and derivative(s) computation
 max_evals = sum_counters(stp.pb) > max_cntrs[:neval_sum]

 # global user limit diagnostic
 stp.meta.resources = max_evals || max_f

 return stp
end

"""
_unbounded_check!: If x gets too big it is likely that the problem is unbounded
                   This is the NLP specialized version that takes into account
                   that the problem might be unbounded if the objective function
                   is unbounded from below.
"""
function _unbounded_check!(stp  :: NLPStopping,
                           x    :: Iterate)

 # check if x is too large
 x_too_large = norm(x,Inf) >= stp.meta.unbounded_x

 if isnan(stp.current_state.fx)
	 stp.current_state.fx = obj(stp.pb, x)
 end
 f_too_large = stp.current_state.fx <= stp.meta.unbounded_threshold

 stp.meta.unbounded = x_too_large || f_too_large

 return stp
end

"""
_optimality_check: If we reached a good approximation of an optimum to our
problem. In it's basic form only checks the norm of the gradient.

This is the NLP specialized version that takes into account the structure of the
NLPStopping where the optimality_check function is an input.
"""
function _optimality_check(stp  :: NLPStopping)

 optimality = stp.optimality_check(stp.pb, stp.current_state)

 return optimality
end

################################################################################
# non linear problems admissibility functions
################################################################################
include("nlp_admissible_functions.jl")
