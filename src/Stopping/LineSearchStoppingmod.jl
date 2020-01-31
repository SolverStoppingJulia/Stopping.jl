export LS_Stopping

################################################################################
# Line search stopping module
################################################################################

"""
LS_Stopping is designed to handle the stopping criterion of line search problems.
Let f:R→Rⁿ, then h(t) = f(x+td) where x and d are vectors and t is a scalar.
h is such that h:R→R.
h is a LineModel defined in SolverTools.jl (https://github.com/JuliaSmoothOptimizers/SolverTools.jl)
It is possible to define those stopping criterion in a NLPStopping except NLPStopping
uses vectors operations. LS_Stopping and it's admissible functions (Armijo and Wolfe are provided with Stopping.jl)
uses scalar operations.
In order to work properly within the Stopping framework, admissible functions must
return a value that will be compare to 0.
For instance, the armijo condition is
h(t)-h(0)-τ₀*t*h'(0) ⩽ 0
therefore armijo(h, h_at_t) returns the maximum between h(t)-h(0)-τ₀*t*h'(0) and 0.
The inputs of an admissible function are :
    - h 	 :: A LineModel
    - h_at_t :: A line search state, defined in State.jl
"""
mutable struct LS_Stopping <: AbstractStopping
    # problem
    pb :: Any # hard to define a proper type to avoid circular dependencies
              # I don't know the right solution to this situation...

    # stopping criterion proper to linesearch
    optimality_check :: Function

    # shared information with linesearch and other stopping
    meta :: AbstractStoppingMeta

    # current information on linesearch
    current_state :: LSAtT

    # Stopping of the main problem, or nothing
    main_stp :: Union{AbstractStopping, Nothing}

    function LS_Stopping(pb             :: Any,
                         admissible     :: Function,
                         current_state  :: LSAtT;
                         meta           :: AbstractStoppingMeta = StoppingMeta(),
                         main_stp       :: Union{AbstractStopping, Nothing} = nothing,
                         kwargs...)

        if !(isempty(kwargs))
           meta = StoppingMeta(;kwargs...)
		end

        return new(pb, admissible, meta, current_state, main_stp)
    end

end

"""
_unbounded_check!: If x gets too big it is likely that the problem is unbounded
                   This is a specialized version that takes into account
                   that the problem might be unbounded if the objective function
                   is unbounded from below.

Warning: evaluate the objective function is state.ht is void.
"""
function _unbounded_check!(stp  :: LS_Stopping,
                           x    :: Iterate)

 # check if x is too large
 x_too_large = norm(x,Inf) >= stp.meta.unbounded_x

 if stp.current_state.ht == nothing
     stp.current_state.ht = obj(stp.pb, x)
 end
 f_too_large = norm(stp.current_state.ht) >= stp.meta.unbounded_threshold

 stp.meta.unbounded = x_too_large || f_too_large

 return stp
end

"""
_resources_check!: Checks if the optimization algorithm has exhausted the resources.

If the stp.pb is an AbstractNLPModel check the number of evaluations of f and sum.
"""
function _resources_check!(stp    :: AbstractStopping,
                           x      :: Iterate)

 max_evals = false
 max_f     = false

 if typeof(stp.pb) <: AbstractNLPModel
  max_f = stp.meta.max_f > neval_obj(stp.pb)
  max_evals = stp.meta.max_eval > sum_counters(stp.pb)
 end

 # global limit diagnostic
 stp.meta.resources = max_evals || max_f

 return stp
end

"""
_optimality_check: If we reached a good approximation of an optimum to our
problem. In it's basic form only checks the norm of the gradient.

This is a specialized version that takes into account the structure of the
LineSearchStopping where the optimality_check function is an input.
"""
function _optimality_check(stp  :: LS_Stopping; kwargs...)

 optimality = stp.optimality_check(stp.pb, stp.current_state; kwargs...)

 return optimality
end

################################################################################
# line search admissibility functions
################################################################################
include("ls_admissible_functions.jl")
