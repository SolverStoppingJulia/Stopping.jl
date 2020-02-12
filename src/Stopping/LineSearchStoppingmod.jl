"""
Type: LS_Stopping (specialization of GenericStopping)
Methods: start!, stop!, update_and_start!, update_and_stop!, fill_in!, reinit!, status

LS_Stopping is designed to handle the stopping criterion of line search problems.
Let f:R→Rⁿ, then h(t) = f(x+td) where x and d are vectors and t is a scalar.
h is such that h:R→R.

Stopping structure for non-linear programming problems using NLPModels.
    Input :
       - pb         : an AbstractNLPModel
       - optimality_check : a stopping criterion through an admissibility function
       - state      : The information relative to the problem, see GenericState
       - (opt) meta : Metadata relative to stopping criterion.
       - (opt) main_stp : Stopping of the main loop in case we consider a Stopping
                          of a subproblem.
                          If not a subproblem, then nothing.

 Note:
 * The pb can be a LineModel defined in SolverTools.jl (https://github.com/JuliaSmoothOptimizers/SolverTools.jl)
 * It is possible to define those stopping criterion in a NLPStopping except NLPStopping
   uses vectors operations. LS_Stopping and it's admissible functions (Armijo and Wolfe are provided with Stopping.jl)
   uses scalar operations.
 * optimality_check(pb, state; kwargs...) -> Float64
   For instance, the armijo condition is: h(t)-h(0)-τ₀*t*h'(0) ⩽ 0
   therefore armijo(h, h_at_t) returns the maximum between h(t)-h(0)-τ₀*t*h'(0) and 0.
 """
mutable struct LS_Stopping <: AbstractStopping
    # problem
    pb :: Any

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
_unbounded_problem_check!: If x gets too big it is likely that the problem is unbounded
                           This is a specialized version that takes into account
                           that the problem might be unbounded if the objective function
                           is unbounded from below.

Note: evaluate the objective function is state.ht is void.
"""
function _unbounded_problem_check!(stp  :: LS_Stopping,
                                   x    :: Iterate)

 if stp.current_state.ht == nothing && typeof(stp.pb) <: AbstractNLPModel
     stp.current_state.ht = obj(stp.pb, x)
 end
 f_too_large = stp.current_state.ht != nothing && norm(stp.current_state.ht) >= stp.meta.unbounded_threshold

 stp.meta.unbounded_pb = f_too_large

 return stp
end

"""
_resources_check!: check if the optimization algorithm has exhausted the resources.

If the problem is an AbstractNLPModel check the number of evaluations of f and sum.
"""
function _resources_check!(stp    :: LS_Stopping,
                           x      :: Iterate)

 max_evals = false
 max_f     = false

 if typeof(stp.pb) <: AbstractNLPModel
  max_f = stp.meta.max_f < neval_obj(stp.pb)
  max_evals = stp.meta.max_eval < sum_counters(stp.pb)
 end

 # global limit diagnostic
 stp.meta.resources = max_evals || max_f

 return stp
end

"""
_optimality_check: compute the optimality score.

This is the NLP specialized version that takes into account the structure of the
LS_Stopping where the optimality_check function is an input.
"""
function _optimality_check(stp  :: LS_Stopping; kwargs...)

 optimality = stp.optimality_check(stp.pb, stp.current_state; kwargs...)

 return optimality
end

################################################################################
# line search admissibility functions
################################################################################
include("ls_admissible_functions.jl")
