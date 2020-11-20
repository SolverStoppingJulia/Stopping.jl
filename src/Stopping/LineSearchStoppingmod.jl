"""
Type: LS_Stopping (specialization of GenericStopping)

Methods: start!, stop!, update\\_and\\_start!, update\\_and\\_stop!, fill\\_in!, reinit!, status,
armijo, wolfe, armijo\\_wolfe, shamanskii_stop, goldstein

Specialization of GenericStopping.
LS_Stopping is designed to handle the stopping criterion of line search problems.
Let f:R→Rⁿ, then h(t) = f(x+td) where x and d are vectors and t is a scalar.
h is such that h:R→R.

Stopping structure for 1D non-linear programming problems.
Input :
- pb         : an Any
- state      : The information relative to the problem, see GenericState
- (opt) meta : Metadata relative to stopping criterion.
- (opt) main_stp : Stopping of the main loop in case we consider a Stopping
                          of a subproblem.
                          If not a subproblem, then nothing.
- (opt) listofstates : ListStates designed to store the history of States.
- (opt) user_specific_struct : Contains any structure designed by the user.

`LS_Stopping(:: Any, :: LSAtT; meta :: AbstractStoppingMeta = StoppingMeta(), main_stp :: Union{AbstractStopping, Nothing} = nothing, user_specific_struct :: Any = nothing, kwargs...)`


 Note:
 * The pb can be a LineModel defined in SolverTools.jl (https://github.com/JuliaSmoothOptimizers/SolverTools.jl)
 * It is possible to define those stopping criterion in a NLPStopping except NLPStopping
   uses vectors operations. LS_Stopping and it's admissible functions (Armijo and Wolfe are provided with Stopping.jl)
   uses scalar operations.
 * optimality\\_check(pb, state; kwargs...) -> Float64 is by default *armijo*
   For instance, the armijo condition is: h(t)-h(0)-τ₀*t*h'(0) ⩽ 0
   therefore armijo(h, h_at_t) returns the maximum between h(t)-h(0)-τ₀*t*h'(0) and 0.

See also GenericStopping, NLPStopping, LSAtT
 """
mutable struct LS_Stopping{Pb, M}  <: AbstractStopping{LSAtT, Pb, M}
    # problem
    pb                   :: Pb

    # shared information with linesearch and other stopping
    meta                 :: M

    # current information on linesearch
    current_state        :: LSAtT

    # Stopping of the main problem, or nothing
    main_stp             :: Union{AbstractStopping, Nothing}

    # History of states
    listofstates         :: Union{ListStates, Nothing}

    # User-specific structure
    user_specific_struct :: Any

    function LS_Stopping(pb             :: Pb,
                         current_state  :: LSAtT;
                         meta           :: M = StoppingMeta(),
                         main_stp       :: Union{AbstractStopping, Nothing} = nothing,
                         list           :: Union{ListStates, Nothing} = nothing,
                         user_specific_struct :: Any = nothing,
                         kwargs...) where {Pb <: Any, M <: AbstractStoppingMeta}

        if !(isempty(kwargs))
           meta = StoppingMeta(;optimality_check = armijo, kwargs...)
        end

        return new{Pb,M}(pb, meta, current_state, main_stp, list, user_specific_struct)
    end

end

"""
\\_unbounded\\_problem\\_check!: If x gets too big it is likely that the problem is unbounded
                           This is a specialized version that takes into account
                           that the problem might be unbounded if the objective function
                           is unbounded from below. This is the LS\\_Stopping specialization.

`_unbounded_problem_check!(stp :: LS_Stopping, x :: Iterate)`

Note: evaluate the objective function is *state.ht* is void.
"""
function _unbounded_problem_check!(stp :: LS_Stopping,
                                   x   :: T) where T <: Union{Number, AbstractVector}

 if isnan(stp.current_state.ht) && typeof(stp.pb) <: AbstractNLPModel && typeof(x) <: AbstractVector
     stp.current_state.ht = obj(stp.pb, x)
 end
 f_too_large = !isnan(stp.current_state.ht) && norm(stp.current_state.ht) >= stp.meta.unbounded_threshold

 stp.meta.unbounded_pb = f_too_large

 return stp
end

"""
\\_resources\\_check!: check if the optimization algorithm has exhausted the resources.
This is the LS\\_Stopping specialization.

`_resources_check!(:: LS_Stopping, :: Iterate)`

Note: If the problem is an AbstractNLPModel check the number of evaluations of *f* and *sum*.
"""
function _resources_check!(stp :: LS_Stopping,
                           x   :: T) where T <: Union{Number, AbstractVector}

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

################################################################################
# line search admissibility functions
#
# TODO: change the ls_admissible_functions and use tol_check et tol_check_neg to
# handle the inequality instead of a max.
################################################################################
include("ls_admissible_functions.jl")
