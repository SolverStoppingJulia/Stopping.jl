"""
Type: NLPStopping

Methods: start!, stop!, update\\_and\\_start!, update\\_and\\_stop!, fill_in!, reinit!, status
KKT, unconstrained\\_check, unconstrained2nd\\_check, optim\\_check\\_bounded

Specialization of GenericStopping. Stopping structure for non-linear programming problems using NLPModels.

Attributes:
- pb         : an AbstractNLPModel
- optimality_check : a stopping criterion via an admissibility function
- state      : The information relative to the problem, see GenericState
- max_cntrs  : Dict contains the maximum number of evaluations
- (opt) meta : Metadata relative to stopping criterion, see *StoppingMeta*.
- (opt) main_stp : Stopping of the main loop in case we consider a Stopping
                          of a subproblem.
                          If not a subproblem, then nothing.

`NLPStopping(:: AbstractNLPModel, :: Function, :: AbstractState; meta :: AbstractStoppingMeta = StoppingMeta(), max_cntrs :: Dict = _init_max_counters(), main_stp :: Union{AbstractStopping, Nothing} = nothing, kwargs...)`

 Note:
- *optimality_check* takes two inputs (*AbstractNLPModel*, *NLPAtX*)
 and returns a *Number* to be compared to *0*.
- designed for *NLPAtX* State. Constructor checks that the State has the
 required entries.
- *optimality_check* does not necessarily fill in the State.

 There is an additional default constructor creating a Stopping where the State is by default and the
 optimality function is the function *KKT()*.

 `NLPStopping(pb :: AbstractNLPModel; kwargs...)`

 Note: Kwargs are forwarded to the classical constructor.
 """
mutable struct NLPStopping <: AbstractStopping

    # problem
    pb :: AbstractNLPModel

    # stopping criterion
    optimality_check :: Function

    # Common parameters
    meta      :: AbstractStoppingMeta
    # Parameters specific to the NLPModels
    max_cntrs :: Dict #contains the max number of evaluations

    # current state of the problem
    current_state :: AbstractState

    # Stopping of the main problem, or nothing
    main_stp :: Union{AbstractStopping, Nothing}

    function NLPStopping(pb             :: AbstractNLPModel,
                         admissible     :: Function,
                         current_state  :: AbstractState;
                         meta           :: AbstractStoppingMeta = StoppingMeta(),
                         max_cntrs      :: Dict = _init_max_counters(),
                         main_stp       :: Union{AbstractStopping, Nothing} = nothing,
                         kwargs...)

        if !(isempty(kwargs))
           meta = StoppingMeta(;kwargs...)
        end

        #current_state is an AbstractState with requirements
        try
            current_state.evals
            current_state.fx, current_state.gx, current_state.Hx
            #if there are bounds:
            current_state.mu
            if pb.meta.ncon > 0 #if there are constraints
               current_state.Jx, current_state.cx, current_state.lambda
            end
        catch
            throw("error: missing entries in the given current_state")
        end

        return new(pb, admissible, meta, max_cntrs, current_state, main_stp)
    end

end

function NLPStopping(pb :: AbstractNLPModel; kwargs...)
 #Create a default NLPAtX
 nlp_at_x = NLPAtX(pb.meta.x0)
 admissible = KKT

 return NLPStopping(pb, admissible, nlp_at_x; kwargs...)
end

"""
\\_init\\_max\\_counters(): initialize the maximum number of evaluations on each of
                        the functions present in the Counters (NLPModels).

`_init_max_counters(; obj :: Int64 = 20000, grad :: Int64 = 20000, cons :: Int64 = 20000, jcon :: Int64 = 20000, jgrad :: Int64 = 20000, jac :: Int64 = 20000, jprod :: Int64 = 20000, jtprod :: Int64 = 20000, hess :: Int64 = 20000, hprod :: Int64 = 20000, jhprod :: Int64 = 20000, sum :: Int64 = 20000*11)`
"""
function _init_max_counters(; obj    :: Int64 = 20000,
                              grad   :: Int64 = 20000,
                              cons   :: Int64 = 20000,
                              jcon   :: Int64 = 20000,
                              jgrad  :: Int64 = 20000,
                              jac    :: Int64 = 20000,
                              jprod  :: Int64 = 20000,
                              jtprod :: Int64 = 20000,
                              hess   :: Int64 = 20000,
                              hprod  :: Int64 = 20000,
                              jhprod :: Int64 = 20000,
                              sum    :: Int64 = 20000*11)

  cntrs = Dict([(:neval_obj,       obj), (:neval_grad,   grad),
                (:neval_cons,     cons), (:neval_jcon,   jcon),
                (:neval_jgrad,   jgrad), (:neval_jac,    jac),
                (:neval_jprod,   jprod), (:neval_jtprod, jtprod),
                (:neval_hess,     hess), (:neval_hprod,  hprod),
                (:neval_jhprod, jhprod), (:neval_sum,    sum)])

 return cntrs
end

"""
\\_init\\_max\\_counters\\_NLS(): initialize the maximum number of evaluations on each of
                          the functions present in the NLSCounters (NLPModels).
https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/NLSModels.jl

`_init_max_counters_NLS(; residual :: Int = 20000, jac_residual :: Int = 20000, jprod_residual :: Int = 20000, jtprod_residual :: Int = 20000, hess_residual :: Int = 20000, jhess_residual :: Int = 20000, hprod_residual :: Int = 20000, kwargs...)`
"""
function _init_max_counters_NLS(; residual        :: Int = 20000,
                                  jac_residual    :: Int = 20000,
                                  jprod_residual  :: Int = 20000,
                                  jtprod_residual :: Int = 20000,
                                  hess_residual   :: Int = 20000,
                                  jhess_residual  :: Int = 20000,
                                  hprod_residual  :: Int = 20000,
                                  kwargs...)

  cntrs_nlp = _init_max_counters(;kwargs...)
  cntrs = Dict([(:neval_residual, residual),
                (:neval_jac_residual, jac_residual),
                (:neval_jprod_residual, jprod_residual),
                (:neval_jtprod_residual, jtprod_residual),
                (:neval_hess_residual, hess_residual),
                (:neval_jhess_residual, jhess_residual),
                (:neval_hprod_residual, hprod_residual)])

 return merge(cntrs_nlp, cntrs)
end

"""
fill_in!: (NLPStopping version) a function that fill in the required values in the *NLPAtX*

`fill_in!( :: NLPStopping, :: Iterate; fx :: Iterate = nothing, gx :: Iterate = nothing, Hx :: Iterate = nothing, cx :: Iterate = nothing, Jx :: Iterate = nothing, lambda :: Iterate = nothing, mu :: Iterate = nothing, matrix_info :: Bool = true, kwargs...)`
"""
function fill_in!(stp  :: NLPStopping,
                  x    :: Iterate;
                  fx   :: Iterate     = nothing,
                  gx   :: Iterate     = nothing,
                  Hx   :: Iterate     = nothing,
                  cx   :: Iterate     = nothing,
                  Jx   :: Iterate     = nothing,
                  lambda :: Iterate   = nothing,
                  mu     :: Iterate   = nothing,
                  matrix_info :: Bool = true,
                  kwargs...)

 gfx = fx == nothing  ? obj(stp.pb, x)   : fx
 ggx = gx == nothing  ? grad(stp.pb, x)  : gx

 if Hx == nothing && matrix_info
   gHx = hess(stp.pb, x)
 else
   gHx = Hx
 end

 if stp.pb.meta.ncon > 0
     gJx = Jx == nothing ? jac(stp.pb, x)  : Jx
     gcx = cx == nothing ? cons(stp.pb, x) : cx
 else
     gJx = stp.current_state.Jx
     gcx = stp.current_state.cx
 end

 #update the Lagrange multiplier if one of the 2 is asked
 if (stp.pb.meta.ncon > 0 || has_bounds(stp.pb)) && (lambda == nothing || mu == nothing)
  lb, lc = _compute_mutliplier(stp.pb, x, ggx, gcx, gJx; kwargs...)
 elseif  stp.pb.meta.ncon == 0 && !has_bounds(stp.pb) && lambda == nothing
  lb, lc = mu, stp.current_state.lambda
 else
  lb, lc = mu, lambda
 end

 return update!(stp.current_state, x=x, fx = gfx,    gx = ggx, Hx = gHx,
                                        cx = gcx,    Jx = gJx, mu = lb,
                                        lambda = lc)
end

"""
\\_resources\\_check!: check if the optimization algorithm has exhausted the resources.
                   This is the NLP specialized version that takes into account
                   the evaluation of the functions following the sum_counters
                   structure from NLPModels.

`_resources_check!(:: NLPStopping, :: Iterate)`

Note:
- function uses counters in *stp.pb*, and update the counters in the state.
- function is compatible with *Counters*, *NLSCounters*, and any type whose entries match the entries of *max_cntrs*.
- all the NLPModels have an attribute *counters* and a function *sum_counters(nlp)*.
"""
function _resources_check!(stp    :: NLPStopping,
                           x      :: Iterate)

  cntrs = stp.pb.counters
  update!(stp.current_state, evals = cntrs)

  max_cntrs = stp.max_cntrs

  # check all the entries in the counter
  max_f = false
  if typeof(stp.pb.counters) == Counters
   for f in fieldnames(Counters)
       max_f = max_f || (getfield(cntrs, f) > max_cntrs[f])
   end
  elseif typeof(stp.pb.counters) == NLSCounters
    for f in fieldnames(NLSCounters)
     max_f = f != :counters ? (max_f || (getfield(cntrs, f) > max_cntrs[f])) : max_f
    end
    for f in fieldnames(Counters)
        max_f = max_f || (getfield(cntrs.counters, f) > max_cntrs[f])
    end
  else #Unknown counters type
   for f in fieldnames(typeof(stp.pb.counters))
    max_f = max_f || (getfield(cntrs, f) > max_cntrs[f])
   end
  end

 # Maximum number of function and derivative(s) computation
 max_evals = sum_counters(stp.pb) > max_cntrs[:neval_sum]

 # global user limit diagnostic
 stp.meta.resources = max_evals || max_f

 return stp
end

"""
\\_unbounded\\_problem\\_check!: This is the NLP specialized version that takes into account
                   that the problem might be unbounded if the objective or the
                   constraint function are unbounded.

`_unbounded_problem_check!(:: NLPStopping, :: Iterate)`

Note: - evaluate the objective function if *state.fx* is *nothing* and store in *state*.
      - evaluate the constraint function if *state.cx* is *nothing* and store in *state*.
"""
function _unbounded_problem_check!(stp  :: NLPStopping,
                                   x    :: Iterate)

 if stp.current_state.fx == nothing
	 stp.current_state.fx = obj(stp.pb, x)
 end
 f_too_large = norm(stp.current_state.fx) >= stp.meta.unbounded_threshold

 c_too_large = false
 if stp.pb.meta.ncon != 0 #if the problems has constraints, check |c(x)|
  if stp.current_state.cx == nothing
   stp.current_state.cx = cons(stp.pb, x)
  end
  c_too_large = norm(stp.current_state.cx) >= abs(stp.meta.unbounded_threshold)
 end

 stp.meta.unbounded_pb = f_too_large || c_too_large

 return stp
end

"""
\\_optimality\\_check: compute the optimality score.

`_optimality_check(:: NLPStopping; kwargs...)`

This is the NLP specialized version that takes into account the structure of the
*NLPStopping* where the *optimality_check* function is an attribute of the Stopping.
"""
function _optimality_check(stp  :: NLPStopping; kwargs...)

 optimality = stp.optimality_check(stp.pb, stp.current_state; kwargs...)
 stp.current_state.current_score = optimality

 return optimality
end

################################################################################
# Nonlinear problems admissibility functions
# Available: unconstrained_check(...), optim_check_bounded(...), KKT
################################################################################
include("nlp_admissible_functions.jl")

################################################################################
# Functions computing Lagrange multipliers of a nonlinear problem
# Available: _compute_mutliplier(...)
################################################################################
include("nlp_compute_multiplier.jl")
