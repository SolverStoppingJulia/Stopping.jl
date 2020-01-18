export NLPStopping, unconstrained, KKT, fill_in!

################################################################################
# Specific stopping module for non linear problems
################################################################################

"""
Stopping structure for non-linear programming problems.
Inputs:
 - pb : An AbstractNLPModel
 - optimality_check : a stopping criterion through an admissibility function
 - meta : StoppingMeta
 - max_cntrs :: Dict contains the max number of evaluations
 - current_state : the current state of the problem (i.e an NLPAtX)

 * optimality_check : takes two inputs (AbstractNLPModel, NLPAtX)
 and returns a Float64 to be compared at 0.
 (Idée: ajouter une nouvelle input main_pb dans le optimality_check?)
 """
mutable struct NLPStopping <: AbstractStopping

    # problem
    pb :: AbstractNLPModel

    # stopping criterion
    optimality_check :: Function # will be put in optimality_check

    # Common parameters
    meta      :: StoppingMeta
    # Parameters specific to the NLPModels
    max_cntrs :: Dict #contains the max number of evaluations

    # current state of the line search Algorithm
    current_state :: AbstractState

    # Stopping of the main problem, or nothing
    main_stp :: Union{AbstractStopping, Nothing}

    function NLPStopping(pb             :: AbstractNLPModel,
                         admissible     :: Function,
                         current_state  :: AbstractState;
                         meta           :: StoppingMeta = StoppingMeta(),
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

"""
NLPStopping(pb): additional default constructor
The function creates a Stopping where the State is by default and the
optimality function is the function KKT().

key arguments are forwarded to the classical constructor.
"""
function NLPStopping(pb :: AbstractNLPModel; kwargs...)
 #Create a default NLPAtX
 nlp_at_x = NLPAtX(pb.meta.x0)
 admissible = (x,y) -> KKT(x,y)

 return NLPStopping(pb, admissible, nlp_at_x; kwargs...)
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
"""
function fill_in!(stp  :: NLPStopping,
                  x    :: Iterate;
                  fx   :: Iterate     = nothing,
                  gx   :: Iterate     = nothing,
                  Hx   :: Iterate     = nothing,
                  cx   :: Iterate     = nothing,
                  Jx   :: Iterate     = nothing,
                  lambda :: Iterate   = nothing,
                  mu   :: Iterate     = nothing,
                  matrix_info :: Bool = true)

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
 if lambda == nothing || mu == nothing
  lb, lc = _compute_mutliplier(stp.pb, x, ggx, gcx, gJx)
 else
  lb, lc = mu, lambda
 end

 return update!(stp.current_state, x=x, fx = gfx,    gx = ggx, Hx = gHx,
                                        cx = gcx,    Jx = gJx, mu = lb,
                                        lambda = lc)
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

 if stp.current_state.fx == nothing
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

"""
Additional function to estimate Lagrange multiplier of the problems
    (guarantee if LICQ holds)
"""
function _compute_mutliplier(pb    :: AbstractNLPModel,
                             x     :: Iterate,
                             gx    :: Iterate,
                             cx    :: Iterate,
                             Jx    :: Any;
                             active_prec_c :: Float64 = 1e-6,
                			 active_prec_b :: Float64 = 1e-6)

 n  = length(x)
 nc = cx == nothing ? 0 : length(cx)

 #active res_bounds
 Ib = findall(x->(norm(x) <= active_prec_b),
			      min(abs.(x - pb.meta.lvar),
				      abs.(x - pb.meta.uvar)))
 if nc != 0
  #active constraints
  Ic = findall(x->(norm(x) <= active_prec_c),
                   min(abs.(cx-pb.meta.ucon),
                   abs.(cx-pb.meta.lcon)))

  Jc = hcat(Matrix(1.0I, n, n)[:,Ib], Jx'[:,Ic])
 else
  Ic = []
  Jc = hcat(Matrix(1.0I, n, n)[:,Ib])
 end


 l = pinv(Jc) * (- gx)

 mu, lambda = zeros(n), zeros(nc)
 mu[Ib], lambda[Ic] = l[1:length(Ib)], l[length(Ib)+1:length(l)]

 return mu, lambda
end
