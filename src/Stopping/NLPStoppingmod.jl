"""
Type: NLPStopping

Methods: start!, stop!, update\\_and\\_start!, update\\_and\\_stop!, fill\\_in!, reinit!, status
KKT, unconstrained\\_check, unconstrained2nd\\_check, optim\\_check\\_bounded

Specialization of GenericStopping. Stopping structure for non-linear programming problems using NLPModels.

Attributes:
- pb         : an AbstractNLPModel
- state      : The information relative to the problem, see GenericState
- (opt) meta : Metadata relative to stopping criterion, see `StoppingMeta`.
- (opt) main_stp : Stopping of the main loop in case we consider a Stopping
                          of a subproblem.
                          If not a subproblem, then nothing.
- (opt) listofstates : ListofStates designed to store the history of States.
- (opt) stopping_user_struct : Contains any structure designed by the user.

`NLPStopping(:: AbstractNLPModel, :: AbstractState; meta :: AbstractStoppingMeta = StoppingMeta(), max_cntrs :: Dict = _init_max_counters(), main_stp :: Union{AbstractStopping, Nothing} = nothing, list :: Union{ListofStates, Nothing} = nothing, stopping_user_struct :: Any = nothing, kwargs...)`

 Note:
- designed for `NLPAtX` State. Constructor checks that the State has the
 required entries.

 There is an additional default constructor creating a Stopping where the State is by default and the
 optimality function is the function `KKT()``.

 `NLPStopping(pb :: AbstractNLPModel; kwargs...)`

 Note: Kwargs are forwarded to the classical constructor.
 """
mutable struct NLPStopping{Pb, M, SRC, T, MStp, LoS
                          }  <: AbstractStopping{Pb, M, SRC, T, MStp, LoS}

  # problem
  pb                   :: Pb

  # Common parameters
  meta                 :: M
  stop_remote          :: SRC

  # current state of the problem
  current_state        :: T

  # Stopping of the main problem, or nothing
  main_stp             :: MStp

  # History of states
  listofstates         :: LoS

  # User-specific structure
  stopping_user_struct :: AbstractDict

end

function NLPStopping(pb             :: Pb,
                     meta           :: M,
                     stop_remote    :: SRC,
                     current_state  :: T;
                     main_stp       :: AbstractStopping = VoidStopping(),
                     list           :: AbstractListofStates = VoidListofStates(),
                     user_struct    :: AbstractDict = Dict(),
                     kwargs...
                     ) where {Pb  <: AbstractNLPModel, 
                              M   <: AbstractStoppingMeta, 
                              SRC <: AbstractStopRemoteControl,
                              T   <: AbstractState}

    return NLPStopping(pb, meta, stop_remote, current_state, 
                       main_stp, list, user_struct)
end

function NLPStopping(pb             :: Pb,
                     meta           :: M,
                     current_state  :: T;
                     main_stp       :: AbstractStopping = VoidStopping(),
                     list           :: AbstractListofStates = VoidListofStates(),
                     user_struct    :: AbstractDict = Dict(),
                     kwargs...
                     ) where {Pb <: AbstractNLPModel, 
                              M  <: AbstractStoppingMeta,
                              T  <: AbstractState}
    
  stop_remote = StopRemoteControl() #main_stp == VoidStopping() ? StopRemoteControl() : cheap_stop_remote_control()

  return NLPStopping(pb, meta, stop_remote, current_state, 
                     main_stp, list, user_struct)
end

function NLPStopping(pb             :: Pb,
                     current_state  :: T;
                     main_stp       :: AbstractStopping = VoidStopping(),
                     list           :: AbstractListofStates = VoidListofStates(),
                     user_struct    :: AbstractDict = Dict(),
                     kwargs...
                     ) where {Pb <: AbstractNLPModel, T <: AbstractState}
    
  if :max_cntrs in keys(kwargs)
    mcntrs = kwargs[:max_cntrs]
  else
    mcntrs = _init_max_counters()
  end
    
  if :optimality_check in keys(kwargs)
    oc = kwargs[:optimality_check]
  else
    oc = KKT
  end

    meta = StoppingMeta(;max_cntrs = mcntrs, optimality_check = oc, kwargs...)
    stop_remote = StopRemoteControl()

  return NLPStopping(pb, meta, stop_remote, current_state, 
                     main_stp, list, user_struct)
end

function NLPStopping(pb :: AbstractNLPModel;
                     n_listofstates :: Int = 0,
                     kwargs...)
  #Create a default NLPAtX
  nlp_at_x = NLPAtX(pb.meta.x0)

  if n_listofstates > 0 && :list ∉ keys(kwargs)
    list = ListofStates(n_listofstates, Val{typeof(nlp_at_x)}())
    return NLPStopping(pb, nlp_at_x, list = list, optimality_check = KKT; kwargs...)
  end

  return NLPStopping(pb, nlp_at_x, optimality_check = KKT; kwargs...)
end

"""
\\_init\\_max\\_counters(): initialize the maximum number of evaluations on each of
                        the functions present in the Counters (NLPModels).

`_init_max_counters(; obj :: Int64 = 20000, grad :: Int64 = 20000, cons :: Int64 = 20000, jcon :: Int64 = 20000, jgrad :: Int64 = 20000, jac :: Int64 = 20000, jprod :: Int64 = 20000, jtprod :: Int64 = 20000, hess :: Int64 = 20000, hprod :: Int64 = 20000, jhprod :: Int64 = 20000, sum :: Int64 = 20000*11)`
"""
function _init_max_counters(; allevals :: T = 20000,
                              obj      :: T = allevals,
                              grad     :: T = allevals,
                              cons     :: T = allevals,
                              jcon     :: T = allevals,
                              jgrad    :: T = allevals,
                              jac      :: T = allevals,
                              jprod    :: T = allevals,
                              jtprod   :: T = allevals,
                              hess     :: T = allevals,
                              hprod    :: T = allevals,
                              jhprod   :: T = allevals,
                              sum      :: T = allevals*11) where {T <: Int}

  cntrs = Dict{Symbol,T}([(:neval_obj,       obj), (:neval_grad,   grad),
                          (:neval_cons,     cons), (:neval_jcon,   jcon),
                          (:neval_jgrad,   jgrad), (:neval_jac,    jac),
                          (:neval_jprod,   jprod), (:neval_jtprod, jtprod),
                          (:neval_hess,     hess), (:neval_hprod,  hprod),
                          (:neval_jhprod, jhprod), (:neval_sum,    sum)])

  return cntrs
end

function max_evals!(stp :: NLPStopping, allevals :: Int)
  stp.meta.max_cntrs = _init_max_counters(allevals = allevals)
  return stp
end

function max_evals!(stp :: NLPStopping; allevals :: T = 20000, kwargs...) where {T <: Int}
  stp.meta.max_cntrs = _init_max_counters(allevals = allevals; kwargs...)
  return stp
end

"""
\\_init\\_max\\_counters\\_NLS(): initialize the maximum number of evaluations on each of
                          the functions present in the NLSCounters (NLPModels).
https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/NLSModels.jl

`_init_max_counters_NLS(; residual :: Int = 20000, jac_residual :: Int = 20000, jprod_residual :: Int = 20000, jtprod_residual :: Int = 20000, hess_residual :: Int = 20000, jhess_residual :: Int = 20000, hprod_residual :: Int = 20000, kwargs...)`
"""
function _init_max_counters_NLS(; allevals        :: T = 20000,
                                  residual        :: T = allevals,
                                  jac_residual    :: T = allevals,
                                  jprod_residual  :: T = allevals,
                                  jtprod_residual :: T = allevals,
                                  hess_residual   :: T = allevals,
                                  jhess_residual  :: T = allevals,
                                  hprod_residual  :: T = allevals,
                                  kwargs...) where {T <: Int}

  cntrs_nlp = _init_max_counters(;kwargs...)
  cntrs = Dict{Symbol,T}([(:neval_residual, residual),
                          (:neval_jac_residual, jac_residual),
                          (:neval_jprod_residual, jprod_residual),
                          (:neval_jtprod_residual, jtprod_residual),
                          (:neval_hess_residual, hess_residual),
                          (:neval_jhess_residual, jhess_residual),
                          (:neval_hprod_residual, hprod_residual)])

  return merge(cntrs_nlp, cntrs)
end

"""
fill_in!: (NLPStopping version) a function that fill in the required values in the `NLPAtX`.

`fill_in!( :: NLPStopping, :: Union{AbstractVector, Nothing}; fx :: Union{AbstractVector, Nothing} = nothing, gx :: Union{AbstractVector, Nothing} = nothing, Hx :: Union{MatrixType, Nothing} = nothing, cx :: Union{AbstractVector, Nothing} = nothing, Jx :: Union{MatrixType, Nothing} = nothing, lambda :: Union{AbstractVector, Nothing} = nothing, mu :: Union{AbstractVector, Nothing} = nothing, matrix_info :: Bool = true, kwargs...)`
"""
function fill_in!(stp         :: NLPStopping{Pb, M, SRC, Stt, MStp, LoS},
                  x           :: AbstractVector;
                  fx          :: Union{AbstractVector, Nothing} = nothing,
                  gx          :: Union{AbstractVector, Nothing} = nothing,
                  Hx          :: Union{MatrixType, Nothing}     = nothing,
                  cx          :: Union{AbstractVector, Nothing} = nothing,
                  Jx          :: Union{MatrixType, Nothing}     = nothing,
                  lambda      :: Union{AbstractVector, Nothing} = nothing,
                  mu          :: Union{AbstractVector, Nothing} = nothing,
                  matrix_info :: Bool    = true,
                  kwargs...) where {Pb, M, SRC, Stt <: NLPAtX, MStp, LoS}

  gfx = isnothing(fx)  ? obj(stp.pb, x)   : fx
  ggx = isnothing(gx)  ? grad(stp.pb, x)  : gx

  if isnothing(Hx) && matrix_info
    gHx = hess(stp.pb, x)
  else
    gHx = Hx
  end

  if stp.pb.meta.ncon > 0
    gJx = isnothing(Jx) ? jac(stp.pb, x)  : Jx
    gcx = isnothing(cx) ? cons(stp.pb, x) : cx
  else
    gJx = stp.current_state.Jx
    gcx = stp.current_state.cx
  end

  #update the Lagrange multiplier if one of the 2 is asked
  if (stp.pb.meta.ncon > 0 || has_bounds(stp.pb)) && (isnothing(lambda) || isnothing(mu))
    lb, lc = _compute_mutliplier(stp.pb, x, ggx, gcx, gJx; kwargs...)
  elseif  stp.pb.meta.ncon == 0 && !has_bounds(stp.pb) && isnothing(lambda)
    lb, lc = mu, stp.current_state.lambda
  else
    lb, lc = mu, lambda
  end

  return update!(stp, x = x, fx = gfx, gx = ggx, Hx = gHx,
                             cx = gcx, Jx = gJx, mu = lb,
                             lambda = lc)
end


function fill_in!(stp         :: NLPStopping{Pb, M, SRC, OneDAtX{S,T}, MStp, LoS},
                  x           :: T;
                  fx          :: Union{T, Nothing} = nothing,
                  gx          :: Union{T, Nothing} = nothing,
                  f₀          :: Union{T, Nothing} = nothing,
                  g₀          :: Union{T, Nothing} = nothing,
                  kwargs...) where {Pb, M, SRC, S, T, MStp, LoS}

 gfx = isnothing(fx) ? obj(stp.pb, x)    : fx
 ggx = isnothing(gx) ? grad(stp.pb, x)   : gx
 gf₀ = isnothing(f₀) ? obj(stp.pb, 0.0)  : f₀
 gg₀ = isnothing(g₀) ? grad(stp.pb, 0.0) : g₀

 return update!(stp.current_state, x=x, fx = gfx, gx = ggx, f₀ = gf₀, g₀ = gg₀)
end


"""
For NLPStopping, `rcounters` set as true also reinitialize the counters.
"""
function reinit!(stp       :: NLPStopping;
                 rstate    :: Bool = false,
                 rlist     :: Bool = true,
                 rcounters :: Bool = false,
                 kwargs...)

  stp.meta.start_time  = NaN
  stp.meta.optimality0 = 1.0

  #reinitialize the boolean status
  reinit!(stp.meta)

  #reinitialize the counter of stop
  stp.meta.nb_of_stop = 0

  #reinitialize the list of states
  if rlist && (typeof(stp.listofstates) != VoidListofStates)
    #TODO: Warning we cannot change the type of ListofStates 
    stp.listofstates = rstate ? VoidListofStates() : ListofStates(stp.current_state)
  end

  #reinitialize the state
  if rstate
    reinit!(stp.current_state; kwargs...)
  end

  #reinitialize the NLPModel Counters
  if rcounters && typeof(stp.pb) <: AbstractNLPModel
    NLPModels.reset!(stp.pb)
  end

  return stp
end

"""
\\_resources\\_check!: check if the optimization algorithm has exhausted the resources.
                   This is the NLP specialized version that takes into account
                   the evaluation of the functions following the sum_counters
                   structure from NLPModels.

`_resources_check!(:: NLPStopping, :: T)`

Note:
- function uses counters in `stp.pb`, and update the counters in the state.     
- function is compatible with `Counters`, `NLSCounters`, and any type whose entries match the entries of `max_cntrs`.   
- all the NLPModels have an attribute `counters` and a function `sum_counters(nlp)`.  
"""
function _resources_check!(stp    :: NLPStopping,
                           x      :: T) where T <: Union{AbstractVector, Number}

  cntrs = stp.pb.counters
  update!(stp, evals = cntrs)

  max_cntrs = stp.meta.max_cntrs

  if max_cntrs == Dict{Symbol,Int64}()
    return stp.meta.resources
  end

  # check all the entries in the counter
  max_f = false
  if typeof(stp.pb.counters) == Counters
    for f in intersect(fieldnames(Counters), keys(max_cntrs))
      max_f = max_f || (getfield(cntrs, f) > max_cntrs[f])
    end
  elseif typeof(stp.pb.counters) == NLSCounters
    for f in intersect(fieldnames(NLSCounters), keys(max_cntrs))
      max_f = f != :counters ? (max_f || (getfield(cntrs, f) > max_cntrs[f])) : max_f
    end
    for f in intersect(fieldnames(Counters), keys(max_cntrs))
      max_f = max_f || (getfield(cntrs.counters, f) > max_cntrs[f])
    end
  else #Unknown counters type
    for f in intersect(fieldnames(typeof(stp.pb.counters)), keys(max_cntrs))
      max_f = max_f || (getfield(cntrs, f) > max_cntrs[f])
    end
  end

  # Maximum number of function and derivative(s) computation
  if :neval_sum in keys(stp.meta.max_cntrs)
    max_evals = sum_counters(stp.pb) > max_cntrs[:neval_sum]
  end

  # global user limit diagnostic
  if (max_evals || max_f) stp.meta.resources = true end

  return stp.meta.resources
end

"""
`_unbounded_problem_check!`: This is the NLP specialized version that takes into account
                   that the problem might be unbounded if the objective or the
                   constraint function are unbounded.

`_unbounded_problem_check!(:: NLPStopping, :: AbstractVector)`

Note:
- evaluate the objective function if `state.fx` for NLPAtX or `state.fx` for OneDAtX is `_init_field` and store in `state`.
- do NOT evaluate the constraint function if `state.cx` is `_init_field` and store in `state`.
- if minimize problem (i.e. nlp.meta.minimize is true) check if
`state.fx <= - meta.unbounded_threshold`,
otherwise check `state.fx ≥ meta.unbounded_threshold`.
- `state.cx` is unbounded if larger than `|meta.unbounded_threshold|`.
"""
function _unbounded_problem_check!(stp  :: NLPStopping{Pb, M, SRC, Stt, MStp, LoS},
                                   x    :: AbstractVector
                                  ) where {Pb, M, SRC, Stt <: NLPAtX, MStp, LoS}

  if isnan(stp.current_state.fx)
	  stp.current_state.fx = obj(stp.pb, x)
  end

  if stp.pb.meta.minimize
    f_too_large = stp.current_state.fx <= - stp.meta.unbounded_threshold
  else
    f_too_large = stp.current_state.fx >=   stp.meta.unbounded_threshold
  end

  c_too_large = false
  #we do not evaluate the constraint if not in the state.
  if stp.pb.meta.ncon != 0 && (stp.current_state.cx != _init_field(typeof(stp.current_state.cx)))
    c_too_large = norm(stp.current_state.cx) >= abs(stp.meta.unbounded_threshold)
  end

  if (f_too_large || c_too_large) stp.meta.unbounded_pb = true end

  return stp.meta.unbounded_pb
end

function _unbounded_problem_check!(stp  :: NLPStopping{Pb, M, SRC, Stt, MStp, LoS},
                                   x    :: Union{AbstractVector, Number}
                                  ) where {Pb, M, SRC, Stt <: OneDAtX, MStp, LoS}
  if isnan(stp.current_state.fx)
	  stp.current_state.fx = obj(stp.pb, x)
  end  

  if stp.pb.meta.minimize
    f_too_large = stp.current_state.fx <= - stp.meta.unbounded_threshold
  else
    f_too_large = stp.current_state.fx >=   stp.meta.unbounded_threshold
  end

  return stp.meta.unbounded_pb
end
"""
\\_infeasibility\\_check!: This is the NLP specialized version.
                       
Note:
  - check wether the `current_score` contains Inf.
  - check the feasibility of an optimization problem in the spirit of a convex
  indicator function.
"""
function _infeasibility_check!(stp  :: NLPStopping,
                               x    :: T) where T
#=
#- evaluate the constraint function if `state.cx` is `nothing` and store in `state`.
#- check the Inf-norm of the violation ≤ stp.meta.atol
 if stp.pb.meta.ncon != 0 #if the problems has constraints, check |c(x)|
  cx = stp.current_state.cx
  if cx == _init_field(typeof(stp.current_state.cx))
   cx = cons(stp.pb, x)
  end
  vio = max.(max.(cx - stp.pb.meta.ucon, 0.), max.(stp.pb.meta.lcon - cx, 0.))
  tol = Inf #stp.meta.atol
  stp.meta.infeasible = _inequality_check(vio, stp.meta.atol, 0.) ? true : stp.meta.infeasible
 end
 =#
 
  if stp.pb.meta.minimize
    vio = any(z-> z ==  Inf, stp.current_state.current_score)
    if vio stp.meta.infeasible = true end
  else
    vio = any(z-> z == -Inf, stp.current_state.current_score)
    if vio stp.meta.infeasible = true end
  end
 
  return stp.meta.infeasible
end

################################################################################
# Nonlinear problems admissibility functions
# Available: unconstrained_check(...), optim_check_bounded(...), KKT
################################################################################
include("nlp_admissible_functions.jl")

################################################################################
# line search admissibility functions
#
# TODO: change the ls_admissible_functions and use tol_check et tol_check_neg to
# handle the inequality instead of a max.
################################################################################
include("ls_admissible_functions.jl")

#=
"""
"""
function feasibility_optim_check(pb, state; kwargs...)
     vio = _feasibility(pb, state)
     tol = Inf #stp.meta.atol
     return _inequality_check(vio, tol, 0.)
end
=#

################################################################################
# Functions computing Lagrange multipliers of a nonlinear problem
# Available: _compute_mutliplier(...)
################################################################################
include("nlp_compute_multiplier.jl")
