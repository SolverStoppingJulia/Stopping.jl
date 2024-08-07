"""
Type: NLPStopping

Methods: `start!`, `stop!`, `update_and_start!`, `update_and_stop!`, `fill_in!`, `reinit!`, `status`, 
`KKT`, `unconstrained_check`, `optim_check_bounded`

Specialization of `GenericStopping`. Stopping structure for non-linear optimization models using `NLPModels` ( https://github.com/JuliaSmoothOptimizers/NLPModels.jl ).

Attributes:
- `pb`         : An `AbstractNLPModel`.
- `current_state`      : The information relative to the problem, see `GenericState` or `NLPAtX`.
- (opt) `meta` : Metadata relative to stopping criteria, see `StoppingMeta`.
- (opt) `main_stp` : Stopping of the main loop in case we consider a Stopping
                          of a subproblem.
                          If not a subproblem, then `VoidStopping`.
- (opt) `listofstates` : ListofStates designed to store the history of States.
- (opt) `stopping_user_struct` : Contains any structure designed by the user.

Constructors: 
- `NLPStopping(pb::AbstractNLPModel, meta::AbstractStoppingMeta, stop_remote::AbstractStopRemoteControl, state::AbstractState; main_stp::AbstractStopping=VoidStopping(), list::AbstractListofStates = VoidListofStates(), user_struct::AbstractDict = Dict(), kwargs...)`
     The default constructor.
- `NLPStopping(pb::AbstractNLPModel, meta::AbstractStoppingMeta, state::AbstractState; main_stp::AbstractStopping=VoidStopping(), list::AbstractListofStates = VoidListofStates(), user_struct::AbstractDict = Dict(), kwargs...)`
     The one passing the `kwargs` to the `stop_remote`.
- `GenericStopping(pb::AbstractNLPModel, state::AbstractState; stop_remote::AbstractStopRemoteControl = StopRemoteControl(), main_stp::AbstractStopping=VoidStopping(), list::AbstractListofStates = VoidListofStates(), user_struct::AbstractDict = Dict(), kwargs...)`
     The one passing the `kwargs` to the `meta`.
- `GenericStopping(pb::AbstractNLPModel; n_listofstates=, kwargs...)`
     The one setting up a default state `NLPAtX` using `pb.meta.x0`, and initializing the list of states if `n_listofstates>0`. The optimality function is the function `KKT` unless `optimality_check` is in the `kwargs`.

 Notes:
- Designed for `NLPAtX` State. Constructor checks that the State has the required entries.

 """
mutable struct NLPStopping{Pb, M, SRC, T, MStp, LoS} <: AbstractStopping{Pb, M, SRC, T, MStp, LoS}

  # problem
  pb::Pb

  # Common parameters
  meta::M
  stop_remote::SRC

  # current state of the problem
  current_state::T

  # Stopping of the main problem, or nothing
  main_stp::MStp

  # History of states
  listofstates::LoS

  # User-specific structure
  stopping_user_struct::AbstractDict
end

get_pb(stp::NLPStopping) = stp.pb
get_meta(stp::NLPStopping) = stp.meta
get_remote(stp::NLPStopping) = stp.stop_remote
get_state(stp::NLPStopping) = stp.current_state
get_main_stp(stp::NLPStopping) = stp.main_stp
get_list_of_states(stp::NLPStopping) = stp.listofstates
get_user_struct(stp::NLPStopping) = stp.stopping_user_struct

function NLPStopping(
  pb::Pb,
  meta::M,
  stop_remote::SRC,
  current_state::T;
  main_stp::AbstractStopping = VoidStopping(),
  list::AbstractListofStates = VoidListofStates(),
  n_listofstates::Integer = 0,
  user_struct::AbstractDict = Dict(),
  kwargs...,
) where {
  Pb <: AbstractNLPModel,
  M <: AbstractStoppingMeta,
  SRC <: AbstractStopRemoteControl,
  T <: AbstractState,
}
  if n_listofstates > 0
    list = ListofStates(n_listofstates, Val{T}())
  end
  return NLPStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct)
end

function NLPStopping(
  pb::Pb,
  meta::M,
  current_state::T;
  main_stp::AbstractStopping = VoidStopping(),
  list::AbstractListofStates = VoidListofStates(),
  n_listofstates::Integer = 0,
  user_struct::AbstractDict = Dict(),
  kwargs...,
) where {Pb <: AbstractNLPModel, M <: AbstractStoppingMeta, T <: AbstractState}
  stop_remote = StopRemoteControl(; kwargs...) #main_stp == VoidStopping() ? StopRemoteControl() : cheap_stop_remote_control()

  if n_listofstates > 0
    list = ListofStates(n_listofstates, Val{T}())
  end

  return NLPStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct)
end

function NLPStopping(
  pb::Pb,
  current_state::T;
  stop_remote::AbstractStopRemoteControl = StopRemoteControl(),
  main_stp::AbstractStopping = VoidStopping(),
  list::AbstractListofStates = VoidListofStates(),
  n_listofstates::Integer = 0,
  user_struct::AbstractDict = Dict(),
  kwargs...,
) where {Pb <: AbstractNLPModel, T <: AbstractState}
  mcntrs = if :max_cntrs in keys(kwargs)
    kwargs[:max_cntrs]
  elseif Pb <: AbstractNLSModel
    init_max_counters_NLS()
  else
    init_max_counters()
  end

  if :optimality_check in keys(kwargs)
    oc = kwargs[:optimality_check]
  else
    oc = KKT
  end

  if n_listofstates > 0
    list = ListofStates(n_listofstates, Val{T}())
  end

  meta = StoppingMeta(; max_cntrs = mcntrs, optimality_check = oc, kwargs...)

  return NLPStopping(pb, meta, stop_remote, current_state, main_stp, list, user_struct)
end

function NLPStopping(pb::AbstractNLPModel; n_listofstates::Integer = 0, kwargs...)
  #Create a default NLPAtX
  initial_guess = copy(pb.meta.x0)
  if get_ncon(pb) > 0
    initial_lag = copy(pb.meta.y0)
    nlp_at_x = NLPAtX(initial_guess, initial_lag)
  else
    nlp_at_x = NLPAtX(initial_guess)
  end

  if n_listofstates > 0 && :list ∉ keys(kwargs)
    list = ListofStates(n_listofstates, Val{typeof(nlp_at_x)}())
    return NLPStopping(pb, nlp_at_x, list = list, optimality_check = KKT; kwargs...)
  end

  return NLPStopping(pb, nlp_at_x, optimality_check = KKT; kwargs...)
end

"""
init\\_max\\_counters: 
initialize the maximum number of evaluations on each of
the functions present in the NLPModels.Counters, e.g.

`init_max_counters(; allevals :: T = typemax(T), obj = allevals, grad = allevals, cons = allevals, jcon = allevals, jgrad = allevals, jac = allevals, jprod = allevals, jtprod = allevals, hess = allevals, hprod = allevals, jhprod = allevals, sum = 11 * allevals, kwargs...)`

`:neval_sum` is by default limited to `|Counters| * allevals`.
"""
function init_max_counters(; allevals::T = typemax(Int), kwargs...) where {T <: Integer}
  entries = [Meta.parse(split("$(f)", '_')[2]) for f in fieldnames(Counters)]
  lim_fields = keys(kwargs)
  cntrs = Dict{Symbol, T}([
    (Meta.parse("neval_$(t)"), t in lim_fields ? kwargs[t] : allevals) for t in entries
  ])
  push!(cntrs, (:neval_sum => :sum in lim_fields ? kwargs[:sum] : typemax(T)))

  return cntrs
end

function max_evals!(
  stp::NLPStopping{Pb, M, SRC, T, MStp, LoS},
  allevals::Integer,
) where {Pb, M, SRC, T, MStp, LoS}
  stp.meta.max_cntrs = if Pb <: AbstractNLSModel
    init_max_counters_NLS(allevals = allevals)
  else
    init_max_counters(allevals = allevals)
  end
  return stp
end

function max_evals!(
  stp::NLPStopping{Pb, M, SRC, T, MStp, LoS};
  allevals::I = typemax(Int),
  kwargs...,
) where {Pb, M, SRC, T, MStp, LoS, I <: Integer}
  stp.meta.max_cntrs = if Pb <: AbstractNLSModel
    init_max_counters_NLS(allevals = allevals; kwargs...)
  else
    init_max_counters(allevals = allevals; kwargs...)
  end
  return stp
end

"""
init\\_max\\_counters\\_NLS: 
initialize the maximum number of evaluations on each of
the functions present in the `NLPModels.NLSCounters`, e.g.

`init_max_counters_NLS(; allevals = typemax(T), residual = allevals, jac_residual = allevals, jprod_residual = allevals, jtprod_residual = allevals, hess_residual = allevals, jhess_residual = allevals, hprod_residual = allevals, kwargs...)`
"""
function init_max_counters_NLS(; allevals::T = typemax(Int), kwargs...) where {T <: Integer}
  cntrs_nlp = init_max_counters(; allevals = allevals, kwargs...)

  entries =
    [Meta.parse(split("$(f)", "neval_")[2]) for f in setdiff(fieldnames(NLSCounters), [:counters])]
  lim_fields = keys(kwargs)
  cntrs = Dict{Symbol, T}([
    (Meta.parse("neval_$(t)"), t in lim_fields ? kwargs[t] : allevals) for t in entries
  ])

  return merge(cntrs_nlp, cntrs)
end

"""
fill_in!: (NLPStopping version) a function that fill in the required values in the `NLPAtX`.

`fill_in!( :: NLPStopping, :: Union{T, Nothing}; fx :: Union{T, Nothing} = nothing, gx :: Union{T, Nothing} = nothing, Hx :: Union{MatrixType, Nothing} = nothing, cx :: Union{T, Nothing} = nothing, Jx :: Union{MatrixType, Nothing} = nothing, lambda :: Union{T, Nothing} = nothing, mu :: Union{T, Nothing} = nothing, matrix_info :: Bool = true, kwargs...)`
"""
function fill_in!(
  stp::NLPStopping{Pb, M, SRC, NLPAtX{Score, S, T}, MStp, LoS},
  x::T;
  fx::Union{eltype(T), Nothing} = nothing,
  gx::Union{T, Nothing} = nothing,
  Hx = nothing,
  cx::Union{T, Nothing} = nothing,
  Jx = nothing,
  lambda::Union{T, Nothing} = nothing,
  mu::Union{T, Nothing} = nothing,
  matrix_info::Bool = true,
  convert::Bool = true,
  kwargs...,
) where {
  Pb,
  M <: AbstractStoppingMeta,
  SRC <: AbstractStopRemoteControl,
  MStp,
  LoS <: AbstractListofStates,
  Score,
  S,
  T,
}
  gfx = isnothing(fx) ? obj(stp.pb, x) : fx
  ggx = isnothing(gx) ? grad(stp.pb, x) : gx

  if isnothing(Hx) && matrix_info
    gHx = hess(stp.pb, x).data
  else
    gHx = isnothing(Hx) ? zeros(eltype(T), 0, 0) : Hx
  end

  if stp.pb.meta.ncon > 0
    gJx = if !isnothing(Jx)
      Jx
    elseif typeof(stp.current_state.Jx) <: LinearOperator
      jac_op(stp.pb, x)
    else # typeof(stp.current_state.Jx) <: SparseArrays.SparseMatrixCSC
      jac(stp.pb, x)
    end
    gcx = isnothing(cx) ? cons(stp.pb, x) : cx
  else
    gJx = stp.current_state.Jx
    gcx = stp.current_state.cx
  end

  #update the Lagrange multiplier if one of the 2 is asked
  if (stp.pb.meta.ncon > 0 || has_bounds(stp.pb)) && (isnothing(lambda) || isnothing(mu))
    lb, lc = _compute_mutliplier(stp.pb, x, ggx, gcx, gJx; kwargs...)
  else
    lb = if isnothing(mu) & has_bounds(stp.pb)
      zeros(eltype(T), get_nvar(stp.pb))
    elseif isnothing(mu) & !has_bounds(stp.pb)
      zeros(eltype(T), 0)
    else
      mu
    end
    lc = isnothing(lambda) ? zeros(eltype(T), get_ncon(stp.pb)) : lambda
  end

  return update!(
    stp,
    x = x,
    fx = gfx,
    gx = ggx,
    Hx = gHx,
    cx = gcx,
    Jx = gJx,
    mu = lb,
    lambda = lc,
    convert = convert,
  )
end

function fill_in!(
  stp::NLPStopping{Pb, M, SRC, OneDAtX{S, T}, MStp, LoS},
  x::T;
  fx::Union{T, Nothing} = nothing,
  gx::Union{T, Nothing} = nothing,
  f₀::Union{T, Nothing} = nothing,
  g₀::Union{T, Nothing} = nothing,
  convert::Bool = true,
  kwargs...,
) where {
  Pb,
  M <: AbstractStoppingMeta,
  SRC <: AbstractStopRemoteControl,
  MStp,
  LoS <: AbstractListofStates,
  S,
  T,
}
  gfx = isnothing(fx) ? obj(stp.pb, x) : fx
  ggx = isnothing(gx) ? grad(stp.pb, x) : gx
  gf₀ = isnothing(f₀) ? obj(stp.pb, 0.0) : f₀
  gg₀ = isnothing(g₀) ? grad(stp.pb, 0.0) : g₀

  return update!(
    stp.current_state,
    x = x,
    fx = gfx,
    gx = ggx,
    f₀ = gf₀,
    g₀ = gg₀,
    convert = convert,
  )
end

"""
For NLPStopping, `rcounters` set as true also reinitialize the counters.
"""
function reinit!(
  stp::NLPStopping;
  rstate::Bool = false,
  rlist::Bool = true,
  rcounters::Bool = false,
  kwargs...,
)
  stp.meta.start_time = NaN
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
`_resources_check!`: check if the optimization algorithm has exhausted the resources.
                   This is the NLP specialized version that takes into account
                   the evaluation of the functions following the `sum_counters`
                   structure from NLPModels.

    _resources_check!(::NLPStopping, ::T)

Note:
- function uses counters in `stp.pb`, and update the counters in the state.     
- function is compatible with `Counters`, `NLSCounters`, and any type whose entries match the entries of `max_cntrs`.   
"""
function _resources_check!(
  stp::NLPStopping{Pb, M, SRC, T, MStp, LoS},
  x::S,
) where {Pb <: AbstractNLPModel, M, SRC, T, MStp, LoS, S}
  max_cntrs = stp.meta.max_cntrs

  if length(max_cntrs) == 0
    return stp.meta.resources
  end

  # check all the entries in the counter
  max_f = check_entries_counters(stp.pb, max_cntrs)

  # Maximum number of function and derivative(s) computation
  if :neval_sum in keys(max_cntrs)
    max_evals = sum_counters(stp.pb) > max_cntrs[:neval_sum]
  end

  # global user limit diagnostic
  if (max_evals || max_f)
    stp.meta.resources = true
  end

  return stp.meta.resources
end

function check_entries_counters(nlp::AbstractNLPModel, max_cntrs)
  for f in keys(max_cntrs)
    if f in fieldnames(Counters)
      if eval(f)(nlp)::Int > max_cntrs[f]
        return true
      end
    end
  end
  return false
end

function check_entries_counters(nlp::AbstractNLSModel, max_cntrs)
  for f in keys(max_cntrs)
    if (f in fieldnames(NLSCounters)) && (f != :counters)
      if eval(f)(nlp)::Int > max_cntrs[f]
        return true
      end
    elseif f in fieldnames(Counters)
      if eval(f)(nlp)::Int > max_cntrs[f]
        return true
      end
    end
  end
  return false
end

"""
    `_unbounded_problem_check!`: This is the NLP specialized version that takes into account
                   that the problem might be unbounded if the objective or the
                   constraint function are unbounded.

    `_unbounded_problem_check!(::NLPStopping, ::T)`

Note:
- evaluate the objective function if `state.fx` for NLPAtX or `state.fx` for OneDAtX is `_init_field` and store in `state`.
- if minimize problem (i.e. nlp.meta.minimize is true) check if `state.fx <= - meta.unbounded_threshold`, otherwise check `state.fx ≥ meta.unbounded_threshold`.
"""
function _unbounded_problem_check!(
  stp::NLPStopping{Pb, M, SRC, NLPAtX{Score, S, T}, MStp, LoS},
  x,
) where {Pb, M, SRC, MStp, LoS, Score, S, T}
  if isnan(get_fx(stp.current_state))
    stp.current_state.fx = obj(stp.pb, x)
  end

  if stp.pb.meta.minimize
    f_too_large = get_fx(stp.current_state) <= -stp.meta.unbounded_threshold
  else
    f_too_large = get_fx(stp.current_state) >= stp.meta.unbounded_threshold
  end

  if f_too_large
    stp.meta.unbounded_pb = true
  end

  return stp.meta.unbounded_pb
end

function _unbounded_problem_check!(
  stp::NLPStopping{Pb, M, SRC, OneDAtX{S, T}, MStp, LoS},
  x,
) where {Pb, M, SRC, MStp, LoS, S, T}
  if isnan(get_fx(stp.current_state))
    stp.current_state.fx = obj(stp.pb, x)
  end

  if stp.pb.meta.minimize
    f_too_large = get_fx(stp.current_state) <= -stp.meta.unbounded_threshold
  else
    f_too_large = get_fx(stp.current_state) >= stp.meta.unbounded_threshold
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
function _infeasibility_check!(stp::NLPStopping, x::T) where {T}
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
    vio = any(z -> z == Inf, stp.current_state.current_score)
    if vio
      stp.meta.infeasible = true
    end
  else
    vio = any(z -> z == -Inf, stp.current_state.current_score)
    if vio
      stp.meta.infeasible = true
    end
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
