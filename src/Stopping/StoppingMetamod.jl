"""
Type: StoppingMeta

Methods: no methods.

Attributes:
- `atol`: absolute tolerance.
- `rtol`: relative tolerance.
- `optimality0`: optimality score at the initial guess.
- `tol_check`: Function of `atol`, `rtol` and `optimality0` testing a score to zero.
- `tol_check_neg`: Function of `atol`, `rtol` and `optimality0` testing a score to zero.
- `check_pos`: pre-allocation for positive tolerance
- `check_neg`: pre-allocation for negative tolerance
- `recomp_tol`: true if tolerances are updated
- `optimality_check`: a stopping criterion via an admissibility function
- `unbounded_threshold`: threshold for unboundedness of the problem.
- `unbounded_x`: threshold for unboundedness of the iterate.
- `max_f`: maximum number of function (and derivatives) evaluations.
- `max_cntrs`: Dict contains the maximum number of evaluations
- `max_eval`:  maximum number of function (and derivatives) evaluations.
- `max_iter`: threshold on the number of stop! call/number of iteration.
- `max_time`: time limit to let the algorithm run.
- `nb_of_stop`: keep track of the number of stop! call/iteration.
- `start_time`: keep track of the time at the beginning.
- `fail_sub_pb`: status.
- `unbounded`: status.
- `unbounded_pb`: status.
- `tired`: status.
- `stalled`: status.
- `iteration_limit`: status.
- `resources`: status.
- `optimal`: status.
- `infeasible`: status.
- `main_pb`: status.
- `domainerror`: status.
- `suboptimal`: status.
- `stopbyuser`: status.
- `exception`: status.
- `meta_user_struct`:  Any
- `user_check_func!`: Function (AbstractStopping, Bool) -> callback.

`StoppingMeta(;atol :: Number = 1.0e-6, rtol :: Number = 1.0e-15, optimality0 :: Number = 1.0, tol_check :: Function = (atol,rtol,opt0) -> max(atol,rtol*opt0), tol_check_neg :: Function = (atol,rtol,opt0) -> -max(atol,rtol*opt0), unbounded_threshold :: Number = 1.0e50, unbounded_x :: Number = 1.0e50, max_f :: Int = typemax(Int), max_eval :: Int = 20000, max_iter :: Int = 5000, max_time :: Number = 300.0, start_time :: Float64 = NaN, meta_user_struct :: Any = nothing, kwargs...)`

an alternative with constant tolerances:

`StoppingMeta(tol_check :: T, tol_check_neg :: T;atol :: Number = 1.0e-6, rtol :: Number = 1.0e-15, optimality0 :: Number = 1.0, unbounded_threshold :: Number = 1.0e50, unbounded_x :: Number = 1.0e50, max_f :: Int = typemax(Int), max_eval :: Int = 20000, max_iter :: Int = 5000, max_time :: Number = 300.0, start_time :: Float64 = NaN, meta_user_struct :: Any = nothing, kwargs...)`

Note:
- It is a mutable struct, therefore we can modify elements of a `StoppingMeta`.
- The `nb_of_stop` is incremented everytime `stop!` or `update_and_stop!` is called
- The `optimality0` is modified once at the beginning of the algorithm (`start!`)
- The `start_time` is modified once at the beginning of the algorithm (`start!`)
      if not precised before.
- The different status: `fail_sub_pb`, `unbounded`, `unbounded_pb`, `tired`, `stalled`,
      `iteration_limit`, `resources`, `optimal`, `main_pb`, `domainerror`, `suboptimal`, `infeasible`
- `fail_sub_pb`, `suboptimal`, and `infeasible` are modified by the algorithm.
- `optimality_check` takes two inputs (`AbstractNLPModel`, `NLPAtX`)
 and returns a `Number` or an `AbstractVector` to be compared to `0`.
- `optimality_check` does not necessarily fill in the State.

Examples: `StoppingMeta()`, `StoppingMeta(1., -1.)`
"""
mutable struct StoppingMeta{
  TolType <: Number,
  CheckType, #Type of the tol_check output
  MUS, #Meta User Struct
  IntType <: Int,
} <: AbstractStoppingMeta

  # problem tolerances
  atol::TolType # absolute tolerance
  rtol::TolType # relative tolerance
  optimality0::TolType # value of the optimality residual at starting point
  tol_check::Union{Function, CheckType} #function of atol, rtol and optimality0
  #by default: tol_check = max(atol, rtol * optimality0)
  #other example: atol + rtol * optimality0
  tol_check_neg::Union{Function, CheckType} # function of atol, rtol and optimality0
  check_pos::CheckType #pre-allocation for positive tolerance
  check_neg::CheckType #pre-allocation for negative tolerance
  optimality_check::Function # stopping criterion
  # Function of (pb, state; kwargs...)
  #return type  :: Union{Number, eltype(stp.meta)}
  recomp_tol::Bool #true if tolerances are updated

  unbounded_threshold::TolType # beyond this value, the problem is declared unbounded
  unbounded_x::TolType # beyond this value, ||x||_\infty is unbounded

  # fine grain control on ressources
  max_f::IntType    # max function evaluations allowed TODO: used?
  max_cntrs::Dict{Symbol, Int} #contains the detailed max number of evaluations

  # global control on ressources
  max_eval::IntType    # max evaluations (f+g+H+Hv) allowed TODO: used?
  max_iter::IntType    # max iterations allowed
  max_time::Float64 # max elapsed time allowed

  #intern Counters
  nb_of_stop::IntType
  #intern start_time
  start_time::Float64

  # stopping properties status of the problem)
  fail_sub_pb::Bool
  unbounded::Bool
  unbounded_pb::Bool
  tired::Bool
  stalled::Bool
  iteration_limit::Bool
  resources::Bool
  optimal::Bool
  infeasible::Bool
  main_pb::Bool
  domainerror::Bool
  suboptimal::Bool
  stopbyuser::Bool
  exception::Bool

  meta_user_struct::MUS
  user_check_func!::Function
end

function StoppingMeta(
  tol_check::CheckType,
  tol_check_neg::CheckType;
  atol::Number = 1.0e-6,
  rtol::Number = 1.0e-15,
  optimality0::Number = 1.0,
  optimality_check::Function = (a, b) -> 1.0,
  recomp_tol::Bool = false,
  unbounded_threshold::Number = 1.0e50, #typemax(Float64)
  unbounded_x::Number = 1.0e50,
  max_f::Int = typemax(Int),
  max_cntrs::Dict{Symbol, Int} = Dict{Symbol, Int}(),
  max_eval::Int = 20000,
  max_iter::Int = 5000,
  max_time::Float64 = 300.0,
  start_time::Float64 = NaN,
  meta_user_struct::Any = nothing,
  user_check_func!::Function = (stp::AbstractStopping, start::Bool) -> nothing,
  kwargs...,
) where {CheckType}
  check_pos = tol_check
  check_neg = tol_check_neg

  # This might be an expansive step.
  # if (true in (check_pos .< check_neg)) #any(x -> x, check_pos .< check_neg)
  #     throw(ErrorException("StoppingMeta: tol_check should be greater than tol_check_neg."))
  # end

  fail_sub_pb = false
  unbounded = false
  unbounded_pb = false
  tired = false
  stalled = false
  iteration_limit = false
  resources = false
  optimal = false
  infeasible = false
  main_pb = false
  domainerror = false
  suboptimal = false
  stopbyuser = false
  exception = false

  nb_of_stop = 0

  #new{TolType, typeof(check_pos), typeof(meta_user_struct)}
  return StoppingMeta(
    atol,
    rtol,
    optimality0,
    tol_check,
    tol_check_neg,
    check_pos,
    check_neg,
    optimality_check,
    recomp_tol,
    unbounded_threshold,
    unbounded_x,
    max_f,
    max_cntrs,
    max_eval,
    max_iter,
    max_time,
    nb_of_stop,
    start_time,
    fail_sub_pb,
    unbounded,
    unbounded_pb,
    tired,
    stalled,
    iteration_limit,
    resources,
    optimal,
    infeasible,
    main_pb,
    domainerror,
    suboptimal,
    stopbyuser,
    exception,
    meta_user_struct,
    user_check_func!,
  )
end

function StoppingMeta(;
  atol::Number = 1.0e-6,
  rtol::Number = 1.0e-15,
  optimality0::Number = 1.0,
  tol_check::Function = (atol::Number, rtol::Number, opt0::Number) -> max(atol, rtol * opt0),
  tol_check_neg::Function = (atol::Number, rtol::Number, opt0::Number) ->
    -tol_check(atol, rtol, opt0),
  optimality_check::Function = (a, b) -> 1.0,
  recomp_tol::Bool = true,
  unbounded_threshold::Number = 1.0e50, #typemax(Float64)
  unbounded_x::Number = 1.0e50,
  max_f::Int = typemax(Int),
  max_cntrs::Dict{Symbol, Int} = Dict{Symbol, Int}(),
  max_eval::Int = 20000,
  max_iter::Int = 5000,
  max_time::Float64 = 300.0,
  start_time::Float64 = NaN,
  meta_user_struct::Any = nothing,
  user_check_func!::Function = (stp::AbstractStopping, start::Bool) -> nothing,
  kwargs...,
)
  check_pos = tol_check(atol, rtol, optimality0)
  check_neg = tol_check_neg(atol, rtol, optimality0)

  # This might be an expansive step.
  # if (true in (check_pos .< check_neg)) #any(x -> x, check_pos .< check_neg)
  #     throw(ErrorException("StoppingMeta: tol_check should be greater than tol_check_neg."))
  # end

  fail_sub_pb = false
  unbounded = false
  unbounded_pb = false
  tired = false
  stalled = false
  iteration_limit = false
  resources = false
  optimal = false
  infeasible = false
  main_pb = false
  domainerror = false
  suboptimal = false
  stopbyuser = false
  exception = false

  nb_of_stop = 0

  #new{TolType, typeof(check_pos), typeof(meta_user_struct)}
  return StoppingMeta(
    atol,
    rtol,
    optimality0,
    tol_check,
    tol_check_neg,
    check_pos,
    check_neg,
    optimality_check,
    recomp_tol,
    unbounded_threshold,
    unbounded_x,
    max_f,
    max_cntrs,
    max_eval,
    max_iter,
    max_time,
    nb_of_stop,
    start_time,
    fail_sub_pb,
    unbounded,
    unbounded_pb,
    tired,
    stalled,
    iteration_limit,
    resources,
    optimal,
    infeasible,
    main_pb,
    domainerror,
    suboptimal,
    stopbyuser,
    exception,
    meta_user_struct,
    user_check_func!,
  )
end

const meta_statuses = [
  :fail_sub_pb,
  :unbounded,
  :unbounded_pb,
  :tired,
  :stalled,
  :iteration_limit,
  :resources,
  :optimal,
  :suboptimal,
  :main_pb,
  :domainerror,
  :infeasible,
  :stopbyuser,
  :exception,
]

"""
`OK_check(meta :: StoppingMeta)`

Return true if one of the decision boolean is true.
"""
function OK_check(
  meta::StoppingMeta{TolType, CheckType, MUS, IntType},
) where {TolType, CheckType, MUS, IntType}
  #13 checks
  OK =
    meta.optimal ||
    meta.tired ||
    meta.iteration_limit ||
    meta.resources ||
    meta.unbounded ||
    meta.unbounded_pb ||
    meta.main_pb ||
    meta.domainerror ||
    meta.suboptimal ||
    meta.fail_sub_pb ||
    meta.stalled ||
    meta.infeasible ||
    meta.stopbyuser
  return OK
end

"""
`tol_check(meta :: StoppingMeta)`

Return the pair of tolerances, recomputed if `meta.recomp_tol` is `true`.
"""
function tol_check(
  meta::StoppingMeta{TolType, CheckType, MUS, IntType},
) where {TolType, CheckType, MUS, IntType}
  if meta.recomp_tol
    atol, rtol, opt0 = meta.atol, meta.rtol, meta.optimality0
    setfield!(meta, :check_pos, meta.tol_check(atol, rtol, opt0))
    setfield!(meta, :check_neg, meta.tol_check_neg(atol, rtol, opt0))
  end

  return (meta.check_pos, meta.check_neg)
end

"""
`update_tol!(meta :: StoppingMeta; atol = meta.atol, rtol = meta.rtol, optimality0 = meta.optimality0)`

Update the tolerances parameters. Set `meta.recomp_tol` as `true`.
"""
function update_tol!(
  meta::StoppingMeta{TolType, CheckType, MUS, IntType};
  atol::TolType = meta.atol,
  rtol::TolType = meta.rtol,
  optimality0::TolType = meta.optimality0,
) where {TolType, CheckType, MUS, IntType}
  setfield!(meta, :recomp_tol, true)
  setfield!(meta, :atol, atol)
  setfield!(meta, :rtol, rtol)
  setfield!(meta, :optimality0, optimality0)

  return meta
end

function reinit!(
  meta::StoppingMeta{TolType, CheckType, MUS, IntType},
) where {TolType, CheckType, MUS, IntType}
  for k in meta_statuses
    setfield!(meta, k, false)
  end

  return meta
end

function checktype(
  meta::StoppingMeta{TolType, CheckType, MUS, IntType},
) where {TolType, CheckType, MUS, IntType}
  return CheckType
end

function toltype(
  meta::StoppingMeta{TolType, CheckType, MUS, IntType},
) where {TolType, CheckType, MUS, IntType}
  return TolType
end

function metausertype(
  meta::StoppingMeta{TolType, CheckType, MUS, IntType},
) where {TolType, CheckType, MUS, IntType}
  return MUS
end

function inttype(
  meta::StoppingMeta{TolType, CheckType, MUS, IntType},
) where {TolType, CheckType, MUS, IntType}
  return IntType
end

import Base.show
function show(io::IO, meta::AbstractStoppingMeta)
  varlines = "$(typeof(meta)) has"
  if OK_check(meta)
    ntrue = 0
    for f in meta_statuses
      if getfield(meta, f) && ntrue == 0
        ntrue += 1
        varlines = string(varlines, @sprintf(" %s", f))
      elseif getfield(meta, f)
        ntrue += 1
        varlines = string(varlines, @sprintf(",\n %s", f))
      end
    end
    varlines =
      ntrue == 1 ? string(varlines, " as only true status.\n") :
      string(varlines, " as true statuses.\n")
  else
    varlines = string(varlines, " now no true statuses.\n")
  end
  varlines = string(varlines, "The return type of tol check functions is $(checktype(meta))")
  if meta.recomp_tol
    varlines = string(varlines, ", and these functions are reevaluated at each stop!.\n")
  else
    varlines = string(varlines, ".\n")
  end
  varlines = string(varlines, "Current tolerances are: \n")
  for k in [
    :atol,
    :rtol,
    :optimality0,
    :unbounded_threshold,
    :unbounded_x,
    :max_f,
    :max_eval,
    :max_iter,
    :max_time,
  ]
    varlines = string(
      varlines,
      @sprintf("%19s: %s (%s) \n", k, getfield(meta, k), typeof(getfield(meta, k)))
    )
  end
  if metausertype(meta) != Nothing
    varlines =
      string(varlines, "The user defined structure in the meta is a $(metausertype(meta)).\n")
  else
    varlines = string(varlines, "There is no user defined structure in the meta.\n")
  end
  println(io, varlines)
end
