import NLPModels: grad, cons, jac

"""
    `unconstrained_check( :: AbstractNLPModel, :: NLPAtX; pnorm :: Real = Inf, kwargs...)`

Return the `pnorm`-norm of the gradient of the objective function.

Require `state.gx` (filled if not provided).

See also `optim_check_bounded`, `KKT`
"""
function unconstrained_check(
  pb::AbstractNLPModel,
  state::NLPAtX{S, T};
  pnorm::eltype(T) = eltype(T)(Inf),
  kwargs...,
) where {S, T}
  if state.gx == _init_field(typeof(state.gx)) # should be filled if empty
    update!(state, gx = grad(pb, state.x))
  end

  return norm(state.gx, pnorm)
end

"""
    `optim_check_bounded( :: AbstractNLPModel, :: NLPAtX; pnorm :: Real = Inf, kwargs...)`

Check the `pnorm`-norm of the gradient of the objective function projected over the bounds.

Require `state.gx` (filled if not provided).

See also `unconstrained_check`, `KKT`
"""
function optim_check_bounded(
  pb::AbstractNLPModel,
  state::NLPAtX{S, T};
  pnorm::eltype(T) = eltype(T)(Inf),
  kwargs...,
) where {S, T}
  if state.gx == _init_field(typeof(state.gx)) # should be filled if void
    update!(state, gx = grad(pb, state.x))
  end

  proj = max.(min.(state.x - state.gx, pb.meta.uvar), pb.meta.lvar)
  gradproj = state.x - proj

  return norm(gradproj, pnorm)
end

"""
constrained: return the violation of the KKT conditions
length(lambda) > 0
"""
function _grad_lagrangian(pb::AbstractNLPModel, state::NLPAtX{S, T}) where {S, T}
  if (pb.meta.ncon == 0) & !has_bounds(pb)
    return state.gx
  elseif pb.meta.ncon == 0
    return state.gx + state.mu
  elseif !has_bounds(pb)
    return state.gx + state.Jx' * state.lambda
  else
    return state.gx + state.mu + state.Jx' * state.lambda
  end
end

function _sign_multipliers_bounds(pb::AbstractNLPModel, state::NLPAtX{S, T}) where {S, T}
  if has_bounds(pb)
    return vcat(
      min.(max.(state.mu, zero(eltype(T))), -state.x + pb.meta.uvar),
      min.(max.(-state.mu, zero(eltype(T))), state.x - pb.meta.lvar),
    )
  else
    return zeros(eltype(T), 0)
  end
end

function _sign_multipliers_nonlin(pb::AbstractNLPModel, state::NLPAtX{S, T}) where {S, T}
  if pb.meta.ncon == 0
    return zeros(eltype(T), 0)
  else
    return vcat(
      min.(max.(state.lambda, zero(eltype(T))), -state.cx + pb.meta.ucon),
      min.(max.(-state.lambda, zero(eltype(T))), state.cx - pb.meta.lcon),
    )
  end
end

function _feasibility(pb::AbstractNLPModel, state::NLPAtX{S, T}) where {S, T}
  if pb.meta.ncon == 0
    return vcat(
      max.(state.x - pb.meta.uvar, zero(eltype(T))),
      max.(-state.x + pb.meta.lvar, zero(eltype(T))),
    )
  else
    return vcat(
      max.(state.cx - pb.meta.ucon, zero(eltype(T))),
      max.(-state.cx + pb.meta.lcon, zero(eltype(T))),
      max.(state.x - pb.meta.uvar, zero(eltype(T))),
      max.(-state.x + pb.meta.lvar, zero(eltype(T))),
    )
  end
end

"""
    `KKT( :: AbstractNLPModel, :: NLPAtX; pnorm :: Real = Inf, kwargs...)`

Check the KKT conditions.

Note: `state.gx` is mandatory + if bounds `state.mu` + if constraints `state.cx`, `state.Jx`, `state.lambda`.

See also `unconstrained_check`, `optim_check_bounded`
"""
function KKT(
  pb::AbstractNLPModel,
  state::NLPAtX{S, T};
  pnorm::eltype(T) = eltype(T)(Inf),
  kwargs...,
) where {S, T}
  if unconstrained(pb) && state.gx == _init_field(typeof(state.gx))
    @warn "KKT needs stp.current_state.gx to be filled-in."
    return eltype(T)(Inf)
  elseif has_bounds(pb) && state.mu == _init_field(typeof(state.mu))
    @warn "KKT needs stp.current_state.mu to be filled-in."
    return eltype(T)(Inf)
  elseif get_ncon(pb) > 0 && (
    state.cx == _init_field(typeof(state.cx)) ||
    state.Jx == _init_field(typeof(state.Jx)) ||
    state.lambda == _init_field(typeof(state.lambda))
  )
    @warn "KKT needs stp.current_state.cx, stp.current_state.Jx and stp.current_state.lambda to be filled-in."
    return eltype(T)(Inf)
  end

  #Check the gradient of the Lagrangian
  gLagx = _grad_lagrangian(pb, state)
  #Check the complementarity condition for the bounds
  dual_res_bounds = _sign_multipliers_bounds(pb, state)
  #Check the complementarity condition for the constraints
  res_nonlin = _sign_multipliers_nonlin(pb, state)
  #Check the feasibility
  feas = _feasibility(pb, state)

  res = vcat(gLagx, feas, dual_res_bounds, res_nonlin)

  return norm(res, pnorm)
end
