import NLPModels: grad, cons, jac

"""
unconstrained_check: return the infinite norm of the gradient of the objective function

`unconstrained_check( :: AbstractNLPModel, :: NLPAtX; pnorm :: Float64 = Inf, kwargs...)`

Require `state.gx` (filled if not provided)

See also `unconstrained2nd_check`, `optim_check_bounded`, `KKT`
"""
function unconstrained_check(pb    :: AbstractNLPModel,
                             state :: NLPAtX;
                             pnorm :: Float64 = Inf,
                             kwargs...)

  if state.gx == _init_field(typeof(state.gx)) # should be filled if empty
    update!(state, gx = grad(pb, state.x))
  end

  return norm(state.gx, pnorm)
end

"""
unconstrained2nd_check: check the norm of the gradient and the smallest
                   eigenvalue of the hessian.

`unconstrained2nd_check( :: AbstractNLPModel, :: NLPAtX; pnorm :: Float64 = Inf, kwargs...)`

Require are `state.gx`, `state.Hx` (filled if not provided).

See also `unconstrained_check`, `optim_check_bounded`, `KKT`
"""
function unconstrained2nd_check(pb    :: AbstractNLPModel,
                                state :: NLPAtX;
                                pnorm :: Float64 = Inf,
                                kwargs...)

  if state.gx == _init_field(typeof(state.gx)) # should be filled if empty
    update!(state, gx = grad(pb, state.x))
  end
  if state.Hx == _init_field(typeof(state.Hx))
    update!(state, Hx = hess(pb, state.x).data)
  end

  res = max(norm(state.gx, pnorm),
            max(- eigmin(Symmetric(state.Hx, :L)), 0.0))

  return res
end

"""
optim\\_check\\_bounded: gradient of the objective function projected

`optim_check_bounded( :: AbstractNLPModel, :: NLPAtX; pnorm :: Float64 = Inf, kwargs...)`

Require `state.gx` (filled if not provided).

See also `unconstrained_check`, `unconstrained2nd_check`, `KKT`
"""
function optim_check_bounded(pb    :: AbstractNLPModel,
                             state :: NLPAtX;
                             pnorm :: Float64 = Inf,
                             kwargs...)

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
function _grad_lagrangian(pb    :: AbstractNLPModel,
                          state :: NLPAtX)

  if (pb.meta.ncon == 0) & !has_bounds(pb)
    return state.gx
  elseif pb.meta.ncon == 0
    return state.gx + state.mu
  else
    return state.gx + state.mu + state.Jx' * state.lambda
  end
end

function _sign_multipliers_bounds(pb    :: AbstractNLPModel,
                                  state :: NLPAtX)
  if has_bounds(pb)
    return vcat(min.(max.( state.mu,0.0), - state.x + pb.meta.uvar),
                min.(max.(-state.mu,0.0),   state.x - pb.meta.lvar))
  else
    return zeros(0)
  end
end

function _sign_multipliers_nonlin(pb    :: AbstractNLPModel,
                                  state :: NLPAtX)
  if pb.meta.ncon == 0
    return zeros(0)
  else
    return vcat(min.(max.( state.lambda,0.0), - state.cx + pb.meta.ucon),
                min.(max.(-state.lambda,0.0),   state.cx - pb.meta.lcon))
  end
end

function _feasibility(pb    :: AbstractNLPModel,
                      state :: NLPAtX)
  if pb.meta.ncon == 0
    return vcat(max.(  state.x  - pb.meta.uvar,0.0),
              max.(- state.x  + pb.meta.lvar,0.0))
  else
    return vcat(max.(  state.cx - pb.meta.ucon,0.0),
                max.(- state.cx + pb.meta.lcon,0.0),
                max.(  state.x  - pb.meta.uvar,0.0),
                max.(- state.x  + pb.meta.lvar,0.0))
  end
end

"""
KKT: verifies the KKT conditions

`KKT( :: AbstractNLPModel, :: NLPAtX; pnorm :: Float64 = Inf, kwargs...)`

Note: `state.gx` is mandatory + if bounds `state.mu` + if constraints `state.cx`, `state.Jx`, `state.lambda`.

See also `unconstrained_check`, `unconstrained2nd_check`, `optim_check_bounded`
"""
function KKT(pb    :: AbstractNLPModel,
             state :: NLPAtX;
             pnorm :: Float64 = Inf,
             kwargs...)

  #Check the gradient of the Lagrangian
  gLagx      = _grad_lagrangian(pb, state)
  #Check the complementarity condition for the bounds
  dual_res_bounds = _sign_multipliers_bounds(pb, state)
  #Check the complementarity condition for the constraints
  res_nonlin = _sign_multipliers_nonlin(pb, state)
  #Check the feasibility
  feas       = _feasibility(pb, state)

  res = vcat(gLagx, feas, dual_res_bounds, res_nonlin)

  return norm(res, pnorm)
end
