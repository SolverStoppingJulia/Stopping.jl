import NLPModels: grad

export unconstrained
"""
unconstrained: return the infinite norm of the gradient of the objective function
"""
function unconstrained(pb    :: AbstractNLPModel,
                       state :: NLPAtX;
                       pnorm :: Float64 = Inf)

    if true in (isnan.(state.gx)) # should be filled if empty
        update!(state, gx = grad(pb, state.x))
    end

    res = norm(state.gx, pnorm)

    return res
end

import NLPModels: grad, cons, jac
"""
constrained: return the violation of the KKT conditions
length(lambda) > 0
"""
function _grad_lagrangian(pb    :: AbstractNLPModel,
                          state :: NLPAtX)
 if pb.meta.ncon == 0 & !has_bounds(pb)
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
"""
function KKT(pb    :: AbstractNLPModel,
             state :: NLPAtX;
             pnorm :: Float64 = Inf)

    #Check the gradient of the Lagrangian
    gLagx      = _grad_lagrangian(pb, state)
    #Check the complementarity condition for the bounds
    res_bounds = _sign_multipliers_bounds(pb, state)
    #Check the complementarity condition for the constraints
    res_nonlin = _sign_multipliers_nonlin(pb, state)
    #Check the feasibility
    feas       = _feasibility(pb, state)

    res = vcat(gLagx, feas, res_bounds, res_nonlin)

    return norm(res, pnorm)
end
