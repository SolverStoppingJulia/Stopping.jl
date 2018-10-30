import NLPModels: grad
"""
unconstrained: return the infinite norm of the gradient of the objective function
"""
function unconstrained(pb    :: AbstractNLPModel,
	                   state :: NLPAtX;
					   pnorm :: Float64 = Inf)

	# print_with_color(:blue, "on rentre dans unconstrained \n")

    if state.gx == nothing #vide il faut remplir
		update!(state, gx = grad(pb, state.x)) #utiliser la fonction fill_in ?
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
 return state.gx + state.mu + state.Jx * state.lambda
end

function _sign_multipliers_bounds(pb    :: AbstractNLPModel,
	                              state :: NLPAtX)
 return vcat(min.(max.(state.mu,0),state.x - pb.meta.uvar),
             min.(max.(-state.mu,0),-state.x + pb.meta.lvar))
end
function _sign_multipliers_nonlin(pb    :: AbstractNLPModel,
	                              state :: NLPAtX)

 return vcat(min.(max.( state.lambda,0),  state.cx - pb.meta.ucon),
             min.(max.(-state.lambda,0),- state.cx + pb.meta.lcon))
end
function _feasibility(pb    :: AbstractNLPModel,
	                  state :: NLPAtX)

 return vcat(max.(state.cx   - pb.meta.ucon,0),
             max.(- state.cx + pb.meta.lcon,0),
			 max.(  state.x - pb.meta.uvar,0),
			 max.(- state.x + pb.meta.lvar,0))
end

function KKT(pb    :: AbstractNLPModel,
	         state :: NLPAtX;
			 pnorm :: Float64 = Inf)

    if state.gx == nothing #vide il faut remplir
		update!(state, gx = grad(pb, state.x)) #utiliser la fonction fill_in ?
	end

	if length(state.lambda) == 0 return norm(state.gx, pnorm) end

	if state.Jx == nothing
		update!(state, Jx = jac(pb, state.x))
	end
    if state.mu != nothing
	  gLagx = _grad_lagrangian(pb, state)
    else #no infos on the Lagrange mulitpliers is available
	  n = length(state.x)
	  Jc = hcat(eye(n),state.Jx)
	  l = pinv(Jc) * (- state.gx)
	  state.mu, state.lambda = l[1:n], l[n:length(l)]
	  gLagx = Jc*l
    end

	res_bounds = _sign_multipliers_bounds(pb, state)
	res_nonlin = _sign_multipliers_nonlin(pb, state)

	if state.cx == nothing update!(state, cx = cons(pb, state.x)) end

	feas       = _feasibility(pb, state)

	res = vcat(gLagx, feas, res_bounds, res_nonlin)

	return norm(res, pnorm)
end
