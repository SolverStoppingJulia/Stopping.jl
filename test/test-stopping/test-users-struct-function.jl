###############################################################################
#
# The Stopping structure eases the implementation of algorithms and the
# stopping criterion.
# We illustrate here the basic features of Stopping.
#
# Stopping has a mysterious attribute "user_specific_struct", we illustrate
# in this example how convenient it can be when specializing your own object.
# stop! also calls a function _user_check! that can also be specialized by the
# user.
# These user-dependent attribute/function allow specializing a new Stopping
# without too much of copy/paste.
#
###############################################################################
#using LinearAlgebra, NLPModels, Main.Stopping, Test

#Let us consider an NLPStopping:
meta = NLPModelMeta(6, x0=ones(6), lvar = -Inf*ones(6), uvar = Inf*ones(6),
                    ncon = 1, y0 = [0.0], lcon = [-Inf], ucon = [6.])
nlp  = ADNLPModel(meta, Counters(), x->x[1],  x-> [sum(x)])

#We create a new structure with two information not available by default in NLPStopping
mutable struct PenaltyNonlinear
    feasible :: Bool    #boolean, true if current iterate is feasible.
    rho      :: Float64 #penalty parameter
end

structtest = PenaltyNonlinear(true, 1e-1)
stop_bnd = NLPStopping(nlp, user_specific_struct = structtest)

@test stop_bnd.user_specific_struct.feasible == true
@test stop_bnd.user_specific_struct.rho == 1e-1

import Main.Stopping._user_check!
#We now redefine _user_check! to verify the feasibility
function _user_check!(stp :: NLPStopping, x :: T) where T <: Union{Number, AbstractVector}
 cx   = stp.current_state.cx
 feas = max.( stp.pb.meta.lcon - cx,
              cx - stp.pb.meta.ucon,
              stp.pb.meta.lvar - x,
              x - stp.pb.meta.uvar )
 tol = max(stp.meta.atol, stp.user_specific_struct.rho)
 stp.user_specific_struct.feasible = norm(feas, Inf) <= tol
end

sol = 2*ones(6)
fill_in!(stop_bnd, sol)
stop!(stop_bnd)
@test stop_bnd.user_specific_struct.feasible == false

sol2 = ones(6); sol2[1] = 1.0 + 1e-1
reinit!(stop_bnd, rstate = true)
fill_in!(stop_bnd, sol2)
stop!(stop_bnd)
#since violation of the constraints smaller than rho.
@test stop_bnd.user_specific_struct.feasible == true

#remove the _user_check!(stp :: NLPStopping, x :: Main.Stopping.Iterate) from the workspace
Base.delete_method(which(_user_check!, (NLPStopping,Union{Number, AbstractVector},)))
