using NLPModels
using Stopping

include("../test-stopping/rosenbrock.jl")

##############################################################################
#
# backtracking LineSearch
# !! The problem (stp.pb) is the 1d objective function
# Requirement: g0 and h0 have been filled in the State.
#
#############################################################################
function backtracking_ls(stp :: LS_Stopping, prms)
 #extract required values in the prms file
 return backtracking_ls(stp :: LS_Stopping, back_update = 0.5, prms = prms)
end

function backtracking_ls(stp :: LS_Stopping; back_update = 0.5, prms = nothing)

 state = stp.current_state; xt = state.x;

 #First call to stopping
 update!(state, x = xt, ht = stp.pb(xt))
 OK = start!(stp)

 #main loop
 while !OK

  xt = xt * back_update

  update!(state, x = xt, ht = stp.pb(xt))

  OK = stop!(stp)

 end

 return stp
end

##############################################################################
#
# Newton method with LineSearch
# Armijo by default
#
#############################################################################
function newton(stp :: NLPStopping, prms)
 #extract required values in the prms file
 #if prms != nothing
 # armijo_prm = prms.armijo_prm
 #end
 return newton(stp; ls_func = (x,y)-> armijo(x,y, τ₀ = 0.01), prms = prms)
end

function newton(stp :: NLPStopping; ls_func = (x,y)-> armijo(x,y, τ₀ = 0.01), prms = nothing)

    state = stp.current_state; xt = state.x; d = zeros(size(xt));

    #First call
    update!(state, x = xt, gx = grad(stp.pb, xt), Hx = hess(stp.pb, xt))
    OK = start!(stp)

    #Initialize the substopping
    h      = x-> obj(stp.pb, xt + x * d)
    lsatx  = LSAtT(1.0)
    lsstp = LS_Stopping(h, ls_func, lsatx)

    #main loop
    while !OK

        d = -inv(state.Hx) * state.gx

        #Prepare the substopping
        reinit!(lsstp)
        lsstp.pb = x-> obj(stp.pb, xt + x * d)
        update!(lsstp.current_state, x = 1.0, g₀ = dot(grad(stp.pb, xt),d), h₀ = obj(stp.pb, xt), ht = lsstp.pb(1.0))

        #solve subproblem
        backtracking_ls(lsstp, prms)
        alpha = lsstp.current_state.x

        #update
        xt = xt + alpha * d

        #update the state
        update!(state, x = xt, gx = grad(stp.pb, xt), Hx = hess(stp.pb, xt))

        OK = stop!(stp)
    end

    return stp
end

##############################################################################
#
# Quadratic penalty algorithm
# fill_in! used instead of update! (works but usually more costly in evaluations)
# subproblems are solved via Newton method
#
#############################################################################
function penalty(stp :: NLPStopping, prms)
 #extract required values in the prms file
 return penalty(stp, rho0 = 100.0, rho_min = 1e-10, rho_update = 0.5, prms = prms)
end

function penalty(stp :: NLPStopping; rho0 = 100.0, rho_min = 1e-10, rho_update = 0.5, prms = nothing)

 #algorithm's parameters
 rho = rho0

 #First call to the stopping
 #Becareful here, some entries must be filled in first.
 fill_in!(stp, stp.current_state.x)
 OK = start!(stp)

 #prepare the subproblem stopping:
 sub_nlp_at_x = NLPAtX(stp.current_state.x)
 sub_stp = NLPStopping(stp.pb, (x,y) -> Stopping.unconstrained(x,y), sub_nlp_at_x, main_stp = stp)

 #main loop
 while !OK

  #solve the subproblem
  reinit!(sub_stp)
  sub_stp.meta.atol = min(rho, sub_stp.meta.atol)
  newton(sub_stp, prms)

  #update!(stp)
  fill_in!(stp, sub_stp.current_state.x)

  #Either stop! is true OR the penalty parameter is too small
  OK = stop!(stp) || rho < rho_min

@show stp.meta.nb_of_stop, OK, rho

  #update the penalty parameter if necessary
  rho = rho * rho_update
 end

 return stp
end

##############################################################################
#
# Algorithmic parameters structure
#
#############################################################################

mutable struct Param

    #parameters for the penalty
    rho0       :: Float64 #initial value of the penalty parameter
    rho_min    :: Float64 #smallest possible parameter
    rho_update :: Float64 #update of the penalty parameter

    #parameters of the unconstrained minimization
    armijo_prm  :: Float64 #Armijo parameter

    #parameters of the 1d minimization
    back_update :: Float64 #backtracking update

    function Param(;rho0 :: Float64 = 100.0,
                    rho_min :: Float64 = sqrt(eps(Float64)),
                    rho_update :: Float64 = 0.5,
                    armijo_prm :: Float64 = 0.01,
                    back_update :: Float64 = 0.5)
        return new(rho0, rho_min,rho_update,armijo_prm,back_update)
    end
end

##############################################################################
#
#
#
#############################################################################

using LinearAlgebra
x0 = 1.5*ones(6)
c(x) = [sum(x)]
nlp2 = ADNLPModel(rosenbrock,  x0,
                 lvar = fill(-10.0,size(x0)), uvar = fill(10.0,size(x0)),
                 y0 = [0.0], c = c, lcon = [-Inf], ucon = [6.])

nlp_at_x_c = NLPAtX(x0, zeros(nlp2.meta.ncon))
stop_nlp_c = NLPStopping(nlp2, (x,y) -> Stopping.KKT(x,y), nlp_at_x_c)

penalty(stop_nlp_c)
status(stop_nlp_c)

#We can check afterwards, the score
Stopping.KKT(stop_nlp_c.pb, stop_nlp_c.current_state)
