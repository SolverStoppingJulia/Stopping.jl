##############################################################################
#
# In this test problem we consider a quadratic penalty method.
# This example features an algorithm with the 3 steps:
# the penalization - the unconstrained min - the 1d min
#
# Note that there is no optimization of the evaluations here.
# The penalization gives an approximation of the gradients, multipliers...
#
# Note the use of a structure for the algorithmic parameters which is
# forwarded to all the 3 steps. If a parameter is not mentioned, then the default
# entry in the algorithm will be taken.
#
#############################################################################

include("uncons.jl")

##############################################################################
#
# Quadratic penalty algorithm
# fill_in! used instead of update! (works but usually more costly in evaluations)
# subproblems are solved via Newton method
#
#############################################################################
function penalty(stp :: NLPStopping, prms)

 #extract required values in the prms file
 r0 = :rho0       ∈ fieldnames(typeof(prms)) ? prms.rho0       : 100.0
 rm = :rho_min    ∈ fieldnames(typeof(prms)) ? prms.rho_min    : 1e-10
 ru = :rho_update ∈ fieldnames(typeof(prms)) ? prms.rho_update : 0.5

 return penalty(stp, rho0 = r0, rho_min = rm, ru = 0.5, prms = prms)
end

function penalty(stp :: NLPStopping; rho0 = 100.0, rho_min = 1e-10,
                                     rho_update = 0.5, prms = nothing)

 #algorithm's parameters
 rho = rho0

 #First call to the stopping
 #Becareful here, some entries must be filled in first.
 fill_in!(stp, stp.current_state.x)
 OK = start!(stp)

 #prepare the subproblem stopping:
 sub_nlp_at_x = NLPAtX(stp.current_state.x)
 sub_stp = NLPStopping(stp.pb, (x,y) -> Stopping.unconstrained(x,y),
                               sub_nlp_at_x, main_stp = stp)

 #main loop
 while !OK

  #solve the subproblem
  reinit!(sub_stp)
  sub_stp.meta.atol = min(rho, sub_stp.meta.atol)
  global_newton(sub_stp, prms)

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
    wolfe_prm   :: Float64 #Wolfe parameter
    onedsolve   :: Function #1D solver
    ls_func     :: Function

    #parameters of the 1d minimization
    back_update :: Float64 #backtracking update
    grad_need   :: Bool

    function Param(;rho0 :: Float64 = 100.0,
                    rho_min :: Float64 = sqrt(eps(Float64)),
                    rho_update :: Float64 = 0.5,
                    armijo_prm  :: Float64 = 0.01,
                    wolfe_prm   :: Float64 = 0.99,
                    onedsolve   :: Function = backtracking_ls,
                    ls_func     :: Function = (x,y)-> armijo(x,y, τ₀ = armijo_prm),
                    back_update :: Float64 = 0.5,
                    grad_need   :: Bool = false)
        return new(rho0, rho_min, rho_update,
                   armijo_prm, wolfe_prm, onedsolve, ls_func,
                   back_update, grad_need)
    end
end

##############################################################################
#
#
#
#############################################################################
printstyled("Constrained optimization: quadratic penalty tutorial.\n", color = :green)
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

printstyled("The End.\n", color = :green)
