###############################################################################
#
# The Stopping structure eases the implementation of algorithms and the
# stopping criterion.
#
# The following examples illustrate solver for optimization:
# - a backtracking 1D optimization solver
# - a globalized Newton for unconstrained optimization solver
# - a bound constraint active-set algorithm
# - a quadratic penalty algorithm for non-linear optimization
#
###############################################################################

using LinearAlgebra, NLPModels, Stopping, Test

include("../test-stopping/rosenbrock.jl")

##############################################################################
#
# Part 1/4
#
#############################################################################

printstyled("How to solve 1D optim problem: \n", color = :red)
include("backls.jl")

printstyled("1D Optimization: backtracking tutorial.\n", color = :green)

x0 = 1.5 * ones(6)
nlp = ADNLPModel(rosenbrock, x0)
g0 = grad(nlp, x0)
h = onedoptim(x -> obj(nlp, x0 - x * g0), x -> -dot(g0, grad(nlp, x0 - x * g0)))

#SCENARIO:
#We create 3 stopping:
#Define the LSAtT with mandatory entries g₀ and h₀.
lsatx = LSAtT(1.0, h₀ = obj(nlp, x0), g₀ = -dot(grad(nlp, x0), grad(nlp, x0)))
lsstp = LS_Stopping(h, lsatx, optimality_check = (x, y) -> armijo(x, y, τ₀ = 0.01))
lsatx2 = LSAtT(1.0, h₀ = obj(nlp, x0), g₀ = -dot(grad(nlp, x0), grad(nlp, x0)))
lsstp2 = LS_Stopping(h, lsatx2, optimality_check = (x, y) -> wolfe(x, y, τ₁ = 0.99))
lsatx3 = LSAtT(1.0, h₀ = obj(nlp, x0), g₀ = -dot(grad(nlp, x0), grad(nlp, x0)))
lsstp3 =
  LS_Stopping(h, lsatx3, optimality_check = (x, y) -> armijo_wolfe(x, y, τ₀ = 0.01, τ₁ = 0.99))

parameters = ParamLS(back_update = 0.5)

printstyled("backtracking line search with Armijo:\n", color = :green)
backtracking_ls(lsstp, parameters)
@show status(lsstp)
@show lsstp.meta.nb_of_stop
@show lsstp.current_state.x

printstyled("backtracking line search with Wolfe:\n", color = :green)
backtracking_ls(lsstp2, parameters)
@show status(lsstp2)
@show lsstp2.meta.nb_of_stop
@show lsstp2.current_state.x

printstyled("backtracking line search with Armijo-Wolfe:\n", color = :green)
backtracking_ls(lsstp3, parameters)
@show status(lsstp3)
@show lsstp3.meta.nb_of_stop
@show lsstp3.current_state.x

printstyled("The End.\n", color = :green)
printstyled("passed ✓ \n", color = :green)

##############################################################################
#
# Part 2/4
#
#############################################################################

printstyled("How to solve unconstrained optim problem: \n", color = :red)
include("uncons.jl")

printstyled("Unconstrained Optimization: globalized Newton.\n", color = :green)

x0 = 1.5 * ones(6)
nlp = ADNLPModel(rosenbrock, x0)

# We use the default builder using the KKT optimality function (which does not
# automatically fill in the State)
stop_nlp = NLPStopping(nlp)
parameters = PrmUn()

printstyled("Newton method with Armijo linesearch.\n", color = :green)
global_newton(stop_nlp, parameters)
@show status(stop_nlp)
#We can check afterwards, the score
@show Stopping.KKT(stop_nlp.pb, stop_nlp.current_state)
@show stop_nlp.meta.nb_of_stop

printstyled("Newton method with Armijo-Wolfe linesearch.\n", color = :green)
reinit!(stop_nlp, rstate = true, x = x0)
reset!(stop_nlp.pb) #reinitialize the counters of the NLP
parameters.ls_func =
  (x, y) -> armijo_wolfe(x, y, τ₀ = parameters.armijo_prm, τ₁ = parameters.wolfe_prm)

global_newton(stop_nlp, parameters)
@show status(stop_nlp)
#We can check afterwards, the score
@show Stopping.KKT(stop_nlp.pb, stop_nlp.current_state)
@show stop_nlp.meta.nb_of_stop

printstyled("The End.\n", color = :green)
printstyled("passed ✓ \n", color = :green)

##############################################################################
#
# Part 3/4
#
#############################################################################

printstyled("How to solve bound constrained optim problem: \n", color = :red)
include("activeset.jl")

printstyled("Constrained optimization: active-set algorithm tutorial.\n", color = :green)
x0 = 1.5 * ones(6);
x0[6] = 1.0;
nlp_bnd = ADNLPModel(rosenbrock, x0, fill(-10.0, size(x0)), fill(1.5, size(x0)))

nlp_bnd_at_x = NLPAtX(x0)
stop_nlp_c = NLPStopping(nlp_bnd, max_iter = 10)

activeset(stop_nlp_c)
@show status(stop_nlp_c)

printstyled("The End.\n", color = :green)

printstyled("passed ✓ \n", color = :green)

##############################################################################
#
# Part 4/4
#
#############################################################################

printstyled("How to solve nonlinear optim problem: \n", color = :red)
include("penalty.jl")

printstyled("Constrained optimization: quadratic penalty tutorial.\n", color = :green)
x0 = 1.5 * ones(6)
c(x) = [sum(x)]
nlp2 = ADNLPModel(rosenbrock, x0, fill(-10.0, size(x0)), fill(10.0, size(x0)), c, [-Inf], [5.0])

nlp_at_x_c = NLPAtX(x0, zeros(nlp2.meta.ncon))
stop_nlp_c = NLPStopping(
  nlp2,
  nlp_at_x_c,
  atol = 1e-3,
  max_cntrs = init_max_counters(obj = 400000, cons = 800000, sum = 1000000),
  optimality_check = (x, y) -> KKT(x, y),
)

penalty(stop_nlp_c)
@show status(stop_nlp_c)

#We can check afterwards, the score
@show KKT(stop_nlp_c.pb, stop_nlp_c.current_state)

printstyled("The End.\n", color = :green)

printstyled("passed ✓ \n", color = :green)
