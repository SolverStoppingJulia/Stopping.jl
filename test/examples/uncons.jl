##############################################################################
#
# In this test problem we consider a globalized Newton method.
# The scenario considers two different stopping criterion to solve the linesearch.
#
# i) This example illustrates how the "structure" handling the algorithmic
# parameters can be passed to the solver of the subproblem.
#
# ii) This algorithm handles a sub-stopping defined by passing the stopping as a
# keyword argument. Note that when a stopping is used several time, it has to
# be reinitialized.
#
# iii) It also shows how we can reuse the information and avoid unnecessary evals.
# Here the objective function of the main problem and sub-problem are the same.
# Warning: the structure onedoptim however does not allow keeping the gradient
# of the main problem. This issue can be corrected by using a specialized State.
#
#############################################################################

include("backls.jl")
#contains the rosenbrock and backtracking_ls functions, and the onedoptim struc
#using LinearAlgebra, NLPModels, Stopping

##############################################################################
#
# Newton method with LineSearch
#
#############################################################################
function global_newton(stp :: NLPStopping, prms)

 lf = :ls_func   ∈ fieldnames(typeof(prms)) ? prms.ls_func : armijo
 os = :onedsolve ∈ fieldnames(typeof(prms)) ? prms.onedsolve : backtracking_ls

 return global_newton(stp, os, lf; prms = prms)
end

function global_newton(stp       :: NLPStopping,
                       onedsolve :: Function,
                       ls_func   :: Function;
                       prms = nothing)

    state = stp.current_state; xt = state.x; nlp = stp.pb
    d = zeros(size(xt));

    #First call
    OK = update_and_start!(stp, x = xt, fx = obj(nlp, xt), gx = grad(nlp, xt), Hx = hess(nlp, xt))

    #Initialize the sub-Stopping with the main Stopping as keyword argument
    h = onedoptim(x -> obj(nlp, xt + x * d),
                  x -> dot(d, grad(nlp, xt + x * d)))
    lsstp = LS_Stopping(h, ls_func, LSAtT(1.0), main_stp = stp)

    #main loop
    while !OK

        d = -inv(state.Hx) * state.gx

        #Prepare the substopping
        #We reinitialize the stopping before each new use
        reinit!(lsstp)
        lsstp.pb = onedoptim(x -> obj(nlp, xt + x * d),
                             x -> dot(d, grad(nlp, xt + x * d)))
        update!(lsstp.current_state, x = 1.0, g₀ = -dot(state.gx,d), h₀ = state.fx)

        #solve subproblem
        onedsolve(lsstp, prms)

        if status(lsstp) == :Optimal
         alpha = lsstp.current_state.x
         #update
         xt = xt + alpha * d
         #Since the onedoptim and the nlp have the same objective function,
         #we save one evaluation.
         update!(stp.current_state, fx = lsstp.current_state.ht)
        else
         stp.meta.fail_sub_pb = true
        end

        OK = update_and_stop!(stp, x = xt, gx = grad(nlp, xt), Hx = hess(nlp, xt))

    end

    return stp
end

##############################################################################
#
#
#
mutable struct PrmUn

    #parameters of the unconstrained minimization
    armijo_prm  :: Float64 #Armijo parameter
    wolfe_prm   :: Float64 #Wolfe parameter
    onedsolve   :: Function #1D solver
    ls_func     :: Function

    #parameters of the 1d minimization
    back_update :: Float64 #backtracking update
    grad_need   :: Bool

    function PrmUn(;armijo_prm  :: Float64 = 0.01,
                    wolfe_prm   :: Float64 = 0.99,
                    onedsolve   :: Function = backtracking_ls,
                    ls_func     :: Function = (x,y)-> armijo(x,y, τ₀ = armijo_prm),
                    back_update :: Float64 = 0.5,
                    grad_need   :: Bool    = false)
        return new(armijo_prm,wolfe_prm,onedsolve,ls_func,back_update,grad_need)
    end
end
#############################################################################
printstyled("Unconstrained Optimization: globalized Newton.\n", color = :green)

x0 = 1.5*ones(6)
nlp = ADNLPModel(rosenbrock,  x0)

stop_nlp = NLPStopping(nlp)
parameters = PrmUn()

printstyled("Newton method with Armijo linesearch.\n", color = :green)
global_newton(stop_nlp, parameters)
@show status(stop_nlp)
#We can check afterwards, the score
@show Stopping.KKT(stop_nlp.pb, stop_nlp.current_state)
@show stop_nlp.meta.nb_of_stop

printstyled("Newton method with Armijo-Wolfe linesearch.\n", color = :green)
reinit!(stop_nlp)
update!(stop_nlp.current_state, x = x0)
parameters.ls_func = (x,y)-> armijo_wolfe(x,y, τ₀ = parameters.armijo_prm,
                                               τ₁ = parameters.wolfe_prm)
parameters.grad_need = true #wolfe condition requires the derivative

global_newton(stop_nlp, parameters)
@show status(stop_nlp)
#We can check afterwards, the score
@show Stopping.KKT(stop_nlp.pb, stop_nlp.current_state)
@show stop_nlp.meta.nb_of_stop

printstyled("The End.\n", color = :green)
