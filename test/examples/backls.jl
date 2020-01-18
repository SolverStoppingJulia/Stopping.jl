##############################################################################
#
# In this test problem we consider a backtracking algorithm for 1D optimization.
# The scenario considers three different stopping criterion to solve a specific
# problem.
#
# This example illustrates how to use a "structure" to handle the algorithmic
# parameters and unify the input. The function
# backtracking_ls(stp :: LS_Stopping, prms)
# serves as a buffer for the real algorithm in the function
# backtracking_ls(stp :: LS_Stopping; back_update :: Float64 = 0.5,
#                                     grad_need :: Bool = true, prms = nothing)
#
# It also shows one way to handle the fact that some stopping criterion require
# more entries in the State than others. In this example, the Wolfe condition
# requires the derivative, while the Armijo condition does not. We use a
# boolean algorithmic parameter to avoid unnecessary evaluations of the
# derivatives.
#############################################################################

using LinearAlgebra, NLPModels, Stopping

##############################################################################
#
# backtracking LineSearch
# !! The problem (stp.pb) is the 1d objective function
# Requirement: g0 and h0 have been filled in the State.
#
#############################################################################
function backtracking_ls(stp :: LS_Stopping;
                         back_update :: Float64 = 0.5,
                         grad_need :: Bool = true,
                         prms = nothing)

 state = stp.current_state; xt = state.x;

 #First call to stopping
 gt = grad_need ? stp.pb.g(xt) : stp.current_state.gt
 OK = update_and_start!(stp, x = xt, ht = stp.pb.f(xt), gt = gt)

 #main loop
 while !OK

  xt = xt * back_update

  #we compute the derivative only if necessary
  gt = grad_need ? stp.pb.g(xt) : state.gt
  OK = update_and_stop!(stp, x = xt, ht = stp.pb.f(xt), gt = gt)

 end

 return stp
end

##############################################################################
#
# Buffer to handle a structure containing the algorithmic parameters.
#
#############################################################################
function backtracking_ls(stp :: LS_Stopping, prms)

 #extract required values in the prms file
 bu = try
  prms.back_update
 catch
  0.5
 end
 gn = try
   prms.grad_need
  catch
   true
  end

 return backtracking_ls(stp :: LS_Stopping, back_update = bu; grad_need = gn, prms = prms)
end

##############################################################################
#
# Scenario: optimization of the rosenbrock function at x0 along the opposite
# of the gradient.
#
# We can also use the LineModel
# available in https://github.com/JuliaSmoothOptimizers/SolverTools.jl
mutable struct onedoptim
    f :: Function
    g :: Function
end
#
# We also store all the algorithmic parameters in a structure.
mutable struct ParamLS

    #parameters of the 1d minimization
    back_update :: Float64 #backtracking update
    grad_need   :: Bool #true, if we need the derivative at each step

    function ParamLS(;back_update :: Float64 = 0.1,
                    grad_need   :: Bool = true)
        return new(back_update,grad_need)
    end
end
#############################################################################

include("../test-stopping/rosenbrock.jl")

x0 = 1.5*ones(6)
nlp = ADNLPModel(rosenbrock,  x0)
g0 = grad(nlp,x0)
h = onedoptim(x -> obj(nlp, x0 - x * g0), x -> - dot(g0,grad(nlp,x0 - x * g0)))

#We create 3 stopping:
#Define the LSAtT with mandatory entries g₀ and h₀.
lsatx  = LSAtT(1.0, h₀ = obj(nlp, x0), g₀ = -dot(grad(nlp, x0),grad(nlp, x0)))
lsstp  = LS_Stopping(h, (x,y)-> armijo(x,y, τ₀ = 0.01), lsatx)
lsatx2 = LSAtT(1.0, h₀ = obj(nlp, x0), g₀ = -dot(grad(nlp, x0),grad(nlp, x0)))
lsstp2 = LS_Stopping(h, (x,y)-> wolfe(x,y, τ₁ = 0.99), lsatx2)
lsatx3 = LSAtT(1.0, h₀ = obj(nlp, x0), g₀ = -dot(grad(nlp, x0),grad(nlp, x0)))
lsstp3 = LS_Stopping(h, (x,y)-> armijo_wolfe(x,y, τ₀ = 0.01, τ₁ = 0.99), lsatx3)

parameters = ParamLS(back_update = 0.5, grad_need = false)

printstyled("1D Optimization: backtracking tutorial.\n", color = :green)
printstyled("backtracking line search with Armijo:\n", color = :green)
backtracking_ls(lsstp, parameters)
@show status(lsstp)
@show lsstp.meta.nb_of_stop
@show lsstp.current_state.x

#For the following two functions we need the derivative:
parameters.grad_need = true

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
