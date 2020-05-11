##############################################################################
#
# In this test problem, we consider a backtracking algorithm for 1D optimization.
# The scenario considers three different stopping criterion to solve a specific
# problem.
#
# This example illustrates how to use a "structure" to handle the algorithmic
# parameters and unify the input. The function
# backtracking_ls(stp :: LS_Stopping, prms)
# serves as a buffer for the real algorithm in the function
# backtracking_ls(stp :: LS_Stopping; back_update :: Float64 = 0.5, prms = nothing)
#
# It also shows that obsolete information in the State (after an update of x)
# must be removed by the algorithm. Otherwise, the optimality_check function
# cannot make the difference between valid and invalid entries.
#############################################################################

##############################################################################
#
# We create a basic structure to handle 1D optimization.
#
# We can also use the LineModel available in
# https://github.com/JuliaSmoothOptimizers/SolverTools.jl

mutable struct onedoptim
    f :: Function
    g :: Function
end

#############################################################################
#
# We specialize three optimality_check functions for 1D optimization to the
# onedoptim type of problem.
#
# The default functions do not fill in automatically the necessary entries.
#
import Stopping: armijo, wolfe, armijo_wolfe

function armijo(h :: onedoptim, h_at_t :: LSAtT; τ₀ :: Float64 = 0.01, kwargs...)

 h_at_t.ht = h_at_t.ht == nothing ? h.f(h_at_t.x) : h_at_t.ht
 h_at_t.h₀ = h_at_t.h₀ == nothing ? h.f(0) : h_at_t.h₀
 h_at_t.g₀ = h_at_t.g₀ == nothing ? h.g(0) : h_at_t.g₀

 hgoal = h_at_t.ht - h_at_t.h₀ - h_at_t.g₀ * h_at_t.x * τ₀

 return max(hgoal, 0.0)
end

function wolfe(h :: onedoptim, h_at_t :: LSAtT; τ₁ :: Float64 = 0.99, kwargs...)

 h_at_t.gt = h_at_t.gt == nothing ? h.g(h_at_t.x) : h_at_t.gt
 h_at_t.g₀ = h_at_t.g₀ == nothing ? h.g(0) : h_at_t.g₀

 wolfe = τ₁ .* h_at_t.g₀ - abs(h_at_t.gt)
 return max(wolfe, 0.0)
end

function armijo_wolfe(h :: onedoptim, h_at_t :: LSAtT; τ₀ :: Float64 = 0.01, τ₁ :: Float64 = 0.99, kwargs...)

 h_at_t.ht = h_at_t.ht == nothing ? h.f(h_at_t.x) : h_at_t.ht
 h_at_t.h₀ = h_at_t.h₀ == nothing ? h.f(0) : h_at_t.h₀
 h_at_t.gt = h_at_t.gt == nothing ? h.g(h_at_t.x) : h_at_t.gt
 h_at_t.g₀ = h_at_t.g₀ == nothing ? h.g(0) : h_at_t.g₀

 return max(armijo(h, h_at_t, τ₀ = τ₀),wolfe(h, h_at_t, τ₁ = τ₁), 0.0)
end

##############################################################################
#
# backtracking LineSearch
# !! The problem (stp.pb) is the 1d objective function
#
# Requirement: g0 and h0 have been filled in the State.
#
#############################################################################
function backtracking_ls(stp :: LS_Stopping;
                         back_update :: Float64 = 0.5,
                         prms = nothing)

 state = stp.current_state; xt = state.x;

 #First call to stopping
 OK = start!(stp)

 #main loop
 while !OK

  xt = xt * back_update

  #after update the infos in the State are no longer valid (except h₀, g₀)
  reinit!(state, xt, h₀ = stp.current_state.h₀, g₀ = stp.current_state.g₀)

  #we call the stop!
  OK = stop!(stp)

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
 bu = :back_update   ∈ fieldnames(typeof(prms)) ? prms.back_update : 0.5

 return backtracking_ls(stp :: LS_Stopping, back_update = bu; prms = prms)
end

##############################################################################
#
# Scenario: optimization of the rosenbrock function at x0 along the opposite
# of the gradient.
#
# We also store all the algorithmic parameters in a structure.
mutable struct ParamLS

    #parameters of the 1d minimization
    back_update :: Float64 #backtracking update

    function ParamLS(;back_update :: Float64 = 0.1)
        return new(back_update)
    end
end
#############################################################################
