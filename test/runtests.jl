using Stopping
#include("../src/Stopping.jl")
using Base.Test

# write your own tests here
using NLPModels
using JuMP

include("line_model.jl")  # For LineFunction  definition
#include("woods.jl")
#nlp = MathProgNLPModel(woods(), name="woods")
include("genrose.jl")
nlp = MathProgNLPModel(genrose(4), name="genrose")

n = nlp.meta.nvar

include("armijo_wolfe.jl")
include("steepestS.jl")

@printf("Problem  Dim  Optim f  Grad norm  f evals g evals Hv evals  iters  outcome         time\n")


s = TStoppingB(nlp,max_iter = 1) # to trigger compilation of steepest
(x, f, gNorm, iter, optimal, tired, status, Stime) = steepest(nlp, s=s, verbose=false)
@printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
        nlp.meta.name, nlp.meta.nvar, f, gNorm,
        nlp.counters.neval_obj, nlp.counters.neval_grad,
        nlp.counters.neval_hprod, iter, status, Stime) 
        
@test status == :UserLimit

reset!(nlp)
s = TStoppingB(nlp,rtol=0.0, max_eval = 500000, max_iter = 100000, max_time = 1.0)
(x, f, gNorm, iter, optimal, tired, status, Stime) = steepest(nlp, s=s, verbose=false)
@printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
        nlp.meta.name, nlp.meta.nvar, f, gNorm,
        nlp.counters.neval_obj, nlp.counters.neval_grad,
        nlp.counters.neval_hprod, iter, status, Stime) 
        
@test status == :Stalled

reset!(nlp)
s = TStoppingB(nlp)
(x, f, gNorm, iter, optimal, tired, status, Stime) = steepest(nlp, s=s, verbose=false, bk_max=2)
@printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
        nlp.meta.name, nlp.meta.nvar, f, gNorm,
        nlp.counters.neval_obj, nlp.counters.neval_grad,
        nlp.counters.neval_hprod, iter, status, Stime) 


@test status == :StalledLinesearch

reset!(nlp)
s = TStoppingB(nlp, optimality_residual = optim_check_U_feasible)
(x, f, gNorm, iter, optimal, tired, status, Stime) = steepest(nlp, s=s, verbose=false)
@printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
        nlp.meta.name, nlp.meta.nvar, f, gNorm,
        nlp.counters.neval_obj, nlp.counters.neval_grad,
        nlp.counters.neval_hprod, iter, status, Stime)
        
@test status == :Unfeasible




# bounded
include("lbfgsB.jl")

reset!(nlp)
sB = TStoppingB(nlp, atol=1.0e-6, rtol = 0.0, max_eval = 10000, max_iter = 300, max_time = 4.0)
(x, f, gNorm, iter, optimal, tired, status, Stime) = lbfgsB(nlp, stp=sB, verbose=false)
@printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
        nlp.meta.name, nlp.meta.nvar, f, gNorm,
        nlp.counters.neval_obj, nlp.counters.neval_grad,
        nlp.counters.neval_hprod, iter, status, Stime) 
        
@test status == :Stalled
reset!(nlp)
sB = TStoppingB(nlp, atol=1.0e-6, rtol = 0.0, max_eval = 10000, max_iter = 300, max_time = 4.0,optimality_residual = optim_check_bounded2)
(x, f, gNorm, iter, optimal, tired, status, Stime) = lbfgsB(nlp, stp=sB, verbose=false)
@printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
        nlp.meta.name, nlp.meta.nvar, f, gNorm,
        nlp.counters.neval_obj, nlp.counters.neval_grad,
        nlp.counters.neval_hprod, iter, status, Stime) 


@test status == :Optimal

