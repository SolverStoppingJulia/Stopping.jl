using Stopping
using Base.Test

# write your own tests here
using NLPModels
using JuMP

include("line_model.jl")  # For LineFunction  definition
include("woods.jl")
nlp = MathProgNLPModel(woods(), name="woods")

include("armijo_wolfe.jl")
include("steepest.jl")

@printf("Problem  Dim  Optim f  Grad norm  f evals g evals Hv evals  iters  outcome         time\n")


s = TStopping(max_iter = 1) # to trigger compilation of steepest
(x, f, gNorm, iter, optimal, tired, status, Stime) = steepest(nlp, stp=s, verbose=false)
@printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
        nlp.meta.name, nlp.meta.nvar, f, gNorm,
        nlp.counters.neval_obj, nlp.counters.neval_grad,
        nlp.counters.neval_hprod, iter, status, Stime) 
        
@test status == :UserLimit

reset!(nlp)
s = TStopping(rtol=0.0, max_eval = 500000, max_iter = 100000, max_time = 1.0)
(x, f, gNorm, iter, optimal, tired, status, Stime) = steepest(nlp, stp=s, verbose=false)
@printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
        nlp.meta.name, nlp.meta.nvar, f, gNorm,
        nlp.counters.neval_obj, nlp.counters.neval_grad,
        nlp.counters.neval_hprod, iter, status, Stime) 
        
@test status == :UserLimit


reset!(nlp)
s = TStopping(rtol=0.0, max_eval = 500000, max_iter = 100000, max_time = 2.0)
(x, f, gNorm, iter, optimal, tired, status, Stime) = steepest(nlp, stp=s, verbose=false)
@printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
        nlp.meta.name, nlp.meta.nvar, f, gNorm,
        nlp.counters.neval_obj, nlp.counters.neval_grad,
        nlp.counters.neval_hprod, iter, status, Stime) 
        
@test status == :Optimal


reset!(nlp)
s = TStopping()
(x, f, gNorm, iter, optimal, tired, status, Stime) = steepest(nlp, stp=s, verbose=false, bk_max=2)
@printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
        nlp.meta.name, nlp.meta.nvar, f, gNorm,
        nlp.counters.neval_obj, nlp.counters.neval_grad,
        nlp.counters.neval_hprod, iter, status, Stime) 
        
@test status == :StalledLinesearch
