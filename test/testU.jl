using Stopping
#include("../src/Stopping.jl")
using Base.Test

# write your own tests here
using NLPModels
using JuMP


#include("CUTEstProblemsB.jl")
#probs = sort(open(readlines,"CUTEstBound.list"))


include("line_model.jl")  # For LineFunction  definition
#include("woods.jl")
#nlp = MathProgNLPModel(woods(), name="woods")
include("genrose.jl")
nlp = MathProgNLPModel(genrose(800), name="genrose")

#nlp = CUTEstModel(probs[6])




# bounded
include("lbfgsB.jl")

#test_probs = (CUTEstModel(p)  for p in probs)

#for nlp in test_probs
#nlp = CUTEstModel("MINSURFO")
#nlp = CUTEstModel("ALLINIT")

    n = nlp.meta.nvar


#sB = TStoppingB(nlp, atol=1.0e-6, rtol = 0.0, max_eval = 100000, max_iter = 10000, max_time = 8.0,
#                optimality_residual =optim_check_U_feasible )

sB = TStoppingB(nlp, atol=1.0e-6, rtol = 0.0, max_eval = 100000, max_iter = 10000, max_time = 8.0,
                optimality_residual = optim_check_bounded2)

#sB = TStoppingB(nlp, atol=1.0e-6, rtol = 0.0, max_eval = 100000, max_iter = 10000, max_time = 8.0,
                optimality_residual = optim_check_unconstrained)


#sB = TStopping( atol=1.0e-6, rtol = 0.0, max_eval = 100000, max_iter = 10000, max_time = 8.0)

(x, f, gNorm, iter, optimal, tired, status, Stime) =  TSlbfgsB(nlp, stp=sB, verbose=false, mem = 5, scaling = true)
    @printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
            nlp.meta.name, nlp.meta.nvar, f, gNorm,
            nlp.counters.neval_obj, nlp.counters.neval_grad,
            nlp.counters.neval_hprod, iter, status, Stime) 
#println("norm pg = ", norm(sB.nlp_at_x.pg,Inf))

reset!(nlp)

include("steepestS.jl")

(x, f, gNorm, iter, optimal, tired, status, Stime) =  steepest(nlp, s=sB, verbose=false, mem = 5, scaling = true)
    
    @printf("%-5s  %3d  %9.2e  %7.1e  %5d  %5d  %6d   %6d  %-20s  %7.3e\n",
            nlp.meta.name, nlp.meta.nvar, f, gNorm,
            nlp.counters.neval_obj, nlp.counters.neval_grad,
            nlp.counters.neval_hprod, iter, status, Stime) 



    finalize(nlp)
#end
#using StatProfilerHTML
#statprofilehtml()

