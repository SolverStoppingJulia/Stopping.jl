## Benchmark

https://juliasmoothoptimizers.github.io/SolverBenchmark.jl/latest/tutorial/
In this tutorial we illustrate the main uses of SolverBenchmark.

The Julia file corresponding to this tutorial can be found [here](https://github.com/Goysa2/Stopping.jl/tree/master/test/examples/benchmark.jl).

```
using LinearAlgebra, NLPModels, Stopping

include("backls.jl")
include("uncons.jl")
```

```
using DataFrames, Printf, SolverBenchmark
```
CUTEst is a collection of test problems
```
using CUTEst
```

```
problems_unconstrained = CUTEst.select(contype="unc")
n = length(problems_unconstrained) #240
```

```
printstyled("Benchmark solvers: \n", color = :green)

#Names of 3 solvers:
names = [:armijo, :wolfe, :armijo_wolfe]
p1 = PrmUn(); p2 = PrmUn(ls_func = wolfe); p3 = PrmUn(ls_func = armijo_wolfe)
paramDict = Dict(:armijo => p1, :wolfe => p2, :armijo_wolfe => p3)
#Initialization of the DataFrame for n problems.
stats = Dict(name => DataFrame(:id => 1:n,
         :name => [@sprintf("prob%s", problems_unconstrained[i]) for i = 1:n],
         :nvar => zeros(Int64, n),
         :status => [:Unknown for i = 1:n],
         :f => NaN*ones(n),
         :t => NaN*ones(n),
         :iter => zeros(Int64, n),
         :eval_f => zeros(Int64, n),
         :eval_g => zeros(Int64, n),
         :eval_H => zeros(Int64, n),
         :score => NaN*ones(n)) for name in names)
```

```
for i=1:n
  nlp_cutest = CUTEst.CUTEstModel(problems_unconstrained[i])
  @show i, problems_unconstrained[i], nlp_cutest.meta.nvar
  #update the stopping with the new problem
  stop_nlp = NLPStopping(nlp_cutest,
                         unconstrained_check,
                         NLPAtX(nlp_cutest.meta.x0),
                         max_iter = 20)

  for name in names

    #solve the problem
    global_newton(stop_nlp, paramDict[name])

    #update the stats from the Stopping
    stats[name].nvar[i] = nlp_cutest.meta.nvar
    stats[name].status[i] = status(stop_nlp)
    stats[name].f[i] = stop_nlp.current_state.fx == nothing ? NaN : stop_nlp.current_state.fx
    stats[name].t[i] = stop_nlp.current_state.current_time == nothing ? NaN : stop_nlp.current_state.current_time - stop_nlp.meta.start_time
    stats[name].iter[i] = stop_nlp.meta.nb_of_stop
    stats[name].score[i] = unconstrained_check(nlp_cutest, stop_nlp.current_state)
    stats[name].eval_f[i] = getfield(stop_nlp.current_state.evals, :neval_obj)
    stats[name].eval_g[i] = getfield(stop_nlp.current_state.evals, :neval_grad)
    stats[name].eval_H[i] = getfield(stop_nlp.current_state.evals, :neval_hess)

    #reinitialize the Stopping and the nlp
    reinit!(stop_nlp, rstate = true, x = nlp_cutest.meta.x0)
    reset!(stop_nlp.pb)
  end

  #finalize nlp
  finalize(nlp_cutest)
end #end of main loop
```

```
for name in names
@show stats[name]
end
```
