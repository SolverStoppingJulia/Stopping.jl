## Use a buffer

We already illustrated the use of Stopping for optimization algorithm,
however, in the case where one algorithm/solver is not Stopping-compatible,
a buffer solver is required to unify the formalism.
We illustrate this situation here with the Ipopt solver.

Remark in the buffer function: in case the solver stops with success
but the stopping condition is not satisfied, one option is to iterate
and reduce the various tolerances.

Documentation for Ipopt options can be found here:
https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_REF

The Julia file corresponding to this tutorial can be found [here](https://github.com/Goysa2/Stopping.jl/tree/master/test/examples/buffer.jl).

```
using Ipopt, NLPModels, NLPModelsIpopt, Stopping

include("../test-stopping/rosenbrock.jl")
x0  = 1.5 * ones(6)
nlp = ADNLPModel(rosenbrock,  x0)
```

The traditional way to solve an optimization problem using NLPModelsIpopt
https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl
```
printstyled("Oth scenario:\n")

stats = ipopt(nlp, print_level = 0, x0 = x0)
```
Use y0 (general), zL (lower bound), zU (upper bound)
for initial guess of Lagrange multipliers.
```
@show stats.solution, stats.status
```

Using Stopping, the idea is to create a buffer function
```
function solveIpopt(stp :: NLPStopping)

 #xk = solveIpopt(stop.pb, stop.current_state.x)
 stats = ipopt(nlp, print_level     = 0,
                    tol             = stp.meta.rtol,
                    x0              = stp.current_state.x,
                    max_iter        = stp.meta.max_iter,
                    max_cpu_time    = stp.meta.max_time,
                    dual_inf_tol    = stp.meta.atol,
                    constr_viol_tol = stp.meta.atol,
                    compl_inf_tol   = stp.meta.atol)

 #Update the meta boolean with the output message
 if stats.status == :first_order stp.meta.suboptimal      = true end
 if stats.status == :acceptable  stp.meta.suboptimal      = true end
 if stats.status == :infeasible  stp.meta.infeasible      = true end
 if stats.status == :small_step  stp.meta.stalled         = true end
 if stats.status == :max_iter    stp.meta.iteration_limit = true end
 if stats.status == :max_time    stp.meta.tired           = true end

 stp.meta.nb_of_stop = stats.iter
 #stats.elapsed_time

 x = stats.solution

 #Not mandatory, but in case some entries of the State are used to stop
 fill_in!(stp, x)

 stop!(stp)

 return stp
end
```
```
nlp_at_x = NLPAtX(x0)
stop = NLPStopping(nlp, unconstrained_check, nlp_at_x)
```

### 1st scenario, we solve again the problem with the buffer solver
```
printstyled("1st scenario:\n")
solveIpopt(stop)
@show stop.current_state.x, status(stop)
nbiter = stop.meta.nb_of_stop
```

### 2nd scenario: we check that we control the maximum iterations.
```
printstyled("2nd scenario:\n")
#rstate is set as true to allow reinit! modifying the State
reinit!(stop, rstate = true, x = x0)
stop.meta.max_iter = max(nbiter-4,1)

solveIpopt(stop)
#Final status is :IterationLimit
@show stop.current_state.x, status(stop)
```
