## Quadratic penalty algorithm

In this test problem we consider a quadratic penalty method.
This example features an algorithm with the 3 steps:
the penalization - the unconstrained min - the 1d min

Note that there is no optimization of the evaluations here.
The penalization gives an approximation of the gradients, multipliers...

Note the use of a structure for the algorithmic parameters which is
forwarded to all the 3 steps. If a parameter is not mentioned, then the default
entry in the algorithm will be taken.

The Julia file corresponding to this tutorial can be found [here](https://github.com/Goysa2/Stopping.jl/tree/master/test/examples/penalty.jl).

```
include("uncons.jl")
```

### Quadratic penalty algorithm
fill_in! used instead of update! (works but usually more costly in evaluations)
subproblems are solved via Newton method

```
function penalty(stp :: NLPStopping; rho0 = 1.0, rho_min = 1e-10,
                                     rho_update = 0.5, prms = nothing)

 #algorithm's parameters
 rho = rho0

 #First call to the stopping
 #Becareful here, some entries must be filled in first.
 fill_in!(stp, stp.current_state.x)
 OK = start!(stp)

 #prepare the subproblem stopping:
 sub_nlp_at_x = NLPAtX(stp.current_state.x)
 sub_pb  = ADNLPModel(x -> obj(stp.pb, x)
                      + 1/rho * norm(max.(cons(stp.pb, x) - stp.pb.meta.ucon, 0.0))^2
                      + 1/rho * norm(max.(- cons(stp.pb, x) + stp.pb.meta.lcon, 0.0))^2,  x0)
 sub_stp = NLPStopping(sub_pb, unconstrained_check,
                               sub_nlp_at_x, main_stp = stp)

 #main loop
 while !OK

  #solve the subproblem
  reinit!(sub_stp)
  sub_stp.meta.atol = min(rho, sub_stp.meta.atol)
  global_newton(sub_stp, prms)

  #Update all the entries of the State
  fill_in!(stp, sub_stp.current_state.x)

  #Either stop! is true OR the penalty parameter is too small
  if rho < rho_min stp.meta.suboptimal = true end
  OK = stop!(stp)

  @show stp.meta.nb_of_stop, OK, rho

  #update the penalty parameter if necessary
  if !OK
   rho = rho * rho_update
   sub_stp.pb  = ADNLPModel(x -> obj(stp.pb, x)
                            + 1/rho * norm(max.(cons(stp.pb, x) - stp.pb.meta.ucon, 0.0))^2
                            + 1/rho * norm(max.(- cons(stp.pb, x) + stp.pb.meta.lcon, 0.0))^2,  x0)
  end
 end

 return stp
end
```


### Quadratic penalty algorithm: buffer function
```
function penalty(stp :: NLPStopping, prms)

 #extract required values in the prms file
 r0 = :rho0       ∈ fieldnames(typeof(prms)) ? prms.rho0       : 1.0
 rm = :rho_min    ∈ fieldnames(typeof(prms)) ? prms.rho_min    : 1e-10
 ru = :rho_update ∈ fieldnames(typeof(prms)) ? prms.rho_update : 0.5

 return penalty(stp, rho0 = r0, rho_min = rm, ru = 0.5, prms = prms)
end
```

### Algorithmic parameters structure
```
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

    function Param(;rho0        :: Float64 = 1.0,
                    rho_min     :: Float64 = sqrt(eps(Float64)),
                    rho_update  :: Float64 = 0.5,
                    armijo_prm  :: Float64 = 0.01,
                    wolfe_prm   :: Float64 = 0.99,
                    onedsolve   :: Function = backtracking_ls,
                    ls_func     :: Function = (x,y)-> armijo(x,y, τ₀ = armijo_prm),
                    back_update :: Float64 = 0.5)
        return new(rho0, rho_min, rho_update,
                   armijo_prm, wolfe_prm, onedsolve, ls_func,
                   back_update)
    end
end
```
