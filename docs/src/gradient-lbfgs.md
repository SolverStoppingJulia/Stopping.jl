## Mixed-algorithms: a ListofStates tutorial

We illustrate here the use of `ListofStates` in dealing with a warm start
procedure. The full code of this tutorial can be found [here](https://github.com/vepiteski/Stopping.jl/blob/master/test/examples/gradient-lbfgs.jl).

`ListofStates` is designed to store the of the iteration process.
In this tutorial, we compare the resolution of a convex unconstrained problem with 3 variants:
 - a steepest descent method
 - an inverse-BFGS method
 - a mix of 5 steps of steepest descent and then switching to a BFGS initialized with the 5 previous steps.

```
using Stopping, NLPModels, LinearAlgebra, Test, Printf
```

First, we introduce our two implementations that both uses an backtracking Armijo linesearch.

```
import Stopping.armijo
function armijo(xk, dk, fk, slope, f)
  t = 1.0
  fk_new = f(xk + dk)
  while f(xk + t * dk) > fk + 1.0e-4 * t * slope
    t /= 1.5
    fk_new = f(xk + t * dk)
  end
  return t, fk_new
end

function steepest_descent(stp :: NLPStopping)

  xk = stp.current_state.x
  fk, gk = objgrad(stp.pb, xk)

  OK = update_and_start!(stp, fx = fk, gx = gk)

  @printf "%2s %9s %7s %7s %7s\n" "k" "fk" "||∇f(x)||" "t" "λ"
  @printf "%2d %7.1e %7.1e\n" stp.meta.nb_of_stop fk norm(stp.current_state.current_score)
  while !OK
    dk = - gk
    slope = dot(dk, gk)
    t, fk = armijo(xk, dk, fk, slope, x->obj(stp.pb, x))
    xk += t * dk
    fk, gk = objgrad(stp.pb, xk)
    
    OK = update_and_stop!(stp, x = xk, fx = fk, gx = gk)

    @printf "%2d %9.2e %7.1e %7.1e %7.1e\n" stp.meta.nb_of_stop fk norm(stp.current_state.current_score) t slope
  end
  return stp
end

function bfgs_quasi_newton_armijo(stp :: NLPStopping; Hk = nothing)

  xk = stp.current_state.x
  fk, gk = objgrad(stp.pb, xk)
  gm = gk

  dk, t = similar(gk), 1.
  if isnothing(Hk)
    Hk = I #start from identity matrix
  end

  OK = update_and_start!(stp, fx = fk, gx = gk)

  @printf "%2s %7s %7s %7s %7s\n" "k" "fk" "||∇f(x)||" "t" "cos"
  @printf "%2d %7.1e %7.1e\n" stp.meta.nb_of_stop fk norm(stp.current_state.current_score)

  while !OK
    if stp.meta.nb_of_stop != 0
      sk = t * dk
      yk = gk - gm
      ρk = 1/dot(yk, sk)
      #we need yk'*sk > 0 for instance yk'*sk ≥ 1.0e-2 * sk' * Hk * sk
      Hk = ρk ≤ 0.0 ? Hk : (I - ρk * sk * yk') * Hk * (I - ρk * yk * sk') + ρk * sk * sk'
      if norm(sk) ≤ 1e-14
        break
      end
      #H2 = Hk + sk * sk' * (dot(sk,yk) + yk'*Hk*yk )*ρk^2 - ρk*(Hk * yk * sk' + sk * yk'*Hk)
    end
    dk = - Hk * gk
    slope = dot(dk, gk) # ≤ -1.0e-4 * norm(dk) * gnorm
    t, fk = armijo(xk, dk, fk, slope, x->obj(stp.pb, x))

    xk = xk + t * dk
    gm = copy(gk)
    gk = grad(stp.pb, xk)

    OK = update_and_stop!(stp, x = xk, fx = fk, gx = gk)
    @printf "%2d %7.1e %7.1e %7.1e %7.1e\n" stp.meta.nb_of_stop fk norm(stp.current_state.current_score) t slope
  end
  stp.stopping_user_struct = Dict( :Hk => Hk)
  return stp
end
```

We consider the following convex unconstrained problem model using `ADNLPModels.jl` and defines a related `NLPStopping`.

```
fH(x) = (x[2]+x[1].^2-11).^2+(x[1]+x[2].^2-7).^2
nlp = ADNLPModel(fH, [10., 20.])
stp = NLPStopping(nlp, optimality_check = unconstrained_check, 
                 atol = 1e-6, rtol = 0.0, max_iter = 100)
```

```
reinit!(stp, rstate = true, x = nlp.meta.x0)
steepest_descent(stp)

@test status(stp) == :Optimal
@test stp.listofstates == VoidListofStates()

@show elapsed_time(stp)
@show nlp.counters

reinit!(stp, rstate = true, x = nlp.meta.x0, rcounters = true)
bfgs_quasi_newton_armijo(stp)

@test status(stp) == :Optimal
@test stp.listofstates == VoidListofStates()

@show elapsed_time(stp)
@show nlp.counters

NLPModels.reset!(nlp)
stp_warm = NLPStopping(nlp, optimality_check = unconstrained_check, 
                      atol = 1e-6, rtol = 0.0, max_iter = 5, 
                      n_listofstates = 5) #shortcut for list = ListofStates(5, Val{NLPAtX{Float64,Array{Float64,1},Array{Float64,2}}}()))
steepest_descent(stp_warm)
@test status(stp_warm) == :IterationLimit
@test length(stp_warm.listofstates) == 5

Hwarm = I
for i=2:5
  sk = stp_warm.listofstates.list[i][1].x - stp_warm.listofstates.list[i-1][1].x 
  yk = stp_warm.listofstates.list[i][1].gx - stp_warm.listofstates.list[i-1][1].gx 
  ρk = 1/dot(yk, sk)
  if ρk > 0.0
    global Hwarm = (I - ρk * sk * yk') * Hwarm * (I - ρk * yk * sk') + ρk * sk * sk'
  end
end

reinit!(stp_warm)
stp_warm.meta.max_iter = 100
bfgs_quasi_newton_armijo(stp_warm, Hk = Hwarm)
status(stp_warm)

@show elapsed_time(stp_warm)
@show nlp.counters
```