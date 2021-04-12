# Checkpointing

In this tutorial, we present the use of Stopping to checkpointing.

When using an optimizer for high-scale problems, the resolution process
might be extremly long. In order to analyze the progress of the algorithm or save ongoing results,
an idea is to introduce checkpointing, i.e. we save the output result in the file
every `n`-steps. Using `Stopping` this operation is now very simple.

```
using ADNLPModels, FileIO, JLD2, LinearAlgebra, NLPModels, Printf, Random, Stopping
```

In this tutorial, we will use the steepest descent method with a fixed stepsize.

```
function fixed_step_steepest_descent(stp :: NLPStopping; t = 1e-5)

  xk = stp.current_state.x

  OK = update_and_start!(stp, gx = grad(stp.pb, xk))

  @printf "%2s %7s\n" "k" "||∇f(x)||"
  @printf "%2d %7.1e\n" stp.meta.nb_of_stop norm(stp.current_state.current_score)
  while !OK
    xk -= t * stp.current_state.gx
    
    OK = update_and_stop!(stp, x = xk, gx = grad(stp.pb, xk))

    @printf "%2d %7.1e\n" stp.meta.nb_of_stop norm(stp.current_state.current_score)
  end
  return stp
end
```

We now generate a regularized least squares problem using `ADNLPModels.jl`.

```
Random.seed!(1234)
m, n = 10_000, 10
A  = rand(m, n)
b  = A * ones(n)
f(x, A, b, λ) = norm(A * x - b)^2 + λ * norm(x)^2
pb = ADNLPModel(x -> f(x, A, b, 1e-2), zeros(n))
```

The final step is now to initialize the `Stopping` and specify user-defined structures
to store the parameter `n_save` set to 50 so that every 50 iterations the current
stopping is saved using the package `JLD2.jl`.

```
save_check(stp, b) = begin 
  if stp.meta.nb_of_stop % stp.stopping_user_struct[:n_save] == 0
    @save "checkpoint_stopping_$(stp.meta.nb_of_stop).jld2" stp
  end
end
n_save = 50
stp = NLPStopping(pb, user_struct = Dict(:n_save => n_save), 
                      user_check_func! = save_check, max_iter = 99)
# Let's go
fixed_step_steepest_descent(stp)
```

The algorithm has now generated two files `checkpoint_stopping_0.jld2` and `checkpoint_stopping_50.jld2` that
can be analyzed.

```
stp0 = load("checkpoint_stopping_0.jld2")["stp"]
stp50 = load("checkpoint_stopping_50.jld2")["stp"]
```
