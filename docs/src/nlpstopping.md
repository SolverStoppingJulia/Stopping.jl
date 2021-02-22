## NLPStopping: A Stopping for NLPModels

The Stopping-structure can be adapted to any problem solved by iterative methods. We discuss here `NLPStopping` a specialization of an `AbstractStopping` for problems of type `NLPModels`. We highlight here the specifities of such instance:
- The problem is an `NLPModel`
- The problem has a funcion-evaluation counter, so we setup a maximum-counters structure in the meta.
- The State is an `NLPAtX` with entries corresponding to usual information for nonlinear optimization models.
- The unboundedness check verifies that the objective function is unbounded below for minimization problems, and above for maximization;
- The problem is declared infeasibility if the score is `Inf` for minimization problems, and `-Inf` for maximization.

```julia
nlp = ADNLPModel(x->sum(x.^2), zeros(5))
nlp_at_x = NLPAtX(zeros(5))
meta  = StoppingMeta(max_cntrs = _init_max_counters())
stp   = NLPStopping(pb, meta, state)
```

By default for `NLPStopping` the optimality function is a function checking the `KKT` conditions using information in the State.
The function `fill_in!` computes all the missing entries in the State. This is an potentially expensive operation, but might be useful.
