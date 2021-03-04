## How Stopping checks for optimality

The solver can let Stopping handles the optimality checks. We see here how it works and how to tune it in.

First, the function `stop!` computes a **score** using `optimality_check` function given in the `meta`. The keywords argument given in `stop!` are passed to this function.
```julia
#Compute the score if !src.optimality_check
score = stp.meta.optimality_check(stp.pb, stp.current_state; kwargs...))
```
The **score** is then stored in `stp.current_state.current_score`. If the **score** doesn't contain any NaN, Stopping proceeds and test whether it is within tolerances given as functions in `meta.tol_check` and `meta.tol_check_neg`.
```julia
#Compute the tolerances
check_pos, check_neg = tol_check(stp.meta)
#Test the score vs the tolerances
optimal = _inequality_check(optimality, check_pos, check_neg)
```
So, overall Stopping does:
```julia
check_pos = stp.meta.tol_check(stp.meta.atol, stp.meta.rtol, stp.meta.optimality0)
check_neg = stp.meta.tol_check_neg(stp.meta.atol, stp.meta.rtol, stp.meta.optimality0)
score     = stp.meta.optimality_check(stp.pb, stp.current_state)
check_pos ≤ score ≤ check_neg
```

### FAQ: Does it work for vector scores as well?

The type of the score and tolerances are respectively initialized in the State and the Meta at the initialization of the Stopping. Hence one can use vectorized scores as long as they can be compared with the tolerances. For instance:
- The score is a vector and tolerances are vectors of the same length or numbers.
- The score is a tuple and tolerances are tuple or a number.

### FAQ: How do I implement AND and OR conditions?
The concatenation of two scores (AND condition) that need to be tested to zero can be represented as a vector.
The disjunction of two score (OR condition) are represented as tuple.

### FAQ: Do Stopping really computes the tolerances each time?

It does unless `meta.recomp_tol` is set as `true`. This entry can be set as true from the beginning as the `tol_check` functions are evaluated once at the initialization of the `meta`.
