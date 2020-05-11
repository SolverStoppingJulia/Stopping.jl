# State
## Types
```@docs
Stopping.GenericState
Stopping.NLPAtX
Stopping.LSAtT
```

## General Functions
```@docs
Stopping.update!
Stopping.reinit!
```

# Stopping
## Types
```@docs
Stopping.GenericStopping
Stopping.NLPStopping
Stopping.LS_Stopping
Stopping.StoppingMeta
```

## General Functions
```@docs
Stopping.start!
Stopping.update_and_start!
Stopping.stop!
Stopping.update_and_stop!
Stopping.reinit!
Stopping.fill_in!
Stopping.status
```

## Non linear admissibility functions
```@docs
Stopping.KKT
Stopping.unconstrained_check
Stopping.unconstrained2nd_check
Stopping.optim_check_bounded
```

## Line search admissibility functions
```@docs
Stopping.armijo
Stopping.wolfe
Stopping.armijo_wolfe
Stopping.shamanskii_stop
Stopping.goldstein
```
