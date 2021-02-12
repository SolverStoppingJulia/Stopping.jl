# Stopping.jl

This package provides general tools for the uniformization of stopping criteria for iterative solvers.
When calling an iterative solver, four outcomes may happen:

1. An approximate solution is obtained;
2. The problem is declared unsolvable (unboundedness, infeasibility, etc);
3. The maximum available resources are not sufficient to compute the solution;
4. An algorithm's dependent failure happens.

There are many advantages in using Stopping:
- Make your code more readable by outsourcing some tests to Stopping;
- Let the user a hand on the stopping criteria;
- Encourage reusability of codes.

Stopping.jl offers several advanced facilities, but already a primary use is beneficial for your code.

## How to install

Install and test the Stopping package with the Julia package manager:
```julia
pkg> add Stopping
pkg> test Stopping
```
You can access the most up-to-date version of the Stopping package using:
```julia
pkg> add https://github.com/vepiteski/Stopping.jl
pkg> test Stopping
```

Stopping.jl most evolved facilities are based on [JuliaSmoothOptimizers' tools](juliasmoothoptimizers.github.io/).

## Stopping.jl in application

Stopping.jl is already used in Julia's codes:
- [StoppingInterface.jl](https://github.com/tmigot/StoppingInterface.jl) an interface between Stopping.jl and the outside world;
- [MPCCSolver.jl](https://github.com/tmigot/MPCCSolver.jl) to solve mathematical programs with complementarity constraints;
- [FletcherPenaltyNLPSolver](https://github.com/tmigot/FletcherPenaltyNLPSolver) solve nonlinear programs with Fletcher's penalty method;
- ...
