###############################################################################
#
# The Stopping structure eases the implementation of algorithms and the
# stopping criterion.
#
# The following examples illustrate solver for optimization:
# - a backtracking 1D optimization solver
# - a globalized Newton for unconstrained optimization solver
# - a bound constraint active-set algorithm
# - a quadratic penalty algorithm for non-linear optimization
#
###############################################################################

using LinearAlgebra, NLPModels, Stopping, Test

printstyled("How to solve 1D optim problem: \n", color = :green)
include("backls.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("How to solve unconstrained optim problem: \n", color = :green)
include("uncons.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("How to solve bound constrained optim problem: \n", color = :green)
include("activeset.jl")
printstyled("passed ✓ \n", color = :green)
printstyled("How to solve nonlinear optim problem: \n", color = :green)
include("penalty.jl")
printstyled("passed ✓ \n", color = :green)
