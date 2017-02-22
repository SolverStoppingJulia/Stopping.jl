# Stopping

Tools to ease the uniformization of stopping criteria in iterative solvers.

When a solver is called on an optimization model, four outcome may happen:
1. the approximate solution is obtained, the problem is considered solved
- the problem is declared unbounded
- the maximum available ressources is not sufficient to compute the solution
- some algorithm dependent failure happens

This tool eases the first 3 items above. It defines a type

    type TStopping
        nlp :: AbstractNLPModel         # the model
        atol :: Float64            
        rtol :: Float64
        unbounded_threshold :: Float64
        max_obj_f :: Int                # fine grain limits
        max_obj_grad :: Int    
        max_obj_hess :: Int
        max_obj_hv :: Int
        max_eval :: Int                 # global limits
                                        # default: 10000
        max_iter :: Int                 # default: 5000
        max_time :: Float64             # default: 600 i.e. 10 minutes
        start_time :: Float64
        optimality :: Float64           # tolerances at start
        optimality_residual :: Function #

which provides default tolerances and maximum ressources. The function `optimality_residual` computes the residual of the optimality conditions. For now, only unconstrained problems are considered, and this functions defaults to the infinity norm of the gradient of the objective function.

Fine grained limits may be specified, namely the maximum number of functions, gradients, hessians and hessian-vector products. Most users will be satisfied with the total, specified by `max_eval`.

The tool provides two functions:
- `start!(s,x0)` initializes the time counter and the tolerance at the starting point. This function is called once at the beginning of an algorithm.
- `stop(s, iter, x, gradf)` verifies if the tolerance is reached for `x` or if the maximum ressources is reached. This function returns booleans optimal, unbounded, tired; moreover, it returns the elapsed time, and fine grain information. Usually, only the four first outputs are used. This function is called at every iteration and, complemented with algorithm specific conditions, is the stopping criterion.

As an example, a naïve version of the steepest descent is provided. Two additionnal conditions are tested within the steepest descent:
1. the direction is a descent direction; not very useful for the steepest descent direction, but the algorithmic pattern could be used to code (quasi) Newton methods and when a computed direction is not a descent, it may be advised to interrupt the algorithm.
2. the line search fails.


Future work will address constrained problems. Then, fine grained information will consists in the number of constraint, their gradient etc. evaluation. The optimality conditions will be based on KKT equations. Separate tolerances for optimality and feasibility will be developed.
