# Stopping

Tools to ease the uniformization of stopping criteria in iterative solvers.

When a solver is called on an optimization model, six outcome may happen:

1. the approximate solution is obtained, the problem is considered solved
2. the problem is declared unbounded
3. the maximum available ressources is not sufficient to compute the solution
4. numerical innacuracies make the algorithm stall
5. a feasible point algorithm generates an unfeasible iterate.
6. some algorithm dependent failure happens

This tool eases the first five items above. It defines a type

     type TStoppingB <: AbstractStopping
     	 atol :: Float64                  # absolute tolerance
	      rtol :: Float64                  # relative tolerance
	      unbounded_threshold :: Float64   # below this value, the problem is declared unbounded
    	  stalled_x_threshold :: Float64
    	  stalled_f_threshold :: Float64
    	  # fine grain control on ressources
    	  max_counters :: NLPModels.Counters
    	  # global control on ressources
    	  max_eval :: Int                  # max evaluations (f+g+H+Hv) allowed
    	  max_iter :: Int                  # max iterations allowed
    	  max_time :: Float64              # max elapsed time allowed
    	  # global information to the stopping manager
    	  iter :: Int
    	  start_time :: Float64            # starting time of the execution of the method
    	  optimality0 :: Float64           # value of the optimality residual at starting point
    	  optimality_residual :: Function  # function to compute the optimality residual
    	  # diagnostic
    	  elapsed_time :: Float64
    	  optimal :: Bool
    	  tired :: Bool
    	  unbounded :: Bool
    	  stalled :: Bool
    	  feasible :: Bool                 # Used for algos stopping when reaching feasibility
    	  unfeasible :: Bool               # Used for algos interrupted when loosing feasibility
    	  #
    	  nlp :: AbstractNLPModel
    	  #
    	  nlp_at_x :: TResult

which provides default tolerances and maximum ressources. The function `optimality_residual` computes the residual of the optimality conditions. For now, only unconstrained and bound constrained problems are considered, and this functions defaults to the infinity norm of the gradient of the objective function.

Fine grained limits may be specified in harmony with the NLPModels counters, namely the maximum number of functions, gradients, hessians and hessian-vector products. Most users will be satisfied with the total, specified by `max_eval`.

The tool provides two functions:
- `start!(s,x0)` initializes the time counter and the tolerance at the starting point. This function is called once at the beginning of an algorithm.
- `stop(s, iter, x, gradf)` verifies if the tolerance is reached for `x` or if the maximum ressources is reached. This function returns a boolean informing if the stopping criterion is satisfied. Information on the reached point is available within TStoppingB. This function is called at every iteration and, complemented with algorithm specific conditions, is the stopping criterion.

As an example, a naïve version of the steepest descent is provided. Two additionnal conditions are tested within the steepest descent:

1. the direction is a descent direction; not very useful for the steepest descent direction, but the algorithmic pattern could be used to code (quasi) Newton methods and when a computed direction is not a descent, it may be advised to interrupt the algorithm.
2. the line search fails.

Another example is a limited memory BFGS variant treating bounded problems. Three distinct "optimality_residual" functions are provided, two based on the norm of the projected gradient, the last computing the norm of the Lagrangian of the problem.

Finally, if an unconstrained algorithm is used on a bounded problem, an "optimality_residual" function détecting unfeasibility is also included.


Future work will address constrained problems. Then, fine grained information will consists in the number of constraint, their gradient etc. evaluation. The optimality conditions will be based on KKT equations. Separate tolerances for optimality and feasibility will be developed.
