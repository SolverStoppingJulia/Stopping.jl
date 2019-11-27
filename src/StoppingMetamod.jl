export AbstractStoppingMeta, StoppingMeta

################################################################################
# Common stopping parameters to all optimization algorithms
################################################################################

"""
AbstractStoppingMeta
Abstract type, if specialized meta for stopping were to be implemented they
would need to be subtypes of AbstractStoppingMeta
"""
abstract type AbstractStoppingMeta end

"""
StoppingMeta
Common stopping criterion for "all" optimization algorithms such as:
    - absolute and relative tolerance
    - threshold for unboundedness
    - time limit to let the algorithm run
    - maximum number of function (and derivatives) evaluations

It's a mutable struct therefore we can modified elements of a StoppingMeta.
	- The nb_of_stop is incremented everytime stop! or update_and_stop! is called
	- The optimality0 is modified once at the beginning of the algorithm (start!)
    - The start_time is modified once at the beginning of the algorithm (start!)
    if not precised before.
	- The different status optimal_sub_pb, unbounded, tired, stalled, optimal and
	  feasible are modified according to the data of the algorithm.
"""
mutable struct StoppingMeta <: AbstractStoppingMeta

 # problem tolerances
 atol                :: Number # absolute tolerance
 rtol                :: Number # relative tolerance
 optimality0         :: Number # value of the optimality residual at starting point

 unbounded_threshold :: Number # below this value, the problem is declared unbounded
 unbounded_x         :: Number # beyond this value, x is unbounded

 # fine grain control on ressources
 max_f               :: Int    # max function evaluations allowed

 # global control on ressources
 max_eval            :: Int    # max evaluations (f+g+H+Hv) allowed
 max_iter            :: Int    # max iterations allowed
 max_time            :: Number # max elapsed time allowed

 #intern Counters
 nb_of_stop :: Int
 #intern start_time
 start_time :: Float64

 # stopping properties status of the problem)
 optimal_sub_pb      :: Bool
 unbounded           :: Bool
 tired               :: Bool
 stalled             :: Bool
 resources 	         :: Bool
 optimal             :: Bool
 feasible            :: Bool
 main_pb            :: Bool

 function StoppingMeta(;atol    :: Number   = 1.0e-6,
			rtol                :: Number   = 1.0e-15,
			optimality0         :: Number   = 1.0,
			unbounded_threshold :: Number   = -1.0e50,
			unbounded_x         :: Number   = 1.0e50,
			max_f               :: Int      = typemax(Int),
			max_eval            :: Int      = 20000,
			max_iter            :: Int      = 5000,
			max_time            :: Number   = 300.0,
            start_time          :: Float64  = NaN,
			kwargs...)

   optimal_sub_pb = false

   unbounded = false
   tired     = false
   stalled   = false
   resources = false
   optimal   = false
   main_pb   = false

   nb_of_stop = 0

   return new(atol, rtol, optimality0, unbounded_threshold, unbounded_x,
              max_f, max_eval, max_iter, max_time, nb_of_stop, start_time,
              optimal_sub_pb, unbounded, tired, stalled, resources, optimal,
              main_pb)
 end
end
