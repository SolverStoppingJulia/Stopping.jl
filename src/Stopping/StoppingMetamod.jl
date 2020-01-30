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
	- The different status fail_sub_pb, unbounded, tired, stalled, optimal,
	  suboptimal, and infeasible are modified according to the data of the algorithm.
"""
mutable struct StoppingMeta <: AbstractStoppingMeta

 # problem tolerances
 atol                :: Number # absolute tolerance
 rtol                :: Number # relative tolerance
 optimality0         :: Number # value of the optimality residual at starting point
 tol_check           :: Function #function of atol, rtol and optimality0
                                 #by default: tol_check = max(atol, rtol * optimality0)
                                 #other example: atol + rtol * optimality0

 unbounded_threshold :: Number # beyond this value, the problem is declared unbounded
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
 fail_sub_pb         :: Bool
 unbounded           :: Bool
 unbounded_pb        :: Bool
 tired               :: Bool
 stalled             :: Bool
 iteration_limit     :: Bool
 resources           :: Bool
 optimal             :: Bool
 infeasible          :: Bool
 main_pb             :: Bool
 domainerror         :: Bool
 suboptimal          :: Bool

 function StoppingMeta(;atol                :: Number   = 1.0e-6,
                        rtol                :: Number   = 1.0e-15,
                        optimality0         :: Number   = 1.0,
                        tol_check           :: Function = (atol,rtol,opt0) -> max(atol,rtol*opt0),
                        unbounded_threshold :: Number   = -1.0e50,
                        unbounded_x         :: Number   = 1.0e50,
                        max_f               :: Int      = typemax(Int),
                        max_eval            :: Int      = 20000,
                        max_iter            :: Int      = 5000,
                        max_time            :: Number   = 300.0,
                        start_time          :: Float64  = NaN,
                        kwargs...)

   fail_sub_pb = false

   try
       tol_check(1,1,1)
   catch
       throw("tol_check must have 3 arguments")
   end

   unbounded       = false
   unbounded_pb    = false
   tired           = false
   stalled         = false
   iteration_limit = false
   resources       = false
   optimal         = false
   infeasible      = false
   main_pb         = false
   domainerror     = false
   suboptimal      = false

   nb_of_stop = 0

   return new(atol, rtol, optimality0, tol_check, unbounded_threshold, unbounded_x,
              max_f, max_eval, max_iter, max_time, nb_of_stop, start_time,
              fail_sub_pb, unbounded, unbounded_pb, tired, stalled,
              iteration_limit, resources, optimal, infeasible, main_pb,
              domainerror, suboptimal)
 end
end
