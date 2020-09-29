"""
Type: StoppingMeta

Methods: no methods.

Attributes:
- atol : absolute tolerance.
- rtol : relative tolerance.
- optimality0 : optimality score at the initial guess.
- tol_check : Function of *atol*, *rtol* and *optimality0* testing a score to zero.
- optimality_check : a stopping criterion via an admissibility function
- unbounded_threshold : threshold for unboundedness of the problem.
- unbounded_x : threshold for unboundedness of the iterate.
- norm_unbounded_x : norm used for the threshold for unboundedness of the iterate.
- max_f :  maximum number of function (and derivatives) evaluations.
- max_cntrs  : Dict contains the maximum number of evaluations
- max_eval :  maximum number of function (and derivatives) evaluations.
- max_iter : threshold on the number of stop! call/number of iteration.
- max_time : time limit to let the algorithm run.
- nb\\_of\\_stop : keep track of the number of stop! call/iteration.
- start_time : keep track of the time at the beginning.
- fail\\_sub\\_pb : status.
- unbounded : status.
- unbounded_pb : status.
- tired : status.
- stalled : status.
- iteration_limit : status.
- resources : status.
- optimal : status.
- infeasible : status.
- main_pb : status.
- domainerror : status.
- suboptimal : status.

`StoppingMeta(;atol :: Number = 1.0e-6, rtol :: Number = 1.0e-15, optimality0 :: Number = 1.0, tol_check :: Function = (atol,rtol,opt0) -> max(atol,rtol*opt0), unbounded_threshold :: Number = 1.0e50, unbounded_x :: Number = 1.0e50, max_f :: Int = typemax(Int), max_eval :: Int = 20000, max_iter :: Int = 5000, max_time :: Number = 300.0, start_time :: Float64 = NaN, kwargs...)`

Note:
- It is a mutable struct, therefore we can modify elements of a *StoppingMeta*.
- The *nb\\_of\\_stop* is incremented everytime *stop!* or *update\\_and\\_stop!* is called
- The *optimality0* is modified once at the beginning of the algorithm (*start!*)
- The *start_time* is modified once at the beginning of the algorithm (*start!*)
      if not precised before.
- The different status: *fail\\_sub\\_pb*, *unbounded*, *unbounded_pb*, *tired*, *stalled*,
      *iteration_limit*, *resources*, *optimal*, *main_pb*, *domainerror*, *suboptimal*, *infeasible*
- *fail\\_sub\\_pb*, *suboptimal*, and *infeasible* are modified by the algorithm.
- *optimality_check* takes two inputs (*AbstractNLPModel*, *NLPAtX*)
 and returns a *Number* to be compared to *0*.
- *optimality_check* does not necessarily fill in the State.

Examples: `StoppingMeta()`
"""
mutable struct StoppingMeta <: AbstractStoppingMeta

 # problem tolerances
 atol                :: Number # absolute tolerance
 rtol                :: Number # relative tolerance
 optimality0         :: Number # value of the optimality residual at starting point
 tol_check           :: Function #function of atol, rtol and optimality0
                                 #by default: tol_check = max(atol, rtol * optimality0)
                                 #other example: atol + rtol * optimality0
 tol_check_neg       :: Function # function of atol, rtol and optimality0
 optimality_check    :: Function # stopping criterion
                                 # Function of (pb, state; kwargs...)

 unbounded_threshold :: Number # beyond this value, the problem is declared unbounded
 unbounded_x         :: Number # beyond this value, ||x|| is unbounded
 norm_unbounded_x    :: Number #norm used to check unboundedness of x.

 # fine grain control on ressources
 max_f               :: Int    # max function evaluations allowed
 max_cntrs           :: Dict #contains the detailed max number of evaluations

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
                        tol_check_neg       :: Function = (atol,rtol,opt0) -> - tol_check(atol,rtol,opt0),
                        optimality_check    :: Function = (a,b) -> Inf,
                        unbounded_threshold :: Number   = 1.0e50,
                        unbounded_x         :: Number   = 1.0e50,
                        norm_unbounded_x    :: Number   = Inf,
                        max_f               :: Int      = typemax(Int),
                        max_cntrs           :: Dict     = Dict(),
                        max_eval            :: Int      = 20000,
                        max_iter            :: Int      = 5000,
                        max_time            :: Number   = 300.0,
                        start_time          :: Float64  = NaN,
                        kwargs...)

   try
       tol_check(1.,1.,1.)
       tol_check_neg(1.,1.,1.)
   catch
       throw("tol_check and tol_check_neg must have 3 arguments")
   end

   fail_sub_pb     = false
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

   return new(atol, rtol, optimality0, tol_check, tol_check_neg, optimality_check,
              unbounded_threshold, unbounded_x, norm_unbounded_x,
              max_f, max_cntrs, max_eval, max_iter, max_time, nb_of_stop, start_time,
              fail_sub_pb, unbounded, unbounded_pb, tired, stalled,
              iteration_limit, resources, optimal, infeasible, main_pb,
              domainerror, suboptimal)
 end
end
