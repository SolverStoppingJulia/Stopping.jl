"""
Type: StoppingMeta

Methods: no methods.

Attributes:
- atol : absolute tolerance.
- rtol : relative tolerance.
- optimality0 : optimality score at the initial guess.
- tol_check : Function of *atol*, *rtol* and *optimality0* testing a score to zero.
- tol_check_neg : Function of *atol*, *rtol* and *optimality0* testing a score to zero.
- check_pos : pre-allocation for positive tolerance
- check_neg : pre-allocation for negative tolerance
- retol : true if tolerances are updated
- optimality_check : a stopping criterion via an admissibility function
- unbounded_threshold : threshold for unboundedness of the problem.
- unbounded_x : threshold for unboundedness of the iterate.
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
 and returns a *Number* or an *AbstractVector* to be compared to *0*.
- *optimality_check* does not necessarily fill in the State.

Examples: `StoppingMeta()`
"""
mutable struct StoppingMeta{TolType <: Number, CheckType} <: AbstractStoppingMeta

 # problem tolerances
 atol                :: TolType # absolute tolerance
 rtol                :: TolType # relative tolerance
 optimality0         :: TolType # value of the optimality residual at starting point
 tol_check           :: Function #function of atol, rtol and optimality0
                                 #by default: tol_check = max(atol, rtol * optimality0)
                                 #other example: atol + rtol * optimality0
 tol_check_neg       :: Function # function of atol, rtol and optimality0
 check_pos           :: CheckType #pre-allocation for positive tolerance
 check_neg           :: CheckType #pre-allocation for negative tolerance
 optimality_check    :: Function # stopping criterion
                                 # Function of (pb, state; kwargs...)
                                 #return type  :: Union{Number, eltype(stp.meta)}
 retol               :: Bool #true if tolerances are updated

 unbounded_threshold :: Number # beyond this value, the problem is declared unbounded
 unbounded_x         :: Number # beyond this value, ||x||_\infty is unbounded

 # fine grain control on ressources
 max_f               :: Int    # max function evaluations allowed
 max_cntrs           :: Dict{Symbol,Int64} #contains the detailed max number of evaluations

 # global control on ressources
 max_eval            :: Int    # max evaluations (f+g+H+Hv) allowed
 max_iter            :: Int    # max iterations allowed
 max_time            :: Float64 # max elapsed time allowed

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

 function StoppingMeta(;atol               :: Number   = 1.0e-6,
                       rtol                :: Number   = 1.0e-15,
                       optimality0         :: Number   = 1.0,
                       tol_check           :: Function = (atol :: Number, rtol :: Number, opt0 :: Number) -> max(atol,rtol*opt0),
                       tol_check_neg       :: Function = (atol :: Number, rtol :: Number, opt0 :: Number) -> - tol_check(atol,rtol,opt0),
                       optimality_check    :: Function = (a,b) -> Inf,
                       retol               :: Bool     = true,
                       unbounded_threshold :: Number   = 1.0e50,
                       unbounded_x         :: Number   = 1.0e50,
                       max_f               :: Int      = typemax(Int),
                       max_cntrs           :: Dict{Symbol,Int} = Dict{Symbol,Int64}(),
                       max_eval            :: Int      = 20000,
                       max_iter            :: Int      = 5000,
                       max_time            :: Float64  = 300.0,
                       start_time          :: Float64  = NaN,
                       kwargs...)

   #throw("Error in StoppingMeta definition: tol_check and tol_check_neg must have 3 arguments")
   check_pos = tol_check(atol, rtol, optimality0)
   check_neg = tol_check_neg(atol, rtol, optimality0)

   if (true in (check_pos .< check_neg))
       @warn "Warning in StoppingMeta definition: tol_check should be greater than tol_check_neg."
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

   return new{typeof(atol), typeof(check_pos)}(atol, rtol, optimality0,
                 tol_check, tol_check_neg,
                 check_pos, check_neg, optimality_check, retol,
                 unbounded_threshold, unbounded_x,
                 max_f, max_cntrs, max_eval, max_iter, max_time, nb_of_stop, start_time,
                 fail_sub_pb, unbounded, unbounded_pb, tired, stalled,
                 iteration_limit, resources, optimal, infeasible, main_pb,
                 domainerror, suboptimal)
 end
end

function tol_check(meta :: StoppingMeta{TolType, CheckType}) where {TolType <: Number, CheckType}

 if meta.retol
   atol, rtol, opt0 = meta.atol, meta.rtol, meta.optimality0
   setfield!(meta, :check_pos, meta.tol_check(atol, rtol, opt0))
   setfield!(meta, :check_neg, meta.tol_check_neg(atol, rtol, opt0))
 end

 return (meta.check_pos, meta.check_neg)
end

function update_tol!(meta        :: StoppingMeta{TolType, CheckType};
                     atol        :: Union{TolType,Nothing} = nothing,
                     rtol        :: Union{TolType,Nothing} = nothing,
                     optimality0 :: Union{TolType,Nothing} = nothing) where {TolType <: Number, CheckType}
 meta.retol = true

 atol == nothing        && setfield!(meta, :atol, atol)
 rtol == nothing        && setfield!(meta, :rtol, rtol)
 optimality0 == nothing && setfield!(meta, :optimality0, optimality0)

 return meta
end
