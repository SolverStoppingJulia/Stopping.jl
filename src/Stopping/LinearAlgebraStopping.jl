"""
Type: LAStopping

Methods: start!, stop!, update\\_and\\_start!, update\\_and\\_stop!, fill_in!, reinit!, status
linear\\_system\\_check, normal\\_equation\\_check

Specialization of GenericStopping. Stopping structure for linear algebra
solving either

A * x = b

or

min\\_{x} 0.5 * || A * x - b ||_2^2.

Attributes:
- pb         : a problem (with pb.A and pb.b available)
- state      : The information relative to the problem, see GenericState
- (opt) meta : Metadata relative to stopping criterion, see *StoppingMeta*.
- (opt) main_stp : Stopping of the main loop in case we consider a Stopping
                          of a subproblem.
                          If not a subproblem, then nothing.

`LAStopping(:: Any, :: AbstractState; meta :: AbstractStoppingMeta = StoppingMeta() main_stp :: Union{AbstractStopping, Nothing} = nothing, kwargs...)`

Note:
- Kwargs are forwarded to the classical constructor.
- Not specific State targeted
- State don't necessarily keep track of evals
- Evals are checked only for pb.A being a LinearOperator
- zero_start is true if 0 is the initial guess (not check automatically)

There is an additional constructor without problem structure:

`LAStopping(:: Union{AbstractLinearOperator, AbstractMatrix}, :: AbstractVector, kwargs...)`

See also GenericStopping, NLPStopping, LS_Stopping
 """
 mutable struct LAStopping <: AbstractStopping

     # problem
     pb :: Any
     # Common parameters
     meta      :: AbstractStoppingMeta
     # current state of the problem
     current_state :: AbstractState
     # Stopping of the main problem, or nothing
     main_stp :: Union{AbstractStopping, Nothing}

     #zero is initial point
     zero_start :: Bool

     function LAStopping(pb             :: Any,
                         current_state  :: AbstractState;
                         meta           :: AbstractStoppingMeta = StoppingMeta(max_cntrs = _init_max_counters_linear_operators(), optimality_check = linear_system_check),
                         main_stp       :: Union{AbstractStopping, Nothing} = nothing,
                         zero_start     :: Bool = false,
                         kwargs...)

         try
             pb.A, pb.b
         catch
            throw("pb must have A and b entries")
         end

         if !(isempty(kwargs))
            meta = StoppingMeta(;max_cntrs = _init_max_counters_linear_operators(), optimality_check = linear_system_check, kwargs...)
         end

         return new(pb, meta, current_state, main_stp, zero_start)
     end
 end

"""
Type: LinearSystem

Minimal structure to store linear algebra problems

`LinearSystem(:: Union{AbstractLinearOperator, AbstractMatrix}, :: AbstractVector)`
"""
mutable struct LinearSystem
  A :: Union{AbstractLinearOperator, AbstractMatrix}
  b :: AbstractVector
end

function LAStopping(A :: Union{AbstractLinearOperator, AbstractMatrix},
                    b :: AbstractVector;
                    kwargs...)
 return LAStopping(LinearSystem(A,b), GenericState(zeros(size(A,2))); kwargs...)
end

 """
 \\_init\\_max\\_counters\\_linear\\_operators(): counters for LinearOperator

 `_init_max_counters_linear_operators(;nprod :: Int64 = 20000, ntprod  :: Int64 = 20000, nctprod :: Int64 = 20000, sum :: Int64 = 20000*11)`
 """
 function _init_max_counters_linear_operators(;nprod   :: Int64 = 20000,
                                               ntprod  :: Int64 = 20000,
                                               nctprod :: Int64 = 20000,
                                               sum     :: Int64 = 20000*11)

   cntrs = Dict([(:nprod,   nprod),   (:ntprod, ntprod),
                 (:nctprod, nctprod), (:neval_sum,    sum)])

  return cntrs
 end

 """
 \\_resources\\_check!: check if the optimization algorithm has exhausted the resources.
                        This is the Linear Algebra specialized version.

 Note:
 * function does _not_ keep track of the evals in the State
 * check :nprod, :ntprod, :nctprod in the LinearOperator entries
 """
 function _resources_check!(stp    :: LAStopping,
                            x      :: Union{Vector, Nothing})

   max_cntrs = stp.meta.max_cntrs

   # check all the entries in the counter
   max_f = false
   sum   = 0
  if typeof(stp.pb.A) <: AbstractLinearOperator
   for f in [:nprod, :ntprod, :nctprod]
    max_f = max_f || (getfield(stp.pb.A, f) > max_cntrs[f])
    sum  += getfield(stp.pb.A, f)
   end
  end

  # Maximum number of function and derivative(s) computation
  max_evals = sum > max_cntrs[:neval_sum]

  # global user limit diagnostic
  stp.meta.resources = max_evals || max_f

  return stp
 end

"""
linear\\_system\\_check: return ||Ax-b||_p

`linear_system_check(pb :: Any state :: AbstractState; pnorm :: Float64 = Inf, kwargs...)`
"""
function linear_system_check(pb    :: Any,
                             state :: AbstractState;
                             pnorm :: Float64 = Inf,
                             kwargs...)
 return norm(pb.A * state.x - pb.b, pnorm)
end

"""
linear\\_system\\_check: return ||A'Ax-A'b||_p

`linear_system_check(pb :: Any state :: AbstractState; pnorm :: Float64 = Inf, kwargs...)`
"""
function normal_equation_check(pb    :: Any,
                               state :: AbstractState;
                               pnorm :: Float64 = Inf,
                               kwargs...)
 return norm(pb.A' * (pb.A * state.x) - pb.A' * pb.b, pnorm)
end
