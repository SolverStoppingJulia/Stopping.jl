###############################################################################
#
# The Stopping structure eases the implementation of algorithms and the
# stopping criterion.
#
# The following examples illustrate a specialized Stopping for the
# LinearOperators type.
# https://github.com/JuliaSmoothOptimizers/LinearOperators.jl
#
using LinearOperators #v.1.0.1
using LinearAlgebra, SparseArrays, Main.Stopping, Test
#
# This tutorial illustrates the different step in preparing the resolution of a
# new problem.
# - This Stopping handles linear system
#  Ax = b with A an m x n matrix.
#  or linear Least Square problem
#  min_{x ∈ ℜⁿ} || Ax - b ||_p^2
# - evaluations are measured by the LinearOperator.
# - we will use the GenericState (a specialized State would keep track of evals)
#
###############################################################################
mutable struct LinearSystem
    A :: Union{AbstractLinearOperator, AbstractMatrix}
    b :: AbstractVector
end

function linear_system_check(pb    :: LinearSystem,
                              state :: AbstractState;
                              pnorm :: Float64 = Inf,
                              kwargs...)
 return norm(pb.A * state.x - pb.b, pnorm)
end

"""
Type: LinearOperatorStopping (specialization of GenericStopping)
Methods: start!, stop!, update_and_start!, update_and_stop!, fill_in!, reinit!, status

Stopping structure for non-linear programming problems using NLPModels.
    Input :
       - pb         : a LinearSystem
       - state      : The information relative to the problem, see GenericState
       - (opt) meta : Metadata relative to stopping criterion.
       - (opt) main_stp : Stopping of the main loop in case we consider a Stopping
                          of a subproblem.
                          If not a subproblem, then nothing.

 Note:
 * Not specific State targeted
 * State don't necessarily keep track of evals
 * Evals are checked only for pb.A being a LinearOperator

 """
mutable struct LinearOperatorStopping <: AbstractStopping

    # problem
    pb :: LinearSystem
    # Common parameters
    meta      :: AbstractStoppingMeta
    # current state of the problem
    current_state :: AbstractState
    # Stopping of the main problem, or nothing
    main_stp :: Union{AbstractStopping, Nothing}

    function LinearOperatorStopping(pb             :: LinearSystem,
                                    current_state  :: AbstractState;
                                    meta           :: AbstractStoppingMeta = StoppingMeta(max_cntrs = _init_max_counters_linear_operators()),
                                    main_stp       :: Union{AbstractStopping, Nothing} = nothing,
                                    kwargs...)

        if !(isempty(kwargs))
           meta = StoppingMeta(;max_cntrs = _init_max_counters_linear_operators(), optimality_check = linear_system_check, kwargs...)
        end

        return new(pb, meta, max_cntrs, current_state, main_stp)
    end
end

"""
_init_max_counters_linear_operators():
"""
function _init_max_counters_linear_operators(;nprod   :: Int64 = 20000,
                                              ntprod  :: Int64 = 20000,
                                              nctprod :: Int64 = 20000,
                                              sum     :: Int64 = 20000*11)

  cntrs = Dict([(:nprod,   nprod),   (:ntprod, ntprod),
                (:nctprod, nctprod), (:neval_sum,    sum)])

 return cntrs
end

import Main.Stopping: _resources_check!, start!, stop!, update_and_start!, update_and_stop!, fill_in!, reinit!, status

"""
_resources_check!: check if the optimization algorithm has exhausted the resources.
                   This is the LinearOperator specialized version that takes
                   into account the evaluation of the functions.

Note:
* function does _not_ keep track of the evals in the State
* check :nprod, :ntprod, :nctprod in the LinearOperator entries
"""
function _resources_check!(stp    :: LinearOperatorStopping,
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

##############################################################
#TEST

n = 5
A = rand(n,n)
xref = 100 * rand(n)
x0 = zeros(n)
b    = A * xref
sA  = sparse(A)
opA = LinearOperator(A)

mLO = LinearSystem(A, b)
sLO = LinearSystem(sA, b)
opLO = LinearSystem(opA, b)
mLOstp = LinearOperatorStopping(mLO, GenericState(x0))
sLOstp = LinearOperatorStopping(sLO, GenericState(x0))
maxcn = _init_max_counters_linear_operators(nprod = 1)
opLOstp = LinearOperatorStopping(opLO, GenericState(x0), max_cntrs = maxcn)
@test start!(mLOstp) == false
@test start!(sLOstp) == false
@test start!(opLOstp) == false

update_and_stop!(mLOstp, x = xref)
@test status(mLOstp) == :Optimal
@test mLOstp.meta.resources == false
update_and_stop!(sLOstp, x = xref)
@test status(sLOstp) == :Optimal
@test sLOstp.meta.resources == false
update_and_stop!(opLOstp, x = xref)
@test status(opLOstp) == :Optimal
@test opLOstp.meta.resources == true

"""
Randomized block Kaczmarz
"""
function RandomizedBlockKaczmarz(stp :: AbstractStopping; kwargs...)

    #A,b = stp.current_state.Jx, stp.current_state.cx
    A,b = stp.pb.A, stp.pb.b
    x0  = stp.current_state.x

    m,n = size(A)
    xk  = x0

    OK = start!(stp)

    while !OK

     i  = Int(floor(rand() * m)+1) #rand a number between 1 and m
     Ai = A[i,:]
     xk  = Ai == 0 ? x0 : x0 - (dot(Ai,x0)-b[i])/dot(Ai,Ai) * Ai
     #xk  = Ai == 0 ? x0 : x0 - (Ai' * x0-b[i])/(Ai' * Ai) * Ai

     OK = update_and_stop!(stp, x = xk)
     x0  = xk

    end

 return stp
end

m, n = 400, 200 #size of A: m x n
A    = 100 * rand(m, n)
xref = 100 * rand(n)
b    = A * xref

#Our initial guess
x0 = zeros(n)

la_stop = LinearOperatorStopping(LinearSystem(A, b), GenericState(x0), max_iter = 150000, rtol = 1e-6)
#Be careful using GenericState(x0) would not work here without forcing convert = true
#in the update function. As the iterate will be an SparseVector to the contrary of initial guess.
#Tangi: maybe start! should send a Warning for such problem !?
sa_stop = LinearOperatorStopping(LinearSystem(sparse(A), b), GenericState(sparse(x0)), max_iter = 150000, rtol = 1e-6)
op_stop = LinearOperatorStopping(LinearSystem(LinearOperator(A), b), GenericState(x0), max_iter = 150000, rtol = 1e-6)

@time RandomizedBlockKaczmarz(la_stop)
@test status(la_stop) == :Optimal
@time RandomizedBlockKaczmarz(sa_stop)
@test status(sa_stop) == :Optimal
#No compatible dot product with LinearOperators.
#@time RandomizedBlockKaczmarz(op_stop)
#@test status(op_stop) == :Optimal
