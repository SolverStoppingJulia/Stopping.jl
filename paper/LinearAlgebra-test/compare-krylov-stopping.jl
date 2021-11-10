using Krylov, Stopping

#https://github.com/JuliaSmoothOptimizers/Krylov.jl

include("random_coordinate_descent_method.jl")
include("stop_random_coordinate_descent_method.jl")
include("cheap_stop_random_coordinate_descent_method.jl")
include("instate_coordinate_descent_method.jl")

#using ProfileView, 
using BenchmarkTools

n = 5000;
A = rand(n, n)
sol = rand(n)
b = A * sol

x0 = zeros(size(A, 2))
pb = LinearSystem(A, b)
s0 = GenericState(x0, similar(b))
mcnt = Stopping._init_max_counters_linear_operators()

###############################################################################
#
# The original algorithm
#
###############################################################################
@time x, OK, k = RandomizedCD(A, b, max_iter = 100)

###############################################################################
#
# Stopping with memory optimization using the State and with cheap_stop!
#
###############################################################################
@time meta3 = StoppingMeta(
  max_cntrs = mcnt,
  atol = 1e-7,
  rtol = 1e-15,
  max_iter = 100,
  retol = false,
  tol_check = (atol, rtol, opt0) -> (atol + rtol * opt0),
  optimality_check = (pb, state) -> state.res,
)
s3 = GenericState(zeros(size(A, 2)), similar(b))
@time stp3 = LAStopping(pb, meta3, s3)
@time StopRandomizedCD2(stp3)

###############################################################################
#
# Stopping version of the algorithm
#
###############################################################################
@time meta2 = StoppingMeta(
  max_cntrs = mcnt,
  atol = 1e-7,
  rtol = 1e-15,
  max_iter = 100,
  retol = false,
  tol_check = (atol, rtol, opt0) -> (atol + rtol * opt0),
  optimality_check = (pb, state) -> state.res,
)
s2 = GenericState(zeros(size(A, 2)), similar(b))
@time stp2 = LAStopping(pb, meta2, s2)
@time StopRandomizedCD(stp2)

###############################################################################
#
# Stopping version of the algorithm with cheap remote control
#
###############################################################################
@time meta1 = StoppingMeta(
  max_cntrs = mcnt,
  atol = 1e-7,
  rtol = 1e-15,
  max_iter = 100,
  retol = false,
  tol_check = (atol, rtol, opt0) -> (atol + rtol * opt0),
  optimality_check = (pb, state) -> state.res,
)
s1 = GenericState(zeros(size(A, 2)), similar(b))
@time stp1 = LAStopping(pb, meta1, cheap_stop_remote_control(), s1)
@time StopRandomizedCD(stp1)

###############################################################################
#
# Stopping version of the algorithm with cheap remote control
#
###############################################################################
@time meta = StoppingMeta(
  max_cntrs = mcnt,
  atol = 1e-7,
  rtol = 1e-15,
  max_iter = 100,
  retol = false,
  tol_check = (atol, rtol, opt0) -> (atol + rtol * opt0),
  optimality_check = (pb, state) -> state.res,
)
s0 = GenericState(zeros(size(A, 2)), similar(b))
@time stp = LAStopping(pb, meta, StopRemoteControl(), s0)
@time StopRandomizedCD(stp)

###############################################################################
#
# Stopping with memory optimization using the State and with cheap_stop!
# uses the @ macro instate.
#
#DOESN'T WORK ??
#
###############################################################################
#=
@time meta4 = StoppingMeta(max_cntrs = mcnt,
                        atol = 1e-7, rtol = 1e-15, max_iter = 100,
                        retol = false,
                        tol_check = (atol, rtol, opt0)->(atol + rtol * opt0),
                        optimality_check = (pb, state) -> state.res)
s4   = GenericState(zeros(size(A,2)), similar(b))
@time stp4 = LAStopping(pb, meta4, s4)
@time stp4 = StopRandomizedCD3(stp4)
=#

Lnrm = [
  norm(stp.current_state.current_score),
  norm(stp1.current_state.current_score),
  norm(stp2.current_state.current_score),
  norm(stp3.current_state.current_score), #norm(stp4.current_state.current_score),
  norm(b - A * x),
]

using Test
@test Lnrm ≈ minimum(Lnrm) * ones(length(Lnrm)) atol = 1e-7

nothing;
