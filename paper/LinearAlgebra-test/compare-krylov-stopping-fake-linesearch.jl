using Krylov, Stopping

include("random_coordinate_descent_method.jl")
include("stop_random_coordinate_descent_method.jl")
include("stop_random_coordinate_descent_method_LS.jl")

n=5000;
A=rand(n,n)
sol=rand(n)
b=A*sol

#@time stp = StopRandomizedCD(A,b, max_iter = 1000)

x0   = zeros(size(A,2))
pb   = LinearSystem(A,b)
s0   = GenericState(x0, similar(b))
mcnt = Stopping._init_max_counters_linear_operators()
#
@time meta = StoppingMeta(max_cntrs = mcnt,
                        atol = 1e-7, rtol = 1e-15, max_iter = 99,
                        retol = false,
                        tol_check = (atol, rtol, opt0)->(atol + rtol * opt0),
                        optimality_check = (pb, state) -> state.res)
s0   = GenericState(x0, similar(b))
@time stp = LAStopping(pb, meta, cheap_stop_remote_control(), s0)
@time StopRandomizedCD(stp)
                        
@time x, OK = RandomizedCD(A,b, max_iter = 100)

###############################################################################
#
# We compare the stopping_randomized_CD with the same version with an 
# artificial substopping created and called at each iteration.
# It appears that the expansive part is to create the SubStopping, and in 
# particular create the StoppingMeta.
#
@time meta = StoppingMeta(max_cntrs = mcnt,
                        atol = 1e-7, rtol = 1e-15, max_iter = 99,
                        retol = false,
                        tol_check = (atol, rtol, opt0)->(atol + rtol * opt0),
                        optimality_check = (pb, state) -> state.res)
s0   = GenericState(x0, similar(b))
@time stp2 = LAStopping(pb, meta, cheap_stop_remote_control(), s0)
@time StopRandomizedCD_LS(stp2)

;