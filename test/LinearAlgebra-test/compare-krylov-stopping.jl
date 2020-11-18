using Krylov, Main.Stopping

#https://github.com/JuliaSmoothOptimizers/Krylov.jl

include("random_coordinate_descent_method.jl")
include("stop_random_coordinate_descent_method.jl")
include("cheap_stop_random_coordinate_descent_method.jl")

using ProfileView, BenchmarkTools

n=100;
A=rand(n,n)
sol=rand(n)
b=A*sol

mutable struct FunctionType{T} <: Function
    f :: Function
end

@time x, OK = RandomizedCD(A,b, max_iter = 1000)

@time stp = StopRandomizedCD(A,b, max_iter = 1000)

@time stp2 = StopRandomizedCD2(A,b, max_iter = 1000)

x0   = zeros(size(A,2))
@time stp3 = LAStopping(LinearSystem(A,b),
                 GenericState(x0),
                 max_cntrs = Main.Stopping._init_max_counters_linear_operators(),
                 atol = 1e-7, rtol = 1e-15, max_iter = 1000,
                 tol_check = (atol, rtol, opt0)->(atol + rtol * opt0),
                 optimality_check = (pb, state) -> state.res,
                 mtype = typeof(x0))
@time StopRandomizedCD2(stp3)
#@code_warntype Main.Stopping._tired_check!(stp, x, time_t = 1.)

#@code_warntype Main.Stopping._optimality_check(stp)

#@code_warntype Main.Stopping._null_test(stp, stp.current_state.current_score)

@profview StopRandomizedCD2(A,b, max_iter = 1000)

nothing;
