using Krylov, Main.Stopping

#https://github.com/JuliaSmoothOptimizers/Krylov.jl

include("random_coordinate_descent_method.jl")
include("stop_random_coordinate_descent_method.jl")
include("cheap_stop_random_coordinate_descent_method.jl")

using ProfileView, BenchmarkTools

n=6000;
A=rand(n,n)
sol=rand(n)
b=A*sol

#@time stp = StopRandomizedCD(A,b, max_iter = 1000)

#@time stp2 = StopRandomizedCD2(A,b, max_iter = 1000)

x0   = zeros(size(A,2))
@time stp3 = LAStopping(LinearSystem(A,b),
                 GenericState(x0, similar(b)),
                 max_cntrs = Main.Stopping._init_max_counters_linear_operators(),
                 atol = 1e-7, rtol = 1e-15, max_iter = 1000,
                 retol = false,
                 tol_check = (atol, rtol, opt0)->(atol + rtol * opt0),
                 optimality_check = (pb, state) -> state.res)
@time StopRandomizedCD2(stp3)
#Before the loop: 0.119287 seconds (214.95 k allocations: 11.009 MiB)
#Overall: 0.197434 seconds (218.95 k allocations: 22.960 MiB)

@time x, OK = RandomizedCD(A,b, max_iter = 1000)

#@btime cheap_stop!(stp3);
#  187.915 ns (0 allocations: 0 bytes)

#@code_warntype stp.meta.optimality_check(stp.pb, stp.current_state)

#@btime LinearSystem(A,b); #66.622 ns (2 allocations: 80 bytes)
#@btime Main.Stopping._init_max_counters_linear_operators(); #213.791 ns (5 allocations: 752 bytes)
#@btime GenericState(x0, similar(b)); #420.106 ns (4 allocations: 4.27 KiB)
#@btime GenericState(zeros(size(A,2)), similar(b)); #1.100 μs (5 allocations: 8.33 KiB)
#265.971 ns (8 allocations: 960 bytes)
#@btime StoppingMeta(;tol_check = (atol, rtol, opt0)->(atol + rtol * opt0), optimality_check = (pb, state) -> state.res, max_cntrs = Main.Stopping._init_max_counters_linear_operators(), atol = 1e-7, rtol = 1e-15, max_iter = 1000, retol = false);
nothing;
