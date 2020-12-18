using Krylov, Main.Stopping

#https://github.com/JuliaSmoothOptimizers/Krylov.jl

include("random_coordinate_descent_method.jl")
include("stop_random_coordinate_descent_method.jl")
include("cheap_stop_random_coordinate_descent_method.jl")
include("instate_coordinate_descent_method.jl")

using ProfileView, BenchmarkTools

n=5000;
A=rand(n,n)
sol=rand(n)
b=A*sol

#@time stp = StopRandomizedCD(A,b, max_iter = 1000)

x0   = zeros(size(A,2))
pb   = LinearSystem(A,b)
s0   = GenericState(x0, similar(b))
mcnt = Main.Stopping._init_max_counters_linear_operators()
#
@time meta = StoppingMeta(max_cntrs = mcnt,
                        atol = 1e-7, rtol = 1e-15, max_iter = 99,
                        retol = false,
                        tol_check = (atol, rtol, opt0)->(atol + rtol * opt0),
                        optimality_check = (pb, state) -> state.res)

#@time stp4 = LAStopping(pb, meta, s0)
#@time StopRandomizedCD3(stp4)

#@time stp3 = LAStopping(pb, meta, s0)#
#=
@time stp3 = LAStopping(pb, s0,
                 max_cntrs = mcnt,
                 atol = 1e-7, rtol = 1e-15, max_iter = 1000,
                 retol = false,
                 tol_check = (atol, rtol, opt0)->(atol + rtol * opt0),
                 optimality_check = (pb, state) -> state.res)
=#
#@time StopRandomizedCD2(stp3)
#Before the loop: 0.119287 seconds (214.95 k allocations: 11.009 MiB)
#Overall: 0.197434 seconds (218.95 k allocations: 22.960 MiB)

@time meta = StoppingMeta(max_cntrs = mcnt,
                        atol = 1e-7, rtol = 1e-15, max_iter = 99,
                        retol = false,
                        tol_check = (atol, rtol, opt0)->(atol + rtol * opt0),
                        optimality_check = (pb, state) -> state.res)
s0   = GenericState(x0, similar(b))
@time stp2 = LAStopping(pb, meta, s0)
@time StopRandomizedCD(stp2)

@time meta = StoppingMeta(max_cntrs = mcnt,
                        atol = 1e-7, rtol = 1e-15, max_iter = 99,
                        retol = false,
                        tol_check = (atol, rtol, opt0)->(atol + rtol * opt0),
                        optimality_check = (pb, state) -> state.res,
                        stop_remote = Main.Stopping.cheap_stop_remote_control())
s0   = GenericState(x0, similar(b))
@time stp = LAStopping(pb, meta, s0)
@time StopRandomizedCD(stp)


@time x, OK = RandomizedCD(A,b, max_iter = 100)

#@btime cheap_stop!(stp3);
#  187.915 ns (0 allocations: 0 bytes)

#@code_warntype stp.meta.optimality_check(stp.pb, stp.current_state)

#@btime LinearSystem(A,b); #66.622 ns (2 allocations: 80 bytes)
#@btime Main.Stopping._init_max_counters_linear_operators(); #213.791 ns (5 allocations: 752 bytes)
#@btime GenericState(x0, similar(b)); #420.106 ns (4 allocations: 4.27 KiB)
#@btime GenericState(zeros(size(A,2)), similar(b)); #1.100 μs (5 allocations: 8.33 KiB)
#265.971 ns (8 allocations: 960 bytes)
#@btime StoppingMeta(;tol_check = (atol, rtol, opt0)->(atol + rtol * opt0), optimality_check = (pb, state) -> state.res, max_cntrs = Main.Stopping._init_max_counters_linear_operators(), atol = 1e-7, rtol = 1e-15, max_iter = 1000, retol = false);

#=
Tanj: Dec. 1st: Why is there such a difference?
      2nd measure is after removing the double Meta creation
      3rd measure is after the new constructors
      
@btime LAStopping(pb, s0,
                        max_cntrs = Main.Stopping._init_max_counters_linear_operators(),
                        atol = 1e-7, rtol = 1e-15, max_iter = 1000,
                        retol = false,
                        tol_check = (atol, rtol, opt0)->(atol + rtol * opt0),
                        optimality_check = (pb, state) -> state.res);
  2.979 μs (42 allocations: 9.95 KiB)
  712.807 ns (13 allocations: 1.22 KiB)
  663.631 ns (12 allocations: 1.20 KiB)
  
  @btime LAStopping(pb, s0);
  1.401 μs (19 allocations: 4.70 KiB)
  312.332 ns (7 allocations: 992 bytes)
  306.650 ns (7 allocations: 992 bytes)
  
  @btime LAStopping(pb, s0, max_cntrs = Main.Stopping._init_max_counters_linear_operators());
  3.000 μs (43 allocations: 9.94 KiB)
  329.270 ns (8 allocations: 1008 bytes)
  332.384 ns (8 allocations: 1008 bytes)
  
  @btime c = Main.Stopping._init_max_counters_linear_operators();
  212.869 ns (5 allocations: 752 bytes)
  @btime LAStopping(pb, s0, max_cntrs = c);
  2.965 μs (39 allocations: 9.22 KiB)
  228.987 ns (4 allocations: 272 bytes)
  235.954 ns (4 allocations: 272 bytes)

=#

nothing;
