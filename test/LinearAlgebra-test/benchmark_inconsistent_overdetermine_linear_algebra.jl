###############################################################################
#
# Benchmark of methods to solve Ax = b with A an m x n matrix.
# A is generated randomly
# b = A * xref
# x0 = 0
#
# Improvements:
#  * specific symmetric positive definite
#  * use matrices from MatrixDepot
#
###############################################################################
using Main.Stopping
using LinearAlgebra, Krylov, LinearOperators, SparseArrays, Main.Stopping

include("random_coordinate_descent_methods.jl")

#https://juliasmoothoptimizers.github.io/SolverBenchmark.jl/latest/tutorial/
#In this tutorial we illustrate the main uses of SolverBenchmark.
using DataFrames, Printf, SolverBenchmark

printstyled("Benchmark linear algebra solvers: \n", color = :green)

N = 3 #number of problems
mi, m, n = 1000, 2000, 100 #size of A: m x n

#Names of solvers:
names = [:Juliainv, :Krylov_cgls, :Krylov_cgne, :PLSSRRand, :RandomizedBlockKaczmarz] #:Krylov_minres,

#Initialization of the DataFrame for n problems.
stats = Dict(name => DataFrame(
         :id     => 1:N,
         :nvar   => zeros(Int64, N),
         :status => [:Unknown for i = 1:N],
         :time   => NaN*ones(N),
         :iter   => zeros(Int64, N),
         :score  => NaN*ones(N)) for name in names)

for i=1:N

  Ai = 100 * rand(m, n)
  xref = 100 * rand(n)
  bi = Ai * xref
  A  = vcat(Ai, -Ai)
  b  = vcat(bi,  bi)

  x0 = zeros(size(A,2))
  la_stop = LinearOperatorStopping(LinearSystem(A, b),
                                   linear_system_check!,
                                   GenericState(x0),
                                   max_iter = 10000,
                                   rtol = sqrt(eps()),
                                   atol = sqrt(eps()))
  @show n, m, cond(A), det(A' * A)
  for name in names

    #solve the problem
    reinit!(la_stop, rstate = true, x = x0)
    la_stop.meta.start_time = time()
    @time eval(name)(la_stop)
    sol = la_stop.current_state.x

    #update the stats from the Stopping
    stats[name].nvar[i]   = n
    stats[name].status[i] = status(la_stop)
    stop_has_time = (la_stop.current_state.current_time != nothing)
    stats[name].time[i]   =  stop_has_time ? la_stop.current_state.current_time - la_stop.meta.start_time : time() - la_stop.meta.start_time
    stats[name].iter[i]   = la_stop.meta.nb_of_stop
    stats[name].score[i]  = la_stop.current_state.current_score

  end

end #end of main loop

for name in names
    @show name
    @show stats[name]
end

#You can export the table in Latex
#latex_table(stdout, stats[:armijo])

#or run a performance profile:
#using Plots
#pyplot()
#cost(df) = (df.status != :Optimal) * Inf + df.t
#p = performance_profile(stats, cost)
#Plots.svg(p, "profile2")

#or a profile wall:
#solved(df) = (def.status .== :Optimal)
#costs = [df -> .!sovled(df) * Inf + df.t, df -> .!sovled(df) * Inf + df.iter]
#costnames = ["Time", "Iterations"]
#p = profile_solvers(stats, costs, costnames)
#Plots.svg(p, "profile3")

printstyled("The End.", color = :green)
