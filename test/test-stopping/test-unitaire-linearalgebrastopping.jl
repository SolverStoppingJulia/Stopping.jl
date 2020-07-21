###############################################################################
#
# The Stopping structure eases the implementation of algorithms and the
# stopping criterion.
#
# The following examples illustrate a specialized Stopping for the
# LinearOperators type.
# https://github.com/JuliaSmoothOptimizers/LinearOperators.jl
#
using LinearAlgebra, SparseArrays, LinearOperators #v.1.0.1

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
mLOstp = LAStopping(mLO, GenericState(x0))
sLOstp = LAStopping(sLO, GenericState(x0))
maxcn = Stopping._init_max_counters_linear_operators(nprod = 1)
opLOstp = LAStopping(opLO, GenericState(x0), max_cntrs = maxcn)
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

la_stop = LAStopping(LinearSystem(A, b), GenericState(x0), max_iter = 150000, rtol = 1e-6)
#Be careful using GenericState(x0) would not work here without forcing convert = true
#in the update function. As the iterate will be an SparseVector to the contrary of initial guess.
#Tangi: maybe start! should send a Warning for such problem !?
sa_stop = LAStopping(LinearSystem(sparse(A), b), GenericState(sparse(x0)), max_iter = 150000, rtol = 1e-6)
op_stop = LAStopping(LinearSystem(LinearOperator(A), b), GenericState(x0), max_iter = 150000, rtol = 1e-6)

@time RandomizedBlockKaczmarz(la_stop)
@test status(la_stop) == :Optimal
@time RandomizedBlockKaczmarz(sa_stop)
@test status(sa_stop) == :Optimal
#No compatible dot product with LinearOperators.
#@time RandomizedBlockKaczmarz(op_stop)
#@test status(op_stop) == :Optimal
