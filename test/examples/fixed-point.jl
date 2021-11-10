###############################################################################
#
# Stopping can also be used for fixed point methods
# Example here concerns the AlternatingDirections Algorithm to find
# a feasible point in the intersection of 2 convex sets A and B.
# This algorithm relies on a fixed point argument, hence it stopped if it finds
# a fixed point.
#
# Example:
# A={ (x,y) | x=y} and B = {(x,y) | y=0}
# Clearly the unique intersection point is (0,0)
#
# Note that in this case the projection on A and the projection on B are trivial
#
# Takeaway: the 2nd scenario illustrates a situation where the algorithm stalls
# as it reached a personal success. (optimal_sub_pb is true)
#
###############################################################################
using LinearAlgebra, NLPModels, Stopping, Test

#Main algorithm
function AlternatingDirections(stp)
  xk = stp.current_state.x
  OK = update_and_start!(stp, cx = cons(stp.pb, x0))
  @show OK, xk

  while !OK

    #First projection
    xk1 = 0.5 * (xk[1] + xk[2]) * ones(2)
    #Second projection
    xk2 = [xk1[1], 0.0]

    #check if we have a fixed point
    Fix = dot(xk - xk2, xk - xk2)
    if Fix <= min(eps(Float64), stp.meta.atol)
      stp.meta.suboptimal = true
    end
    #call the stopping
    OK = update_and_stop!(stp, x = xk2, cx = cons(stp.pb, xk2))

    xk = xk2
    @show OK, xk
  end

  return stp
end

# We model the problem using the NLPModels without objective function
#Formulate the problem with NLPModels
c(x) = [x[1] - x[2], x[2]]
lcon = [0.0, 0.0]
ucon = [0.0, 0.0]
nlp = ADNLPModel(x -> 0.0, zeros(2), c = c, lcon = lcon, ucon = ucon)

#1st scenario: we solve the problem
printstyled("1st scenario:\n")
#Prepare the Stopping
x0 = [0.0, 5.0]
state = NLPAtX(x0)
#Recall that for the optimality_check function x is the pb and y is the state
#Here we take the infinite norm of the residual.
stop = NLPStopping(nlp, state, optimality_check = (x, y) -> norm(y.cx, Inf))

AlternatingDirections(stop)
@show status(stop)
@test status(stop) == :Optimal

#2nd scenario: the user gives an irrealistic optimality condition
printstyled("2nd scenario:\n")
reinit!(stop, rstate = true, x = x0)
stop.meta.optimality_check = (x, y) -> norm(y.cx, Inf) + 0.5

AlternatingDirections(stop)
#In this scenario, the algorithm stops because it attains a fixed point
#Hence, status is :SubOptimal.
@show status(stop)
@test status(stop) == :SubOptimal
