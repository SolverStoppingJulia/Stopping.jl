using NLPModels
using State
using Stopping

# We create a simple function to test
A = rand(5, 5);
Q = A' * A

f(x) = 0.5 * x' * Q * x
x0 = ones(5)
c(x) = sum(x)
nlp = ADNLPModel(f,  x0,
                 lvar = fill(-10.0,size(x0)), uvar = fill(10.0,size(x0)),
                 y0 = [0.0], c = c, lcon = [-Inf], ucon = [1])
@show "Warning: initialize a constrained State"
nlp_at_x = NLPAtX(ones(5))
@show "Warning: KKT is not working properly"
stop_nlp = NLPStopping(nlp, (x,y) -> Stopping.KKT(x,y), nlp_at_x)

function penalty(stp :: NLPStopping)

#algorithm parameters
 rho, rho_min = 100.0, 1e-7

#start!(stp)

#prepare the subproblem stopping:
# sub_nlp_at_x = NLPAtX(ones(5))
# sub_stp = NLPStopping(nlp, (x,y) -> Stopping.unconstrained(x,y), sub_nlp_at_x)
#main loop
 while !OK
  #solve the subproblem
  #TODO
  # newton(sub_stp)

  #update the penalty parameter if necessary
  rho = rho / 2

  #check
  #update!(stp)
  #OK = stop!(stp)
  OK  = rho < rho_min
 end

 return stp
end
