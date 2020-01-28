###############################################################################
#
# The data used through the algorithmic process in the Stopping framework
# are stored in a State.
# We illustrate here the NLPAtX which is a specialization of the State for
# non-linear programming.
#
###############################################################################

using NLPModels
using Stopping

include("../test-stopping/rosenbrock.jl")
#Formulate the problem with NLPModels
x0 = ones(6)
y0 = ones(1)
c(x) = [x[1] - x[2]]
lcon = [0.0]
ucon = [0.0]

#Here we provide y0 = [1.0]
#Note that the default value is [0.0]
nlp = ADNLPModel(x->rosenbrock(x), x0, y0 = y0,
                 c=c, lcon=lcon, ucon=ucon,
                 lvar=zeros(6), uvar = Inf * ones(6))
nlp2 = ADNLPModel(x->rosenbrock(x), x0,
                 lvar=zeros(6), uvar = Inf * ones(6))

fx = rosenbrock(x0)

#There are two main constructor for the States:
#The unconstrained:
state_unc = NLPAtX(x0)
#The constrained:
state_con = NLPAtX(x0, y0)

#By default, all the values in the State are set to nothing except x and lambda
#In the unconstrained case lambda is a vector of length 0
printstyled("Is lambda void? ",state_unc.lambda == nothing,"\n")
printstyled("Is fx void? ",state_unc.fx == nothing,"\n")
#Apart from x and lambda, the counters are also initialized by default
printstyled("Is counter void? ", state_unc.evals  == nothing,"\n")

#Note that the constructor proceeds to a size checking on gx, Hx, mu, cx, Jx.
#It returns an error if this test fails.
try
  NLPAtX(x0, Jx = ones(1,1))
catch
  printstyled("NLPAtX(x0, Jx = ones(1,1)) is invalid as length(lambda)=0\n")
end

#For algorithmic use, it might be conveninent to fill in all the entries of then
#State. In this case, we can use the Stopping:
stop = NLPStopping(nlp, (x,y) -> Stopping.unconstrained(x,y), state_unc)
#Note that the fill_in! can receive known informations via keywords.
#If we don't want to store the hessian matrix, we turn the keyword
#matrix_info as false.
fill_in!(stop, x0, matrix_info = false)

printstyled("Hx has not been updated: ",stop.current_state.Hx == nothing,"\n")

# We can now use the updated step in the algorithmic procedure
start!(stop)
