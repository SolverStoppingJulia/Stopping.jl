###############################################################################
#
# The data used through the algorithmic process in the Stopping framework
# are stored in a State.
# We illustrate here the NLPAtX which is a specialization of the State for
# non-linear programming.
#
###############################################################################
#using Test, NLPModels, Stopping

include("../test-stopping/rosenbrock.jl")
#Formulate the problem with NLPModels
x0 = ones(6)
y0 = ones(1)
c(x) = [x[1] - x[2]]
lcon = [0.0]
ucon = [0.0]

#We can create a NLPAtX for constrained optimization.
#Here we provide y0 = [1.0]
#Note that the default value is [0.0]
nlp = ADNLPModel(x->rosenbrock(x), x0, y0 = y0,
                 c=c, lcon=lcon, ucon=ucon,
                 lvar=zeros(6), uvar = Inf * ones(6))
#We can create a NLPAtX for bounds-constrained optimization:
nlp2 = ADNLPModel(x->rosenbrock(x), x0,
                 lvar=zeros(6), uvar = Inf * ones(6))
#We can create a NLPAtX for unconstrained optimization:
nlp3 = ADNLPModel(x->rosenbrock(x), x0)

###############################################################################
#I. Initialize a NLPAtX:
#
#There are two main constructor for the States.
#The unconstrained:
state_unc = NLPAtX(x0)
#The constrained:
state_con = NLPAtX(x0, y0)

#By default, all the values in the State are set to nothing except x and lambda
#In the unconstrained case lambda is a vector of length 0
@test !(state_unc.lambda == nothing)
#From the default initialization, all the other entries are void:
@test state_unc.mu == nothing && state_con.mu == nothing
@test state_unc.fx == nothing && state_con.fx == nothing
#exception is the counters which is initialized as a default Counters:
@test (sum_counters(state_unc.evals) + sum_counters(state_con.evals)) == 0

#Note that the constructor proceeds to a size checking on gx, Hx, mu, cx, Jx.
#It returns an error if this test fails.
try
  NLPAtX(x0, Jx = ones(1,1))
  @test false
catch
  #printstyled("NLPAtX(x0, Jx = ones(1,1)) is invalid as length(lambda)=0\n")
  @test true
end

###############################################################################
#II. Update the entries
#
#At the creation of a NLPAtX, keyword arguments populate the state:
state_bnd = NLPAtX(x0, mu = zeros(6))
@test state_bnd.mu == zeros(6) #initialize multipliers with bounds constraints

#The NLPAtX has two functions: update! and reinit!
#The update! has the same behavior as in the GenericState:
update!(state_bnd, fx = 1.0, blah = 1) #update! ignores unnecessary keywords
@test state_bnd.mu == zeros(6) && state_bnd.fx == 1.0 && state_bnd.x == x0

#reinit! by default reuse x and lambda and reset all the entries at their
#default values (void or empty Counters):
reinit!(state_bnd, mu = ones(6))
@test state_bnd.mu == ones(6) && state_bnd.fx == nothing
@test state_bnd.x == x0 && state_bnd.lambda == zeros(0)
#Trying to inherit reinit!(AbstractState, Vector) would not work here as
#lambda is a mandatory entry.
try
 reinit!(state_bnd, 2 * ones(6))
 @test false
catch
 @test true
end
#However, we can specify both entries
reinit!(state_bnd, 2 * ones(6), zeros(0))
@test state_bnd.x == 2*ones(6) && state_bnd.lambda == zeros(0)
@test state_bnd.mu == nothing && sum_counters(state_bnd.evals) == 0
#Giving a new Counters update as well:
test = Counters(); setfield!(test, :neval_obj, 102)
reinit!(state_bnd, evals = test)
@test getfield(state_bnd.evals, :neval_obj) == 102
@test sum_counters(state_bnd.evals) - 102 == 0

###############################################################################
#III. Domain Error
#Similar to the GenericState we can use _domain_check to verify there are no NaN
@test Stopping._domain_check(state_bnd) == false
update!(state_bnd, fx = NaN)
@test Stopping._domain_check(state_bnd) == true

###############################################################################
#IV. Use the NLPAtX
#
#For algorithmic use, it might be conveninent to fill in all the entries of then
#State. In this case, we can use the Stopping:
stop = NLPStopping(nlp, (x,y) -> unconstrained_check(x,y), state_unc)
#Note that the fill_in! can receive known informations via keywords.
#If we don't want to store the hessian matrix, we turn the keyword
#matrix_info as false.
fill_in!(stop, x0, matrix_info = false)

#printstyled("Hx has not been updated: ",stop.current_state.Hx == nothing,"\n")
@test stop.current_state.Hx == nothing

# We can now use the updated step in the algorithmic procedure
@test start!(stop) #return true
