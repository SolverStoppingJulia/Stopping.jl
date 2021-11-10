###############################################################################
#
# # ListofStates tutorial : 1/2
#
# We illustrate here the use of ListofStates in dealing with a warm start
# procedure.
#
# ListofStates can also prove the user history over the iteration process.
#
###############################################################################

using NLPModels, Stopping, Test

# Random search in [0,1] of the global minimum for unconstrained optimization
function algo_rand(stp::NLPStopping)
  x0 = stp.current_state.x
  n = length(x0)
  OK = start!(stp)

  while !OK
    x = rand(n)
    OK = update_and_stop!(stp, x = x, fx = obj(nlp, x), gx = grad(nlp, x))
  end

  return stp
end

include("../test-stopping/rosenbrock.jl")
x0 = 1.5 * ones(6)
nlp = ADNLPModel(rosenbrock, x0, zeros(6), ones(6))

state = NLPAtX(x0)
stop_lstt = NLPStopping(
  nlp,
  state,
  list = ListofStates(state),
  max_iter = 10,
  optimality_check = optim_check_bounded,
)
algo_rand(stop_lstt)
print(stop_lstt.listofstates, print_sym = [:fx, :x])
@test length(stop_lstt.listofstates.list) == 12

#Note the difference if the length of the ListofStates is limited
reinit!(stop_lstt, rstate = true, x = x0)
stop_lstt.listofstates = ListofStates(state, n = 5)
algo_rand(stop_lstt)
print(stop_lstt.listofstates, print_sym = [:fx, :x])
@test length(stop_lstt.listofstates.list) == 5

#Take the best out of 5:
bestfx, best = findmax([stop_lstt.listofstates[i].fx for i = 1:length(stop_lstt.listofstates)])
best_state = copy(stop_lstt.listofstates[best])
reinit!(stop_lstt)
stop_lstt.current_state = best_state
stop_lstt.listofstates = ListofStates(best_state, n = 5)
algo_rand(stop_lstt)
print(stop_lstt.listofstates, print_sym = [:fx, :x])
@test length(stop_lstt.listofstates.list) == 5
